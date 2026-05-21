"""Step-batched chain execution engine backed by CARL.

Architecture
------------
**Step-batched execution** — all *N* samples advance through step *k* together
before any sample starts step *k+1*.  This yields homogeneous LLM request
batches (same prompt structure, similar length) which vLLM can batch far more
efficiently than per-sample execution.

**CARL integration**

- ``mmar_carl.ReasoningContext`` — per-sample state object (outer context,
  history, system prompt, tool registry).
- ``GigaEvoPromptTemplate`` (from ``carl_bridge``) — prompt assembly using
  CARL's ``PromptTemplate`` with gigaevo-specific English templates.
- ``GigaEvoClientAdapter`` (from ``carl_bridge``) — wraps gigaevo's callable
  LLM client as CARL's ``LLMClientBase``.  Each sample gets its own copy via
  ``client.copy()`` to ensure thread-safe concurrent calls.
- ``mmar_carl.LLMStepDescription`` / ``ToolStepDescription`` — CARL execution
  types produced by ``validate_chain_spec`` and consumed here.

**What stays gigaevo-specific**

- ``_strip_thinking`` — strips ``<think>…</think>`` blocks from LLM outputs
  before they enter history or tool $-references.
- ``_resolve_reference`` / ``_resolve_dependencies`` — raw step-output
  indexing for tool step ``input_mapping`` resolution.  CARL's
  ``resolve_context_reference`` maps ``$history[N]`` to formatted history
  entries; gigaevo's tool steps need the *raw* output (e.g. a bare answer
  string for BM25).  Keeping the original resolver preserves exact backward
  semantics.
- ``step_max_tokens`` per-step override — passed directly to the underlying
  client call as a keyword argument, bypassing CARL's per-step ``llm_config``
  mechanism (which requires an OpenAI-compatible client).

Public API (unchanged signatures)
----------------------------------
``run_chain_on_dataset``            — legacy single-registry sync wrapper.
``run_chain_on_dataset_stepwise``   — production sync wrapper with batch-tool
                                      registry + per-step max_tokens.
"""

import asyncio
from collections.abc import Callable
import re
import sys
import time

from mmar_carl import LLMStepDescription, ToolStepDescription
from mmar_carl.models import Language, ReasoningContext

from problems.chains.carl_bridge import GigaEvoClientAdapter, GigaEvoPromptTemplate
from problems.chains.runner_config import FeedbackMode, RunnerConfig, StepExecutionMode
from problems.chains.types import ChainResult, ChainSpec

# ---------------------------------------------------------------------------
# Thinking-mode stripping
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Strip ``<think>…</think>`` blocks from LLM thinking-mode output.

    Must be applied to all LLM step outputs before they are stored in
    ``step_outputs`` or formatted into history, so that:

    - BM25 queries resolved via ``$history[-1]`` are not polluted with
      reasoning traces.
    - Subsequent LLM steps receive clean factual context rather than the
      model's internal monologue.

    Handles two cases:

    - Well-formed: ``<think>…</think>`` — stripped by first sub.
    - Truncated (max_tokens cutoff mid-block): ``<think>…`` (no closing tag)
      — stripped by second sub, which removes from ``<think>`` to
      end-of-string.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# $-reference resolution for tool steps
# ---------------------------------------------------------------------------


def _resolve_reference(
    ref: str,
    outer_context: str,
    step_outputs: list[str],
    sample: dict | None = None,
) -> str:
    """Resolve a ``$``-reference to a concrete value.

    Used exclusively for tool step ``input_mapping`` resolution.  Resolves
    against *raw* step outputs (not formatted history entries) so that BM25
    and other tools receive the bare answer text, not the
    ``"Step N. Title\\nResult: …\\n"`` wrapper.

    Supported syntax::

        $outer_context    — original sample context string
        $history[-1]      — last completed step's raw output
        $history[N]       — step raw output at position N (0-based)
        $sample.foo       — field from the current sample (supports dot path)

    Args:
        ref: The ``$``-reference string from ``input_mapping``.
        outer_context: Sample context string.
        step_outputs: Raw outputs accumulated so far (0-indexed).

    Returns:
        Resolved string value.
    """
    if ref == "$outer_context":
        return outer_context

    if ref == "$history[-1]":
        return step_outputs[-1] if step_outputs else ""

    if ref.startswith("$sample."):
        if sample is None:
            return ""
        value: object = sample
        for part in ref[len("$sample.") :].split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return ""
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    match = re.match(r"\$history\[(\d+)\]", ref)
    if match:
        idx = int(match.group(1))
        return step_outputs[idx] if idx < len(step_outputs) else ""

    raise ValueError(f"Unknown reference syntax: {ref}")


def _resolve_dependencies(
    step_deps: list[int],
    history: list[str],
    step_outputs: list[str],
) -> tuple[list[str], dict[int, str]]:
    """Filter history and outputs to the steps a given step depends on.

    Used for LLM prompt building: a step should only see the history of steps
    it directly depends on (or all prior steps when ``step_deps`` is empty).

    Args:
        step_deps: 1-based step numbers this step depends on.  Empty = all prior.
        history: Full accumulated history entries.
        step_outputs: Full accumulated raw step outputs.

    Returns:
        ``(visible_history, visible_outputs)`` filtered by dependencies.
    """
    n_completed = len(step_outputs)

    if not step_deps:
        visible_history = history[:n_completed]
        visible_outputs = {i + 1: step_outputs[i] for i in range(n_completed)}
    else:
        visible_history = []
        visible_outputs = {}
        for dep in step_deps:
            idx = dep - 1  # 1-based → 0-based
            if 0 <= idx < n_completed:
                visible_history.append(history[idx])
                visible_outputs[dep] = step_outputs[idx]

    return visible_history, visible_outputs


# ---------------------------------------------------------------------------
# RunnerConfig helpers — feedback modes and execution modes
# ---------------------------------------------------------------------------


async def _call_llm_with_client(
    client,
    prompt: str,
    semaphore: asyncio.Semaphore,
    **kwargs: object,
) -> str:
    """Single LLM call with semaphore guard (creates a thread-safe client copy)."""
    async with semaphore:
        return await client.copy()(prompt, **kwargs)


async def _apply_self_critic(
    results: list[str],
    prompts: list[str],
    step,
    client,
    semaphore: asyncio.Semaphore,
    cfg: RunnerConfig,
    overrides: dict,
) -> list[str]:
    """SELF_CRITIC: evaluate each LLM output and retry rejected ones.

    Runs up to ``cfg.self_critic.max_revisions`` evaluate → regenerate cycles.
    Stops early if all outputs are APPROVE-d.
    """
    aim = getattr(step, "aim", "complete the task")
    for _ in range(cfg.self_critic.max_revisions):
        eval_prompts = [
            cfg.self_critic.evaluator_prompt_template.format(aim=aim, output=r)
            for r in results
        ]
        evals = list(
            await asyncio.gather(
                *[
                    _call_llm_with_client(client, p, semaphore, **overrides)
                    for p in eval_prompts
                ]
            )
        )
        rejected = [i for i, e in enumerate(evals) if "REJECT" in e.upper()]
        if not rejected:
            break
        retry_prompts = [
            prompts[i]
            + "\n\n"
            + cfg.self_critic.disapprove_feedback_template.format(
                reason=evals[i].split("REJECT:", 1)[-1].strip()
                if ":" in evals[i]
                else evals[i]
            )
            for i in rejected
        ]
        retried = list(
            await asyncio.gather(
                *[
                    _call_llm_with_client(client, p, semaphore, **overrides)
                    for p in retry_prompts
                ]
            )
        )
        for j, i in enumerate(rejected):
            results[i] = _strip_thinking(retried[j])
    return results


async def _apply_simple_retry(
    results: list[str],
    prompts: list[str],
    client,
    semaphore: asyncio.Semaphore,
    cfg: RunnerConfig,
    overrides: dict,
) -> list[str]:
    """SIMPLE: retry outputs that match any bad pattern up to max_retries times."""
    patterns = cfg.simple_retry.bad_patterns
    case_insensitive = not cfg.simple_retry.case_sensitive
    feedback = cfg.simple_retry.feedback_message
    for _ in range(cfg.simple_retry.max_retries):
        bad = [
            i
            for i, r in enumerate(results)
            if any(
                (p.lower() in r.lower() if case_insensitive else p in r)
                for p in patterns
            )
        ]
        if not bad:
            break
        retry_prompts = [prompts[i] + "\n\n" + feedback for i in bad]
        retried = list(
            await asyncio.gather(
                *[
                    _call_llm_with_client(client, p, semaphore, **overrides)
                    for p in retry_prompts
                ]
            )
        )
        for j, i in enumerate(bad):
            results[i] = _strip_thinking(retried[j])
    return results


async def _apply_metric_feedback(
    results: list[str],
    prompts: list[str],
    client,
    semaphore: asyncio.Semaphore,
    cfg: RunnerConfig,
    overrides: dict,
) -> list[str]:
    """METRICS: score outputs and retry those below threshold up to max_retries times."""
    metric_fn = cfg.metric_feedback.metric_fn
    min_len = cfg.metric_feedback.min_output_length
    if metric_fn is None:

        def metric_fn(outputs: list[str]) -> list[float]:
            return [1.0 if len(t.split()) >= min_len else 0.0 for t in outputs]

    feedback = cfg.metric_feedback.feedback_message
    threshold = cfg.metric_feedback.threshold
    for _ in range(cfg.metric_feedback.max_retries):
        scores = metric_fn(results)
        bad = [i for i, s in enumerate(scores) if s < threshold]
        if not bad:
            break
        retry_prompts = [prompts[i] + "\n\n" + feedback for i in bad]
        retried = list(
            await asyncio.gather(
                *[
                    _call_llm_with_client(client, p, semaphore, **overrides)
                    for p in retry_prompts
                ]
            )
        )
        for j, i in enumerate(bad):
            results[i] = _strip_thinking(retried[j])
    return results


# ---------------------------------------------------------------------------
# Async core — step-batched execution
# ---------------------------------------------------------------------------


async def _run_chain_on_dataset_stepwise(
    chain: ChainSpec,
    client,
    dataset: list[dict],
    outer_context_builder: Callable[[dict], str],
    tool_registry: dict[str, Callable] | None = None,
    batch_tool_registry: dict[str, Callable] | None = None,
    step_max_tokens: dict[int, int] | None = None,
    max_concurrent: int = 300,
    runner_config: RunnerConfig | None = None,
) -> list[ChainResult]:
    """Step-batched async core: all samples advance through each step together.

    CARL integration points
    -----------------------
    - One ``ReasoningContext`` per sample holds ``outer_context``, ``history``,
      and ``system_prompt``.  The ``api`` is a ``GigaEvoClientAdapter``
      wrapping a thread-safe copy of the gigaevo client.
    - Prompts are built with ``GigaEvoPromptTemplate`` (CARL's
      ``PromptTemplate`` subclass) which matches gigaevo's exact English
      template format.
    - History entries are appended to each context via
      ``ReasoningContext.add_to_history``.
    - Tool step ``$``-references are resolved from ``all_step_outputs`` (raw
      outputs), not from ``context.history`` (formatted entries), preserving
      the semantics existing chains depend on.

    Args:
        chain: Validated ``ChainSpec`` with CARL step types.
        client: Gigaevo LLM client (callable + ``.copy()``).
        dataset: List of sample dicts.
        outer_context_builder: Builds the data context string from a sample.
        tool_registry: ``{tool_name: fn(**kwargs) -> str}`` for per-sample tool
            calls (run concurrently via ``asyncio.to_thread``).
        batch_tool_registry: ``{tool_name: fn(list[dict]) -> list[str]}`` for
            vectorised batch tool calls (single call for all *N* samples).
        step_max_tokens: ``{step_number: max_tokens}`` overrides forwarded to
            the LLM client as keyword arguments.
        max_concurrent: Semaphore limit for parallel LLM calls per step.

    Returns:
        Ordered list of ``ChainResult`` (one per sample).
    """

    cfg: RunnerConfig = runner_config if runner_config is not None else RunnerConfig()

    def _log(msg: str) -> None:
        sys.__stderr__.write(f"[chain] {msg}\n")
        sys.__stderr__.flush()

    n = len(dataset)
    total_steps = len(chain.steps)
    prompt_template = GigaEvoPromptTemplate()
    semaphore = asyncio.Semaphore(max_concurrent)
    chain_t0 = time.time()

    # --- Per-sample state ---------------------------------------------------
    # ReasoningContext manages outer_context, history, and system_prompt.
    # Each context gets its own client copy for thread-safe concurrent calls.
    contexts: list[ReasoningContext] = [
        ReasoningContext(
            outer_context=outer_context_builder(sample),
            api=GigaEvoClientAdapter(client.copy()),
            system_prompt=chain.system_prompt,
            language=Language.ENGLISH,
        )
        for sample in dataset
    ]

    # Raw step outputs — used for tool $-reference resolution and ChainResult.
    # Separate from context.history which holds formatted history entries.
    all_step_outputs: list[list[str]] = [[] for _ in range(n)]

    # -----------------------------------------------------------------------
    for step_idx, step in enumerate(chain.steps):
        step_t0 = time.time()
        step_type = "tool" if isinstance(step, ToolStepDescription) else "llm"
        _log(
            f"step {step_idx + 1}/{total_steps} "
            f"({step_type}) '{step.title}' — {n} samples"
        )

        if isinstance(step, LLMStepDescription):
            # --- Build all prompts for this step ----------------------------
            # Use dependency-filtered history so each step only sees the
            # history entries of its declared dependencies.
            overrides: dict = {}
            if step_max_tokens and step.number in step_max_tokens:
                overrides["max_tokens"] = step_max_tokens[step.number]

            prompts: list[str] = []
            for i in range(n):
                visible_history, _ = _resolve_dependencies(
                    step.dependencies,
                    contexts[i].history,
                    all_step_outputs[i],
                )
                step_prompt = prompt_template.format_step_prompt(
                    step,
                    contexts[i].outer_context,
                    Language.ENGLISH,
                )
                full_prompt = prompt_template.format_chain_prompt(
                    outer_context=contexts[i].outer_context,
                    current_task=step_prompt,
                    history="\n".join(visible_history),
                    language=Language.ENGLISH,
                    system_prompt=chain.system_prompt,
                )
                prompts.append(full_prompt)

            # --- Fire all LLM calls concurrently (step-batched) -------------
            raw_results = list(
                await asyncio.gather(
                    *[
                        _call_llm_with_client(client, p, semaphore, **overrides)
                        for p in prompts
                    ]
                )
            )
            results = [_strip_thinking(r) for r in raw_results]

            # --- Apply execution mode and per-step feedback mode ------------
            if cfg.execution_mode == StepExecutionMode.SELF_CRITIC:
                results = await _apply_self_critic(
                    results, prompts, step, client, semaphore, cfg, overrides
                )
            if cfg.feedback_mode == FeedbackMode.SIMPLE:
                results = await _apply_simple_retry(
                    results, prompts, client, semaphore, cfg, overrides
                )
            elif cfg.feedback_mode == FeedbackMode.METRICS:
                results = await _apply_metric_feedback(
                    results, prompts, client, semaphore, cfg, overrides
                )

            # --- Update per-sample state ------------------------------------
            for i in range(n):
                all_step_outputs[i].append(results[i])
                contexts[i].add_to_history(
                    prompt_template.format_history_entry(
                        step.number, step.title, results[i]
                    )
                )

        elif isinstance(step, ToolStepDescription):
            if tool_registry is None and batch_tool_registry is None:
                raise ValueError(
                    f"Tool step {step.number} encountered but no tool registry provided"
                )

            tool_name = step.config.tool_name

            # Resolve $-references using raw step outputs (not context.history)
            # so that tools receive bare answer text, not formatted entries.
            all_resolved: list[dict[str, str]] = []
            for i in range(n):
                resolved: dict[str, str] = {
                    param: _resolve_reference(
                        ref, contexts[i].outer_context, all_step_outputs[i]
                    )
                    for param, ref in step.config.input_mapping.items()
                }
                all_resolved.append(resolved)

            if batch_tool_registry and tool_name in batch_tool_registry:
                # Vectorised: one call for all N samples (e.g. batch BM25).
                results = batch_tool_registry[tool_name](all_resolved)
            elif tool_registry and tool_name in tool_registry:
                # Per-sample: run concurrently in thread pool.
                results = list(
                    await asyncio.gather(
                        *[
                            asyncio.to_thread(tool_registry[tool_name], **kw)
                            for kw in all_resolved
                        ]
                    )
                )
            else:
                raise ValueError(
                    f"Tool '{tool_name}' not found in any registry. "
                    f"Available: tool={list(tool_registry or {})}, "
                    f"batch={list(batch_tool_registry or {})}"
                )

            for i in range(n):
                all_step_outputs[i].append(results[i])
                contexts[i].add_to_history(
                    prompt_template.format_history_entry(
                        step.number, step.title, results[i]
                    )
                )

        else:
            raise ValueError(f"Unknown step type: {type(step).__name__}")

        elapsed = time.time() - step_t0
        total_elapsed = time.time() - chain_t0
        remaining_steps = total_steps - step_idx - 1
        eta = (total_elapsed / (step_idx + 1)) * remaining_steps
        _log(
            f"step {step_idx + 1}/{total_steps} done in {elapsed:.1f}s "
            f"(total {total_elapsed:.1f}s, ETA ~{eta:.0f}s)"
        )

    # --- DATASET feedback (post-loop) ----------------------------------------
    # Applies only if the last chain step is an LLM step.
    # Two modes:
    #   (a) Constraint-based (IFBench): if samples have "instruction_id_list",
    #       run the real constraint checker and report violated constraints.
    #   (b) Answer-based (GSM8K, HoVer): check if expected answer appears in output.
    if cfg.feedback_mode == FeedbackMode.DATASET:
        last_step = chain.steps[-1] if chain.steps else None
        if isinstance(last_step, LLMStepDescription):
            answer_key = cfg.dataset_feedback.answer_key
            has_constraints = bool(dataset and "instruction_id_list" in dataset[0])
            # Preserve per-step max_tokens on the correction call so that the
            # retry uses the same budget as the original step.
            correction_overrides: dict = {}
            if step_max_tokens and last_step.number in step_max_tokens:
                correction_overrides["max_tokens"] = step_max_tokens[last_step.number]

            # ifbench constraint checker pulls in optional deps (emoji,
            # langdetect, ...). Import once up front (not once per retry) only
            # when actually needed.
            if has_constraints:
                from problems.chains.ifbench.utils.evaluation import (
                    get_violated_constraints,
                )

            for _ in range(cfg.dataset_feedback.max_retries):
                wrong: list[int] = []
                wrong_feedback: dict[int, str] = {}

                if has_constraints:
                    for i in range(n):
                        final = all_step_outputs[i][-1] if all_step_outputs[i] else ""
                        if not final.strip():
                            wrong.append(i)
                            wrong_feedback[i] = (
                                "Your response was empty. Provide a substantive "
                                "response that satisfies all constraints in the prompt."
                            )
                            continue
                        violated = get_violated_constraints(dataset[i], final)
                        if violated:
                            wrong.append(i)
                            viol_str = "; ".join(violated[:5])
                            wrong_feedback[i] = (
                                f"Your response violated {len(violated)} constraint(s): "
                                f"{viol_str}. "
                                "Rewrite your response to satisfy ALL constraints "
                                "specified in the prompt."
                            )
                else:
                    for i in range(n):
                        expected = str(dataset[i].get(answer_key, ""))
                        final = all_step_outputs[i][-1] if all_step_outputs[i] else ""
                        if expected and expected.lower() not in final.lower():
                            wrong.append(i)
                            actual = final[:200] if final else ""
                            wrong_feedback[i] = (
                                cfg.dataset_feedback.feedback_template.format(
                                    expected=expected, actual=actual
                                )
                            )

                if not wrong:
                    break

                correction_data: list[tuple[int, str]] = []
                for i in wrong:
                    feedback = wrong_feedback[i]
                    visible_history, _ = _resolve_dependencies(
                        last_step.dependencies,
                        contexts[i].history[:-1],
                        all_step_outputs[i][:-1],
                    )
                    step_prompt = prompt_template.format_step_prompt(
                        last_step, contexts[i].outer_context, Language.ENGLISH
                    )
                    correction_prompt = prompt_template.format_chain_prompt(
                        outer_context=contexts[i].outer_context,
                        current_task=step_prompt + "\n\n" + feedback,
                        history="\n".join(visible_history),
                        language=Language.ENGLISH,
                        system_prompt=chain.system_prompt,
                    )
                    correction_data.append((i, correction_prompt))
                corrected = list(
                    await asyncio.gather(
                        *[
                            _call_llm_with_client(
                                client, p, semaphore, **correction_overrides
                            )
                            for _, p in correction_data
                        ]
                    )
                )
                for (i, _), corrected_raw in zip(correction_data, corrected):
                    corrected_out = _strip_thinking(corrected_raw)
                    all_step_outputs[i][-1] = corrected_out
                    if contexts[i].history:
                        contexts[i].history[-1] = prompt_template.format_history_entry(
                            last_step.number, last_step.title, corrected_out
                        )

    return [
        ChainResult(
            history=contexts[i].history,
            final_output=all_step_outputs[i][-1] if all_step_outputs[i] else "",
            step_outputs=all_step_outputs[i],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Legacy async core (run_chain_on_dataset)
# ---------------------------------------------------------------------------


async def _run_chain_on_dataset_async(
    chain: ChainSpec,
    client,
    dataset: list[dict],
    outer_context_builder: Callable[[dict], str],
    tool_registry: dict[str, Callable] | None = None,
    max_concurrent: int = 300,
    runner_config: RunnerConfig | None = None,
) -> list[ChainResult]:
    """Async core for the legacy ``run_chain_on_dataset`` entry-point.

    Delegates to ``_run_chain_on_dataset_stepwise`` treating ``tool_registry``
    as a per-sample registry (no batch registry, no step_max_tokens).
    """
    return await _run_chain_on_dataset_stepwise(
        chain=chain,
        client=client,
        dataset=dataset,
        outer_context_builder=outer_context_builder,
        tool_registry=tool_registry,
        batch_tool_registry=None,
        step_max_tokens=None,
        max_concurrent=max_concurrent,
        runner_config=runner_config,
    )


# ---------------------------------------------------------------------------
# Public sync wrappers
# ---------------------------------------------------------------------------


def run_chain_on_dataset(
    chain: ChainSpec,
    client,
    dataset: list[dict],
    outer_context_builder: Callable[[dict], str],
    tool_registry: dict[str, Callable] | None = None,
    max_concurrent: int = 300,
    runner_config: RunnerConfig | None = None,
) -> list[ChainResult]:
    """Run chain on dataset using step-batched execution (sync wrapper).

    Legacy entry-point.  For new code prefer ``run_chain_on_dataset_stepwise``
    which supports batch tool registries and per-step max_tokens overrides.

    Args:
        chain: Validated ``ChainSpec``.
        client: Gigaevo LLM client.
        dataset: List of sample dicts.
        outer_context_builder: Builds data context string from a sample.
        tool_registry: ``{tool_name: fn(**kwargs) -> str}`` per-sample tools.
        max_concurrent: Max parallel LLM calls per step.
        runner_config: Optional feedback/execution mode config.  Defaults to
            ``RunnerConfig()`` (NONE feedback, FAST execution).

    Returns:
        Ordered list of ``ChainResult`` (one per sample).
    """
    return asyncio.run(
        _run_chain_on_dataset_async(
            chain,
            client,
            dataset,
            outer_context_builder,
            tool_registry,
            max_concurrent,
            runner_config,
        )
    )


def run_chain_on_dataset_stepwise(
    chain: ChainSpec,
    client,
    dataset: list[dict],
    outer_context_builder: Callable[[dict], str],
    tool_registry: dict[str, Callable] | None = None,
    batch_tool_registry: dict[str, Callable] | None = None,
    step_max_tokens: dict[int, int] | None = None,
    max_concurrent: int = 300,
    runner_config: RunnerConfig | None = None,
) -> list[ChainResult]:
    """Run chain on dataset using step-batched execution (sync wrapper).

    All *N* samples advance through step *k* together before any starts step
    *k+1*, producing homogeneous LLM request batches for efficient vLLM use.

    Args:
        chain: Validated ``ChainSpec`` with CARL step types.
        client: Gigaevo LLM client (callable + ``.copy()``).
        dataset: List of sample dicts.
        outer_context_builder: Builds data context string from a sample.
        tool_registry: ``{tool_name: fn(**kwargs) -> str}`` — called per
            sample via ``asyncio.to_thread``.
        batch_tool_registry: ``{tool_name: fn(list[dict]) -> list[str]}`` —
            called once with all *N* resolved-kwargs dicts.  Takes precedence
            over ``tool_registry`` when both register the same tool name.
        step_max_tokens: ``{step_number: max_tokens}`` forwarded to the client
            as ``max_tokens=`` keyword argument for that step only.
        max_concurrent: Semaphore limit for parallel LLM calls per step.
        runner_config: Optional feedback/execution mode config.  Defaults to
            ``RunnerConfig()`` (NONE feedback, FAST execution).

    Returns:
        Ordered list of ``ChainResult`` (one per sample).
    """
    return asyncio.run(
        _run_chain_on_dataset_stepwise(
            chain,
            client,
            dataset,
            outer_context_builder,
            tool_registry,
            batch_tool_registry,
            step_max_tokens,
            max_concurrent,
            runner_config,
        )
    )
