"""Cached worker-local Sudoku prompt evaluator for GigaEvo validators."""

from __future__ import annotations

import atexit
import os
from dataclasses import asdict, dataclass
from typing import Optional

from problems.prompts.sudoku.config import (
    MODEL_CONFIG,
    PROMPT_CONFIG,
    USER_PROMPT_TEMPLATE,
    VALIDATION_CONFIG,
    resolve_dataset_path,
)


def _load_env_defaults() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


_load_env_defaults()

_IMPORT_ERROR: Exception | None = None

try:
    from problems.prompts.sudoku.local_runtime.action_parser import BasicActionParser
    from problems.prompts.sudoku.local_runtime.dataset import GoldPath, SudokuAdapter
    from problems.prompts.sudoku.local_runtime.grid import Grid, SudokuSpec
    from problems.prompts.sudoku.local_runtime.models import (
        BacktrackAction,
        DoneAction,
        Node,
        NodeAction,
        PathContext,
    )
    from problems.prompts.sudoku.local_runtime.solver import (
        GenerationConfig,
        LocalVLLMSolver,
    )
    from problems.prompts.sudoku.local_runtime.validator import SudokuValidator
except Exception as exc:  # pragma: no cover - exercised only in missing-env setups
    _IMPORT_ERROR = exc


SUPPORTED_LAYOUTS = {"one_line", "rows", "rows_sep"}


def _raise_if_runtime_unavailable() -> None:
    if _IMPORT_ERROR is None:
        return

    raise RuntimeError(
        "Sudoku prompt validation requires the local Sudoku runtime bundled in "
        "this problem plus the Python dependencies for local vLLM inference. "
        "Make sure the current environment can import `torch`, `transformers`, "
        "and `vllm`."
    ) from _IMPORT_ERROR


@dataclass(frozen=True)
class PromptVariant:
    layout: str
    empty_symbol: str

    @property
    def key(self) -> str:
        symbol = self.empty_symbol
        if symbol == ".":
            symbol = "dot"
        elif symbol == "_":
            symbol = "underscore"
        elif symbol == " ":
            symbol = "space"
        return f"{self.layout}__empty_{symbol}"


@dataclass
class ExampleResult:
    example_id: int
    initial_grid: str
    target_grid: str
    success: bool
    steps: int
    backtracks: int
    solved_via: Optional[str]
    failure_reason: Optional[str]
    raw_last_output: Optional[str]
    parsed_last_grid: Optional[str]


def _next_node_id(nodes: list[Node]) -> int:
    for node in reversed(nodes):
        if isinstance(node.action, NodeAction):
            return node.action.node_id + 1
    return 0


def _build_node_id_to_index(nodes: list[Node]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for idx, node in enumerate(nodes):
        if isinstance(node.action, NodeAction):
            mapping[node.action.node_id] = idx
    return mapping


def _render_grid(canonical_grid: str, variant: PromptVariant) -> str:
    rendered = canonical_grid.replace(".", variant.empty_symbol)
    rows = [rendered[i : i + 4] for i in range(0, 16, 4)]

    if variant.layout == "one_line":
        return rendered
    if variant.layout == "rows":
        return "\n".join(rows)
    if variant.layout == "rows_sep":
        pretty_rows = [f"{row[0]} {row[1]} | {row[2]} {row[3]}" for row in rows]
        return "\n".join(
            [pretty_rows[0], pretty_rows[1], "----+----", pretty_rows[2], pretty_rows[3]]
        )

    raise ValueError(f"Unknown layout: {variant.layout}")


def _parse_grid_from_text(text: str, empty_symbols: set[str]) -> Optional[str]:
    tokens: list[str] = []
    for ch in text:
        if ch in "1234":
            tokens.append(ch)
        elif ch in empty_symbols:
            tokens.append(".")

    if len(tokens) != 16:
        return None

    return "".join(tokens)


def _grid_with_pivots(canonical: str, spec: SudokuSpec, pivots_mask: int) -> Grid:
    grid = Grid.from_string(canonical, spec=spec)
    grid._pivots = pivots_mask  # type: ignore[attr-defined]
    return grid


def _evaluate_single_example(
    *,
    solver: LocalVLLMSolver,
    variant: PromptVariant,
    validator: SudokuValidator,
    gold_path: GoldPath,
    example_id: int,
    max_steps: int,
    parse_empty_symbols: set[str],
    spec: SudokuSpec,
) -> ExampleResult:
    initial_grid = gold_path.nodes[0].action.text.strip()
    target_grid = gold_path.nodes[-1].action.text.strip()

    initial_state = Grid.from_string(initial_grid, spec=spec)
    pivots_mask = initial_state.get_pivots_mask()

    root_node = Node(
        parent=None,
        action=NodeAction(0, _render_grid(initial_grid, variant)),
        state=initial_state,
    )
    context_nodes = [root_node]
    node_id_to_index = {0: 0}

    raw_last_output: Optional[str] = None
    parsed_last_grid: Optional[str] = None
    backtracks = 0

    def fail(reason: str, steps: int) -> ExampleResult:
        return ExampleResult(
            example_id=example_id,
            initial_grid=initial_grid,
            target_grid=target_grid,
            success=False,
            steps=steps,
            backtracks=backtracks,
            solved_via=None,
            failure_reason=reason,
            raw_last_output=raw_last_output,
            parsed_last_grid=parsed_last_grid,
        )

    for step_idx in range(1, max_steps + 1):
        context = PathContext(nodes=context_nodes)

        try:
            generated = solver.inference(context=context, max_actions=1)
        except Exception as exc:
            return fail(f"inference_error:{type(exc).__name__}", steps=step_idx - 1)

        if not generated:
            return fail("no_output", steps=step_idx - 1)

        raw_last_output = str(generated[0]).strip()
        expected_node_id = _next_node_id(context_nodes)
        parsed_action = BasicActionParser.parse(raw_last_output, node_id=expected_node_id)

        if isinstance(parsed_action, BacktrackAction):
            backtracks += 1
            target_id = parsed_action.target_id
            if target_id not in node_id_to_index:
                return fail("invalid_backtrack_target", steps=step_idx)
            target_idx = node_id_to_index[target_id]
            context_nodes = context_nodes[: target_idx + 1]
            node_id_to_index = _build_node_id_to_index(context_nodes)
            continue

        parsed_last_grid = _parse_grid_from_text(
            parsed_action.text,
            empty_symbols=parse_empty_symbols,
        )
        if parsed_last_grid is None:
            return fail("parse_grid_failed", steps=step_idx)

        is_done = isinstance(parsed_action, DoneAction)
        canonical_action = (
            DoneAction(parsed_last_grid)
            if is_done
            else NodeAction(expected_node_id, parsed_last_grid)
        )

        validation_result = validator.validate(
            action=canonical_action,
            context=PathContext(nodes=context_nodes),
        )
        if validation_result is not None and not validation_result.valid:
            return fail(
                f"validation_failed:{validation_result.comment}",
                steps=step_idx,
            )

        if parsed_last_grid == target_grid and "." not in parsed_last_grid:
            return ExampleResult(
                example_id=example_id,
                initial_grid=initial_grid,
                target_grid=target_grid,
                success=True,
                steps=step_idx,
                backtracks=backtracks,
                solved_via="done" if is_done else "node",
                failure_reason=None,
                raw_last_output=raw_last_output,
                parsed_last_grid=parsed_last_grid,
            )

        if is_done:
            return fail("wrong_final_grid", steps=step_idx)

        next_state = _grid_with_pivots(parsed_last_grid, spec=spec, pivots_mask=pivots_mask)
        next_node = Node(
            parent=context_nodes[-1],
            action=NodeAction(expected_node_id, _render_grid(parsed_last_grid, variant)),
            state=next_state,
        )
        context_nodes.append(next_node)
        node_id_to_index[expected_node_id] = len(context_nodes) - 1

    return fail("max_steps", steps=max_steps)


def _build_metrics(results: list[ExampleResult]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        return {
            "fitness": 0.0,
            "avg_parse_failures": 1.0,
            "avg_validation_failures": 1.0,
            "avg_no_output_failures": 1.0,
            "avg_wrong_final_failures": 1.0,
            "avg_max_steps_failures": 1.0,
            "is_valid": 1.0,
        }

    def rate(predicate) -> float:
        return sum(1 for item in results if predicate(item)) / total

    fitness = rate(lambda item: item.success)

    return {
        "fitness": fitness,
        "avg_parse_failures": rate(
            lambda item: item.failure_reason == "parse_grid_failed"
        ),
        "avg_validation_failures": rate(
            lambda item: (item.failure_reason or "").startswith("validation_failed:")
        ),
        "avg_no_output_failures": rate(lambda item: item.failure_reason == "no_output"),
        "avg_wrong_final_failures": rate(
            lambda item: item.failure_reason == "wrong_final_grid"
        ),
        "avg_max_steps_failures": rate(lambda item: item.failure_reason == "max_steps"),
        "is_valid": 1.0,
    }


class SudokuPromptRuntime:
    """Worker-local Sudoku evaluator with a cached vLLM-backed solver."""

    def __init__(self) -> None:
        _raise_if_runtime_unavailable()

        self.variant = PromptVariant(
            layout=PROMPT_CONFIG["layout"],
            empty_symbol=PROMPT_CONFIG["empty_symbol"],
        )
        if self.variant.layout not in SUPPORTED_LAYOUTS:
            raise ValueError(f"Unsupported layout: {self.variant.layout}")

        generation_config = GenerationConfig(**MODEL_CONFIG["generation"])

        self.solver = LocalVLLMSolver(
            model_name=MODEL_CONFIG["model_name"],
            generation_config=generation_config,
            gpu_memory_utilization=MODEL_CONFIG["gpu_memory_utilization"],
            max_model_len=MODEL_CONFIG["max_model_len"],
            bf16=MODEL_CONFIG["bf16"],
            system_prompt="placeholder",
            user_prompt=USER_PROMPT_TEMPLATE,
        )
        self.validator = SudokuValidator()
        self._dataset_cache: dict[tuple[str, int | None], tuple[SudokuSpec, list[GoldPath]]] = {}

    def close(self) -> None:
        self.solver.close()

    def _load_gold_paths(
        self,
        *,
        split: str,
        max_examples: int | None,
    ) -> tuple[SudokuSpec, list[GoldPath]]:
        cache_key = (split, max_examples)
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]

        dataset_path = resolve_dataset_path(split)
        adapter = SudokuAdapter(dataset_path=dataset_path)
        if adapter.spec.size != 4:
            raise ValueError(
                f"Sudoku prompt problem expects 4x4 datasets. Got size={adapter.spec.size} "
                f"from {dataset_path}."
            )

        gold_paths = adapter.load_gold_paths(size=max_examples)
        if not gold_paths:
            raise ValueError(f"No Sudoku examples loaded from dataset: {dataset_path}")

        payload = (adapter.spec, gold_paths)
        self._dataset_cache[cache_key] = payload
        return payload

    def evaluate_prompt(
        self,
        prompt_template: str,
        *,
        split: str = "train",
        max_examples: int | None = None,
        max_steps: int | None = None,
    ) -> tuple[dict[str, float], dict[str, object]]:
        if not isinstance(prompt_template, str):
            raise TypeError("Prompt template must be a string")
        if not prompt_template.strip():
            raise ValueError("Prompt template must be a non-empty string")

        if max_examples is None:
            if split == "train":
                max_examples = VALIDATION_CONFIG["train_examples"]
            elif split == "test":
                max_examples = VALIDATION_CONFIG["test_examples"]
            else:
                raise ValueError(f"Unknown split: {split}")

        if max_steps is None:
            max_steps = VALIDATION_CONFIG["max_steps"]

        spec, gold_paths = self._load_gold_paths(split=split, max_examples=max_examples)
        parse_empty_symbols = {self.variant.empty_symbol, ".", "0", "_"}
        self.solver.system_prompt = prompt_template
        self.solver.user_prompt = USER_PROMPT_TEMPLATE

        results: list[ExampleResult] = []
        for idx, gold_path in enumerate(gold_paths):
            results.append(
                _evaluate_single_example(
                    solver=self.solver,
                    variant=self.variant,
                    validator=self.validator,
                    gold_path=gold_path,
                    example_id=idx,
                    max_steps=max_steps,
                    parse_empty_symbols=parse_empty_symbols,
                    spec=spec,
                )
            )

        metrics = _build_metrics(results)
        artifact = {
            "split": split,
            "variant": self.variant.key,
            "num_examples": len(results),
            "results": [asdict(result) for result in results],
        }
        return metrics, artifact


_RUNTIME: SudokuPromptRuntime | None = None


def get_runtime() -> SudokuPromptRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = SudokuPromptRuntime()
        atexit.register(_RUNTIME.close)
    return _RUNTIME


def validate_prompt(prompt_template: str) -> dict[str, float]:
    metrics, _ = get_runtime().evaluate_prompt(prompt_template, split="train")
    return metrics


def evaluate_prompt(
    prompt_template: str,
    *,
    split: str = "train",
    max_examples: int | None = None,
    max_steps: int | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    return get_runtime().evaluate_prompt(
        prompt_template,
        split=split,
        max_examples=max_examples,
        max_steps=max_steps,
    )
