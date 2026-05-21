"""Bridge between gigaevo's chain infrastructure and CARL.

Provides two thin adapters:

``GigaEvoClientAdapter``
    Wraps gigaevo's callable LLM client (``await client(prompt, **kw)``) as a
    CARL ``LLMClientBase``, so CARL's step executors can call it uniformly.

``GigaEvoPromptTemplate``
    Subclasses CARL's ``PromptTemplate`` with gigaevo-specific English
    templates (no Russian text, no RAG context queries, no prescriptive
    response format).  Also adds ``format_history_entry`` — a helper used by
    the step-batched runner to produce history strings compatible with gigaevo's
    existing ``PromptBuilder`` format.
"""

from mmar_carl import LLMClientBase
from mmar_carl.models import PromptTemplate


class GigaEvoClientAdapter(LLMClientBase):
    """Wrap a gigaevo callable LLM client as a CARL ``LLMClientBase``.

    Gigaevo clients are async callables with signature::

        async def __call__(prompt: str, **kwargs) -> str

    and expose a ``.copy()`` method for thread-safe concurrent use.
    CARL's executors expect ``get_response`` / ``get_response_with_retries``.

    The adapter delegates both methods directly to the underlying callable.
    Retry logic lives in gigaevo's client, so retries passed to
    ``get_response_with_retries`` are intentionally ignored here.

    Args:
        client: A gigaevo LLM client (callable accepting prompt + kwargs).
        max_tokens: Optional per-call token limit.  When set, it is forwarded
            as ``max_tokens=`` to the underlying client call.
    """

    def __init__(self, client, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens

    async def get_response(self, prompt: str) -> str:
        """Call the underlying gigaevo client."""
        if self._max_tokens is not None:
            return await self._client(prompt, max_tokens=self._max_tokens)
        return await self._client(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Call the underlying gigaevo client (retries handled internally)."""
        return await self.get_response(prompt)


class GigaEvoPromptTemplate(PromptTemplate):
    """CARL ``PromptTemplate`` with gigaevo-specific English templates.

    Differences from CARL's default template:
    - No ``{context_queries}`` placeholder — gigaevo steps don't use RAG
      extraction queries.
    - Chain template contains only ``Data:`` prefix without prescriptive
      instructions ("Respond concisely…") that CARL adds by default.
    - History and step templates match gigaevo's ``PromptBuilder`` defaults
      so existing chains produce identical prompts after migration.

    When CARL's ``format_step_prompt`` passes ``context_queries=`` to
    ``str.format()``, the extra keyword is silently ignored because the
    template string has no ``{context_queries}`` placeholder.
    """

    # Override CARL's English templates to match gigaevo's prompt format.
    en_step_template: str = (
        "Step {step_number}. {step_title}\n"
        "Objective: {aim}\n"
        "Task: {stage_action}\n"
        "Questions: {reasoning_questions}\n"
        "Example reasoning: {example_reasoning}"
    )

    # Plain data wrapper — no prescriptive instructions appended.
    en_chain_template: str = "Data:\n{outer_context}\n\n{step_prompt}"

    en_history_template: str = (
        "Previous steps:\n{history}\n\n"
        "Based on the results of previous steps, "
        "perform the following task:\n{current_task}"
    )

    def format_history_entry(self, number: int, title: str, result: str) -> str:
        """Format a completed step's output as a history entry string.

        Produces the same format used by gigaevo's ``PromptBuilder``::

            "Step {number}. {title}\\nResult: {result}\\n"

        Args:
            number: 1-based step number.
            title: Human-readable step title.
            result: Raw string output of the step.

        Returns:
            Formatted history entry ready to append to
            ``ReasoningContext.history``.
        """
        return f"Step {number}. {title}\nResult: {result}\n"
