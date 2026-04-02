from copy import copy
import json
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from gigaevo.memory.ideas_tracker.components.data_components import IncomingIdeas
from gigaevo.memory.ideas_tracker.components.fabrics.llm_clients_fabric import LLMClient
from gigaevo.memory.ideas_tracker.utils.it_logger import IdeasTrackerLogger

load_dotenv()


class IdeaAnalyzer:
    """
    Analyzes and classifies ideas from programs using LLM-based classification.

    Compares incoming ideas against existing idea banks to determine if they
    are new or already known, using prompt-based LLM classification.
    """

    def __init__(
        self,
        model: str = "deepseek/deepseek-v3.2",
        reasoning: dict[str, Any] | None = None,
        base_url: str | None = None,
        description_rewriting: bool = False,
        retry_attempts: int = 10,
    ) -> None:
        """
        Initialize IdeaAnalyzer with LLM model.

        Args:
            model: Name of the LLM model to use for classification.
            reasoning: Optional OpenRouter reasoning settings, e.g. {"effort": "low"}.
            base_url: Optional OpenAI-compatible base URL from config.
            description_rewriting: If True, use ``classify_ext`` (IDs as ``shortId:seq``).
                If False, use ``classify`` (bare short IDs only; incompatible with
                :meth:`_extract_ideas_v2` without extra handling).
        """
        self.model = model
        self.reasoning = reasoning or {}
        self.base_url = str(base_url).strip() if base_url is not None else None
        self._is_openrouter = False
        self.logger: IdeasTrackerLogger | None = None
        self.description_rewriting = description_rewriting
        self.retry_attempts = retry_attempts
        self._init_analyzer()

    def _init_analyzer(self) -> None:
        """
        Initialize OpenAI client and prompt manager.

        Sets up the LLM client using environment variables and creates
        a PromptManager instance for loading classification prompts.
        """
        self.llm = LLMClient(model=self.model, base_url=self.base_url)

        if self.logger is not None:
            self.logger.log_init(
                component="IdeaAnalyzer",
                model_name=self.model,
            )

    def call_llm(self, step_name: str, prompt_content: str) -> str:
        """
        Call LLM with system and user prompts for a given step.

        Args:
            step_name: Name of the step (e.g., "classify") used to load prompts.
            prompt_content: Content to insert into the user prompt template.

        Returns:
            LLM response content as a string, or empty string if no content.
        """
        return self.llm.call_llm(step_name, prompt_content, self.reasoning)

    async def call_llm_async(self, step_name: str, prompt_content: str) -> str:
        """
        Asynchronous chat completion for code paths that use the async API (e.g. refinement).
        """
        return await self.llm.call_llm_async(step_name, prompt_content, self.reasoning)

    def classify_ideas(
        self, program_changes: str, known_ideas: str
    ) -> dict[str, list[str]]:
        """
        Classify ideas as new or already known using LLM.

        Args:
            program_changes: Text representation of ideas from a program.
            known_ideas: Text representation of existing ideas to compare against.

        Returns:
            Dictionary with keys:
            - "present_ideas": List of short_ids of ideas that match known ideas
            - "new_ideas": List of descriptions of ideas that are new
        """
        prompt_content = (
            f" Existing Ideas: \n {known_ideas} \n Incoming Ideas: \n {program_changes}"
        )
        classified_ideas = {"present_ideas": [], "new_ideas": [], "updated_ideas": []}
        for _ in range(self.retry_attempts):
            try:
                response = self.call_llm("classify_ext", prompt_content)
                classified_ideas = json.loads(response)
                return classified_ideas
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")
                continue
        return classified_ideas

    def short_id_to_full_id(
        self, short_id: str, ideas_list: list[dict[str, str]]
    ) -> str:
        """
        Convert a short UUID identifier to its full UUID.

        Args:
            short_id: Short UUID identifier (first part of full UUID).
            ideas_list: List of idea dictionaries with "short_id" and "id" keys.

        Returns:
            Full UUID string if found, empty string otherwise.
        """
        for idea in ideas_list:
            if idea["short_id"] == short_id:
                return idea["id"]
        return ""

    def _split_id(self, idea_ref: str) -> tuple[str, int]:
        """
        Parse ``shortId:sequence`` from ``classify_ext`` output (or ``classify`` bare ID).

        If the model omits ``:sequence``, returns sequence ``1`` (best-effort; may be
        wrong when several incoming ideas are classified in one call).
        """
        raw = idea_ref.strip()
        if ":" not in raw:
            return raw.strip("[]"), 1
        left, right = raw.split(":", 1)
        idea_short_id = left.strip("[]")
        idea_sequence_number = int(right.strip("[]"))
        return idea_short_id, idea_sequence_number

    def _extract_ideas_v2(
        self,
        program_changes: IncomingIdeas,
        bank_data: dict[int, dict[str, list[dict[str, str]] | str]],
    ) -> IncomingIdeas:
        ideas_data = copy(program_changes)
        for idea_block in bank_data.values():
            block_text = idea_block["text"]
            new_ideas_text = ideas_data.get_list_of_ideas()
            parsed_ideas = self.classify_ideas(new_ideas_text, block_text)
            for idea in parsed_ideas.get("present_ideas", []):
                idea_short_id, idea_sequence_number = self._split_id(idea)
                idea_full_id = self.short_id_to_full_id(
                    idea_short_id, idea_block["descriptions"]
                )
                if not idea_full_id:
                    continue
                ideas_data.update_idea(idea_sequence_number, idea_full_id, False)

            for idea in parsed_ideas.get("updated_ideas", []):
                idea_short_id, idea_sequence_number = self._split_id(idea["id"])
                idea_full_id = self.short_id_to_full_id(
                    idea_short_id, idea_block["descriptions"]
                )
                if not idea_full_id:
                    continue
                ideas_data.update_idea(idea_sequence_number, idea_full_id, True)

            ideas_data.update_mapping()
            if ideas_data.new_ideas_count == 0:
                break

        return ideas_data

    def process_ideas(
        self,
        program_changes: IncomingIdeas,
        ideas_active_bank: dict[int, dict[str, list[dict[str, str]] | str]],
        inactive_ideas_bank: dict[int, dict[str, list[dict[str, str]] | str]],
    ) -> dict[str, list[str] | dict[str, str]]:
        """
        Process ideas from a program against active and inactive idea banks.

        Checks active bank first, then inactive bank for remaining new ideas
        to classify which ideas are new and which already exist.

        Args:
            program_changes: IncomingIdeas object containing idea descriptions from the program.
            ideas_active_bank: Dictionary mapping list indices to active bank data.
            inactive_ideas_bank: Dictionary mapping list indices to inactive bank data.

        Returns:
            IncomingIdeas object with updated classification status for each idea.
        """
        classified_ideas = copy(program_changes)
        classified_ideas = self._extract_ideas_v2(classified_ideas, ideas_active_bank)
        if classified_ideas.new_ideas_count > 0:
            classified_ideas = self._extract_ideas_v2(
                classified_ideas, inactive_ideas_bank
            )
        return classified_ideas
