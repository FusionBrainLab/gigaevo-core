import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from gigaevo.llm.ideas_tracker.components.prompt_manager import PromptManager
from gigaevo.llm.ideas_tracker.utils.it_logger import IdeasTrackerLogger

load_dotenv()


class IdeaAnalyzer:
    """
    Analyzes and classifies ideas from programs using LLM-based classification.

    Compares incoming ideas against existing idea banks to determine if they
    are new or already known, using prompt-based LLM classification.
    """

    def __init__(self, model: str = "deepseek/deepseek-v3.2") -> None:
        """
        Initialize IdeaAnalyzer with LLM model.

        Args:
            model: Name of the LLM model to use for classification.
        """
        # LLM model name can be overridden via external configuration
        self.model = model
        self.logger: IdeasTrackerLogger | None = None
        self._init_analyzer()

    def _init_analyzer(self) -> None:
        """
        Initialize OpenAI client and prompt manager.

        Sets up the LLM client using environment variables and creates
        a PromptManager instance for loading classification prompts.
        """
        self.llm = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["BASE_URL"]
        )
        self.prompt_manager = PromptManager()

        if self.logger is not None:
            # Log analyzer initialization as part of IdeaTracker lifecycle
            self.logger.log_init(
                component="IdeaAnalyzer",
                model_name=self.model,
            )

    def _call_llm(self, step_name: str, prompt_content: str) -> str:
        """
        Call LLM with system and user prompts for a given step.

        Args:
            step_name: Name of the step (e.g., "classify") used to load prompts.
            prompt_content: Content to insert into the user prompt template.

        Returns:
            LLM response content as a string, or empty string if no content.
        """
        prompt_system_name = f"{step_name}__system"
        prompt_user_name = f"{step_name}__user"
        prompt_system = self.prompt_manager.load_prompt(prompt_name=prompt_system_name)
        prompt_user = self.prompt_manager.load_prompt(
            prompt_name=prompt_user_name, insert_data=prompt_content
        )
        response = self.llm.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            model=self.model,
            temperature=0,
        )
        if not response.choices[0].message.content:
            return ""
        return response.choices[0].message.content

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
        retry_attempts = 10
        generated = False
        classified_ideas = {"present_ideas": [], "new_ideas": []}
        while not generated:
            try:
                response = self._call_llm("classify", prompt_content)
                classified_ideas = json.loads(response)
                generated = True
            except Exception:
                retry_attempts -= 1
                if retry_attempts == 0:
                    break
        return classified_ideas

    def _ideas_to_text(self, changes: list[str]) -> str:
        """
        Convert a list of idea descriptions to formatted text.

        Args:
            changes: List of idea description strings.

        Returns:
            Formatted text with numbered list of ideas.
        """
        text = ""
        c = 1
        for change in changes:
            text += f"{c}) {change} \n"
        return text

    def _extract_ideas(
        self,
        new_ideas: list[str],
        bank_data: dict[int, dict[str, list[dict[str, str]] | str]],
    ) -> tuple[list[str], list[str]]:
        """
        Walk over all idea lists in a bank and:
        - accumulate ids of ideas that are already known (by short id);
        - progressively narrow down the list of truly new ideas.
        Does not mutate the input list.

        Args:
            new_ideas: List of idea descriptions to check.
            bank_data: Dictionary mapping list indices to idea data with "text" key.

        Returns:
            Tuple of (known_ideas_ids, remaining_new_ideas):
            - known_ideas_ids: List of short_ids found in the bank
            - remaining_new_ideas: List of idea descriptions not found in the bank
        """
        known_ideas_ids = []
        # Work on a shallow copy so callers' lists are not mutated.
        new_ideas_candidates = list(new_ideas)
        for ideas_block in bank_data.values():
            block_text = ideas_block["text"]
            new_ideas_text = self._ideas_to_text(new_ideas_candidates)
            parsed_ideas = self.classify_ideas(new_ideas_text, block_text)
            # present_ideas are expected to be short_ids of known ideas
            known_ideas_ids.extend(parsed_ideas.get("present_ideas", []))
            # new_ideas are descriptions of ideas that were not found in this block
            new_ideas_candidates = [idea for idea in parsed_ideas.get("new_ideas", [])]
        return known_ideas_ids, new_ideas_candidates

    def process_ideas(
        self,
        program_chages: list[str],
        ideas_active_bank: dict[int, dict[str, list[dict[str, str]] | str]],
        inactive_ideas_bank: dict[int, dict[str, list[dict[str, str]] | str]],
    ) -> tuple[list[str], list[str]]:
        """
        Process ideas from a program against active and inactive idea banks.

        Checks all idea lists in both banks to determine which ideas are new
        and which already exist. Checks active bank first, then inactive bank
        for any remaining new ideas.

        Args:
            program_chages: List of idea descriptions from the program.
            ideas_active_bank: Dictionary mapping list indices to active bank data.
            inactive_ideas_bank: Dictionary mapping list indices to inactive bank data.

        Returns:
            Tuple of (new_ideas, known_ideas_ids):
            - new_ideas: List of idea descriptions not found in any bank
            - known_ideas_ids: List of short_ids of ideas found in either bank
        """
        # Start from a copy so the caller's list is not mutated.
        remaining_new_ideas = list(program_chages)
        known_ideas_ids: list[str] = []

        # First, check against the active ideas bank.
        active_known_ids, remaining_new_ideas = self._extract_ideas(
            remaining_new_ideas, ideas_active_bank
        )
        known_ideas_ids.extend(active_known_ids)

        # Then, any ideas still considered new are checked against the inactive bank.
        if remaining_new_ideas:
            inactive_known_ids, remaining_new_ideas = self._extract_ideas(
                remaining_new_ideas, inactive_ideas_bank
            )
            known_ideas_ids.extend(inactive_known_ids)

        # remaining_new_ideas: descriptions of ideas not found in any bank
        # known_ideas_ids: list of short_ids of ideas found in either active or inactive banks
        return remaining_new_ideas, known_ideas_ids
