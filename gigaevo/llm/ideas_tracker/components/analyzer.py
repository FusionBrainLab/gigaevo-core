import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from gigaevo.llm.ideas_tracker.components.prompt_manager import PromptManager

load_dotenv()


class IdeaAnalyzer:
    def __init__(self, model: str = "deepseek/deepseek-v3.2"):
        # LLM model name can be overridden via external configuration
        self.model = model
        self._init_analyzer()

    def _init_analyzer(self):
        self.llm = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["BASE_URL"]
        )
        self.prompt_manager = PromptManager()

    def _call_llm(self, step_name: str, prompt_content: str) -> str:
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

    def classify_ideas(self, program_changes: str, known_ideas: str):
        """
        Receive list of ideas extracted from program and information about program and existing ideas.
        Returns two list: list of ideas that already known and list of ideas that considered new.
        """
        prompt_content = (
            f" Existing Ideas: \n {known_ideas} \n Incoming Ideas: \n {program_changes}"
        )
        response = self._call_llm("classify", prompt_content)
        classified_ideas = json.loads(response)
        return classified_ideas

    def _ideas_to_text(self, changes: list):
        text = ""
        c = 1
        for change in changes:
            text += f"{c}) {change} \n"
        return text

    def _extract_ideas(self, new_ideas, bank_data):
        """
        Walk over all idea lists in a bank and:
        - accumulate ids of ideas that are already known (by short id);
        - progressively narrow down the list of truly new ideas.
        Does not mutate the input list.
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
        self, program_chages: list, ideas_active_bank: dict, inactive_ideas_bank: dict
    ):
        """
        Receive list of ideas from program and all lists of ideas from ideas banks.
        Checks all lists if any of provided ideas from program are present in them.
        Returns:
        - list of new idea descriptions;
        - list of short_ids of ideas that are already known.
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
