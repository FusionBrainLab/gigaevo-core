from __future__ import annotations

from typing import Any

from gigaevo.memory.ideas_tracker.components.data_components import RecordBank
from gigaevo.memory.ideas_tracker.utils.it_logger import IdeasTrackerLogger


class RecordManager:
    """
    Manages active and inactive idea banks with movement and querying capabilities.

    Provides high-level operations for adding, modifying ideas, moving ideas
    between banks based on activity, and formatting ideas for LLM processing.
    """

    def __init__(self, list_max_ideas: int = 5) -> None:
        """
        Initialize RecordManager with empty active and inactive banks.

        Args:
            list_max_ideas: Maximum number of ideas per list in each bank.
        """
        self.record_bank: RecordBank = RecordBank(list_max_ideas=list_max_ideas)

        self.logger: IdeasTrackerLogger | None = None

    def add_new_idea(
        self,
        description: str,
        linked_program: str,
        generation: int,
        category: str = "",
        strategy: str = "",
        task_description: str = "",
        change_motivation: str = "",
    ) -> None:
        """
        Add a new idea to the active ideas bank with logging.

        Args:
            description: Idea description text.
            linked_program: Program ID where this idea was first seen.
            generation: Generation number when idea appeared.
            category: Optional idea category.
            strategy: Optional strategy used.
            task_description: Optional task context description.
            change_motivation: Optional explanation of the change.
        """
        self.record_bank.add_idea(
            description,
            linked_program,
            generation,
            category,
            strategy,
            task_description,
            change_motivation,
        )

        if self.logger is not None:
            self.logger.log_new_idea(
                description=description,
                generation=generation,
                linked_program=linked_program,
                category=category,
                strategy=strategy,
            )

    def modify_idea(
        self,
        idea_id: str,
        new_programs: list[str] | None,
        generation: int | None,
        new_description: str | None = None,
        change_motivation: str | None = None,
    ) -> None:
        """
        Update an idea's programs, generation, and optionally description.

        If idea is in inactive bank, moves it to active bank after update.

        Args:
            idea_id: UUID of idea to modify.
            new_programs: List of program IDs to add, or None.
            generation: Generation number to update if greater, or None.
            new_description: Optional new description text.
            change_motivation: Optional explanation of the modification.

        Raises:
            ValueError: If idea_id not found in either bank.
        """
        idea_bank = None
        old_description = None
        if idea_id in self.record_bank.uuids:
            old_description = self.record_bank.get_idea(idea_id).description
            self.record_bank.modify_idea(
                idea_id, new_programs, generation, new_description, change_motivation
            )
            idea_bank = self.record_bank
        else:
            raise ValueError(f"No idea with id {idea_id} found!")

        if self.logger is not None and idea_bank is not None:
            idea = idea_bank.get_idea(idea_id)
            extra: dict[str, Any] = {}
            if hasattr(idea, "category"):
                extra["category"] = getattr(idea, "category", "")
            if hasattr(idea, "strategy"):
                extra["strategy"] = getattr(idea, "strategy", "")
            if hasattr(idea, "programs"):
                extra["programs"] = list(getattr(idea, "programs", []))
            self.logger.log_modify_idea(
                idea_id=idea.id,
                old_description=old_description,
                new_description=idea.description,
                new_linked_programs=new_programs or [],
                **extra,
            )

    def ideas_as_text(self, ideas_data: list[dict[str, str]]) -> str:
        """
        Format list of idea dicts as text block with [short_id]: description format.

        Args:
            ideas_data: List of dicts with "short_id" and "description" keys.

        Returns:
            Formatted text string with one idea per line.
        """
        final_text = ""
        for idea_description in ideas_data:
            short_id = idea_description.get("short_id", "")
            description = idea_description.get("description", "")
            new_line = f"[{short_id}]: {description} \n "
            final_text += new_line
        return final_text

    def ideas_groups_texts(
        self, use_inactive: bool = False
    ) -> dict[int, dict[str, list[dict[str, str]] | str]]:
        """
        Get idea groups with both structured and text representations.

        Args:
            use_inactive: If True, use inactive bank; otherwise use active bank.

        Returns:
            Dictionary mapping list indices to dicts with "descriptions" list
            and "text" string representation.
        """

        bank = self.record_bank
        all_ideas = bank.all_ideas_short()
        return {
            k: {**v, "text": self.ideas_as_text(v["descriptions"])}
            for k, v in all_ideas.items()
        }

    def enrich_idea_metadata(
        self,
        idea_id: str,
        keywords: list[str] | None = None,
        summary: str | None = None,
        task_description_summary: str | None = None,
    ) -> None:
        """
        Update enrichment metadata on an idea in the active or inactive bank.

        Args:
            idea_id: UUID of the idea to update.
            keywords: Optional keyword list to set.
            summary: Optional explanation summary to set.
            task_description_summary: Optional task-description summary to set.

        Raises:
            ValueError: If idea_id not found in either bank.
        """
        if idea_id in self.record_bank.uuids:
            self.record_bank.modify_idea_metadata(
                idea_id, keywords, summary, task_description_summary
            )
        else:
            raise ValueError(f"No idea with id {idea_id} found!")

    @staticmethod
    def get_full_id(
        short_id: str,
        ideas_desc_lists: dict[int, dict[str, list[dict[str, str]] | str]],
    ) -> str:
        """
        Resolve a short_id to its full UUID by searching through idea description lists.

        Args:
            short_id: Short UUID identifier (first part of full UUID).
            ideas_desc_lists: Dictionary mapping list indices to idea data containing
                "descriptions" key with list of idea dicts.

        Returns:
            Full UUID string if found, empty string otherwise.
        """
        for desc_list in ideas_desc_lists.values():
            for description in desc_list["descriptions"]:
                if description["short_id"] == short_id:
                    return description["id"]
        return ""
