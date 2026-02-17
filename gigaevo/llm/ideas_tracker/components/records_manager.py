from typing import Any

from gigaevo.llm.ideas_tracker.components.data_components import RecordBank
from gigaevo.llm.ideas_tracker.utils.it_logger import IdeasTrackerLogger


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
        self.inactive_record_bank: RecordBank = RecordBank(
            list_max_ideas=list_max_ideas
        )
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
        elif idea_id in self.inactive_record_bank.uuids:
            old_description = self.inactive_record_bank.get_idea(idea_id).description
            self.inactive_record_bank.modify_idea(
                idea_id, new_programs, generation, new_description, change_motivation
            )
            self.move_to_active(idea_id)
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

    def move_inactive(self, generation: int, delta: int) -> None:
        """
        Move inactive ideas from active bank to inactive bank based on generation threshold.

        Ideas with |last_generation - generation| > delta are considered inactive.

        Args:
            generation: Current generation number.
            delta: Generation threshold for considering ideas inactive.
        """
        inactive_ideas = self.record_bank.get_inactive_ideas(
            generation, delta, persist=True
        )
        for inact_idea in inactive_ideas:
            self.inactive_record_bank.import_idea(inact_idea)
            if self.logger is not None:
                programs_list = list(
                    getattr(inact_idea, "programs", None)
                    or getattr(inact_idea, "linked_programs", [])
                )
                self.logger.log_move_idea(
                    idea_id=inact_idea.id,
                    description=inact_idea.description,
                    linked_programs=programs_list,
                    destination="inactive_bank",
                )
        for inact_idea in inactive_ideas:
            self.record_bank.remove_idea(inact_idea.id)

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
        if use_inactive:
            bank = self.inactive_record_bank
        else:
            bank = self.record_bank
        all_ideas = bank.all_ideas_short()
        return {
            k: {**v, "text": self.ideas_as_text(v["descriptions"])}
            for k, v in all_ideas.items()
        }

    def move_to_inactive(self, idea_id: str) -> None:
        """
        Move idea from active bank to inactive bank.

        Args:
            idea_id: UUID of idea to move.

        Raises:
            ValueError: If idea_id not found in active bank.
        """
        idea = self.record_bank.get_idea(idea_id)
        self.inactive_record_bank.import_idea(idea)
        self.record_bank.remove_idea(idea_id)

    def move_to_active(self, idea_id: str) -> None:
        """
        Move idea from inactive bank to active bank.

        Args:
            idea_id: UUID of idea to move.

        Raises:
            ValueError: If idea_id not found in inactive bank.
        """
        idea = self.inactive_record_bank.get_idea(idea_id)
        self.record_bank.import_idea(idea)
        self.inactive_record_bank.remove_idea(idea_id)

    def get_rankings(
        self, inactive: bool = False
    ) -> list[dict[str, str | list[str] | list[float]]]:
        """
        Get idea rankings from active or inactive bank.

        Args:
            inactive: If True, return rankings from inactive bank; otherwise from active bank.

        Returns:
            List of dicts with keys: id, description, programs, count, fitness, an_fitness.
        """
        if not inactive:
            return self.record_bank.rankings()
        else:
            return self.inactive_record_bank.rankings()

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
