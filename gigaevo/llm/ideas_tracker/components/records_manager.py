from typing import Any

from gigaevo.llm.ideas_tracker.components.data_components import RecordBank
from gigaevo.llm.ideas_tracker.utils.it_logger import IdeasTrackerLogger


class RecordManager:
    """Manages active and inactive idea banks; supports moving ideas between them and querying."""

    def __init__(self, list_max_ideas: int = 5) -> None:
        """Create empty active and inactive RecordBanks."""
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
        """Add a new idea to the main (active) ideas bank."""
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
        """Update a RecordCard's linked_programs and/or last_generation; move from inactive to active if in inactive bank."""
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
            # Fetch updated idea to log its description change and new links
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
        """Move ideas deemed inactive (by generation/delta) from active bank to inactive bank."""
        inactive_ideas = self.record_bank.get_inactive_ideas(
            generation, delta, persist=True
        )
        for inact_idea in inactive_ideas:
            self.inactive_record_bank.import_idea(inact_idea)
            if self.logger is not None:
                # Support both RecordCard (linked_programs) and RecordCardExtended (programs)
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
        """Convert a list of idea dicts (short_id, description) into a single text block.
        Missing keys default to empty string."""
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
        """Return per-list short descriptions and a text version for the active or inactive bank."""
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
        """Move the idea with the given id from the active bank to the inactive bank.
        Raises ValueError if idea_id is not in the active bank."""
        idea = self.record_bank.get_idea(idea_id)
        self.inactive_record_bank.import_idea(idea)
        self.record_bank.remove_idea(idea_id)

    def move_to_active(self, idea_id: str) -> None:
        """Move the idea with the given id from the inactive bank to the active bank.
        Raises ValueError if idea_id is not in the inactive bank."""
        idea = self.inactive_record_bank.get_idea(idea_id)
        self.record_bank.import_idea(idea)
        self.inactive_record_bank.remove_idea(idea_id)

    def get_rankings(
        self, inactive: bool = False
    ) -> list[dict[str, str | list[str] | list[float]]]:
        """
        Return ideas ranking from main bank.

        Args:
            inactive: If True, return rankings from inactive ideas bank instead.

        Returns:
            List of dictionaries with idea ranking information.
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
