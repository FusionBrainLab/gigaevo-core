from dataclasses import asdict, dataclass, field, fields
from typing import Any
from uuid import uuid4


@dataclass
class ProgramRecord:
    """
    Storage for individual program data (fitness, generation, parents, insights).
    """

    id: str = ""
    fitness: float = 0.0
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)


@dataclass
class RecordCard:
    """
    Holds information about a single record (idea or program insight).
    Mutable after init: linked_programs, last_generation, description.
    """

    id: str = ""
    description: str = ""
    linked_programs: list[str] = field(default_factory=list)
    last_generation: int = 0


@dataclass
class RecordCardExtended:
    id: str = ""
    category: str = ""
    description: str = ""
    task_description: str = ""
    strategy: str = ""
    aliases: list[dict[str, str]] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    evolution_statistics: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, list[str] | str] = field(default_factory=dict)
    works_with: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    usage: dict[str, str] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        required_fields = [
            "id",
            "category",
            "description",
            "task_descriptio",
            "strategy",
            "programs",
        ]
        if not all(field in kwargs for field in required_fields):
            missing_fields = [field for field in required_fields if field not in kwargs]
            raise ValueError(f"Missing required fields: {missing_fields}")

        names = set([f.name for f in fields(self)])
        for arg, value in kwargs.items():
            if arg in names:
                setattr(self, arg, value)


@dataclass
class RecordList:
    """
    Holds a bounded number of RecordCard instances (ideas), up to max_ideas.
    """

    ideas: list[RecordCard] = field(default_factory=list)
    num_ideas: int = 0
    max_ideas: int = 5

    def add_idea(self, idea_dict: dict[str, Any]) -> None:
        """Append a new idea from a dict of RecordCard fields. Raises if list is full."""
        if self.num_ideas >= self.max_ideas:
            raise ValueError("Can't add new idea to list since it full")
        new_idea = RecordCard(**idea_dict)
        self.ideas.append(new_idea)
        self.num_ideas = len(self.ideas)

    def find_idea_index(self, idea_id: str) -> int | None:
        """Return index of idea with given id, or None if not found."""
        for index, idea in enumerate(self.ideas):
            if idea.id == idea_id:
                return index
        return None

    def modify_idea(
        self,
        idea_id: str,
        new_programs: list[str] | None = None,
        new_generation: int | None = None,
        new_description: str | None = None,
    ) -> bool:
        """Update linked_programs (extend) and/or last_generation (if greater). Returns True if found."""
        idea_index = self.find_idea_index(idea_id)
        if idea_index is None:
            return False
        idea = self.ideas[idea_index]
        if new_programs is not None:
            idea.linked_programs.extend(new_programs)
        if new_generation is not None and idea.last_generation < new_generation:
            idea.last_generation = new_generation
        if new_description is not None:
            idea.description = new_description
        return True

    def get_idea(self, idea_id: str) -> RecordCard:
        """Return the RecordCard with the given id. Raises ValueError if not found."""
        idea_index = self.find_idea_index(idea_id)
        if idea_index is None:
            raise ValueError(f"Can't find idea with id {idea_id}")
        return self.ideas[idea_index]

    def remove_idea(self, idea_id: str) -> bool:
        """Remove the idea with the given id. Returns True if found and removed."""
        idea_index = self.find_idea_index(idea_id)
        if idea_index is None:
            return False
        self.ideas.pop(idea_index)
        self.num_ideas = len(
            self.ideas
        )  # keep in sync in case list was mutated elsewhere
        return True

    def import_idea(self, new_idea: RecordCard) -> None:
        """Append an existing RecordCard. Raises if list is full."""
        if self.num_ideas >= self.max_ideas:
            raise ValueError("Can't import idea: list is full")
        self.ideas.append(new_idea)
        self.num_ideas = len(self.ideas)

    def exclude_inactive_ideas(
        self, generation: int, delta: int, persist_ideas: bool = False
    ) -> list[RecordCard]:
        """Return ideas where |last_generation - generation| > delta; optionally remove them from this list."""
        excluded_ideas = []
        excluded_ideas_id = []
        for idea in self.ideas:
            if abs(idea.last_generation - generation) > delta:
                excluded_ideas.append(idea)
                excluded_ideas_id.append(idea.id)
        if not persist_ideas:
            for exc_idea_id in excluded_ideas_id:
                self.remove_idea(exc_idea_id)
        return excluded_ideas

    def all_ideas_short(self) -> list[dict[str, str]]:
        """Return a short representation of each idea: [{"id","short_id", "description"}, ...]."""
        ideas = []
        for idea in self.ideas:
            ideas.append(
                {
                    "id": idea.id,
                    "short_id": idea.id.split("-")[
                        0
                    ],  # first part of the id is the short id
                    "description": idea.description,
                }
            )
        return ideas

    def change_max_ideas(self, new_val: int) -> None:
        """Set max_ideas to new_val. Raises if new_val is negative or less than current size."""
        if new_val < 0:
            raise ValueError(f"New max_ideas value {new_val} smaller than 0")
        if new_val < self.num_ideas:
            raise ValueError(
                f"New max_ideas value {new_val} smaller than size of list {self.num_ideas}"
            )
        self.max_ideas = new_val

    def is_full(self) -> bool:
        """Return True if this list has reached max_ideas."""
        if self.num_ideas < self.max_ideas:
            return False
        else:
            return True

    def rankings(self) -> list[dict[str, str | list[str] | int]]:
        """Return a short representation of each idea: [{"id", "description", "programs", "count"}, ...]."""
        ideas = []
        for idea in self.ideas:
            ideas.append(
                {
                    "id": idea.id,
                    "description": idea.description,
                    "programs": idea.linked_programs,
                    "count": len(idea.linked_programs),
                }
            )
        return ideas


@dataclass
class RecordBank:
    """
    Holds multiple RecordList instances; manages unique ids and idea add/remove/import.
    """

    ideas_lists: list[RecordList] = field(default_factory=list)
    num_lists: int = 0
    generation: int = 0
    uuids: list[str] = field(default_factory=list)
    list_max_ideas: int = 5

    def _unique_id_pair(self) -> str:
        """Return a uuid not present in uuids."""
        generated = False
        new_uuid = ""
        while not generated:
            new_uuid = str(uuid4())
            if new_uuid not in self.uuids:
                generated = True
        return new_uuid

    def _append_record_list(self, idea_data: dict[str, Any]) -> None:
        """Append idea to first non-full list, or create a new RecordList and append there."""
        for index, record_list in enumerate(self.ideas_lists):
            if not record_list.is_full():
                self.ideas_lists[index].add_idea(idea_data)
                return
        new_list = RecordList(max_ideas=self.list_max_ideas)
        new_list.add_idea(idea_data)
        self.ideas_lists.append(new_list)
        self.num_lists += 1
        return

    def _remove_id_from_bank(self, idea_id: str) -> None:
        """Remove idea_id from uuids at same index."""
        try:
            idx = self.uuids.index(idea_id)
        except ValueError:
            return
        self.uuids.pop(idx)

    def add_idea(self, description: str, linked_program: str, generation: int) -> None:
        """Create a new idea with a unique id and append it to a list in the bank."""
        idea_id = self._unique_id_pair()
        self.uuids.append(idea_id)
        idea_dict = {
            "id": idea_id,
            "description": description,
            "linked_programs": [linked_program],
            "last_generation": generation,
        }
        self._append_record_list(idea_dict)

    def modify_idea(
        self,
        idea_id: str,
        new_programs: list[str] | None = None,
        new_generation: int | None = None,
        new_description: str | None = None,
    ) -> None:
        """Update the RecordCard's linked_programs (extend) and/or last_generation (if greater)."""
        if idea_id not in self.uuids:
            raise ValueError(f"No idea with id {idea_id} found")
        for index, record_list in enumerate(self.ideas_lists):
            if record_list.find_idea_index(idea_id) is not None:
                self.ideas_lists[index].modify_idea(
                    idea_id, new_programs, new_generation, new_description
                )
                return

    def import_idea(self, new_idea: RecordCard) -> None:
        """Append an existing RecordCard; assign new id if its id already exists in the bank."""
        if new_idea.id in self.uuids:
            new_uuid = self._unique_id_pair()
            new_idea.id = new_uuid
        self.uuids.append(new_idea.id)
        idea_dict = asdict(new_idea)
        idea_dict["linked_programs"] = list(idea_dict["linked_programs"])
        self._append_record_list(idea_dict)

    def get_idea(self, idea_id: str) -> RecordCard:
        """Return the RecordCard with the given id. Raises ValueError if not found."""
        if idea_id not in self.uuids:
            raise ValueError(f"No idea with id {idea_id} found")
        for index, record_list in enumerate(self.ideas_lists):
            if record_list.find_idea_index(idea_id) is not None:
                return self.ideas_lists[index].get_idea(idea_id)
        raise ValueError(f"No idea with id {idea_id} found in any list")

    def get_inactive_ideas(
        self, generation: int, delta: int, persist: bool = False
    ) -> list[RecordCard]:
        """Return ideas where |last_generation - generation| > delta; if not persist, remove them from the bank."""
        excluded_ideas: list[RecordCard] = []
        for list_num in range(len(self.ideas_lists)):
            list_excluded_ideas = self.ideas_lists[list_num].exclude_inactive_ideas(
                generation=generation, delta=delta, persist_ideas=persist
            )
            excluded_ideas.extend(list_excluded_ideas)
        if not persist:
            for card in excluded_ideas:
                self._remove_id_from_bank(card.id)
        return excluded_ideas

    def remove_idea(self, idea_id: str) -> None:
        """Remove the idea with the given id from its list and from uuids. Raises if not found."""
        if idea_id not in self.uuids:
            raise ValueError(f"No idea with id {idea_id} found")
        for index, record_list in enumerate(self.ideas_lists):
            if record_list.find_idea_index(idea_id) is not None:
                self.ideas_lists[index].remove_idea(idea_id)
                self._remove_id_from_bank(idea_id)
                return

    def bank_size(self) -> int:
        """Return the number of RecordList instances in this bank."""
        return self.num_lists

    def get_record_list(self, list_num: int) -> RecordList:
        """Return the RecordList at the given index. Raises if index out of range."""
        if list_num > len(self.ideas_lists) - 1 or list_num < 0:
            raise ValueError("No records list with such id exists")
        return self.ideas_lists[list_num]

    def all_ideas_short(self) -> dict[int, dict[str, list[dict[str, str]]]]:
        """Return {list_index: {"descriptions": [{"short_id", "description"}, ...]}} for each list."""
        ideas_by_list: dict[int, dict[str, list[dict[str, str]]]] = {}
        for index, ideas_list in enumerate(self.ideas_lists):
            short_descriptions = ideas_list.all_ideas_short()
            ideas_by_list[index] = {"descriptions": short_descriptions}
        return ideas_by_list

    def all_ideas_cards(self) -> list[RecordCard]:
        """Return all ideas in the bank."""
        ideas = []
        for ideas_list in self.ideas_lists:
            ideas.extend(ideas_list.ideas)
        return ideas

    def rankings(self) -> list[dict[str, str | list[str] | list[float]]]:
        """
        Returns:
            List of dictionaries with keys: id, description, programs, fitness, an_fitness.
        """
        ideas_rankings = []
        for ideas_list in self.ideas_lists:
            ideas_with_scores = ideas_list.rankings()
            for idea in ideas_with_scores:
                idea_dict = {**idea}
                idea_dict["fitness"] = []
                idea_dict["an_fitness"] = []
                ideas_rankings.append(idea_dict)
        return ideas_rankings
