from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from uuid import uuid4

from loguru import logger

_DESCRIPTION_KEYS = (
    "description",
    "summary",
    "title",
    "change",
    "what_changed",
    "pattern",
    "improvement",
    "name",
)
_EXPLANATION_KEYS = (
    "explanation",
    "rationale",
    "reason",
    "why",
    "motivation",
    "expected_effect",
    "impact",
    "details",
    "justification",
)


def _stringify_improvement_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _stringify_improvement_value(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = [_stringify_improvement_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def normalize_improvement_item(idea: Any) -> dict[str, str]:
    """Coerce mutation change payloads into the tracker's legacy shape."""
    if isinstance(idea, str):
        description = idea.strip()
        return {"description": description, "explanation": ""}

    if not isinstance(idea, dict):
        description = _stringify_improvement_value(idea)
        return {"description": description or "Unspecified change", "explanation": ""}

    description = ""
    for key in _DESCRIPTION_KEYS:
        description = _stringify_improvement_value(idea.get(key))
        if description:
            break

    explanation = ""
    for key in _EXPLANATION_KEYS:
        explanation = _stringify_improvement_value(idea.get(key))
        if explanation:
            break

    extras: list[str] = []
    for key, value in idea.items():
        if key in _DESCRIPTION_KEYS or key in _EXPLANATION_KEYS:
            continue
        text = _stringify_improvement_value(value)
        if text:
            extras.append(f"{key}: {text}")

    if not description and extras:
        description = extras[0]
        extras = extras[1:]
    if not explanation and extras:
        explanation = "; ".join(extras)
    if not description:
        description = explanation or "Unspecified change"

    return {"description": description, "explanation": explanation}


def normalize_improvements(ideas: Any) -> list[dict[str, str]]:
    if ideas is None:
        return []
    if isinstance(ideas, list):
        return [normalize_improvement_item(idea) for idea in ideas]
    return [normalize_improvement_item(ideas)]


@dataclass
class ProgramRecord:
    """
    Represents a single program in the evolutionary system.

    Stores program metadata including fitness metrics, generation, lineage (parents),
    insights used, improvements made, and contextual information.
    """

    id: str = ""
    fitness: float = 0.0
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)
    category: str = ""
    strategy: str = ""
    task_description: str = ""
    task_description_summary: str = ""
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert ProgramRecord to a dictionary."""
        return {
            "id": self.id,
            "fitness": self.fitness,
            "generation": self.generation,
            "parents": self.parents,
            "insights": self.insights,
            "improvements": self.improvements,
            "category": self.category,
            "strategy": self.strategy,
            "task_description": self.task_description,
            "task_description_summary": self.task_description_summary,
            "code": self.code,
        }


@dataclass
class RecordCardEmbedding:
    """
    Represents a tracked idea with embedding vector.
    """

    id: str = ""
    description: str = ""
    embedding: list[float] = field(default_factory=list)
    source_program_id: str = ""
    cluster_id: str = ""
    change_motivation: str = ""


@dataclass
class ClusterCard:
    """
    A cluster of RecordCardEmbedding instances for embedding/LLM grouping.

    Members share references with the global card list; cluster_id on each card
    should match this cluster's cluster_id when the card belongs here.
    """

    cluster_id: str = ""
    center: list[float] = field(default_factory=list)
    members: list[RecordCardEmbedding] = field(default_factory=list)
    index_to_card: dict[int, RecordCardEmbedding] = field(default_factory=dict)
    #: When ``False``, refinement loop skips this cluster (eligible only if ``True``).
    has_changed: bool = True

    @property
    def size(self) -> int:
        return len(self.members)

    def rebuild_index_to_card(self) -> None:
        self.index_to_card = {i + 1: card for i, card in enumerate(self.members)}

    def prune_stale_members(self) -> None:
        """Remove members whose cluster_id does not match this cluster."""
        self.members = [c for c in self.members if c.cluster_id == self.cluster_id]
        self.rebuild_index_to_card()

    def numbered_ideas_text(self) -> str:
        """Numbered descriptions (same shape as IncomingIdeas.get_list_of_ideas)."""
        lines: list[str] = []
        for i, card in enumerate(self.members, start=1):
            lines.append(f"{i}) {card.description} \n")
        return "".join(lines)

    def numbered_idea_groups(self, subgroup_size: int = 20) -> list[str]:
        """
        Split the cluster into fixed-size subgroups (last group may be short).

        Each returned string uses the same line format as :meth:`numbered_ideas_text`,
        with **continuous global** 1-based indices across groups (first group 1..k,
        next k+1.., etc.).
        """
        if subgroup_size < 1:
            raise ValueError("subgroup_size must be >= 1")
        n = len(self.members)
        if n == 0:
            return []
        groups: list[str] = []
        if subgroup_size >= n:
            return [self.numbered_ideas_text()]
        for i in range(0, n, subgroup_size):
            chunk = self.members[i : i + subgroup_size]
            lines: list[str] = []
            for j, card in enumerate(chunk):
                g = i + j + 1
                lines.append(f"{g}) {card.description} \n")
            groups.append("".join(lines))
        return groups

    def add_member(self, card: RecordCardEmbedding) -> None:
        """Append a card and set its cluster_id to this cluster."""
        card.cluster_id = self.cluster_id
        self.members.append(card)
        self.rebuild_index_to_card()


@dataclass
class RecordCardExtended:
    """
    Extended idea record with comprehensive metadata and evolution tracking.

    Includes categorization, task context, strategy, aliases (historical versions),
    keywords, statistics, explanations, related ideas, and usage information.
    """

    id: str = ""
    category: str = ""
    description: str = ""
    task_description: str = ""
    task_description_summary: str = ""
    strategy: str = ""
    last_generation: int = 0
    aliases: list[dict[str, dict[str, str | list[str]]]] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    evolution_statistics: dict[str, Any] = field(default_factory=dict)
    explanation: dict[str, Any] = field(default_factory=dict)
    works_with: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        required_fields = [
            "id",
            "category",
            "description",
            "task_description",
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

        self.explanation = {
            "explanations": [kwargs["change_motivation"]],
            "summary": "",
        }
        self.aliases = []

    def update_idea(
        self,
        experiment_id: str,
        program_id: str | list[str] | None,
        generation: int | None,
        new_description: str | None = None,
        change_motivation: str | None = None,
    ) -> None:
        """
        Update idea with new programs, generation, and optionally new description.

        When description changes, archives current version to aliases before updating.

        Args:
            experiment_id: Experiment identifier for alias key.
            program_id: Single program ID or list of program IDs to add.
            generation: Generation number to update if greater than current.
            new_description: Optional new description (archives old version if provided).
            change_motivation: Optional explanation to add to explanation history.
        """
        if program_id is None:
            program_id = []
        elif isinstance(program_id, str):
            program_id = [program_id]

        if new_description is not None:
            current_description = new_description
            current_programs = copy(self.programs)
            current_description = copy(self.description)
            current_explanations = copy(self.explanation["explanations"])
            new_alias_key = f"{experiment_id}-{program_id[0]}"
            self.aliases.append(
                {
                    new_alias_key: {
                        "description": current_description,
                        "programs": current_programs,
                        "explanations": current_explanations,
                    }
                }
            )
            self.description = new_description
        if generation is not None and generation > self.last_generation:
            self.last_generation = generation
        self.programs.extend(program_id)
        if change_motivation is not None:
            self.explanation["explanations"].append(change_motivation)

    def add_explanation(self, explanation: str) -> None:
        """Add an explanation to the idea's explanation history."""
        self.explanation["explanations"].append(explanation)

    def update_metadata(
        self,
        keywords: list[str] | None = None,
        evolution_statistics: dict[str, Any] | None = None,
        works_with: list[str] | None = None,
        links: list[str] | None = None,
        usage: dict[str, Any] | None = None,
        summary: str | None = None,
        task_description_summary: str | None = None,
    ) -> None:
        """
        Update optional metadata fields for the idea.

        Args:
            keywords: List of keyword tags for the idea.
            evolution_statistics: Statistical data about idea evolution.
            works_with: List of related idea IDs that work well together.
            links: List of reference links or resources.
            usage: Dictionary of usage patterns or examples.
            summary: Explanation summary text.
            task_description_summary: Compact summary of the task description text.
        """
        if keywords is not None:
            self.keywords = keywords
        if evolution_statistics is not None:
            self.evolution_statistics = evolution_statistics
        if works_with is not None:
            self.works_with = works_with
        if links is not None:
            self.links = links
        if usage is not None:
            self.usage = usage
        if summary is not None:
            self.explanation["summary"] = summary
        if task_description_summary is not None:
            self.task_description_summary = task_description_summary


@dataclass
class RecordListV2:
    """
    Bounded list holding RecordCardExtended instances with capacity management.

    Provides operations for adding, finding, modifying, removing ideas,
    and identifying inactive ideas based on generation thresholds.
    """

    ideas: list[RecordCardExtended] = field(default_factory=list)
    num_ideas: int = 0
    max_ideas: int = 5

    def add_idea(self, idea_dict: dict[str, Any]) -> None:
        """Append a new idea from a dict of RecordCardExtended fields. Raises if list is full."""
        if self.num_ideas >= self.max_ideas:
            raise ValueError("Can't add new idea to list since it full")
        new_idea = RecordCardExtended(**idea_dict)
        self.ideas.append(new_idea)
        self.num_ideas = len(self.ideas)

    def add_idea_forced(self, idea_card: RecordCardExtended) -> None:
        """Append a new idea from a RecordCardExtended. Raises if list is full."""
        if self.num_ideas >= self.max_ideas:
            raise ValueError("Can't add new idea to list since it full")
        self.ideas.append(idea_card)
        self.num_ideas = len(self.ideas)

    def find_idea_index(self, idea_id: str) -> int | None:
        """Return index of idea with given id, or None if not found."""
        for index, idea in enumerate(self.ideas):
            if idea.id == idea_id:
                return index
        return None

    def get_idea(self, idea_id: str) -> RecordCardExtended:
        """Return the RecordCardExtended with the given id. Raises ValueError if not found."""
        idea_index = self.find_idea_index(idea_id)
        if idea_index is None:
            raise ValueError(f"Can't find idea with id {idea_id}")
        return self.ideas[idea_index]

    def modify_idea(
        self,
        idea_id: str,
        new_programs: list[str] | None = None,
        new_generation: int | None = None,
        new_description: str | None = None,
        change_motivation: str | None = None,
    ) -> bool:
        """Update last_generation and/or description (if greater). Returns True if found."""
        idea_index = self.find_idea_index(idea_id)
        if idea_index is None:
            return False
        idea = self.ideas[idea_index]
        idea.update_idea(
            experiment_id=idea_id,
            program_id=new_programs,
            generation=new_generation,
            new_description=new_description,
            change_motivation=change_motivation,
        )
        return True

    def modify_idea_metadata(
        self,
        idea_id: str,
        new_keywords: list[str] | None = None,
        new_evolution_statistics: dict[str, Any] | None = None,
        new_works_with: list[str] | None = None,
        new_links: list[str] | None = None,
        new_usage: dict[str, Any] | None = None,
        new_summary: str | None = None,
        new_task_description_summary: str | None = None,
    ) -> bool:
        """Update metadata fields via RecordCardExtended.update_metadata. Returns True if found."""
        idea_index = self.find_idea_index(idea_id)
        if idea_index is None:
            return False
        idea = self.ideas[idea_index]
        idea.update_metadata(
            keywords=new_keywords,
            evolution_statistics=new_evolution_statistics,
            works_with=new_works_with,
            links=new_links,
            usage=new_usage,
            summary=new_summary,
            task_description_summary=new_task_description_summary,
        )
        return True

    def remove_idea(self, idea_id: str) -> bool:
        """Remove the idea with the given id. Returns True if found and removed."""
        idea_index = self.find_idea_index(idea_id)
        if idea_index is None:
            return False
        self.ideas.pop(idea_index)
        self.num_ideas = len(self.ideas)
        return True

    def all_ideas_short(self) -> list[dict[str, str]]:
        """
        Return short representation of all ideas.

        Returns:
            List of dicts with keys: id (full UUID), short_id (first UUID segment),
            and description.
        """
        ideas = []
        for idea in self.ideas:
            ideas.append(
                {
                    "id": idea.id,
                    "short_id": idea.id.split("-")[0],
                    "description": idea.description,
                }
            )
        return ideas

    def is_full(self) -> bool:
        """Return True if this list has reached max_ideas."""
        if self.num_ideas < self.max_ideas:
            return False
        else:
            return True


@dataclass
class RecordBank:
    """
    Container for multiple RecordList instances with UUID management.

    Manages idea distribution across multiple bounded lists, ensures unique UUIDs,
    and provides operations for adding, modifying, removing, and importing ideas.
    Supports identifying inactive ideas based on generation thresholds.
    """

    ideas_lists: list[RecordListV2] = field(default_factory=list)
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

    def _append_record_list(
        self, idea_data: dict[str, Any] | RecordCardExtended, is_forced: bool = False
    ) -> None:
        """Append idea to first non-full list, or create a new RecordList and append there."""
        for index, record_list in enumerate(self.ideas_lists):
            if not record_list.is_full():
                if is_forced and isinstance(idea_data, RecordCardExtended):
                    self.ideas_lists[index].add_idea_forced(idea_data)
                elif not is_forced and isinstance(idea_data, dict):
                    self.ideas_lists[index].add_idea(idea_data)
                else:
                    raise ValueError(f"Invalid idea data type: {type(idea_data)}")
                return
        new_list = RecordListV2(max_ideas=self.list_max_ideas)
        if is_forced and isinstance(idea_data, RecordCardExtended):
            new_list.add_idea_forced(idea_data)
        elif not is_forced and isinstance(idea_data, dict):
            new_list.add_idea(idea_data)
        else:
            raise ValueError(f"Invalid idea data type: {type(idea_data)}")
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

    def add_idea(
        self,
        description: str,
        linked_program: str,
        generation: int,
        category: str = "",
        strategy: str = "",
        task_description: str = "",
        change_motivation: str = "",
    ) -> None:
        """Create a new idea with a unique id and append it to a list in the bank."""
        idea_id = self._unique_id_pair()
        self.uuids.append(idea_id)
        idea_dict = {
            "id": idea_id,
            "description": description,
            "linked_programs": [linked_program],
            "programs": [linked_program],
            "category": category,
            "strategy": strategy,
            "task_description": task_description,
            "last_generation": generation,
            "change_motivation": change_motivation,
        }
        self._append_record_list(idea_dict)

    def modify_idea(
        self,
        idea_id: str,
        new_programs: list[str] | None = None,
        new_generation: int | None = None,
        new_description: str | None = None,
        change_motivation: str | None = None,
    ) -> None:
        """Update the RecordCard's linked_programs (extend) and/or last_generation (if greater)."""
        if idea_id not in self.uuids:
            raise ValueError(f"No idea with id {idea_id} found")
        for index, record_list in enumerate(self.ideas_lists):
            if record_list.find_idea_index(idea_id) is not None:
                self.ideas_lists[index].modify_idea(
                    idea_id,
                    new_programs,
                    new_generation,
                    new_description,
                    change_motivation,
                )
                return

    def modify_idea_metadata(
        self,
        idea_id: str,
        new_keywords: list[str] | None = None,
        new_summary: str | None = None,
        new_task_description_summary: str | None = None,
    ) -> None:
        """Update keywords and/or explanation summary on a RecordCardExtended in the bank."""
        if idea_id not in self.uuids:
            raise ValueError(f"No idea with id {idea_id} found")
        for record_list in self.ideas_lists:
            if record_list.find_idea_index(idea_id) is not None:
                record_list.modify_idea_metadata(
                    idea_id,
                    new_keywords=new_keywords,
                    new_summary=new_summary,
                    new_task_description_summary=new_task_description_summary,
                )
                return

    def import_idea_extended(
        self, new_idea: RecordCardExtended, is_forced: bool = False
    ) -> None:
        """Append an existing RecordCardExtended; assign new id if its id already exists in the bank."""
        if new_idea.id in self.uuids:
            new_uuid = self._unique_id_pair()
            new_idea.id = new_uuid
        self.uuids.append(new_idea.id)
        if is_forced:
            self._append_record_list(new_idea, is_forced=is_forced)
        else:
            idea_dict = asdict(new_idea)
            idea_dict["programs"] = list(idea_dict["programs"])
            self._append_record_list(idea_dict, is_forced=is_forced)

    def get_idea(self, idea_id: str) -> RecordCardExtended:
        """Return the RecordCard with the given id. Raises ValueError if not found."""
        if idea_id not in self.uuids:
            raise ValueError(f"No idea with id {idea_id} found")
        for index, record_list in enumerate(self.ideas_lists):
            if record_list.find_idea_index(idea_id) is not None:
                return self.ideas_lists[index].get_idea(idea_id)
        raise ValueError(f"No idea with id {idea_id} found in any list")

    def remove_idea(self, idea_id: str) -> None:
        """Remove the idea with the given id from its list and from uuids. Raises if not found."""
        if idea_id not in self.uuids:
            raise ValueError(f"No idea with id {idea_id} found")
        for index, record_list in enumerate(self.ideas_lists):
            if record_list.find_idea_index(idea_id) is not None:
                self.ideas_lists[index].remove_idea(idea_id)
                self._remove_id_from_bank(idea_id)
                return

    def get_record_list(self, list_num: int) -> RecordListV2:
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

    def all_ideas_cards(self) -> list[RecordCardExtended]:
        """Return all ideas in the bank."""
        ideas = []
        for ideas_list in self.ideas_lists:
            ideas.extend(ideas_list.ideas)
        return ideas


@dataclass
class IncomingIdeas:
    """
    Container for processing and classifying incoming program ideas.

    Tracks classification status, target idea mappings, and rewrite flags
    for each incoming idea during the classification workflow.
    """

    ideas: list[dict[str, Any]] = field(default_factory=list)
    mapping: dict[int, str] = field(default_factory=dict)

    def __init__(self, ideas: list[dict[str, str]]) -> None:
        """
        Initialize with list of incoming ideas.

        Args:
            ideas: List of dicts with keys "description" and "explanation".
        """
        self.ideas = []
        for idea in normalize_improvements(ideas):
            idea_dict = {
                "description": idea["description"],
                "change_motivation": idea["explanation"],
                "target_idea_id": "",
                "rewrite": False,
                "classified": False,
            }
            self.ideas.append(idea_dict)
        self.update_mapping()

    def update_mapping(self) -> None:
        """Rebuild mapping from sequence numbers to unclassified idea descriptions."""
        mapping: dict[int, str] = {}
        c = 1
        for idea in self.ideas:
            if not idea["classified"]:
                mapping[c] = idea["description"]
                c += 1
        self.mapping = mapping

    def get_list_of_ideas(self) -> str:
        """
        Format unclassified ideas as numbered text list.

        Returns:
            String with format "1) description\\n2) description\\n..." for unclassified ideas.
        """
        text = ""
        c = 1
        for idea in self.ideas:
            if not idea["classified"]:
                text += f"{c}) {idea['description']} \n"
                c += 1
        return text

    def update_idea(self, idea_number: int, target_idea_id: str, rewrite: bool) -> None:
        """
        Mark idea as classified with target ID and rewrite flag.

        Args:
            idea_number: Sequence number from mapping (1-indexed).
            target_idea_id: UUID of matching existing idea.
            rewrite: Whether this idea rewrites the target idea's description.
        """

        idea_description = self.mapping.get(idea_number)
        if idea_description is None:
            logger.warning(f"No idea with number {idea_number} found")
            return
        for index, idea in enumerate(self.ideas):
            if idea["description"] == idea_description:
                self.ideas[index]["target_idea_id"] = target_idea_id
                self.ideas[index]["rewrite"] = rewrite
                self.ideas[index]["classified"] = True
                break

    @property
    def new_ideas_count(self) -> int:
        """Count of unclassified (new) ideas."""
        return len([idea for idea in self.ideas if not idea["classified"]])

    @property
    def present_ideas_count(self) -> int:
        """Count of classified (already known) ideas."""
        return len([idea for idea in self.ideas if idea["classified"]])
