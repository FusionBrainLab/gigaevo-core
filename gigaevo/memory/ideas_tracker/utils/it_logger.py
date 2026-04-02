from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gigaevo.memory.ideas_tracker.components.data_components import (
        RecordCardExtended,
    )
    from gigaevo.memory.ideas_tracker.components.records_manager import RecordManager


class IdeasTrackerLogger:
    """
    Structured logger for ideas_tracker.py components.

    Records initialization parameters, idea additions/modifications/movements,
    rankings, bank states, programs, and statistics. Each session creates a
    timestamped directory with log.txt and JSON snapshot files.
    """

    def __init__(
        self, ideas_tracker_path: str | Path, *, logs_dir: str | Path | None = None
    ):
        """
        Initialize logger with timestamped session directory.

        Args:
            ideas_tracker_path: Path to ideas_tracker.py file (parent becomes logs root).
        """
        self.ideas_tracker_path = Path(ideas_tracker_path)
        self.logs_dir = (
            Path(logs_dir)
            if logs_dir is not None
            else (self.ideas_tracker_path.parent / "logs")
        )
        self.session_dir: Path | None = None
        self.log_file: Path | None = None
        self.rankings_file: Path | None = None
        self.banks_file: Path | None = None
        self.programs_file: Path | None = None
        self.best_ideas_file: Path | None = None
        self.memory_usage_updates_file: Path | None = None

        os.makedirs(self.logs_dir, exist_ok=True)
        self._create_session_dir()

    def _create_session_dir(self) -> None:
        """
        Create timestamped session directory and initialize log files.

        Format: YYYY-MM-DD_HH-MM-SS. Initializes log.txt and empty JSON arrays
        for rankings, banks, programs, and best_ideas.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.logs_dir / timestamp
        os.makedirs(self.session_dir, exist_ok=True)

        self.log_file = self.session_dir / "log.txt"
        self.rankings_file = self.session_dir / "rankings.json"
        self.banks_file = self.session_dir / "banks.json"
        self.programs_file = self.session_dir / "programs.json"
        self.best_ideas_file = self.session_dir / "best_ideas.json"
        self.memory_usage_updates_file = self.session_dir / "memory_usage_updates.json"

        if not self.rankings_file.exists():
            with open(self.rankings_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.banks_file.exists():
            with open(self.banks_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.programs_file.exists():
            with open(self.programs_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.best_ideas_file.exists():
            with open(self.best_ideas_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.memory_usage_updates_file.exists():
            with open(self.memory_usage_updates_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _get_timestamp(self) -> str:
        """
        Get current timestamp string.

        Returns:
            Timestamp in format "YYYY-MM-DD HH:MM:SS".
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _idea_to_dict(self, idea: RecordCardExtended) -> dict[str, Any]:
        """
        Serialize idea to JSON-serializable dict.

        Supports both RecordCard (linked_programs) and RecordCardExtended (programs, etc.).

        Args:
            idea: RecordCard or RecordCardExtended instance.

        Returns:
            Dictionary with all idea fields.
        """
        if hasattr(idea, "programs"):
            return {
                "id": idea.id,
                "category": getattr(idea, "category", ""),
                "description": idea.description,
                "task_description": getattr(idea, "task_description", ""),
                "task_description_summary": getattr(
                    idea, "task_description_summary", ""
                ),
                "strategy": getattr(idea, "strategy", ""),
                "last_generation": idea.last_generation,
                "programs": list(getattr(idea, "programs", [])),
                "aliases": getattr(idea, "aliases", []),
                "keywords": getattr(idea, "keywords", []),
                "evolution_statistics": getattr(idea, "evolution_statistics", {}),
                "explanation": getattr(idea, "explanation", {}),
                "works_with": getattr(idea, "works_with", []),
                "links": getattr(idea, "links", []),
                "usage": getattr(idea, "usage", {}),
            }
        return {
            "id": idea.id,
            "description": idea.description,
            "linked_programs": list(getattr(idea, "linked_programs", [])),
            "last_generation": idea.last_generation,
        }

    def _write_log(
        self, program_id: str, destination: str, action: str, parameters: dict[str, Any]
    ) -> None:
        """
        Write structured log entry to log.txt.

        Args:
            program_id: Identifier for the program/component.
            destination: Target bank or component.
            action: Action being performed.
            parameters: Dictionary of action parameters to log.
        """
        timestamp = self._get_timestamp()
        log_entry = f'[{timestamp}]: "{program_id}" -> "{destination}"/"{action}"\n'
        log_entry += "parameters:\n"

        for key, value in parameters.items():
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
                log_entry += f"{key}: [{value_str}]\n"
            else:
                log_entry += f"{key}: {value}\n"

        log_entry += "\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_init(self, **init_params: Any) -> None:
        """
        Log initialization parameters.

        Args:
            **init_params: Keyword arguments representing init parameters
        """
        self._write_log(
            program_id="init",
            destination="ideas_tracker",
            action="initialization",
            parameters=init_params,
        )

    def log_new_idea(
        self,
        description: str,
        generation: int,
        linked_program: str,
        *,
        category: str = "",
        strategy: str = "",
    ) -> None:
        """
        Log addition of a new idea to active bank.

        Args:
            description: Idea description (not logged separately, context only).
            generation: Generation number when idea appeared.
            linked_program: Program ID where idea was first seen.
            category: Optional category for RecordCardExtended.
            strategy: Optional strategy for RecordCardExtended.
        """
        parameters: dict[str, Any] = {
            "generation": generation,
            "linked_program": linked_program,
        }
        if category:
            parameters["category"] = category
        if strategy:
            parameters["strategy"] = strategy

        self._write_log(
            program_id=linked_program,
            destination="active_bank",
            action="add_new_idea",
            parameters=parameters,
        )

    def log_modify_idea(
        self,
        idea_id: str,
        old_description: str | None,
        new_description: str,
        new_linked_programs: list[str],
        *,
        category: str = "",
        strategy: str = "",
        programs: list[str] | None = None,
    ) -> None:
        """
        Log modification of an existing idea.

        Args:
            idea_id: UUID of modified idea.
            old_description: Previous description (if available).
            new_description: Updated description.
            new_linked_programs: Newly added program IDs.
            category: Optional category for RecordCardExtended.
            strategy: Optional strategy for RecordCardExtended.
            programs: Optional full programs list for RecordCardExtended.
        """
        program_id = new_linked_programs[0] if new_linked_programs else idea_id
        parameters: dict[str, Any] = {
            "idea_id": idea_id,
            "old_description": old_description,
            "new_description": new_description,
            "new_linked_programs": new_linked_programs,
        }
        if category:
            parameters["category"] = category
        if strategy:
            parameters["strategy"] = strategy
        if programs is not None:
            parameters["programs"] = programs
        self._write_log(
            program_id=program_id,
            destination="active_bank",
            action="modify_idea",
            parameters=parameters,
        )

    def log_banks(
        self,
        active_bank: list[dict[str, Any]] | dict[str, Any],
    ) -> None:
        """
        Append timestamped snapshot of both banks to banks.json.

        Args:
            active_bank: Active bank state as list of idea dicts or dict.
            inactive_bank: Inactive bank state as list of idea dicts or dict.
        """
        banks_state = {
            "active_bank": active_bank,
            "timestamp": self._get_timestamp(),
        }

        # Read existing banks
        if self.banks_file.exists():
            with open(self.banks_file, encoding="utf-8") as f:
                existing_banks = json.load(f)
        else:
            existing_banks = []

        # Append new banks state
        existing_banks.append(banks_state)

        # Write back
        with open(self.banks_file, "w", encoding="utf-8") as f:
            json.dump(existing_banks, f, indent=4)

    def log_programs(self, programs: list[dict[str, Any]]) -> None:
        """
        Append programs snapshot to programs.json.

        Args:
            programs: List of program dictionaries to persist.
        """
        if self.programs_file is None:
            return

        snapshot = {
            "timestamp": self._get_timestamp(),
            "programs": programs,
        }

        # Read existing snapshots
        if self.programs_file.exists():
            with open(self.programs_file, encoding="utf-8") as f:
                try:
                    existing_programs = json.load(f)
                except json.JSONDecodeError:
                    existing_programs = []
        else:
            existing_programs = []

        # Append new snapshot
        existing_programs.append(snapshot)

        # Write back
        with open(self.programs_file, "w", encoding="utf-8") as f:
            json.dump(existing_programs, f, indent=4)

    def log_best_ideas(self, statistics: dict[str, Any]) -> None:
        """
        Append best ideas statistics to best_ideas.json.

        Args:
            statistics: Result of get_statistics() (e.g. top_fitness_ideas, top_delta_ideas).
        """
        if self.best_ideas_file is None:
            return

        snapshot = {
            "timestamp": self._get_timestamp(),
            **statistics,
        }

        # Read existing snapshots
        if self.best_ideas_file.exists():
            with open(self.best_ideas_file, encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        existing.append(snapshot)

        with open(self.best_ideas_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4)

    def log_memory_usage_updates(
        self, usage_updates: dict[str, dict[str, Any]]
    ) -> None:
        """
        Append memory usage updates snapshot to memory_usage_updates.json.

        Args:
            usage_updates: Mapping card_id -> usage payload snapshot.
        """
        if self.memory_usage_updates_file is None:
            return

        snapshot = {
            "timestamp": self._get_timestamp(),
            "usage_updates": usage_updates,
        }

        if self.memory_usage_updates_file.exists():
            with open(self.memory_usage_updates_file, encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
        else:
            existing = []

        existing.append(snapshot)
        with open(self.memory_usage_updates_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4)

    def dump_final_state(self, record_manager: RecordManager) -> None:
        """
        Extract and log final state of both idea banks.

        Serializes all ideas from active and inactive banks and writes to banks.json
        and log.txt.

        Args:
            record_manager: RecordManager instance containing both banks.
        """
        active_bank_data = []
        for list_idx in range(record_manager.record_bank.num_lists):
            record_list = record_manager.record_bank.get_record_list(list_idx)
            for idea in record_list.ideas:
                active_bank_data.append(self._idea_to_dict(idea))

        banks_state = {
            "active_bank": active_bank_data,
        }

        self.log_banks(banks_state["active_bank"])

        self._write_log(
            program_id="final_state",
            destination="both_banks",
            action="dump_final_state",
            parameters={
                "active_bank_size": len(active_bank_data),
            },
        )
