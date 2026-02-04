from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Optional

from gigaevo.llm.ideas_tracker.components.records_manager import RecordManager


class IdeasTrackerLogger:
    """
    Custom logger for ideas_tracker.py components.
    Records init parameters, idea additions, modifications, movements, and final state.
    """

    def __init__(self, ideas_tracker_path: str | Path):
        """
        Initialize the logger.

        Args:
            ideas_tracker_path: Path to ideas_tracker.py file
        """
        self.ideas_tracker_path = Path(ideas_tracker_path)
        self.logs_dir = self.ideas_tracker_path.parent / "logs"
        self.session_dir: Optional[Path] = None
        self.log_file: Optional[Path] = None
        self.rankings_file: Optional[Path] = None
        self.banks_file: Optional[Path] = None
        self.programs_file: Optional[Path] = None

        # Create logs directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create session directory with timestamp
        self._create_session_dir()

    def _create_session_dir(self) -> None:
        """Create a new session directory with timestamp format: year-month-date_hours_minutes_seconds"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.logs_dir / timestamp
        os.makedirs(self.session_dir, exist_ok=True)

        # Initialize log files
        self.log_file = self.session_dir / "log.txt"
        self.rankings_file = self.session_dir / "rankings.json"
        self.banks_file = self.session_dir / "banks.json"
        self.programs_file = self.session_dir / "programs.json"

        # Initialize JSON files as empty arrays if they don't exist
        if not self.rankings_file.exists():
            with open(self.rankings_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.banks_file.exists():
            with open(self.banks_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.programs_file.exists():
            with open(self.programs_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _get_timestamp(self) -> str:
        """Get current timestamp in the format [time]"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_log(
        self, program_id: str, destination: str, action: str, parameters: dict[str, Any]
    ) -> None:
        """
        Write a log entry to log.txt in the specified format.

        Format:
        [time]: "[program_id]" -> "[destination]/["action"]"
        parameters:
        param1: ...
        param2: ...
        ...
        """
        timestamp = self._get_timestamp()
        log_entry = f'[{timestamp}]: "{program_id}" -> "{destination}"/"{action}"\n'
        log_entry += "parameters:\n"

        for key, value in parameters.items():
            # Format list values nicely
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
        self, description: str, generation: int, linked_program: str
    ) -> None:
        """
        Log addition of a new idea.

        Args:
            description: Idea description
            generation: Generation number
            linked_program: Program ID linked to this idea
        """
        self._write_log(
            program_id=linked_program,
            destination="active_bank",
            action="add_new_idea",
            parameters={
                "description": description,
                "generation": generation,
                "linked_program": linked_program,
            },
        )

    def log_modify_idea(
        self, idea_id: str, description: str, new_linked_programs: list[str]
    ) -> None:
        """
        Log modification of an existing idea.

        Args:
            idea_id: ID of the modified idea
            description: Idea description
            new_linked_programs: List of newly linked program IDs
        """
        # Use the first program ID if available, otherwise use idea_id
        program_id = new_linked_programs[0] if new_linked_programs else idea_id

        self._write_log(
            program_id=program_id,
            destination="active_bank",
            action="modify_idea",
            parameters={
                "idea_id": idea_id,
                "description": description,
                "new_linked_programs": new_linked_programs,
            },
        )

    def log_move_idea(
        self,
        idea_id: str,
        description: str,
        linked_programs: list[str],
        destination: str,
    ) -> None:
        """
        Log movement of an idea between banks.

        Args:
            idea_id: ID of the moved idea
            description: Idea description
            linked_programs: List of linked program IDs
            destination: Destination bank ("inactive_bank" or "active_bank")
        """
        # Use the first program ID if available, otherwise use idea_id
        program_id = linked_programs[0] if linked_programs else idea_id

        self._write_log(
            program_id=program_id,
            destination=destination,
            action="move_idea",
            parameters={
                "idea_id": idea_id,
                "description": description,
                "linked_programs": linked_programs,
                "destination": destination,
            },
        )

    def log_rankings(self, rankings: list[dict[str, Any]]) -> None:
        """
        Append rankings to rankings.json.

        Args:
            rankings: List of ranking dictionaries
        """
        # Read existing rankings
        if self.rankings_file.exists():
            with open(self.rankings_file, "r", encoding="utf-8") as f:
                existing_rankings = json.load(f)
        else:
            existing_rankings = []

        # Append new rankings
        existing_rankings.append(rankings)

        # Write back
        with open(self.rankings_file, "w", encoding="utf-8") as f:
            json.dump(existing_rankings, f, indent=2)

    def log_banks(
        self, active_bank: dict[str, Any], inactive_bank: dict[str, Any]
    ) -> None:
        """
        Append banks state to banks.json.

        Args:
            active_bank: Dictionary representing the active bank state
            inactive_bank: Dictionary representing the inactive bank state
        """
        banks_state = {
            "active_bank": active_bank,
            "inactive_bank": inactive_bank,
            "timestamp": self._get_timestamp(),
        }

        # Read existing banks
        if self.banks_file.exists():
            with open(self.banks_file, "r", encoding="utf-8") as f:
                existing_banks = json.load(f)
        else:
            existing_banks = []

        # Append new banks state
        existing_banks.append(banks_state)

        # Write back
        with open(self.banks_file, "w", encoding="utf-8") as f:
            json.dump(existing_banks, f, indent=2)

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
            with open(self.programs_file, "r", encoding="utf-8") as f:
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
            json.dump(existing_programs, f, indent=2)

    def dump_final_state(self, record_manager: RecordManager) -> None:
        """
        Dump final state of idea banks.

        Args:
            record_manager: RecordManager instance to extract bank states from
        """
        # Extract active bank state
        active_bank_data = []
        for list_idx in range(record_manager.record_bank.num_lists):
            record_list = record_manager.record_bank.get_record_list(list_idx)
            for idea in record_list.ideas:
                active_bank_data.append(
                    {
                        "id": idea.id,
                        "short_id": idea.short_id,
                        "description": idea.description,
                        "linked_programs": idea.linked_programs,
                        "last_generation": idea.last_generation,
                    }
                )

        # Extract inactive bank state
        inactive_bank_data = []
        for list_idx in range(record_manager.inactive_record_bank.num_lists):
            record_list = record_manager.inactive_record_bank.get_record_list(list_idx)
            for idea in record_list.ideas:
                inactive_bank_data.append(
                    {
                        "id": idea.id,
                        "short_id": idea.short_id,
                        "description": idea.description,
                        "linked_programs": idea.linked_programs,
                        "last_generation": idea.last_generation,
                    }
                )

        banks_state = {
            "active_bank": active_bank_data,
            "inactive_bank": inactive_bank_data,
        }

        self.log_banks(banks_state["active_bank"], banks_state["inactive_bank"])

        # Also log this as a final state dump action
        self._write_log(
            program_id="final_state",
            destination="both_banks",
            action="dump_final_state",
            parameters={
                "active_bank_size": len(active_bank_data),
                "inactive_bank_size": len(inactive_bank_data),
            },
        )
