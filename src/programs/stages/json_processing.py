"""Simple dictionary processing stages for MetaEvolve."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.exceptions import StageError
from src.programs.program import Program
from src.programs.stages.base import Stage
from src.programs.core_types import ProgramStageResult, StageState
from src.programs.utils import build_stage_result
from src.runner.stage_registry import StageRegistry


@StageRegistry.register(
    description="Merge two Python dictionaries using simple dict merge"
)
class MergeDictStage(Stage):
    """Stage that merges two Python dictionaries using simple dict merge syntax.
    
    This stage takes two dictionary inputs and merges them using {**first, **second}
    which means the second dictionary overwrites any conflicting keys from the first.
    """

    @classmethod
    def mandatory_inputs(cls) -> List[str]:
        return ["first_dict", "second_dict"]

    @classmethod
    def optional_inputs(cls) -> List[str]:
        return []

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Execute the dictionary merge operation."""
        try:
            logger.debug(
                f"[{self.stage_name}] Program {program.id}: Merging dictionaries"
            )

            first_dict = self.get_input("first_dict")
            second_dict = self.get_input("second_dict")

            if not isinstance(first_dict, dict):
                raise StageError("first_dict must be a dictionary")
            if not isinstance(second_dict, dict):
                raise StageError("second_dict must be a dictionary")

            merged_result = {**first_dict, **second_dict}

            logger.info(
                f"[{self.stage_name}] Successfully merged dictionaries for program {program.id}"
            )

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=merged_result,
                stage_name=self.stage_name,
                metadata={
                    "merged_dict": merged_result,
                    "first_input_size": len(str(first_dict)),
                    "second_input_size": len(str(second_dict)),
                    "merged_output_size": len(str(merged_result)),
                },
            )

        except Exception as e:
            logger.error(
                f"[{self.stage_name}] Dictionary merge failed for program {program.id}: {e}"
            )
            raise StageError(
                f"Dictionary merge operation failed: {e}",
                stage_name=self.stage_name,
                stage_type="data_processing",
                cause=e,
            ) from e


@StageRegistry.register(
    description="Convert JSON string to Python object"
)
class ParseJSONStage(Stage):
    """Stage that parses a JSON string into a Python object."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @classmethod
    def mandatory_inputs(cls) -> List[str]:
        return ["json_string"]

    @classmethod
    def optional_inputs(cls) -> List[str]:
        return []

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Execute the JSON parsing operation."""
        try:
            logger.debug(
                f"[{self.stage_name}] Program {program.id}: Parsing JSON string"
            )

            # Get input using the stage's input mechanism
            json_string = self.get_input("json_string")

            if not isinstance(json_string, str):
                raise StageError("json_string must be a string")

            # Parse JSON
            try:
                parsed_json = json.loads(json_string)
            except json.JSONDecodeError as e:
                raise StageError(f"Invalid JSON string: {e}") from e

            # Store result in program metadata
            program.set_metadata("parsed_json", parsed_json)

            logger.info(
                f"[{self.stage_name}] Successfully parsed JSON for program {program.id}"
            )

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=parsed_json,
                stage_name=self.stage_name,
                metadata={
                    "parsed_json": parsed_json,
                    "input_length": len(json_string),
                    "parsed_type": type(parsed_json).__name__,
                },
            )

        except Exception as e:
            logger.error(
                f"[{self.stage_name}] JSON parsing failed for program {program.id}: {e}"
            )
            raise StageError(
                f"JSON parsing operation failed: {e}",
                stage_name=self.stage_name,
                stage_type="data_processing",
                cause=e,
            ) from e


@StageRegistry.register(
    description="Convert Python object to JSON string"
)
class StringifyJSONStage(Stage):
    """Stage that converts a Python object to a JSON string."""

    def __init__(
        self,
        indent: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.indent = indent

    @classmethod
    def mandatory_inputs(cls) -> List[str]:
        return ["python_object"]

    @classmethod
    def optional_inputs(cls) -> List[str]:
        return []

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Execute the JSON stringification operation."""
        try:
            logger.debug(
                f"[{self.stage_name}] Program {program.id}: Converting to JSON string"
            )

            python_object = self.get_input("python_object")

            try:
                json_string = json.dumps(python_object, indent=self.indent)
            except (TypeError, ValueError) as e:
                raise StageError(f"Cannot convert to JSON: {e}") from e

            logger.info(
                f"[{self.stage_name}] Successfully converted to JSON string for program {program.id}"
            )

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output=json_string,
                stage_name=self.stage_name,
                metadata={
                    "json_string": json_string,
                    "output_length": len(json_string),
                    "input_type": type(python_object).__name__,
                },
            )

        except Exception as e:
            logger.error(
                f"[{self.stage_name}] JSON stringification failed for program {program.id}: {e}"
            )
            raise StageError(
                f"JSON stringification operation failed: {e}",
                stage_name=self.stage_name,
                stage_type="data_processing",
                cause=e,
            ) from e

