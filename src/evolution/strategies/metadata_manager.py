from loguru import logger

from src.database.redis_program_storage import RedisProgramStorage
from src.programs.program import Program


class MetadataManager:
    def __init__(self, program_storage: RedisProgramStorage):
        self.program_storage = program_storage

    async def set_current_island(
        self, program: Program, island_id: str
    ) -> None:
        try:
            program.metadata.setdefault("home_island", island_id)
            program.metadata["current_island"] = island_id
            await self.program_storage.update(program)
            logger.debug(
                f"MetadataManager: Set current_island={island_id} for program {program.id}"
            )
        except Exception as e:
            logger.warning(
                f"MetadataManager: Failed to set current_island for program {program.id}: {e}"
            )

    async def clear_current_island(self, program: Program) -> None:
        try:
            if program.metadata.get("current_island"):
                program.metadata["current_island"] = None
                await self.program_storage.update(program)
                logger.debug(
                    f"MetadataManager: Cleared current_island for program {program.id}"
                )
        except Exception as e:
            logger.warning(
                f"MetadataManager: Failed to clear current_island for program {program.id}: {e}"
            )
