from loguru import logger

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.programs.program import Program


class MetadataManager:
    def __init__(self, program_storage: RedisProgramStorage):
        self.program_storage = program_storage

    async def set_current_island(self, program: Program, island_id: str) -> None:
        program.metadata.setdefault("home_island", island_id)
        program.metadata["current_island"] = island_id
        await self.program_storage.update(program)
        logger.debug(
            f"MetadataManager: Set current_island={island_id} for program {program.id}"
        )

    async def clear_current_island(self, program: Program) -> None:
        if program.metadata.get("current_island"):
            program.metadata["current_island"] = None
            await self.program_storage.update(program)
            logger.debug(
                f"MetadataManager: Cleared current_island for program {program.id}"
            )
