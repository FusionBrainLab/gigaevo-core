import asyncio
import contextlib

from loguru import logger

from gigaevo.evolution.engine import EvolutionEngine


class EvolutionRunner:
    def __init__(self, engine: EvolutionEngine) -> None:
        self._engine = engine
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._engine.run(), name="evolution-engine")
        logger.info("[EngineDriver] Evolution engine started")

    async def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception as e:
            logger.warning("[EngineDriver] stop() raised: {}", e)
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
            logger.info("[EngineDriver] Evolution engine stopped")

    def pause(self) -> None:
        self._engine.pause()

    def resume(self) -> None:
        self._engine.resume()

    def is_running(self) -> bool:
        return self._engine.is_running()

    async def get_status(self) -> dict[str, object]:
        return await self._engine.get_status()

    @property
    def task(self) -> asyncio.Task | None:
        return self._task
