# src/programs/stages/persistent.py
from __future__ import annotations

from typing import Any, ClassVar, Self, override
import threading
from loguru import logger

from src.programs.stages.base import Stage


class PersistentStage(Stage):
    _instances: ClassVar[dict[tuple[type[Self], str], Self]] = {}
    _registry_lock: ClassVar[threading.RLock] = threading.RLock()

    @override
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        key = cls._extract_key(kwargs)
        with cls._registry_lock:
            inst = cls._instances.get((cls, key))
            if inst is not None:
                return inst
            inst = super().__new__(cls)
            inst._persistence_key = key  # type: ignore[attr-defined]
            inst._initialized = False     # type: ignore[attr-defined]
            inst._init_lock = threading.Lock()  # type: ignore[attr-defined]
            cls._instances[(cls, key)] = inst
            return inst

    @override
    def __init__(
        self,
        *,
        timeout: float,
        persistence_key: str,
        stage_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if getattr(self, "_initialized", False):  # type: ignore[attr-defined]
            return
        with self._init_lock:  # type: ignore[attr-defined]
            if getattr(self, "_initialized", False):  # type: ignore[attr-defined]
                return
            super().__init__(timeout=timeout, stage_name=stage_name or self.__class__.__name__)
            self._memory: dict[str, Any] = {}
            self._initialized = True  # type: ignore[attr-defined]
            logger.debug(
                "[{stage}] PersistentStage initialized (key={key})",
                stage=self.stage_name,
                key=self._persistence_key,  # type: ignore[attr-defined]
            )

    @classmethod
    def _extract_key(cls, kwargs: dict[str, Any]) -> str:
        key = kwargs.get("persistence_key")
        if not key:
            raise ValueError("persistence_key is required for PersistentStage")
        return str(key)

    @property
    def persistence_key(self) -> str:
        return self._persistence_key  # type: ignore[attr-defined]

    def get_memory(self) -> dict[str, Any]:
        return self._memory

    def reset_memory(self) -> None:
        self._memory.clear()
        logger.debug("[{stage}] memory cleared", stage=self.stage_name)

    @classmethod
    def get_instance(cls, *, persistence_key: str) -> Self | None:
        with cls._registry_lock:
            inst = cls._instances.get((cls, persistence_key))
            return inst if isinstance(inst, cls) else None

    @classmethod
    def destroy_instance(cls, *, persistence_key: str) -> bool:
        with cls._registry_lock:
            inst = cls._instances.pop((cls, persistence_key), None)
        if inst is not None:
            logger.debug("[{stage}] PersistentStage destroyed (key={key})", stage=inst.stage_name, key=persistence_key)
            return True
        return False

    @classmethod
    def destroy_all(cls) -> int:
        with cls._registry_lock:
            keys = [k for k in cls._instances.keys() if k[0] is cls]
            for k in keys:
                cls._instances.pop(k, None)
        for _, key in [(cls, k[1]) for k in keys]:
            logger.debug("[{stage}] PersistentStage destroyed (key={key})", stage=cls.__name__, key=key)
        return len(keys)
