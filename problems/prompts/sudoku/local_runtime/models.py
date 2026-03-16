from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class Action:
    @property
    def text(self) -> str:
        raise NotImplementedError


class NodeAction(Action):
    __slots__ = ("node_id", "_text")

    def __init__(self, node_id: int, text: str):
        self.node_id = int(node_id)
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def __str__(self) -> str:
        return f"<node>{self.node_id} {self.text}</node>"


class BacktrackAction(Action):
    __slots__ = ("target_id", "reason")

    def __init__(self, target_id: int, reason: Optional[str] = None):
        self.target_id = int(target_id)
        self.reason = reason or f"backtrack to {target_id}"

    @property
    def text(self) -> str:
        return ""

    def __str__(self) -> str:
        return f"<backtrack>{self.target_id}</backtrack>"


class DoneAction(Action):
    __slots__ = ("answer",)

    def __init__(self, answer: str):
        self.answer = answer

    @property
    def text(self) -> str:
        return self.answer

    def __str__(self) -> str:
        return f"<done>{self.text}</done>"


@dataclass(slots=True)
class Node:
    parent: Optional["Node"]
    action: Action
    state: Any = None


@dataclass(slots=True)
class PathContext:
    nodes: list[Node]

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    @property
    def last_node(self) -> Node:
        return self.nodes[-1]


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    comment: str = ""
