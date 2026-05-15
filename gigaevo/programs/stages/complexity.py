from __future__ import annotations

import ast
from collections import Counter
import math
from typing import Any

from loguru import logger

from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import FloatDictContainer
from gigaevo.programs.stages.stage_registry import StageRegistry


class NumericalComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.call_count = 0
        self.binop_count = 0
        self.subscript_count = 0
        self.loop_count = 0
        self.condition_count = 0
        self.function_def_count = 0
        self.class_def_count = 0
        self.identifiers: set[str] = set()
        self.current_depth = 0
        self.max_depth = 0

    def visit_Call(self, node):
        self.call_count += 1
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.binop_count += 1
        self.generic_visit(node)

    def visit_Subscript(self, node):
        self.subscript_count += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.loop_count += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.loop_count += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.condition_count += 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        # Ternary `a if cond else b` is a branch decision.
        self.condition_count += 1
        self.generic_visit(node)

    def visit_Match(self, node):
        # Each case branch is a discrete condition, mirroring how if/elif
        # chains are counted (each If node contributes 1).
        self.condition_count += len(node.cases)
        self.generic_visit(node)

    def visit_Try(self, node):
        # Each except-handler is a branch path off the try body.
        self.condition_count += len(node.handlers)
        self.generic_visit(node)

    def visit_TryStar(self, node):
        # `except*` (PEP 654) — same branching semantics as Try.
        self.condition_count += len(node.handlers)
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.loop_count += 1
        self.generic_visit(node)

    def _visit_comprehension(self, node):
        # Comprehensions are loops in disguise. Each generator contributes a
        # loop, and each `if` clause inside it is an extra condition.
        self.loop_count += len(node.generators)
        self.condition_count += sum(len(gen.ifs) for gen in node.generators)
        self.generic_visit(node)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_FunctionDef(self, node):
        self.function_def_count += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.function_def_count += 1
        self.generic_visit(node)

    def visit_Lambda(self, node):
        # Lambda is an anonymous function definition; count it alongside def.
        self.function_def_count += 1
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.class_def_count += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        self.identifiers.add(node.id)
        self.generic_visit(node)

    def generic_visit(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        super().generic_visit(node)
        self.current_depth -= 1


def compute_numerical_complexity(code_str: str) -> dict[str, int | float]:
    tree = ast.parse(code_str)
    visitor = NumericalComplexityVisitor()
    visitor.visit(tree)

    node_types = [type(node).__name__ for node in ast.walk(tree)]
    total_nodes = len(node_types)
    type_counts = Counter(node_types)
    entropy = (
        -sum(
            (count / total_nodes) * math.log2((count / total_nodes) + 1e-12)
            for count in type_counts.values()
            if count > 0
        )
        if total_nodes > 0
        else 0.0
    )

    return {
        "call_count": visitor.call_count,
        "binop_count": visitor.binop_count,
        "subscript_count": visitor.subscript_count,
        "loop_count": visitor.loop_count,
        "condition_count": visitor.condition_count,
        "function_def_count": visitor.function_def_count,
        "class_def_count": visitor.class_def_count,
        "unique_identifiers": len(visitor.identifiers),
        "max_depth": visitor.max_depth,
        "ast_entropy": float(entropy),
        # total_nodes should be the size of the AST, not the sum of a small
        # subset of visitor counts (which under-reported by ~5-10x for any
        # non-trivial program). Reuse the ast.walk result already computed
        # above for entropy.
        "total_nodes": total_nodes,
    }


_COMPLEXITY_WEIGHTS: dict[str, float] = {
    "call_count": 0.15,
    "binop_count": 0.10,
    "loop_count": 0.15,
    "condition_count": 0.15,
    "function_def_count": 0.10,
    "class_def_count": 0.05,
    "max_depth": 0.20,
    "unique_identifiers": 0.10,
}

_COMPLEXITY_CAPS: dict[str, int] = {
    "call_count": 50,
    "binop_count": 50,
    "loop_count": 20,
    "condition_count": 20,
    "function_def_count": 10,
    "class_def_count": 5,
    "max_depth": 15,
    "unique_identifiers": 50,
}


def compute_complexity_score(features: dict[str, Any]) -> float:
    score = 0.0
    for k, w in _COMPLEXITY_WEIGHTS.items():
        v = min(int(features.get(k, 0) or 0), _COMPLEXITY_CAPS[k])
        score += v * w
    return float(score)


@StageRegistry.register(description="Get the length of program code")
class GetCodeLengthStage(Stage):
    InputsModel = VoidInput
    OutputModel = FloatDictContainer

    async def compute(self, program: Program) -> FloatDictContainer:
        length = float(len(program.code))
        logger.debug("[{}] code_length={}", self.stage_name, length)
        return FloatDictContainer(data={"code_length": length})


@StageRegistry.register(description="Compute code complexity metrics")
class ComputeComplexityStage(Stage):
    InputsModel = VoidInput
    OutputModel = FloatDictContainer

    async def compute(self, program: Program) -> FloatDictContainer:
        feats = compute_numerical_complexity(program.code)
        score = compute_complexity_score(feats)
        return FloatDictContainer(
            data={
                "call_count": feats["call_count"],
                "binop_count": feats["binop_count"],
                "subscript_count": feats["subscript_count"],
                "loop_count": feats["loop_count"],
                "condition_count": feats["condition_count"],
                "function_def_count": feats["function_def_count"],
                "class_def_count": feats["class_def_count"],
                "unique_identifiers": feats["unique_identifiers"],
                "max_depth": feats["max_depth"],
                "ast_entropy": feats["ast_entropy"],
                "total_nodes": feats["total_nodes"],
                "complexity_score": score,
                "negative_complexity_score": -score,
            }
        )
