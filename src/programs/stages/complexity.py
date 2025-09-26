"""Code complexity analysis for MetaEvolve."""

import ast
from collections import Counter
from datetime import datetime
import math
from typing import Any, Dict

from loguru import logger

from src.exceptions import StageError
from src.programs.program import Program
from src.programs.stages.state import ProgramStageResult, StageState
from src.programs.utils import build_stage_result
from src.runner.stage_registry import StageRegistry

from src.programs.stages.worker_pool import WorkerPoolStage


@StageRegistry.register(
    description="Get the length of program code"
)
class GetCodeLengthStage(WorkerPoolStage):
    """A stage for getting the length of the code."""

    def _work(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Synchronous computation executed inside the shared thread pool."""
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Computing complexity metrics"
        )
        try:
            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output={"code_length": len(program.code)},
            )

        except Exception as e:
            raise StageError(
                f"Code length computation failed: {e}",
                stage_name=self.stage_name,
                stage_type="analysis",
                cause=e,
            )


class NumericalComplexityVisitor(ast.NodeVisitor):
    """Enhanced AST visitor for comprehensive complexity analysis."""

    def __init__(self):
        self.call_count = 0
        self.binop_count = 0
        self.subscript_count = 0
        self.loop_count = 0
        self.condition_count = 0
        self.function_def_count = 0
        self.class_def_count = 0
        self.expr_depths = []
        self.identifiers = set()
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

    def visit_FunctionDef(self, node):
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


def compute_numerical_complexity(code_str: str) -> Dict[str, float]:
    """Compute comprehensive numerical complexity metrics."""
    try:
        tree = ast.parse(code_str)
        visitor = NumericalComplexityVisitor()
        visitor.visit(tree)

        # Compute AST entropy
        node_types = [type(node).__name__ for node in ast.walk(tree)]
        total_nodes = len(node_types)
        type_counts = Counter(node_types)
        entropy = -sum(
            (count / total_nodes) * math.log2(count / total_nodes + 1e-12)
            for count in type_counts.values()
            if count > 0
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
            "ast_entropy": entropy,
            "total_nodes": sum(
                [
                    visitor.call_count,
                    visitor.binop_count,
                    visitor.subscript_count,
                    visitor.loop_count,
                    visitor.condition_count,
                    visitor.function_def_count,
                    visitor.class_def_count,
                ]
            ),
        }

    except SyntaxError as e:
        return {
            "error": f"Syntax error: {e}",
            "call_count": 0,
            "binop_count": 0,
            "subscript_count": 0,
            "loop_count": 0,
            "condition_count": 0,
            "function_def_count": 0,
            "class_def_count": 0,
            "unique_identifiers": 0,
            "max_depth": 0,
            "ast_entropy": 0,
            "total_nodes": 0,
        }
    except Exception as e:
        return {
            "error": f"Analysis failed: {e}",
            "call_count": 0,
            "binop_count": 0,
            "subscript_count": 0,
            "loop_count": 0,
            "condition_count": 0,
            "function_def_count": 0,
            "class_def_count": 0,
            "unique_identifiers": 0,
            "max_depth": 0,
            "ast_entropy": 0,
            "total_nodes": 0,
        }


def compute_complexity_score(features: Dict[str, Any]) -> float:
    """Compute normalized complexity score from features."""
    if "error" in features:
        return 0.0

    # Weighted complexity calculation
    weights = {
        "call_count": 0.15,  # Use of abstraction / reusability
        "binop_count": 0.1,  # Raw computational work
        "loop_count": 0.15,  # Iterative complexity
        "condition_count": 0.15,  # Branching complexity
        "function_def_count": 0.1,  # Structural modularity
        "class_def_count": 0.05,  # OOP, if applicable
        "max_depth": 0.2,  # Nesting depth = semantic burden
        "unique_identifiers": 0.1,  # Diversity of state/symbols
    }

    # Optional: normalize some features to prevent extreme scaling
    normalized_features = {
        "call_count": min(features.get("call_count", 0), 50),
        "binop_count": min(features.get("binop_count", 0), 50),
        "loop_count": min(features.get("loop_count", 0), 20),
        "condition_count": min(features.get("condition_count", 0), 20),
        "function_def_count": min(features.get("function_def_count", 0), 10),
        "class_def_count": min(features.get("class_def_count", 0), 5),
        "max_depth": min(features.get("max_depth", 0), 15),
        "unique_identifiers": min(features.get("unique_identifiers", 0), 50),
    }

    score = 0.0
    for feature, weight in weights.items():
        value = normalized_features.get(feature, 0)
        score += value * weight

    return score

@StageRegistry.register(
    description="Compute code complexity metrics"
)
class ComputeComplexityStage(WorkerPoolStage):
    """Stage for computing code complexity metrics."""

    def _work(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Synchronous computation executed inside the shared thread pool."""
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Computing complexity metrics"
        )

        try:
            features = compute_numerical_complexity(program.code)
            score = compute_complexity_score(features)

            return build_stage_result(
                status=StageState.COMPLETED,
                started_at=started_at,
                output={
                    **features,
                    "complexity_score": score,
                    "negative_complexity_score": -score,
                },
            )

        except Exception as e:  # pragma: no cover – unexpected
            raise StageError(
                f"Complexity computation failed: {e}",
                stage_name=self.stage_name,
                stage_type="analysis",
                cause=e,
            ) from e
