"""Program Contract Validation Utilities.

This module provides utilities to help users understand and validate
the contract between program completion and evolution engine acceptance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from src.programs.program import Program


class ContractViolationType(str, Enum):
    """Types of contract violations."""

    INCOMPLETE_PROGRAM = "incomplete_program"
    EMPTY_METRICS = "empty_metrics"
    MISSING_REQUIRED_KEYS = "missing_required_keys"
    STAGE_FAILURE = "stage_failure"
    INVALID_METRICS = "invalid_metrics"


@dataclass
class ContractViolation:
    """Represents a specific contract violation."""

    violation_type: ContractViolationType
    program_id: str
    description: str
    suggestion: str
    stage_context: Optional[str] = None


class ProgramContractChecker:
    """Utility class to check program contracts before evolution."""

    def __init__(self, required_behavior_keys: Optional[Set[str]] = None):
        self.required_behavior_keys = required_behavior_keys or set()

    def check_program(
        self, program: Program
    ) -> Tuple[bool, List[ContractViolation]]:
        """
        Check if a program meets the evolution contract.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Check completion status
        if not program.is_complete:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.INCOMPLETE_PROGRAM,
                    program_id=program.id,
                    description="Program has not completed DAG pipeline execution",
                    suggestion="Ensure the program has been processed through the DAG pipeline and all stages have finished",
                )
            )

        # Check metrics presence
        if not program.metrics:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.EMPTY_METRICS,
                    program_id=program.id,
                    description="Program has empty metrics dictionary",
                    suggestion="Check if validation stages failed. Use FactoryMetricsStage for fallback metrics, or debug DAG stage execution",
                )
            )
        elif self.required_behavior_keys:
            # Check required keys
            missing_keys = self.required_behavior_keys - set(
                program.metrics.keys()
            )
            if missing_keys:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.MISSING_REQUIRED_KEYS,
                        program_id=program.id,
                        description=f"Program missing required behavior keys: {sorted(missing_keys)}",
                        suggestion=f"Configure your DAG to compute these metrics, or remove them from required_behavior_keys in EngineConfig",
                    )
                )

        # Check for stage failures that might explain empty metrics
        if program.is_complete and not program.metrics:
            failed_stages = []
            for stage_name, result in program.stage_results.items():
                if result.is_failed():
                    failed_stages.append(stage_name)

            if failed_stages:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.STAGE_FAILURE,
                        program_id=program.id,
                        description=f"Program completed but has failed stages: {failed_stages}",
                        suggestion="Check stage logs for errors. Consider using FactoryMetricsStage for fallback metrics when validation fails",
                        stage_context=f"Failed stages: {', '.join(failed_stages)}",
                    )
                )

        # Check metric value validity
        if program.metrics:
            invalid_metrics = []
            for key, value in program.metrics.items():
                if not isinstance(value, (int, float)):
                    invalid_metrics.append(f"{key}={type(value).__name__}")

            if invalid_metrics:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.INVALID_METRICS,
                        program_id=program.id,
                        description=f"Program has non-numeric metrics: {invalid_metrics}",
                        suggestion="Ensure all metrics are numeric (int or float) values for evolution compatibility",
                    )
                )

        return len(violations) == 0, violations

    def check_programs_batch(
        self, programs: List[Program]
    ) -> Dict[str, Tuple[bool, List[ContractViolation]]]:
        """Check multiple programs and return results by program ID."""
        results = {}
        for program in programs:
            is_valid, violations = self.check_program(program)
            results[program.id] = (is_valid, violations)
        return results

    def generate_contract_report(self, programs: List[Program]) -> str:
        """Generate a comprehensive contract validation report."""
        results = self.check_programs_batch(programs)

        valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
        invalid_count = len(results) - valid_count

        report = []
        report.append("=" * 60)
        report.append("PROGRAM CONTRACT VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Total Programs: {len(programs)}")
        report.append(f"✅ Valid Programs: {valid_count}")
        report.append(f"❌ Invalid Programs: {invalid_count}")

        if self.required_behavior_keys:
            report.append(
                f"Required Behavior Keys: {sorted(self.required_behavior_keys)}"
            )
        else:
            report.append(
                "Required Behavior Keys: None (any non-empty metrics accepted)"
            )

        report.append("")

        # Group violations by type
        violation_summary = {}
        for program_id, (is_valid, violations) in results.items():
            if not is_valid:
                for violation in violations:
                    violation_type = violation.violation_type
                    if violation_type not in violation_summary:
                        violation_summary[violation_type] = []
                    violation_summary[violation_type].append(violation)

        if violation_summary:
            report.append("VIOLATION SUMMARY:")
            report.append("-" * 40)
            for violation_type, violations in violation_summary.items():
                report.append(
                    f"\n{violation_type.value.upper().replace('_', ' ')} ({len(violations)} programs):"
                )
                for violation in violations[:3]:  # Show first 3 examples
                    report.append(
                        f"  • {violation.program_id}: {violation.description}"
                    )
                    report.append(f"    💡 {violation.suggestion}")
                if len(violations) > 3:
                    report.append(
                        f"  ... and {len(violations) - 3} more programs"
                    )

        # Show valid programs summary
        if valid_count > 0:
            report.append(f"\n✅ VALID PROGRAMS ({valid_count}):")
            report.append("-" * 40)
            valid_programs = [p for p in programs if results[p.id][0]]
            for program in valid_programs[:5]:  # Show first 5
                metrics_summary = f"{len(program.metrics)} metrics: {sorted(program.metrics.keys())}"
                report.append(f"  • {program.id}: {metrics_summary}")
            if len(valid_programs) > 5:
                report.append(
                    f"  ... and {len(valid_programs) - 5} more valid programs"
                )

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def diagnose_empty_metrics(program: Program) -> str:
    """Diagnose why a program has empty metrics despite being complete."""
    if not program.is_complete:
        return "Program is not marked as complete - DAG execution may not have finished"

    if program.metrics:
        return "Program has metrics - this function is for diagnosing empty metrics"

    diagnosis = []
    diagnosis.append(f"EMPTY METRICS DIAGNOSIS for Program {program.id}")
    diagnosis.append("-" * 50)
    diagnosis.append(
        f"Program Status: is_complete={program.is_complete}, is_discarded={program.is_discarded}"
    )
    diagnosis.append(f"Stage Results: {len(program.stage_results)} stages")

    # Analyze stage results
    completed_stages = []
    failed_stages = []
    skipped_stages = []
    running_stages = []

    for stage_name, result in program.stage_results.items():
        if result.is_completed():
            completed_stages.append(stage_name)
        elif result.is_failed():
            failed_stages.append(stage_name)
        elif result.is_skipped():
            skipped_stages.append(stage_name)
        elif result.is_running():
            running_stages.append(stage_name)

    diagnosis.append(f"✅ Completed Stages: {completed_stages}")
    diagnosis.append(f"❌ Failed Stages: {failed_stages}")
    diagnosis.append(f"⏭️  Skipped Stages: {skipped_stages}")
    diagnosis.append(f"🔄 Running Stages: {running_stages}")

    # Likely causes
    diagnosis.append("\nLIKELY CAUSES:")
    if failed_stages:
        diagnosis.append("1. ❌ Validation or metrics stages failed")
        diagnosis.append(
            "   💡 Check stage error messages and fix validation logic"
        )

    if skipped_stages:
        diagnosis.append(
            "2. ⏭️  Critical stages were skipped due to dependencies"
        )
        diagnosis.append(
            "   💡 Check DAG dependencies and ensure prerequisite stages succeed"
        )

    if not any(
        "metric" in stage.lower() or "validat" in stage.lower()
        for stage in completed_stages
    ):
        diagnosis.append("3. 🔍 No metrics or validation stages found")
        diagnosis.append(
            "   💡 Ensure your DAG includes UpdateMetricsStage or FactoryMetricsStage"
        )

    diagnosis.append("\nRECOMMENDED ACTIONS:")
    diagnosis.append(
        "• Use FactoryMetricsStage instead of UpdateMetricsStage for fallback metrics"
    )
    diagnosis.append("• Check validator function for errors")
    diagnosis.append("• Verify DAG stage dependencies and execution order")
    diagnosis.append("• Enable debug logging to see detailed stage execution")

    return "\n".join(diagnosis)


def create_contract_summary(
    required_behavior_keys: Optional[Set[str]] = None,
) -> str:
    """Create a summary of the program-evolution contract."""
    summary = []
    summary.append("PROGRAM-EVOLUTION CONTRACT SUMMARY")
    summary.append("=" * 50)
    summary.append("For a program to be accepted by the Evolution Engine:")
    summary.append("")
    summary.append("1. ✅ Program Completion:")
    summary.append("   • program.is_complete must be True")
    summary.append("   • program.is_discarded must be False")
    summary.append("   • This is set by DAG execution when pipeline finishes")
    summary.append("")
    summary.append("2. ✅ Metrics Presence:")
    summary.append("   • program.metrics must be non-empty dictionary")
    summary.append("   • All metric values should be numeric (int/float)")
    summary.append("   • Empty metrics = automatic rejection")
    summary.append("")

    if required_behavior_keys:
        summary.append("3. ✅ Required Behavior Keys:")
        summary.append(
            f"   • These keys must exist in metrics: {sorted(required_behavior_keys)}"
        )
        summary.append("   • Missing any required key = automatic rejection")
        summary.append("   • Configure your DAG to compute these metrics")
    else:
        summary.append("3. ✅ Required Behavior Keys:")
        summary.append(
            "   • No specific keys required (any non-empty metrics accepted)"
        )
        summary.append(
            "   • Consider setting required_behavior_keys in EngineConfig"
        )

    summary.append("")
    summary.append("COMMON ISSUES:")
    summary.append("• Program is complete but metrics are empty")
    summary.append("  → Validation stage failed, check stage results")
    summary.append("• Program has dummy metrics but missing required keys")
    summary.append("  → Update validator to compute required metrics")
    summary.append("• Program never becomes complete")
    summary.append("  → DAG execution not running or stages are failing")

    return "\n".join(summary)
