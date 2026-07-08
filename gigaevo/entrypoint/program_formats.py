"""Program-format features for default pipeline builders."""

from __future__ import annotations

from gigaevo.entrypoint.default_pipelines import PipelineBuilder, PipelineFeature
from gigaevo.programs.stages.json_genome import ParseJsonProgram


class JsonDocumentEvaluationFeature(PipelineFeature):
    """Treat ``program.code`` as a JSON document instead of Python source.

    JSON-document problems store the candidate genome directly in
    ``program.code``. The syntax gate and payload producer are therefore the
    same operation: parse the JSON text and pass the parsed object to
    ``validate.py``.
    """

    name = "json_document_evaluation"
    description = (
        "Replace Python source validation/execution with JSON parsing for both "
        "syntax checking and validator payload production."
    )

    def apply(self, builder: PipelineBuilder) -> None:
        builder.replace_stage(
            "ValidateCodeStage",
            lambda: ParseJsonProgram(timeout=builder._stage_timeout),
        )
        builder.replace_stage(
            "CallProgramFunction",
            lambda: ParseJsonProgram(timeout=builder._stage_timeout),
        )
