#!/usr/bin/env python3
"""
FastAPI backend for DAG Builder GUI.

Provides endpoints for:
- Getting stage registry information
- Exporting DAGs as PipelineBuilder code
- Validating DAG structures
"""

from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yaml

from gigaevo.programs.dag.automata import DAGAutomata
from gigaevo.programs.dag.automata import DataFlowEdge as DAGDataFlowEdge
from gigaevo.programs.dag.automata import (
    ExecutionOrderDependency as DAGExecutionOrderDependency,
)

# Import all stages to register them
from gigaevo.programs.stages.stage_registry import StageRegistry

app = FastAPI(title="GigaEvo DAG Builder API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class StageInfoResponse(BaseModel):
    name: str
    description: str
    class_name: str
    import_path: str
    mandatory_inputs: List[str]
    optional_inputs: List[str]
    input_types: dict  # Dict mapping input name to type string
    output_fields: List[str]
    output_model_name: str
    cacheable: bool  # Whether this stage's results can be cached


class DataFlowEdgeRequest(BaseModel):
    source_stage: str
    destination_stage: str
    input_name: str


class ExecutionDependencyRequest(BaseModel):
    stage: str
    dependency_type: str  # "on_success", "on_failure", "always_after"
    target_stage: str


class StageRequest(BaseModel):
    name: str
    custom_name: Optional[str] = None
    display_name: str
    description: str = ""
    notes: str = ""


class DAGRequest(BaseModel):
    stages: List[StageRequest]  # List of stage objects with metadata
    data_flow_edges: List[DataFlowEdgeRequest]
    execution_dependencies: List[ExecutionDependencyRequest]


class DAGExportResponse(BaseModel):
    code: str
    validation_errors: List[str]


class DAGValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class YAMLExportResponse(BaseModel):
    yaml: str
    validation_errors: List[str]


class YAMLConfigInfo(BaseModel):
    filename: str
    path: str
    display_name: str


class LoadYAMLRequest(BaseModel):
    yaml_path: str


class LoadYAMLResponse(BaseModel):
    stages: List[StageRequest]
    data_flow_edges: List[DataFlowEdgeRequest]
    execution_dependencies: List[ExecutionDependencyRequest]
    errors: List[str]
    warnings: List[str]


@app.get("/")
async def root():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GigaEvo DAG Builder</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #333; margin-bottom: 10px; }
            .header p { color: #666; }
            .main-content { display: flex; gap: 20px; }
            .stage-library {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: 300px;
                height: fit-content;
            }
            .stage-library h3 { margin-top: 0; color: #333; }
            .stage-item {
                background: #f8f9fa;
                padding: 12px;
                margin: 8px 0;
                border-radius: 6px;
                cursor: pointer;
                border: 1px solid #e9ecef;
                transition: all 0.2s;
            }
            .stage-item:hover {
                background: #e9ecef;
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .stage-name { font-weight: bold; color: #495057; margin-bottom: 4px; }
            .stage-description { font-size: 0.9em; color: #6c757d; margin-bottom: 6px; }
            .stage-inputs { font-size: 0.8em; color: #868e96; }
            .dag-canvas {
                flex: 1;
                background: white;
                border: 2px dashed #dee2e6;
                border-radius: 8px;
                min-height: 500px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .canvas-placeholder {
                text-align: center;
                color: #6c757d;
            }
            .canvas-placeholder h3 { margin-bottom: 10px; }
            .canvas-placeholder p { margin: 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏗️ GigaEvo DAG Builder</h1>
                <p>Drag stages from the library to build your execution pipeline</p>
            </div>

            <div class="main-content">
                <div class="stage-library">
                    <h3>📚 Stage Library</h3>
                    <div id="stages-container">
                        <p>Loading stages...</p>
                    </div>
                </div>

                <div class="dag-canvas" id="dag-canvas">
                    <div class="canvas-placeholder">
                        <h3>🎯 DAG Canvas</h3>
                        <p>Drag stages here to build your pipeline</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Simple JavaScript for now - will be replaced with React
            fetch('/api/stages')
                .then(response => response.json())
                .then(stages => {
                    const container = document.getElementById('stages-container');
                    container.innerHTML = '';

                    stages.forEach(stage => {
                        const div = document.createElement('div');
                        div.className = 'stage-item';
                        div.draggable = true;
                        div.innerHTML = `
                            <div class="stage-name">${stage.name}</div>
                            <div class="stage-description">${stage.description}</div>
                            <div class="stage-inputs">
                                <strong>Inputs:</strong><br>
                                Mandatory: ${stage.mandatory_inputs.join(', ') || 'none'}<br>
                                Optional: ${stage.optional_inputs.join(', ') || 'none'}<br>
                                <strong>Output:</strong> ${stage.output_model_name}<br>
                                Fields: ${stage.output_fields.join(', ') || 'none'}
                            </div>
                        `;

                        // Add drag event
                        div.addEventListener('dragstart', (e) => {
                            e.dataTransfer.setData('text/plain', stage.name);
                        });

                        container.appendChild(div);
                    });
                })
                .catch(error => {
                    console.error('Error loading stages:', error);
                    document.getElementById('stages-container').innerHTML = '<p>Error loading stages</p>';
                });

            // Canvas drop functionality
            const canvas = document.getElementById('dag-canvas');
            canvas.addEventListener('dragover', (e) => {
                e.preventDefault();
                canvas.style.backgroundColor = '#f8f9fa';
            });

            canvas.addEventListener('dragleave', (e) => {
                canvas.style.backgroundColor = 'white';
            });

            canvas.addEventListener('drop', (e) => {
                e.preventDefault();
                canvas.style.backgroundColor = 'white';

                const stageName = e.dataTransfer.getData('text/plain');
                if (stageName) {
                    // For now, just show a simple message
                    canvas.innerHTML = `
                        <div class="canvas-placeholder">
                            <h3>🎯 DAG Canvas</h3>
                            <p>Dropped stage: <strong>${stageName}</strong></p>
                            <p><em>React Flow integration coming soon...</em></p>
                        </div>
                    `;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/stages", response_model=List[StageInfoResponse])
async def get_stages():
    """Get all available stages."""
    stages = StageRegistry.get_all_stages()
    return [
        StageInfoResponse(
            name=info.name,
            description=info.description,
            class_name=info.class_name,
            import_path=info.import_path,
            mandatory_inputs=info.mandatory_inputs,
            optional_inputs=info.optional_inputs,
            input_types=info.input_types,
            output_fields=info.output_fields,
            output_model_name=info.output_model_name,
            cacheable=info.cacheable,
        )
        for info in stages.values()
    ]


@app.get("/api/stages/{stage_name}", response_model=StageInfoResponse)
async def get_stage(stage_name: str):
    """Get a specific stage by name."""
    stage_info = StageRegistry.get_stage(stage_name)
    if not stage_info:
        raise HTTPException(status_code=404, detail=f"Stage '{stage_name}' not found")

    return StageInfoResponse(
        name=stage_info.name,
        description=stage_info.description,
        class_name=stage_info.class_name,
        import_path=stage_info.import_path,
        mandatory_inputs=stage_info.mandatory_inputs,
        optional_inputs=stage_info.optional_inputs,
        input_types=stage_info.input_types,
        output_fields=stage_info.output_fields,
        output_model_name=stage_info.output_model_name,
        cacheable=stage_info.cacheable,
    )


@app.post("/api/validate-dag", response_model=DAGValidationResponse)
async def validate_dag(dag_request: DAGRequest):
    """Validate a DAG structure using DAGAutomata.validate_structure()."""
    try:
        errors = []
        warnings = []

        # Basic validation (unique names, etc.)
        basic_errors = validate_dag_structure(dag_request)
        if basic_errors:
            errors.extend(basic_errors)
            return DAGValidationResponse(
                is_valid=False, errors=errors, warnings=warnings
            )

        # Get stage classes (not instances!) from registry
        available_stages = StageRegistry.get_all_stages()
        stage_classes = {}

        for stage_req in dag_request.stages:
            stage_info = available_stages.get(stage_req.name)
            if not stage_info:
                errors.append(f"Stage type '{stage_req.name}' not found in registry")
                continue

            # Import the stage class
            try:
                module_path, class_name = stage_info.import_path.rsplit(".", 1)
                import importlib

                module = importlib.import_module(module_path)
                stage_class = getattr(module, class_name)
                stage_classes[stage_req.display_name] = stage_class
            except Exception as e:
                errors.append(
                    f"Could not import stage {stage_req.display_name}: {str(e)}"
                )

        if errors:
            return DAGValidationResponse(
                is_valid=False, errors=errors, warnings=warnings
            )

        # Convert edges to DAG format
        dag_data_edges = []
        for edge in dag_request.data_flow_edges:
            dag_data_edges.append(
                DAGDataFlowEdge.create(
                    source=edge.source_stage,
                    destination=edge.destination_stage,
                    input_name=edge.input_name,
                )
            )

        # Convert execution dependencies
        dag_exec_deps = {}
        for dep in dag_request.execution_dependencies:
            if dep.stage not in dag_exec_deps:
                dag_exec_deps[dep.stage] = []

            if dep.dependency_type == "on_success":
                dag_exec_deps[dep.stage].append(
                    DAGExecutionOrderDependency.on_success(dep.target_stage)
                )
            elif dep.dependency_type == "on_failure":
                dag_exec_deps[dep.stage].append(
                    DAGExecutionOrderDependency.on_failure(dep.target_stage)
                )
            elif dep.dependency_type == "always_after":
                dag_exec_deps[dep.stage].append(
                    DAGExecutionOrderDependency.always_after(dep.target_stage)
                )

        # Use DAGAutomata.validate_structure() - clean and simple!
        validation_errors = DAGAutomata.validate_structure(
            stage_classes, dag_data_edges, dag_exec_deps
        )

        if validation_errors:
            return DAGValidationResponse(
                is_valid=False, errors=validation_errors, warnings=warnings
            )

        return DAGValidationResponse(is_valid=True, errors=[], warnings=warnings)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/api/export-dag", response_model=DAGExportResponse)
async def export_dag(dag_request: DAGRequest):
    """Export a DAG as PipelineBuilder code."""
    try:
        # Validate the DAG structure
        validation_errors = validate_dag_structure(dag_request)

        if validation_errors:
            return DAGExportResponse(code="", validation_errors=validation_errors)

        # Generate PipelineBuilder code
        code = generate_pipeline_builder_code(dag_request)

        return DAGExportResponse(code=code, validation_errors=[])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/api/export-yaml", response_model=YAMLExportResponse)
async def export_yaml(dag_request: DAGRequest):
    """Export a DAG as YAML configuration."""
    try:
        # Validate the DAG structure
        validation_errors = validate_dag_structure(dag_request)

        if validation_errors:
            return YAMLExportResponse(yaml="", validation_errors=validation_errors)

        # Generate YAML configuration
        yaml_content = generate_yaml_config(dag_request)

        return YAMLExportResponse(yaml=yaml_content, validation_errors=[])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YAML export failed: {str(e)}")


def validate_dag_structure(dag_request: DAGRequest) -> List[str]:
    """Validate the DAG structure."""
    errors = []

    # Check that all referenced stages exist
    available_stages = StageRegistry.get_all_stages()

    # Check for unique stage names (display names)
    stage_display_names = [stage.display_name for stage in dag_request.stages]
    seen_names = set()
    for display_name in stage_display_names:
        if display_name in seen_names:
            errors.append(
                f"Duplicate stage name '{display_name}' - each stage must have a unique name"
            )
        seen_names.add(display_name)

    # Check that all stage types exist in the registry
    stage_names = [stage.name for stage in dag_request.stages]
    for stage_name in stage_names:
        if stage_name not in available_stages:
            errors.append(f"Stage type '{stage_name}' not found in registry")

    # Check that all edge references valid stages (using display names)
    stage_display_names_set = set(stage_display_names)
    for edge in dag_request.data_flow_edges:
        if edge.source_stage not in stage_display_names_set:
            errors.append(
                f"Data flow edge source '{edge.source_stage}' not in stages list"
            )
        if edge.destination_stage not in stage_display_names_set:
            errors.append(
                f"Data flow edge destination '{edge.destination_stage}' not in stages list"
            )

    # Check that all dependency references valid stages (using display names)
    for dep in dag_request.execution_dependencies:
        if dep.stage not in stage_display_names_set:
            errors.append(
                f"Execution dependency stage '{dep.stage}' not in stages list"
            )
        if dep.target_stage not in stage_display_names_set:
            errors.append(
                f"Execution dependency target '{dep.target_stage}' not in stages list"
            )

    return errors


def generate_pipeline_builder_code(dag_request: DAGRequest) -> str:
    """Generate PipelineBuilder code for the DAG."""
    # Get all stages from registry to extract imports dynamically
    all_stages = StageRegistry.get_all_stages()

    # Extract unique import paths from the stages we're using
    used_stages = {stage.name for stage in dag_request.stages}
    import_paths = set()
    stage_classes = {}

    for stage_name in used_stages:
        if stage_name in all_stages:
            stage_info = all_stages[stage_name]
            import_paths.add(stage_info.import_path)
            stage_classes[stage_name] = stage_info.class_name

    code_lines = [
        "#!/usr/bin/env python3",
        '"""',
        "Generated Pipeline from GigaEvo DAG Builder",
        "This file contains a complete pipeline configuration.",
        '"""',
        "",
        "# Core imports",
        "from gigaevo.runner.pipeline_factory import PipelineBuilder, PipelineContext",
        "from gigaevo.programs.automata import DataFlowEdge, ExecutionOrderDependency",
        "from gigaevo.runner.dag_blueprint import DAGBlueprint",
        "",
        "# Stage imports - dynamically generated from registry",
    ]

    # Add imports dynamically
    for import_path in sorted(import_paths):
        code_lines.append(f"from {import_path} import *")

    code_lines.extend(
        [
            "",
            "# Configuration variables - Customize these for your pipeline",
            'PIPELINE_NAME = "my_custom_pipeline"',
            'PIPELINE_DESCRIPTION = "Generated from DAG Builder"',
            'PIPELINE_VERSION = "1.0.0"',
            "",
            "# Stage configuration variables",
            "DEFAULT_TIMEOUT = 300  # seconds",
            "MAX_RETRIES = 3",
            "ENABLE_LOGGING = True",
            "",
            "def create_custom_pipeline(ctx: PipelineContext) -> DAGBlueprint:",
            '    """',
            "    Create a custom pipeline from DAG Builder configuration.",
            "    ",
            "    Args:",
            "        ctx: Pipeline context containing runtime information",
            "        ",
            "    Returns:",
            "        DAGBlueprint: Complete pipeline specification",
            '    """',
            "    builder = PipelineBuilder(ctx)",
            "",
            "    # Add stages with configuration",
        ]
    )

    # Add stage creation with enhanced configuration
    for stage in dag_request.stages:
        if stage.name in stage_classes:
            display_name = stage.display_name
            class_name = stage_classes[stage.name]
            stage_info = all_stages.get(stage.name)

            # Add stage comment with metadata
            code_lines.append(f"    # {display_name}")
            if stage.custom_name:
                code_lines.append(f"    # Custom name: {stage.custom_name}")
            if stage.description:
                code_lines.append(f"    # Description: {stage.description}")
            if stage.notes:
                code_lines.append(f"    # Notes: {stage.notes}")

            # Add input/output documentation
            if stage_info:
                if stage_info.mandatory_inputs:
                    code_lines.append(
                        f"    # Required inputs: {', '.join(stage_info.mandatory_inputs)}"
                    )
                if stage_info.optional_inputs:
                    code_lines.append(
                        f"    # Optional inputs: {', '.join(stage_info.optional_inputs)}"
                    )
                if (
                    stage_info.output_fields
                    and stage_info.output_model_name != "VoidOutput"
                ):
                    code_lines.append(
                        f"    # Output model: {stage_info.output_model_name}"
                    )
                    code_lines.append(
                        f"    # Output fields: {', '.join(stage_info.output_fields)}"
                    )

            code_lines.append("    builder.add_stage(")
            code_lines.append(f'        "{stage.name}",')
            code_lines.append(f"        lambda: {class_name}(")
            code_lines.append(f'            stage_name="{display_name}",')
            code_lines.append(
                "            # Add any additional required arguments here"
            )
            code_lines.append("        )")
            code_lines.append("    )")
            code_lines.append("")

    # Add data flow edges
    if dag_request.data_flow_edges:
        code_lines.append("    # Add data flow edges")
        for edge in dag_request.data_flow_edges:
            code_lines.append("    builder.add_data_flow_edge(")
            code_lines.append(f'        "{edge.source_stage}",')
            code_lines.append(f'        "{edge.destination_stage}",')
            code_lines.append(f'        "{edge.input_name}"')
            code_lines.append("    )")
        code_lines.append("")

    # Add execution dependencies
    if dag_request.execution_dependencies:
        code_lines.append("    # Add execution dependencies")
        for dep in dag_request.execution_dependencies:
            if dep.dependency_type == "on_success":
                code_lines.append("    builder.add_exec_dep(")
                code_lines.append(f'        "{dep.stage}",')
                code_lines.append(
                    f'        ExecutionOrderDependency.on_success("{dep.target_stage}")'
                )
                code_lines.append("    )")
            elif dep.dependency_type == "on_failure":
                code_lines.append("    builder.add_exec_dep(")
                code_lines.append(f'        "{dep.stage}",')
                code_lines.append(
                    f'        ExecutionOrderDependency.on_failure("{dep.target_stage}")'
                )
                code_lines.append("    )")
            elif dep.dependency_type == "always_after":
                code_lines.append("    builder.add_exec_dep(")
                code_lines.append(f'        "{dep.stage}",')
                code_lines.append(
                    f'        ExecutionOrderDependency.always_after("{dep.target_stage}")'
                )
                code_lines.append("    )")
        code_lines.append("")

    code_lines.extend(
        [
            "    return builder.build_spec()",
            "",
            "",
            "# Example usage and configuration",
            "def main():",
            '    """',
            "    Example usage of the generated pipeline.",
            '    """',
            "    # Create pipeline context",
            "    ctx = PipelineContext(",
            "        pipeline_name=PIPELINE_NAME,",
            "        pipeline_description=PIPELINE_DESCRIPTION,",
            "        pipeline_version=PIPELINE_VERSION,",
            "        enable_logging=ENABLE_LOGGING",
            "    )",
            "    ",
            "    # Create and run pipeline",
            "    dag_spec = create_custom_pipeline(ctx)",
            "    ",
            "    # Optional: Print pipeline information",
            '    print(f"Pipeline: {PIPELINE_NAME}")',
            '    print(f"Description: {PIPELINE_DESCRIPTION}")',
            '    print(f"Version: {PIPELINE_VERSION}")',
            '    print(f"Stages: {len(dag_spec.stages)}")',
            "    ",
            "    return dag_spec",
            "",
            "",
            'if __name__ == "__main__":',
            "    # Run the pipeline",
            "    pipeline = main()",
            "    ",
            "    # You can also import and use this function in other modules:",
            "    # from this_module import create_custom_pipeline",
            "    # dag_spec = create_custom_pipeline(your_context)",
        ]
    )

    return "\n".join(code_lines)


def generate_yaml_config(dag_request: DAGRequest) -> str:
    """Generate YAML configuration matching config/pipeline/custom.yaml structure."""
    # Get all stages from registry
    all_stages = StageRegistry.get_all_stages()

    yaml_lines = [
        "# @package _global_",
        "",
        "# DAG Pipeline Configuration",
        "# Generated by DAG Builder",
        "# TODO: Fill in the required parameters marked with <FILL_ME>",
        "",
        "# Evolution context configuration",
        "evolution_context:",
        "  _target_: gigaevo.entrypoint.evolution_context.EvolutionContext",
        "  problem_ctx: ${problem_context}",
        "  llm_wrapper: ${ref:llm}",
        "  storage: ${ref:redis_storage}",
        "",
        "dag_blueprint:",
        "  _target_: gigaevo.runner.dag_blueprint.DAGBlueprint",
        "",
        "  # Stage definitions",
        "  nodes:",
    ]

    # Add stages
    for stage in dag_request.stages:
        if stage.name in all_stages:
            stage_info = all_stages[stage.name]
            display_name = stage.display_name

            # Stage header comment
            yaml_lines.append(f"    # {display_name}")
            if stage.description:
                yaml_lines.append(f"    # {stage.description}")

            # Add input/output documentation
            if stage_info.mandatory_inputs:
                yaml_lines.append(
                    f"    # Required inputs: {', '.join(stage_info.mandatory_inputs)}"
                )
            if stage_info.optional_inputs:
                yaml_lines.append(
                    f"    # Optional inputs: {', '.join(stage_info.optional_inputs)}"
                )
            if (
                stage_info.output_fields
                and stage_info.output_model_name != "VoidOutput"
            ):
                yaml_lines.append(
                    f"    # Output: {stage_info.output_model_name}({', '.join(stage_info.output_fields)})"
                )

            yaml_lines.append(f"    {display_name}:")
            yaml_lines.append(f"      _target_: {stage_info.import_path}")
            yaml_lines.append("      _partial_: true")

            # Add parameter placeholders
            yaml_lines.append(
                f"      # TODO: Add required parameters for {stage_info.name}"
            )
            yaml_lines.append(
                "      timeout: ${stage_timeout}  # or specific value like 300.0"
            )

            # Add stage-specific notes if present
            if stage.notes:
                for note_line in stage.notes.split("\n"):
                    if note_line.strip():
                        yaml_lines.append(f"      # Note: {note_line.strip()}")

            yaml_lines.append("")

    # Add data flow edges
    yaml_lines.append("  # Data flow edges (source -> destination with input name)")
    yaml_lines.append("  data_flow_edges:")
    if dag_request.data_flow_edges:
        for edge in dag_request.data_flow_edges:
            yaml_lines.append(f"    - source_stage: {edge.source_stage}")
            yaml_lines.append(f"      destination_stage: {edge.destination_stage}")
            yaml_lines.append(f"      input_name: {edge.input_name}")
            yaml_lines.append("")
    else:
        yaml_lines.append("    # No data flow edges defined")
        yaml_lines.append("")

    # Add execution dependencies in Hydra format
    yaml_lines.append("  # Execution order dependencies")
    yaml_lines.append("  exec_order_deps:")
    if dag_request.execution_dependencies:
        # Group by stage
        deps_by_stage = {}
        for dep in dag_request.execution_dependencies:
            if dep.stage not in deps_by_stage:
                deps_by_stage[dep.stage] = []
            deps_by_stage[dep.stage].append(dep)

        for stage_name, deps in deps_by_stage.items():
            yaml_lines.append(f"    {stage_name}:")
            for dep in deps:
                # Map dependency types to condition values
                condition_map = {
                    "on_success": "success",
                    "on_failure": "failure",
                    "always_after": "always",
                }
                condition = condition_map.get(dep.dependency_type, "success")

                yaml_lines.append(
                    "      - _target_: gigaevo.programs.dag.automata.ExecutionOrderDependency"
                )
                yaml_lines.append(f"        stage_name: {dep.target_stage}")
                yaml_lines.append(f"        condition: {condition}")
                yaml_lines.append("")
    else:
        yaml_lines.append("    # No execution order dependencies defined")
        yaml_lines.append("")

    # Add DAG-level settings
    yaml_lines.extend(
        [
            "  # DAG-level settings",
            "  max_parallel_stages: ${dag_concurrency}  # or specific value like 8",
            "  dag_timeout: ${dag_timeout}  # or specific value like 2400.0",
        ]
    )

    return "\n".join(yaml_lines)


@app.get("/api/yaml-configs", response_model=List[YAMLConfigInfo])
async def list_yaml_configs():
    """List available Hydra YAML pipeline configs with dag_blueprint section."""
    try:
        config_dir = project_root / "config" / "pipeline"

        if not config_dir.exists():
            return []

        configs = []
        for yaml_file in config_dir.glob("*.yaml"):
            if yaml_file.name.startswith("_"):
                continue  # Skip private configs

            # Check if this config has a dag_blueprint section
            try:
                with open(yaml_file, "r") as f:
                    config_data = yaml.safe_load(f)

                # Only include configs that have dag_blueprint with nodes
                if not config_data:
                    continue

                dag_blueprint = config_data.get("dag_blueprint", {})
                if not dag_blueprint or not dag_blueprint.get("nodes"):
                    continue

                configs.append(
                    YAMLConfigInfo(
                        filename=yaml_file.name,
                        path=str(yaml_file.relative_to(project_root)),
                        display_name=yaml_file.stem.replace("_", " ").title(),
                    )
                )
            except Exception as e:
                # Skip files that can't be parsed
                print(f"Skipping {yaml_file.name}: {e}")
                continue

        return sorted(configs, key=lambda x: x.filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list configs: {str(e)}")


@app.post("/api/load-yaml", response_model=LoadYAMLResponse)
async def load_yaml_config(request: LoadYAMLRequest):
    """Load and parse a Hydra YAML config into DAG builder format."""
    try:
        yaml_path = project_root / request.yaml_path

        if not yaml_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Config file not found: {request.yaml_path}"
            )

        # Load YAML
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        errors = []
        warnings = []
        stages = []
        data_flow_edges = []
        execution_dependencies = []

        # Extract DAG blueprint
        dag_blueprint = config.get("dag_blueprint", {})

        if not dag_blueprint:
            errors.append("No 'dag_blueprint' section found in YAML")
            return LoadYAMLResponse(
                stages=[],
                data_flow_edges=[],
                execution_dependencies=[],
                errors=errors,
                warnings=warnings,
            )

        # Parse nodes (stages)
        nodes = dag_blueprint.get("nodes", {})
        for stage_name, stage_config in nodes.items():
            target = stage_config.get("_target_", "")

            if not target:
                warnings.append(f"Stage '{stage_name}' has no _target_")
                continue

            # Extract class name from import path
            class_name = target.split(".")[-1]

            # Check if stage is registered
            stage_info = StageRegistry.get_stage(class_name)
            description = stage_info.description if stage_info else ""

            stages.append(
                StageRequest(
                    name=class_name,
                    custom_name=stage_name if stage_name != class_name else None,
                    display_name=stage_name,
                    description=description,
                    notes=f"Imported from {request.yaml_path}",
                )
            )

        # Parse data flow edges
        edges = dag_blueprint.get("data_flow_edges", [])
        for edge in edges:
            data_flow_edges.append(
                DataFlowEdgeRequest(
                    source_stage=edge.get("source_stage", ""),
                    destination_stage=edge.get("destination_stage", ""),
                    input_name=edge.get("input_name", ""),
                )
            )

        # Parse execution order dependencies
        exec_deps = dag_blueprint.get("exec_order_deps", {})
        for stage_name, deps_list in exec_deps.items():
            for dep in deps_list:
                condition = dep.get("condition", "success")
                target_stage = dep.get("stage_name", "")

                # Map condition to dependency_type
                dep_type_map = {
                    "success": "on_success",
                    "failure": "on_failure",
                    "always": "always_after",
                }
                dep_type = dep_type_map.get(condition, "on_success")

                execution_dependencies.append(
                    ExecutionDependencyRequest(
                        stage=stage_name,
                        dependency_type=dep_type,
                        target_stage=target_stage,
                    )
                )

        if not stages:
            warnings.append("No stages found in config")

        return LoadYAMLResponse(
            stages=stages,
            data_flow_edges=data_flow_edges,
            execution_dependencies=execution_dependencies,
            errors=errors,
            warnings=warnings,
        )

    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting GigaEvo DAG Builder API...")
    print("📱 Open http://localhost:8081 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8081, reload=True)
