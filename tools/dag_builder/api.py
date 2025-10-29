#!/usr/bin/env python3
"""
FastAPI backend for DAG Builder GUI.

Provides endpoints for:
- Getting stage registry information
- Exporting DAGs as PipelineBuilder code
- Validating DAG structures
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.programs.automata import DataFlowEdge, ExecutionOrderDependency
from src.programs.stages.complexity import ComputeComplexityStage, GetCodeLengthStage

# Import all stages to register them
from src.programs.stages.execution import (
    RunConstantPythonCode,
    RunProgramCodeWithOptionalProducedData,
    ValidatorCodeExecutor,
)
from src.programs.stages.insights import GenerateLLMInsightsStage
from src.programs.stages.insights_lineage import GenerateLineageInsightsStage
from src.programs.stages.llm_score import GenerateLLMScoreStage
from src.programs.stages.metrics import EnsureMetricsStage, NormalizeMetricsStage
from src.programs.stages.validation import ValidateCodeStage
from src.programs.stages.worker_pool import WorkerPoolStage
from src.runner.pipeline_factory import PipelineBuilder, PipelineContext
from src.runner.stage_registry import StageInfo, StageRegistry

app = FastAPI(title="MetaEvolve DAG Builder API", version="1.0.0")

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


@app.get("/")
async def root():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MetaEvolve DAG Builder</title>
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
                <h1>🏗️ MetaEvolve DAG Builder</h1>
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
                                Mandatory: ${stage.mandatory_inputs.join(', ') || 'none'}<br>
                                Optional: ${stage.optional_inputs.join(', ') || 'none'}
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
    )


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
        "Generated Pipeline from MetaEvolve DAG Builder",
        "This file contains a complete pipeline configuration.",
        '"""',
        "",
        "# Core imports",
        "from src.runner.pipeline_factory import PipelineBuilder, PipelineContext",
        "from src.programs.automata import DataFlowEdge, ExecutionOrderDependency",
        "from src.runner.dag_blueprint import DAGBlueprint",
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

            # Add stage comment with metadata
            code_lines.append(f"    # {display_name}")
            if stage.custom_name:
                code_lines.append(f"    # Custom name: {stage.custom_name}")
            if stage.description:
                code_lines.append(f"    # Description: {stage.description}")
            if stage.notes:
                code_lines.append(f"    # Notes: {stage.notes}")

            code_lines.append(f"    builder.add_stage(")
            code_lines.append(f'        "{stage.name}",')
            code_lines.append(f"        lambda: {class_name}(")
            code_lines.append(f'            stage_name="{display_name}",')
            code_lines.append(
                f"            # Add any additional required arguments here"
            )
            code_lines.append(f"        )")
            code_lines.append(f"    )")
            code_lines.append("")

    # Add data flow edges
    if dag_request.data_flow_edges:
        code_lines.append("    # Add data flow edges")
        for edge in dag_request.data_flow_edges:
            code_lines.append(f"    builder.add_data_flow_edge(")
            code_lines.append(f'        "{edge.source_stage}",')
            code_lines.append(f'        "{edge.destination_stage}",')
            code_lines.append(f'        "{edge.input_name}"')
            code_lines.append(f"    )")
        code_lines.append("")

    # Add execution dependencies
    if dag_request.execution_dependencies:
        code_lines.append("    # Add execution dependencies")
        for dep in dag_request.execution_dependencies:
            if dep.dependency_type == "on_success":
                code_lines.append(f"    builder.add_exec_dep(")
                code_lines.append(f'        "{dep.stage}",')
                code_lines.append(
                    f'        ExecutionOrderDependency.on_success("{dep.target_stage}")'
                )
                code_lines.append(f"    )")
            elif dep.dependency_type == "on_failure":
                code_lines.append(f"    builder.add_exec_dep(")
                code_lines.append(f'        "{dep.stage}",')
                code_lines.append(
                    f'        ExecutionOrderDependency.on_failure("{dep.target_stage}")'
                )
                code_lines.append(f"    )")
            elif dep.dependency_type == "always_after":
                code_lines.append(f"    builder.add_exec_dep(")
                code_lines.append(f'        "{dep.stage}",')
                code_lines.append(
                    f'        ExecutionOrderDependency.always_after("{dep.target_stage}")'
                )
                code_lines.append(f"    )")
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


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting MetaEvolve DAG Builder API...")
    print("📱 Open http://localhost:8081 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8081, reload=True)
