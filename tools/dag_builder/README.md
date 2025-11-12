# GigaEvo DAG Builder

A modern visual interface for building execution pipelines in GigaEvo using React and React Flow.

## Features

### Core Functionality
- **Visual Stage Library**: Browse all available stages with descriptions and input requirements
- **Drag & Drop Interface**: Build DAGs by dragging stages onto the canvas
- **Real-time Validation**: Validate connections and DAG structure with unique name enforcement
- **Stage Editor**: Customize stage names, descriptions, and notes
- **Unique Name Management**: Automatic counter appending and validation to prevent duplicate stage names
- **Quick Add**: Press 'A' or use Quick Add button to rapidly add stages to your canvas

### Export & Import
- **Multiple Export Formats**:
  - **Python Code**: Generate PipelineBuilder code from your visual DAG
  - **YAML Config**: Export as Hydra YAML configuration compatible with GigaEvo
  - **PDF Document**: Export canvas as a high-quality PDF for documentation
- **Config Loading**: Load and edit existing pipeline configurations from your project
- **Auto-Discovery**: Automatically finds all pipeline configs in `config/pipeline/`

### Editing & Navigation
- **Undo/Redo**: Full history management for all DAG changes (Ctrl+Z / Ctrl+Y)
- **Keyboard Shortcuts**: Quick navigation with F (fit), C (center), 0 (reset), Z (zoom to selected), A (quick add)
- **Toast Notifications**: Real-time feedback for all operations and errors
- **Smart Positioning**: Auto-layout and intelligent node positioning

## Installation & Setup

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **GigaEvo project** (this tool is part of the GigaEvo codebase)

### 1. Backend Setup (Python/FastAPI)

The backend runs a FastAPI server that provides stage registry and export functionality.

```bash
# Navigate to the DAG builder directory
cd tools/dag_builder

# Install Python dependencies (if not already installed)
pip install fastapi uvicorn pydantic

# Start the backend server
python run.py
```

The backend will start on **http://localhost:8081** with:
- Main interface: http://localhost:8081
- API documentation: http://localhost:8081/docs

### 2. Frontend Setup (React)

The frontend is a modern React application with React Flow for the visual DAG interface.

```bash
# Navigate to the frontend directory
cd tools/dag_builder/frontend

# Install npm dependencies
npm install

# Start the development server
npm start
```

The frontend will start on **http://localhost:8082** and automatically open in your browser.

### 3. Quick Start Script

The project includes a convenient startup script that runs both backend and frontend:

```bash
# Start both backend and frontend
bash tools/dag_builder/start.sh
```

## Usage Guide

### Building a DAG

1. **Browse Stages**: Look through the stage library on the left panel
2. **Add Stages**:
   - Click the `+` button on any stage card to add it to the canvas
   - Press `A` to open Quick Add dialog for rapid stage insertion
   - Search and filter stages by name, description, or inputs/outputs
3. **Connect Stages**: Drag from output ports to input ports to create data flow connections
4. **Execution Dependencies**: Connect execution ports (top/bottom) to define execution order
5. **Customize**: Click on any stage to edit its name, description, and notes
6. **Undo/Redo**: Use Ctrl+Z / Ctrl+Y or toolbar buttons to navigate your edit history

### Stage Management

- **Unique Names**: The system automatically prevents duplicate stage names by appending counters (`StageName_1`, `StageName_2`, etc.)
- **Custom Names**: Set custom display names while keeping the original stage type
- **Real-time Validation**: Get immediate feedback when setting duplicate names
- **Quick Search**: Use the Quick Add dialog (press `A`) to search through all available stages

### Loading Existing Configurations

1. Click the **Load Config** button in the toolbar
2. Browse available pipeline configurations from `config/pipeline/`
3. Select a config to load it into the visual editor
4. Modify the DAG as needed and re-export

### Exporting Your DAG

The DAG Builder supports multiple export formats:

#### 1. Export as Python Code
1. Click the **Export Code** button in the toolbar
2. The system validates your DAG structure
3. Download the generated PipelineBuilder Python code
4. Use the code in your GigaEvo project

#### 2. Export as YAML Config
1. Click the **Export YAML** button in the toolbar
2. The system generates a Hydra-compatible YAML configuration
3. Download the YAML file
4. Place it in your `config/pipeline/` directory

#### 3. Export as PDF
1. Click the **Export PDF** button in the toolbar
2. The entire canvas is rendered as a high-quality PDF document
3. Perfect for documentation, presentations, or archiving

### Keyboard Shortcuts

#### Navigation
- **F**: Fit view to all nodes
- **C**: Center view (preserve zoom)
- **0**: Reset zoom and center
- **Z**: Zoom to selected node

#### Editing
- **A**: Open Quick Add dialog
- **Ctrl+Z** / **Cmd+Z**: Undo last action
- **Ctrl+Y** / **Cmd+Y**: Redo last undone action
- **Delete** / **Backspace**: Delete selected node or edge

#### Quick Add Dialog (when open)
- **Up/Down Arrow**: Navigate through filtered stages
- **Enter**: Add selected stage
- **Escape**: Close dialog

## Architecture

### Backend (FastAPI)
- **Stage Registry**: Automatically imports all stages from GigaEvo using `@StageRegistry.register` decorators
- **DAG Validation**: Validates DAG structure and ensures unique stage names
- **Multiple Export Formats**:
  - Python PipelineBuilder code generation
  - YAML Hydra configuration export
  - PDF document rendering
- **Config Management**: Load and parse existing pipeline configurations
- **Auto-Discovery**: Scans `config/pipeline/` for available configurations
- **CORS Support**: Configured for frontend communication

### Frontend (React + React Flow)
- **React Flow**: Modern node-based visual editor with advanced viewport controls
- **Stage Library**: Dynamic stage browser with fuzzy search
- **Quick Add Dialog**: Keyboard-driven rapid stage insertion (press `A`)
- **Stage Editor**: Inline editing for stage properties with validation
- **Undo/Redo System**: Full history management for all DAG operations
- **Toast Notifications**: Real-time user feedback system
- **Multi-Format Export**: Code, YAML, and PDF export capabilities
- **Config Loading**: Import and visualize existing pipeline configurations

## Development

### Project Structure

```
tools/dag_builder/
├── api.py                      # FastAPI backend server with all endpoints
├── run.py                      # Backend entry point
├── start.sh                    # Startup script for both backend and frontend
├── requirements.txt            # Python dependencies
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.js              # Main application with undo/redo logic
│   │   ├── index.js            # React entry point
│   │   ├── index.css           # Global styles
│   │   ├── components/         # React components
│   │   │   ├── StageLibrary.js            # Stage browser
│   │   │   ├── StageNode.js               # Individual stage node
│   │   │   ├── StageEditor.js             # Stage editing panel
│   │   │   ├── NodeDetails.js             # Stage details panel
│   │   │   ├── QuickAdd.js                # Quick stage insertion (press A)
│   │   │   ├── Toolbar.js                 # Top toolbar with export/import
│   │   │   ├── Toast.js                   # Notification system
│   │   │   ├── DataEdge.js                # Data flow edge component
│   │   │   ├── ExecutionEdge.js           # Execution dependency edge
│   │   │   └── ConnectionLineWithTooltip.js
│   │   ├── services/           # API service layer
│   │   │   └── api.js          # Backend API client
│   │   └── utils/              # Utility functions
│   │       ├── stageUtils.js   # Stage-related utilities
│   │       └── helpers.js      # General helper functions
│   ├── package.json            # npm dependencies
│   └── public/                 # Static assets
│       └── index.html          # HTML template
└── README.md                   # This file
```

### Adding New Stages

The tool automatically discovers stages from the GigaEvo codebase. To add a new stage:

1. Create your stage class in the appropriate module
2. Use the `@StageRegistry.register` decorator
3. The stage will automatically appear in the DAG builder

### Customizing the Interface

- **Stage Colors**: Modify `getStageColor()` in `utils/stageUtils.js`
- **Stage Icons**: Update `getStageIcon()` in `utils/stageUtils.js`
- **Validation Rules**: Extend validation in `api.py` and `StageEditor.js`

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Backend (port 8081)
lsof -ti:8081 | xargs kill -9

# Frontend (port 3000)
lsof -ti:8082 | xargs kill -9
```

**npm Install Issues**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Python Import Errors**
```bash
# Ensure you're in the GigaEvo project root
cd /path/to/gigaevo
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Development Mode

For development with hot reloading:

```bash
# Terminal 1: Backend with auto-reload
cd tools/dag_builder
python run.py

# Terminal 2: Frontend with hot reload
cd tools/dag_builder/frontend
npm start
```

## API Reference

### Backend Endpoints

#### Stage Management
- `GET /api/stages` - Get all available stages from the registry
- `GET /api/stages/{name}` - Get detailed information for a specific stage

#### DAG Operations
- `POST /api/validate-dag` - Validate DAG structure and connections
  - Request body: `DAGRequest` (stages, data_flow_edges, execution_dependencies)
  - Returns: `DAGValidationResponse` (is_valid, errors, warnings)

#### Export Endpoints
- `POST /api/export-dag` - Export DAG as Python PipelineBuilder code
  - Request body: `DAGRequest`
  - Returns: `DAGExportResponse` (code, validation_errors)

- `POST /api/export-yaml` - Export DAG as Hydra YAML configuration
  - Request body: `DAGRequest`
  - Returns: `YAMLExportResponse` (yaml, validation_errors)

#### Config Management
- `GET /api/yaml-configs` - List all available pipeline configurations in `config/pipeline/`
  - Returns: List of `YAMLConfigInfo` (filename, path, display_name)

- `POST /api/load-yaml` - Load and parse a YAML config into DAG builder format
  - Request body: `LoadYAMLRequest` (yaml_path)
  - Returns: `LoadYAMLResponse` (stages, edges, dependencies, errors, warnings)

### Frontend Components

#### Core Components
- `App.js` - Main application with undo/redo state management
- `Toolbar.js` - Top toolbar with export, import, and history controls
- `Toast.js` - Toast notification system for user feedback

#### Stage Components
- `StageLibrary.js` - Browsable stage library with search
- `StageNode.js` - Individual stage node with ports
- `StageEditor.js` - Right panel for editing stage properties
- `NodeDetails.js` - Stage details display panel
- `QuickAdd.js` - Quick stage insertion dialog (keyboard-driven)

#### Edge Components
- `DataEdge.js` - Data flow edge rendering
- `ExecutionEdge.js` - Execution dependency edge rendering
- `ConnectionLineWithTooltip.js` - Connection preview with tooltips

#### Services
- `services/api.js` - API client for backend communication

#### Utilities
- `utils/stageUtils.js` - Stage color, icons, and name generation
- `utils/helpers.js` - General helper functions

## Tips & Best Practices

### Workflow Recommendations

1. **Start with Quick Add**: Press `A` to quickly search and add stages instead of scrolling through the library
2. **Use Keyboard Shortcuts**: Master `F` (fit view), `Z` (zoom to selected), and `Ctrl+Z` (undo) for efficient editing
3. **Save Your Work**: Export to YAML regularly to save your progress and version control your pipelines
4. **Validate Early**: The system validates in real-time, but use the validation endpoint for detailed error messages
5. **Name Meaningfully**: Use descriptive custom names for stages to make your DAG more readable

### Performance Tips

- **Large DAGs**: For DAGs with 50+ nodes, use the minimap (bottom right) for navigation
- **Clean Layout**: Use `F` to auto-fit view after adding multiple stages
- **Undo History**: The undo/redo system keeps up to 50 states - use it liberally

### Export Format Comparison

| Format | Use Case | Best For |
|--------|----------|----------|
| **Python Code** | Integration with custom Python scripts | Development, custom pipelines |
| **YAML Config** | Hydra configuration management | Production, config-driven workflows |
| **PDF** | Documentation and presentations | Sharing, archiving, documentation |

## Recent Updates

### Version 2.0 Features
- **YAML Import/Export**: Full support for Hydra configuration files
- **Config Loading**: Load and edit existing pipeline configurations
- **PDF Export**: Generate documentation-ready PDFs
- **Quick Add Dialog**: Keyboard-driven stage insertion with fuzzy search
- **Undo/Redo System**: Full edit history with Ctrl+Z / Ctrl+Y support
- **Toast Notifications**: Real-time feedback for all operations
- **Enhanced Validation**: More detailed error messages and warnings
- **Auto-Discovery**: Automatic detection of pipeline configs in project

### Coming Soon
- Drag-to-reorder in stage library
- Multi-select for batch operations
- Stage templates and snippets
- Collaborative editing support
- Git integration for config versioning

## Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure unique name validation works correctly
5. Test export/import functionality with real configs
6. Verify keyboard shortcuts work across platforms

## License

Part of the GigaEvo project. See the main project license for details.
