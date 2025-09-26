# 🏗️ MetaEvolve DAG Builder

A modern visual interface for building execution pipelines in MetaEvolve using React and React Flow.

## ✨ Features

- **🎨 Visual Stage Library**: Browse all available stages with descriptions and input requirements
- **🖱️ Drag & Drop Interface**: Build DAGs by dragging stages onto the canvas
- **✅ Real-time Validation**: Validate connections and DAG structure with unique name enforcement
- **📤 Code Export**: Generate PipelineBuilder code from your visual DAG
- **🔧 Stage Editor**: Customize stage names, descriptions, and notes
- **🎯 Unique Name Management**: Automatic counter appending and validation to prevent duplicate stage names
- **⌨️ Keyboard Shortcuts**: Quick navigation with F (fit), C (center), 0 (reset), Z (zoom to selected)

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **MetaEvolve project** (this tool is part of the MetaEvolve codebase)

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
cd tools/dag_builder
./start.sh
```

## 📖 Usage Guide

### Building a DAG

1. **Browse Stages**: Look through the stage library on the left panel
2. **Add Stages**: Click the `+` button on any stage card to add it to the canvas
3. **Connect Stages**: Drag from output ports to input ports to create data flow connections
4. **Execution Dependencies**: Connect execution ports (top/bottom) to define execution order
5. **Customize**: Click on any stage to edit its name, description, and notes

### Stage Management

- **Unique Names**: The system automatically prevents duplicate stage names by appending counters (`StageName_1`, `StageName_2`, etc.)
- **Custom Names**: Set custom display names while keeping the original stage type
- **Real-time Validation**: Get immediate feedback when setting duplicate names

### Exporting Your DAG

1. Click the **Export** button in the toolbar
2. The system validates your DAG structure
3. Download the generated Python pipeline code
4. Use the code in your MetaEvolve project

### Keyboard Shortcuts

- **F**: Fit view to all nodes
- **C**: Center view (preserve zoom)
- **0**: Reset zoom and center
- **Z**: Zoom to selected node

## 🏗️ Architecture

### Backend (FastAPI)
- **Stage Registry**: Automatically imports all stages from MetaEvolve using `@StageRegistry.register` decorators
- **DAG Validation**: Validates DAG structure and ensures unique stage names
- **Code Generation**: Exports visual DAGs as PipelineBuilder Python code
- **CORS Support**: Configured for frontend communication

### Frontend (React + React Flow)
- **React Flow**: Modern node-based visual editor
- **Stage Library**: Dynamic stage browser with search functionality
- **Stage Editor**: Inline editing for stage properties
- **Real-time Validation**: Client-side validation with server-side backup

## 🔧 Development

### Project Structure

```
tools/dag_builder/
├── api.py                 # FastAPI backend server
├── run.py                 # Backend entry point
├── start.sh              # Startup script for both backend and frontend
├── requirements.txt      # Python dependencies
├── frontend/             # React frontend
│   ├── src/
│   │   ├── App.js        # Main React component
│   │   ├── components/   # React components
│   │   ├── services/    # API service layer
│   │   └── utils/       # Utility functions
│   ├── package.json     # npm dependencies
│   └── public/          # Static assets
└── README.md            # This file
```

### Adding New Stages

The tool automatically discovers stages from the MetaEvolve codebase. To add a new stage:

1. Create your stage class in the appropriate module
2. Use the `@StageRegistry.register` decorator
3. The stage will automatically appear in the DAG builder

### Customizing the Interface

- **Stage Colors**: Modify `getStageColor()` in `utils/stageUtils.js`
- **Stage Icons**: Update `getStageIcon()` in `utils/stageUtils.js`
- **Validation Rules**: Extend validation in `api.py` and `StageEditor.js`

## 🐛 Troubleshooting

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
# Ensure you're in the MetaEvolve project root
cd /path/to/metaevolve
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

## 📚 API Reference

### Backend Endpoints

- `GET /api/stages` - Get all available stages
- `GET /api/stages/{name}` - Get specific stage info
- `POST /api/export-dag` - Export DAG as PipelineBuilder code

### Frontend Components

- `App.js` - Main application component
- `StageLibrary.js` - Stage browser component
- `StageNode.js` - Individual stage node component
- `StageEditor.js` - Stage editing panel
- `NodeDetails.js` - Stage details panel

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure unique name validation works correctly

## 📄 License

Part of the MetaEvolve project. See the main project license for details.
