// App.js
import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  ReactFlowProvider,
  useReactFlow,
  getNodesBounds,
  Panel,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

import StageLibrary from './components/StageLibrary';
import StageNode from './components/StageNode';
import NodeDetails from './components/NodeDetails';
import StageEditor from './components/StageEditor';
import Toolbar from './components/Toolbar';
import ExecutionEdge from './components/ExecutionEdge';
import DataEdge from './components/DataEdge';
import { getStages, exportDAG } from './services/api';
import { generateNodeId, generateEdgeId, getStageColor, generateUniqueStageName } from './utils/stageUtils';

const nodeTypes = { stageNode: StageNode };
const edgeTypes = { execution: ExecutionEdge, dataFlow: DataEdge };
const defaultEdgeOptions = {
  markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
};

/* ----------------------- UTIL: wait for node measurement ----------------------- */
async function waitForMeasurement(getNodes, { maxFrames = 30 } = {}) {
  let frame = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const nodes = getNodes();
    const ok =
      nodes.length > 0 &&
      nodes.every((n) => n.measured && n.measured.width && n.measured.height);
    if (ok) return true;
    if (frame++ >= maxFrames) return false;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => requestAnimationFrame(r));
  }
}

/* --------------------------- Viewport Toolbar (pinned) --------------------------- */
function ViewportToolbar({ nodeCount, rfReady, selectedNode }) {
  const { fitView, setCenter, getNodes, getZoom } = useReactFlow();

  const ensureReady = useCallback(() => {
    if (!rfReady) {
      console.warn('[Viewport] ReactFlow not initialized yet');
      return false;
    }
    return true;
  }, [rfReady]);

  const handleFitView = useCallback(async () => {
    if (!ensureReady()) return;
    const nodes = getNodes();
    if (!nodes.length) return;
    await waitForMeasurement(getNodes);
    requestAnimationFrame(() => {
      try {
        fitView({ padding: 0.2, includeHiddenNodes: false, duration: 450 });
      } catch (e) {
        console.error('[Viewport] fitView error:', e);
      }
    });
  }, [ensureReady, fitView, getNodes]);

  const handleCenter = useCallback(async () => {
    if (!ensureReady()) return;
    const nodes = getNodes();
    if (!nodes.length) return;
    await waitForMeasurement(getNodes);
    const b = getNodesBounds(nodes);
    const cx = b.x + b.width / 2;
    const cy = b.y + b.height / 2;
    const zoom = getZoom();
    requestAnimationFrame(() => {
      try {
        setCenter(cx, cy, { zoom, duration: 450 });
      } catch (e) {
        console.error('[Viewport] setCenter error:', e);
      }
    });
  }, [ensureReady, getNodes, getZoom, setCenter]);

  const handleReset = useCallback(async () => {
    if (!ensureReady()) return;
    const nodes = getNodes();
    await waitForMeasurement(getNodes);
    if (nodes.length) {
      const b = getNodesBounds(nodes);
      const cx = b.x + b.width / 2;
      const cy = b.y + b.height / 2;
      requestAnimationFrame(() => setCenter(cx, cy, { zoom: 1, duration: 250 }));
    }
  }, [ensureReady, getNodes, setCenter]);

  const handleZoomToSelected = useCallback(async () => {
    if (!ensureReady() || !selectedNode) return;
    await waitForMeasurement(getNodes);
    requestAnimationFrame(() => {
      try {
        // Fit only the selected node; padding gives a pleasing frame
        fitView({ nodes: [selectedNode], padding: 0.6, duration: 350 });
      } catch (e) {
        console.error('[Viewport] fit selected error:', e);
      }
    });
  }, [ensureReady, fitView, getNodes, selectedNode]);

  const btnBase = {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: '8px 12px', borderRadius: 8, border: '1px solid #e5e7eb',
    background: 'white', fontSize: 13, fontWeight: 600, color: '#374151',
    cursor: 'pointer', boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
    transition: 'transform 0.1s ease, box-shadow 0.2s ease, background 0.2s ease',
  };
  const btn = (enabled = true) =>
    enabled
      ? btnBase
      : { ...btnBase, color: '#9ca3af', background: '#f9fafb', cursor: 'not-allowed' };

  return (
    <Panel position="top-right">
      <div
        style={{
          display: 'flex',
          gap: 8,
          padding: 8,
          background: 'rgba(255,255,255,0.9)',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          boxShadow: '0 8px 20px rgba(0,0,0,0.08)',
          backdropFilter: 'blur(6px)',
        }}
      >
        <button style={btn(!!nodeCount)} onClick={handleFitView} disabled={!nodeCount} title="F">
          Fit
        </button>
        <button style={btn(!!nodeCount)} onClick={handleCenter} disabled={!nodeCount} title="C">
          Center
        </button>
        <button style={btn(!!nodeCount)} onClick={handleReset} disabled={!nodeCount} title="0">
          Reset
        </button>
        <button
          style={btn(!!selectedNode)}
          onClick={handleZoomToSelected}
          disabled={!selectedNode}
          title="Z"
        >
          Zoom to selected
        </button>
      </div>
    </Panel>
  );
}

/* -------------------- Auto-fit once when nodes first appear -------------------- */
function AutoFitOnFirstGraph({ rfReady }) {
  const didAutoFitRef = useRef(false);
  const { fitView, getNodes } = useReactFlow();

  useEffect(() => {
    if (!rfReady || didAutoFitRef.current) return;
    const nodes = getNodes();
    if (!nodes.length) return;
    (async () => {
      const ok = await waitForMeasurement(getNodes);
      didAutoFitRef.current = true;
      requestAnimationFrame(() => {
        fitView({ padding: 0.2, includeHiddenNodes: false, duration: ok ? 300 : 0 });
      });
    })();
  }, [rfReady, fitView, getNodes]);

  return null;
}

/* -------------------- Hotkeys (must live inside <ReactFlow/>) -------------------- */
function ViewportHotkeys({ rfReady, selectedNodeRef }) {
  const { fitView, setCenter, getNodes, getZoom } = useReactFlow();

  useEffect(() => {
    if (!rfReady) return;

    const onKey = async (e) => {
      // Ignore when typing in inputs/textareas
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.isComposing) return;

      // F: Fit all
      if (e.key.toLowerCase() === 'f') {
        const nodes = getNodes();
        if (!nodes.length) return;
        await waitForMeasurement(getNodes);
        requestAnimationFrame(() =>
          fitView({ padding: 0.2, includeHiddenNodes: false, duration: 350 })
        );
      }

      // C: Center (preserve zoom)
      if (e.key.toLowerCase() === 'c') {
        const nodes = getNodes();
        if (!nodes.length) return;
        await waitForMeasurement(getNodes);
        const b = getNodesBounds(nodes);
        const cx = b.x + b.width / 2;
        const cy = b.y + b.height / 2;
        const z = getZoom();
        requestAnimationFrame(() => setCenter(cx, cy, { zoom: z, duration: 300 }));
      }

      // 0: Reset zoom & center
      if (e.key === '0') {
        const nodes = getNodes();
        if (!nodes.length) return;
        await waitForMeasurement(getNodes);
        const b = getNodesBounds(nodes);
        const cx = b.x + b.width / 2;
        const cy = b.y + b.height / 2;
        requestAnimationFrame(() => setCenter(cx, cy, { zoom: 1, duration: 250 }));
      }

      // Z: Zoom to selected node
      if (e.key.toLowerCase() === 'z') {
        const n = selectedNodeRef.current;
        if (!n) return;
        await waitForMeasurement(getNodes);
        requestAnimationFrame(() => fitView({ nodes: [n], padding: 0.6, duration: 300 }));
      }
    };

    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [rfReady, fitView, setCenter, getNodes, getZoom, selectedNodeRef]);

  return null;
}

/* ---------------------------------- App ---------------------------------- */

function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [stages, setStages] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const selectedNodeRef = useRef(null);
  const [editingNode, setEditingNode] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [rfReady, setRfReady] = useState(false);

  useEffect(() => {
    selectedNodeRef.current = selectedNode;
  }, [selectedNode]);

  // Load stages on mount
  useEffect(() => {
    (async () => {
      try {
        setIsLoading(true);
        const stagesData = await getStages();
        setStages(stagesData);
        setError(null);
      } catch (err) {
        setError('Failed to load stages: ' + (err?.message || String(err)));
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  // Validate connection compatibility to prevent snapping to wrong port types
  const isValidConnection = useCallback(
    (connection) => {
      const { sourceHandle, targetHandle, source, target } = connection || {};
      if (!sourceHandle || !targetHandle || !source || !target) return false;

      const isExecSource = sourceHandle === 'exec-source';
      const isExecTarget = targetHandle?.startsWith('exec-');

      // Enforce exec↔exec, data↔data
      if (isExecSource !== isExecTarget) return false;

      if (isExecSource && isExecTarget) {
        // Block duplicate execution edge from same source → same target handle
        const dupExec = edges.some(
          (e) =>
            e.type === 'execution' &&
            e.source === source &&
            e.target === target &&
            e.targetHandle === targetHandle
        );
        return !dupExec;
      }

      // Data: source must be 'output' and target must NOT be exec-*
      if (sourceHandle !== 'output') return false;

      // Block multiple wires into the same data input handle
      const occupied = edges.some(
        (e) =>
          e.type === 'dataFlow' &&
          e.target === target &&
          e.targetHandle === targetHandle
      );
      return !occupied;
    },
    [edges]
  );

  const onConnect = useCallback(
    (params) => {
      const { sourceHandle, targetHandle, source, target } = params || {};
      if (!sourceHandle || !targetHandle || !source || !target) return;

      const isExec = sourceHandle === 'exec-source' && targetHandle?.startsWith('exec-');
      const isData = sourceHandle === 'output' && !targetHandle?.startsWith('exec-');

      // Hard-stop if invalid (keeps behavior consistent with drag preview)
      if (!isExec && !isData) return;

      // Dedup guards
      if (isExec) {
        const exists = edges.some(
          (e) =>
            e.type === 'execution' &&
            e.source === source &&
            e.target === target &&
            e.targetHandle === targetHandle
        );
        if (exists) return;

        const executionType = targetHandle.replace('exec-', '');
        const executionColors = { always: '#6c757d', success: '#28a745', failure: '#dc3545' };

        const edge = {
          id: generateEdgeId(),
          ...params,
          type: 'execution',
          animated: true,
          style: {
            stroke: executionColors[executionType] || '#6c757d',
            strokeWidth: 3,
            strokeDasharray: '5,5',
          },
          data: { executionType },
        };
        setEdges((eds) => addEdge(edge, eds));
        return;
      }

      if (isData) {
        // Only one edge may occupy a given input handle
        const occupied = edges.some(
          (e) =>
            e.type === 'dataFlow' &&
            e.target === target &&
            e.targetHandle === targetHandle
        );
        if (occupied) return;

        const targetInputName = targetHandle;
        const edge = {
          id: generateEdgeId(),
          ...params,
          type: 'dataFlow',
          animated: false,
          style: { stroke: '#94a3b8', strokeWidth: 2 },
          data: { inputName: targetInputName },
        };
        setEdges((eds) => addEdge(edge, eds));
      }
    },
    [edges, setEdges]
  );

  const onNodeClick = useCallback((_, node) => {
    setSelectedNode(node);
    setEditingNode(null);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setEditingNode(null);
  }, []);

  const handleEditNode = useCallback(
    (nodeId, editedData) => {
      setNodes((nds) =>
        nds.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  ...editedData,
                },
              }
            : node
        )
      );
      setEditingNode(null);
    },
    [setNodes]
  );

  const openEditor = useCallback((node) => {
    setEditingNode(node);
    setSelectedNode(null);
  }, []);

  const addStageToCanvas = useCallback(
    (stageInfo, position = null) => {
      let nodePosition;
      if (position) {
        nodePosition = position;
      } else if (nodes.length === 0) {
        nodePosition = { x: 0, y: 0 };
      } else {
        const lastNode = nodes[nodes.length - 1];
        nodePosition = {
          x: lastNode.position.x + 250,
          y: lastNode.position.y + (nodes.length % 2 === 0 ? 0 : 100),
        };
      }

      // Generate a unique name for the stage to prevent duplicates
      const uniqueName = generateUniqueStageName(stageInfo.name, nodes);

      const newNode = {
        id: generateNodeId(),
        type: 'stageNode',
        position: nodePosition,
        data: {
          ...stageInfo,
          originalName: stageInfo.name, // Preserve original stage type
          name: uniqueName, // Use unique name as the actual name
          incomingExecutionType: null,
        },
      };

      setNodes((nds) => {
        // Double-check that we don't already have a node with this unique name
        const existingNames = nds.map((node) => node.data?.name).filter(Boolean);
        if (existingNames.includes(uniqueName)) {
          console.warn('Duplicate name detected, skipping:', uniqueName);
          return nds;
        }
        return [...nds, newNode];
      });
    },
    [setNodes, nodes]
  );

  const exportPipeline = useCallback(async () => {
    try {
      const dagData = {
        stages: nodes.map((node) => ({
          name: node.data.originalName || node.data.name, // Use original stage type for backend
          custom_name: node.data.customName || null,
          display_name: node.data.customName || node.data.name, // Use unique name for display
          description: node.data.description || '',
          notes: node.data.notes || '',
        })),
        data_flow_edges: edges
          .filter((edge) => edge.type === 'dataFlow')
          .map((edge) => ({
            source_stage: nodes.find((n) => n.id === edge.source)?.data?.name, // Use unique name
            destination_stage: nodes.find((n) => n.id === edge.target)?.data?.name, // Use unique name
            input_name: edge.data?.inputName || edge.targetHandle || 'default',
          })),
        execution_dependencies: edges
          .filter((edge) => edge.type === 'execution')
          .map((edge) => ({
            stage: nodes.find((n) => n.id === edge.source)?.data?.name, // Use unique name
            dependency_type: edge.data?.executionType || 'always',
            target_stage: nodes.find((n) => n.id === edge.target)?.data?.name, // Use unique name
          })),
      };

      const result = await exportDAG(dagData);
      const blob = new Blob([result.code], { type: 'text/python' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'generated_pipeline.py';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to export DAG: ' + (err?.message || String(err)));
    }
  }, [nodes, edges]);

  const clearCanvas = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNode(null);
  }, [setNodes, setEdges]);

  const handleEdgesChange = useCallback(
    (changes) => {
      onEdgesChange(changes);
      // If you later mirror edge state into node.data (e.g., marking an input as "occupied"),
      // this is the right place to react to `remove` changes and recompute.
    },
    [onEdgesChange]
  );

  if (isLoading) {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
          fontSize: 18,
          color: '#666',
        }}
      >
        Loading stages...
      </div>
    );
  }

  /* --------------------- LAYOUT: prevent page from scrolling --------------------- */
  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        fontFamily:
          'Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          background: 'white',
          borderBottom: '1px solid #e5e7eb',
          padding: '12px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
          flex: '0 0 auto',
        }}
      >
        <h1 style={{ margin: 0, fontSize: 22, color: '#111827', letterSpacing: 0.2 }}>
          🏗️ MetaEvolve DAG Builder
        </h1>
        <Toolbar
          onExport={exportPipeline}
          onClear={clearCanvas}
          nodeCount={nodes.length}
          edgeCount={edges.length}
        />
      </div>

      {/* Error message */}
      {error && (
        <div
          style={{
            background: '#fef2f2',
            color: '#991b1b',
            padding: '8px 20px',
            borderBottom: '1px solid #fecaca',
            fontWeight: 600,
          }}
        >
          {error}
        </div>
      )}

      {/* Main content */}
      <div style={{ flex: '1 1 auto', display: 'flex', minHeight: 0 }}>
        {/* Stage Library (scrolls independently) */}
        <div
          style={{
            flex: '0 0 auto',
            overflowY: 'auto',
            borderRight: '1px solid #e5e7eb',
            maxWidth: 360,
          }}
        >
          <StageLibrary stages={stages} onAddStage={addStageToCanvas} />
        </div>

        {/* Canvas area (pinned, never page-scrolls) */}
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden', minWidth: 0 }}>
          {/* Overlay hint */}
          {nodes.length === 0 && (
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
                color: '#6b7280',
                zIndex: 2,
                pointerEvents: 'none',
              }}
            >
              <div style={{ fontSize: 48, marginBottom: 10 }}>🎯</div>
              <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Empty Canvas</div>
              <div style={{ fontSize: 14 }}>
                Add stages from the library to start building your pipeline
              </div>
            </div>
          )}

          {/* Count badge */}
          {nodes.length > 0 && (
            <div
              style={{
                position: 'absolute',
                top: 16,
                left: 16,
                background: 'rgba(255, 255, 255, 0.9)',
                padding: '8px 12px',
                borderRadius: 10,
                border: '1px solid #e5e7eb',
                fontSize: 12,
                fontWeight: 600,
                color: '#374151',
                zIndex: 2,
                boxShadow: '0 4px 14px rgba(0,0,0,0.08)',
                backdropFilter: 'blur(4px)',
              }}
            >
              📊 {nodes.length} stages • {edges.length} connections
            </div>
          )}

          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={handleEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            isValidConnection={isValidConnection}
            minZoom={0.3}
            maxZoom={1.5}
            fitView
            fitViewOptions={{ padding: 0.2, includeHiddenNodes: false }}
            attributionPosition="bottom-left"
            defaultEdgeOptions={defaultEdgeOptions}
            connectionLineStyle={{ stroke: '#94a3b8', strokeWidth: 2 }}
            elevateEdgesOnSelect
            onInit={() => setRfReady(true)}
            style={{ width: '100%', height: '100%' }}
            // Canvas ergonomics
            panOnScroll
            panOnDrag
            zoomOnDoubleClick={false}
            selectionOnDrag
            multiSelectionKeyCode="Shift"
            snapToGrid
            snapGrid={[10, 10]}
          >
            <Background variant="dots" gap={20} size={1} color="#e5e7eb" />
            <Controls
              position="bottom-left"
              showInteractive={false}
              style={{
                background: 'white',
                border: '1px solid #e5e7eb',
                borderRadius: 10,
                boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
              }}
            />
            <MiniMap
              nodeColor={(node) =>
                node.data?.originalName ? getStageColor(node.data.originalName) : '#d1d5db'
              }
              style={{
                background: '#f9fafb',
                border: '1px solid #e5e7eb',
                borderRadius: 10,
                boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
              }}
              position="bottom-right"
              pannable
              zoomable
            />

            {/* Pinned viewport toolbar (no +/-) */}
            <ViewportToolbar nodeCount={nodes.length} rfReady={rfReady} selectedNode={selectedNode} />

            {/* Hotkeys tied to this RF instance */}
            <ViewportHotkeys rfReady={rfReady} selectedNodeRef={selectedNodeRef} />

            {/* Auto-fit once when nodes first appear */}
            <AutoFitOnFirstGraph rfReady={rfReady} />
          </ReactFlow>
        </div>

        {/* Node Details Panel */}
        {selectedNode && (
          <div
            style={{
              flex: '0 0 360px',
              maxWidth: 360,
              borderLeft: '1px solid #e5e7eb',
              overflowY: 'auto',
            }}
          >
            <NodeDetails
              node={selectedNode}
              onClose={() => setSelectedNode(null)}
              onEdit={openEditor}
            />
          </div>
        )}

        {/* Stage Editor Panel */}
        {editingNode && (
          <div
            style={{
              flex: '0 0 420px',
              maxWidth: 420,
              borderLeft: '1px solid #e5e7eb',
              overflowY: 'auto',
            }}
          >
            <StageEditor
              node={editingNode}
              nodes={nodes}
              onSave={handleEditNode}
              onClose={() => setEditingNode(null)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default function AppWithProvider() {
  return (
    <ReactFlowProvider>
      <App />
    </ReactFlowProvider>
  );
}
