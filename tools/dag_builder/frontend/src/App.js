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
import QuickAdd from './components/QuickAdd';
import Toast from './components/Toast';
import { getStages, exportYAML, validateDAG, listYAMLConfigs, loadYAMLConfig } from './services/api';
import { generateNodeId, generateEdgeId, getStageColor, generateUniqueStageName } from './utils/stageUtils';
import { toPng } from 'html-to-image';
import { jsPDF } from 'jspdf';

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
  const [validationStatus, setValidationStatus] = useState(null);
  const [showConfigLoader, setShowConfigLoader] = useState(false);
  const [availableConfigs, setAvailableConfigs] = useState([]);
  const [loadingConfig, setLoadingConfig] = useState(false);
  const [validationErrors, setValidationErrors] = useState([]);

  // History for undo/redo
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [copiedNode, setCopiedNode] = useState(null);

  // Quick Add modal
  const [showQuickAdd, setShowQuickAdd] = useState(false);
  const [quickAddPosition, setQuickAddPosition] = useState(null);
  const { project } = useReactFlow();

  // Toast notifications
  const [toast, setToast] = useState(null);

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

  // Track if we're in the middle of undo/redo to avoid capturing those states
  const isUndoRedoRef = useRef(false);

  // History management - automatically capture state changes
  useEffect(() => {
    // Don't capture history during undo/redo or if no nodes
    if (isUndoRedoRef.current || isLoading) return;

    // Debounce history capture to avoid intermediate states
    const timeoutId = setTimeout(() => {
      const snapshot = {
        nodes: JSON.parse(JSON.stringify(nodes)),
        edges: JSON.parse(JSON.stringify(edges)),
      };

      // Don't push if this is the same as the current history entry
      if (historyIndex >= 0 && history[historyIndex]) {
        const current = history[historyIndex];
        const isSame =
          JSON.stringify(current.nodes) === JSON.stringify(snapshot.nodes) &&
          JSON.stringify(current.edges) === JSON.stringify(snapshot.edges);
        if (isSame) return;
      }

      // Remove any future history if we're not at the end
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push(snapshot);

      // Limit history to 50 entries
      if (newHistory.length > 50) {
        newHistory.shift();
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);
      } else {
        setHistory(newHistory);
        setHistoryIndex(newHistory.length - 1);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timeoutId);
  }, [nodes, edges, history, historyIndex, isLoading]);

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

  const onNodesDelete = useCallback((deletedNodes) => {
    // If the deleted node was selected, close the sidebar
    if (selectedNode && deletedNodes.some(node => node.id === selectedNode.id)) {
      setSelectedNode(null);
    }
  }, [selectedNode]);

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

      // Select the newly added node
      setSelectedNode(newNode);
    },
    [setNodes, nodes]
  );

  // Helper function to build DAG data structure
  const buildDAGData = useCallback(() => {
    // Helper to get the effective name (customName takes precedence)
    const getEffectiveName = (node) => node?.data?.customName || node?.data?.name;

    return {
      stages: nodes.map((node) => ({
        name: node.data.originalName || node.data.name, // Use original stage type for backend
        custom_name: node.data.customName || null,
        display_name: getEffectiveName(node), // Use custom name if set, otherwise use unique name
        description: node.data.description || '',
        notes: node.data.notes || '',
      })),
      data_flow_edges: edges
        .filter((edge) => edge.type === 'dataFlow')
        .map((edge) => ({
          source_stage: getEffectiveName(nodes.find((n) => n.id === edge.source)), // Use effective name
          destination_stage: getEffectiveName(nodes.find((n) => n.id === edge.target)), // Use effective name
          input_name: edge.data?.inputName || edge.targetHandle || 'default',
        })),
      execution_dependencies: edges
        .filter((edge) => edge.type === 'execution')
        .map((edge) => {
          // IMPORTANT: Execution edge direction is reversed!
          // Visual edge: source → target means "source must run before target"
          // So target depends on source, thus: stage=target, target_stage=source

          // Map frontend executionType to backend dependency_type format
          const typeMap = {
            'success': 'on_success',
            'failure': 'on_failure',
            'always': 'always_after'
          };
          const execType = edge.data?.executionType || 'always';

          return {
            stage: getEffectiveName(nodes.find((n) => n.id === edge.target)), // Stage that has dependency
            dependency_type: typeMap[execType] || 'always_after',
            target_stage: getEffectiveName(nodes.find((n) => n.id === edge.source)), // Stage it depends on
          };
        }),
    };
  }, [nodes, edges]);

  // Automatic validation when nodes or edges change
  useEffect(() => {
    // Debounce validation to avoid calling too frequently
    const timeoutId = setTimeout(async () => {
      if (nodes.length === 0) {
        setValidationStatus(null);
        setValidationErrors([]);
        return;
      }

      try {
        const dagData = buildDAGData();
        const result = await validateDAG(dagData);
        setValidationStatus(result);

        if (!result.is_valid) {
          setValidationErrors(result.errors || []);
        } else {
          setValidationErrors([]);
        }
      } catch (err) {
        console.error('Validation error:', err);
        setValidationErrors([err?.message || String(err)]);
      }
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  }, [nodes, edges, buildDAGData]);

  const handleValidate = useCallback(async () => {
    try {
      const dagData = buildDAGData();
      const result = await validateDAG(dagData);
      setValidationStatus(result);

      if (!result.is_valid) {
        setValidationErrors(result.errors || []);
      } else {
        setValidationErrors([]);
      }
    } catch (err) {
      setValidationErrors([err?.message || String(err)]);
      setValidationStatus(null);
    }
  }, [buildDAGData]);

  const exportYAMLConfig = useCallback(async () => {
    try {
      const dagData = buildDAGData();
      const result = await exportYAML(dagData);
      const blob = new Blob([result.yaml], { type: 'text/yaml' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'generated_pipeline.yaml';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      setToast({ message: 'YAML config exported successfully!', type: 'success' });
    } catch (err) {
      setError('Failed to export YAML: ' + (err?.message || String(err)));
      setToast({ message: 'Failed to export YAML', type: 'error' });
    }
  }, [buildDAGData]);

  const exportPDF = useCallback(() => {
    const reactFlowElement = document.querySelector('.react-flow__viewport');
    if (!reactFlowElement) {
      setToast({ message: 'Canvas not found', type: 'error' });
      return;
    }

    // Get the viewport element for better quality capture
    const viewportElement = reactFlowElement;

    // Hide controls and minimap temporarily for cleaner export
    const flowWrapper = document.querySelector('.react-flow');
    const controls = flowWrapper?.querySelector('.react-flow__controls');
    const minimap = flowWrapper?.querySelector('.react-flow__minimap');
    const background = flowWrapper?.querySelector('.react-flow__background');
    const panel = flowWrapper?.querySelector('.react-flow__panel');

    const originalControlsDisplay = controls?.style.display;
    const originalMinimapDisplay = minimap?.style.display;
    const originalPanelDisplay = panel?.style.display;
    const originalBackgroundOpacity = background?.style.opacity;

    if (controls) controls.style.display = 'none';
    if (minimap) minimap.style.display = 'none';
    if (panel) panel.style.display = 'none';
    if (background) background.style.opacity = '0.3';

    // Use higher pixel ratio and cacheBust for better quality
    toPng(flowWrapper, {
      backgroundColor: '#ffffff',
      quality: 1.0,
      pixelRatio: 3, // Even higher resolution for crisp output
      cacheBust: true,
      style: {
        transform: 'none', // Remove any transforms
      },
    })
      .then((dataUrl) => {
        // Restore hidden elements
        if (controls) controls.style.display = originalControlsDisplay;
        if (minimap) minimap.style.display = originalMinimapDisplay;
        if (panel) panel.style.display = originalPanelDisplay;
        if (background) background.style.opacity = originalBackgroundOpacity;

        const img = new Image();
        img.onload = () => {
          // Create PDF in landscape if width > height, portrait otherwise
          const imgWidth = img.width;
          const imgHeight = img.height;
          const isLandscape = imgWidth > imgHeight;

          // A4 dimensions in mm
          const pdf = new jsPDF({
            orientation: isLandscape ? 'landscape' : 'portrait',
            unit: 'mm',
            format: 'a4',
            compress: false, // Don't compress for better quality
          });

          const pageWidth = pdf.internal.pageSize.getWidth();
          const pageHeight = pdf.internal.pageSize.getHeight();

          // Calculate dimensions to fit page while maintaining aspect ratio
          const imgRatio = imgWidth / imgHeight;
          const pageRatio = pageWidth / pageHeight;

          let finalWidth, finalHeight;
          if (imgRatio > pageRatio) {
            // Image is wider than page ratio
            finalWidth = pageWidth - 20; // 10mm margin on each side
            finalHeight = finalWidth / imgRatio;
          } else {
            // Image is taller than page ratio
            finalHeight = pageHeight - 20; // 10mm margin on each side
            finalWidth = finalHeight * imgRatio;
          }

          const x = (pageWidth - finalWidth) / 2;
          const y = (pageHeight - finalHeight) / 2;

          // Use PNG format with no compression for best quality
          pdf.addImage(dataUrl, 'PNG', x, y, finalWidth, finalHeight, undefined, 'FAST');
          pdf.save('dag-pipeline.pdf');

          setToast({ message: 'PDF exported successfully!', type: 'success' });
        };
        img.src = dataUrl;
      })
      .catch((err) => {
        // Restore hidden elements on error
        if (controls) controls.style.display = originalControlsDisplay;
        if (minimap) minimap.style.display = originalMinimapDisplay;
        if (panel) panel.style.display = originalPanelDisplay;
        if (background) background.style.opacity = originalBackgroundOpacity;

        console.error('Failed to export PDF:', err);
        setToast({ message: 'Failed to export PDF', type: 'error' });
      });
  }, []);

  const clearCanvas = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNode(null);
  }, [setNodes, setEdges]);

  // Helper function to apply layout algorithm to nodes (pure function)
  const applyLayoutToNodes = useCallback((nodesToLayout, edgesToLayout) => {
    if (!nodesToLayout || nodesToLayout.length === 0) return nodesToLayout;

    // Build adjacency list for topological sort
    const graph = new Map();
    const inDegree = new Map();

    nodesToLayout.forEach(node => {
      graph.set(node.id, []);
      inDegree.set(node.id, 0);
    });

    edgesToLayout.forEach(edge => {
      if (edge.type === 'dataFlow' || edge.type === 'execution') {
        if (!graph.has(edge.source)) graph.set(edge.source, []);
        graph.get(edge.source).push(edge.target);
        inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
      }
    });

    // Topological sort to get layers
    const layers = [];
    const queue = nodesToLayout.filter(n => inDegree.get(n.id) === 0).map(n => n.id);
    const layerMap = new Map();

    queue.forEach(id => layerMap.set(id, 0));

    while (queue.length > 0) {
      const nodeId = queue.shift();
      const layer = layerMap.get(nodeId);

      if (!layers[layer]) layers[layer] = [];
      layers[layer].push(nodeId);

      const neighbors = graph.get(nodeId) || [];
      neighbors.forEach(targetId => {
        const newInDegree = inDegree.get(targetId) - 1;
        inDegree.set(targetId, newInDegree);

        if (newInDegree === 0) {
          layerMap.set(targetId, layer + 1);
          queue.push(targetId);
        }
      });
    }

    // Handle nodes with no edges
    nodesToLayout.forEach(node => {
      if (!layerMap.has(node.id)) {
        const lastLayer = layers.length;
        if (!layers[lastLayer]) layers[lastLayer] = [];
        layers[lastLayer].push(node.id);
        layerMap.set(node.id, lastLayer);
      }
    });

    // Position nodes in layers
    const layerSpacing = 450;
    const nodeSpacing = 180;

    return nodesToLayout.map(node => {
      const nodeLayer = layerMap.get(node.id) || 0;
      const nodesInLayer = layers[nodeLayer] || [];
      const indexInLayer = nodesInLayer.indexOf(node.id);
      const layerHeight = nodesInLayer.length * nodeSpacing;

      return {
        ...node,
        position: {
          x: 100 + nodeLayer * layerSpacing,
          y: 100 + indexInLayer * nodeSpacing - layerHeight / 2 + window.innerHeight / 3,
        },
      };
    });
  }, []);

  const autoLayout = useCallback((nodesToLayout = null, edgesToLayout = null) => {
    // Use provided nodes/edges or fall back to state
    const currentNodes = nodesToLayout || nodes;
    const currentEdges = edgesToLayout || edges;

    if (currentNodes.length === 0) return;

    // Apply layout and update state
    const layoutedNodes = applyLayoutToNodes(currentNodes, currentEdges);
    setNodes(layoutedNodes);
  }, [nodes, edges, setNodes, applyLayoutToNodes]);

  const undo = useCallback(() => {
    if (historyIndex <= 0) return;

    isUndoRedoRef.current = true;
    const newIndex = historyIndex - 1;
    const snapshot = history[newIndex];

    setNodes(snapshot.nodes);
    setEdges(snapshot.edges);
    setHistoryIndex(newIndex);
    setSelectedNode(null);

    // Reset flag after a short delay to allow state to settle
    setTimeout(() => {
      isUndoRedoRef.current = false;
    }, 100);
  }, [history, historyIndex, setNodes, setEdges]);

  const redo = useCallback(() => {
    if (historyIndex >= history.length - 1) return;

    isUndoRedoRef.current = true;
    const newIndex = historyIndex + 1;
    const snapshot = history[newIndex];

    setNodes(snapshot.nodes);
    setEdges(snapshot.edges);
    setHistoryIndex(newIndex);
    setSelectedNode(null);

    // Reset flag after a short delay to allow state to settle
    setTimeout(() => {
      isUndoRedoRef.current = false;
    }, 100);
  }, [history, historyIndex, setNodes, setEdges]);

  // Copy/Paste functionality
  const copySelectedNode = useCallback(() => {
    if (!selectedNode) return;
    setCopiedNode(selectedNode);
  }, [selectedNode]);

  const pasteNode = useCallback(() => {
    if (!copiedNode) return;

    // Create a duplicate with offset position
    const newNode = {
      ...copiedNode,
      id: generateNodeId(),
      position: {
        x: copiedNode.position.x + 50,
        y: copiedNode.position.y + 50,
      },
      data: {
        ...copiedNode.data,
        customName: copiedNode.data.customName
          ? generateUniqueStageName(copiedNode.data.customName, nodes)
          : null,
      },
      selected: false,
    };

    setNodes((nds) => [...nds, newNode]);
    setSelectedNode(newNode);
  }, [copiedNode, nodes, setNodes]);

  // Global keyboard shortcuts for copy/paste/undo/redo/quick-add
  useEffect(() => {
    const handleKeyDown = (e) => {
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const cmdOrCtrl = isMac ? e.metaKey : e.ctrlKey;

      // Ignore if typing in an input/textarea (except for "/" which opens quick add)
      const isTyping = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';
      if (isTyping && e.key !== '/') {
        return;
      }

      // "/" - Quick Add (open at center)
      if (e.key === '/' && !cmdOrCtrl) {
        e.preventDefault();
        setQuickAddPosition({ x: 250, y: 250 }); // Will be centered
        setShowQuickAdd(true);
        return;
      }

      // Cmd/Ctrl+K - Quick Add (open at center)
      if (cmdOrCtrl && e.key === 'k') {
        e.preventDefault();
        setQuickAddPosition({ x: 250, y: 250 }); // Will be centered
        setShowQuickAdd(true);
        return;
      }

      // Cmd/Ctrl+C - Copy
      if (cmdOrCtrl && e.key === 'c' && !e.shiftKey) {
        e.preventDefault();
        copySelectedNode();
      }

      // Cmd/Ctrl+V - Paste
      if (cmdOrCtrl && e.key === 'v' && !e.shiftKey) {
        e.preventDefault();
        pasteNode();
      }

      // Cmd/Ctrl+D - Duplicate (same as copy+paste)
      if (cmdOrCtrl && e.key === 'd' && !e.shiftKey) {
        e.preventDefault();
        // Only duplicate if a node is actually selected
        if (selectedNode) {
          copySelectedNode();
          // Paste immediately after copying
          setTimeout(() => pasteNode(), 10);
        }
      }

      // Cmd/Ctrl+Z - Undo
      if (cmdOrCtrl && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      }

      // Cmd/Ctrl+Shift+Z - Redo
      if (cmdOrCtrl && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        redo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [copySelectedNode, pasteNode, undo, redo]);

  const handleLoadConfig = useCallback(async () => {
    try {
      setLoadingConfig(true);
      const configs = await listYAMLConfigs();
      setAvailableConfigs(configs);
      setShowConfigLoader(true);
    } catch (err) {
      setError('Failed to load config list: ' + (err?.message || String(err)));
    } finally {
      setLoadingConfig(false);
    }
  }, []);

  const handleSelectConfig = useCallback(async (configPath) => {
    try {
      setLoadingConfig(true);
      setShowConfigLoader(false);

      const result = await loadYAMLConfig(configPath);

      if (result.errors && result.errors.length > 0) {
        setError('Config load errors: ' + result.errors.join(', '));
        return;
      }

      // Clear current canvas
      setNodes([]);
      setEdges([]);
      setSelectedNode(null);

      // Add stages to canvas
      const newNodes = [];
      const stageMap = new Map();

      result.stages.forEach((stageReq, idx) => {
        const nodeId = generateNodeId();
        const displayName = stageReq.display_name || stageReq.custom_name || stageReq.name;

        // Find stage info from registry
        const stageInfo = stages.find(s => s.name === stageReq.name);

        newNodes.push({
          id: nodeId,
          type: 'stageNode',
          position: { x: 100 + (idx % 4) * 300, y: 100 + Math.floor(idx / 4) * 200 },
          data: {
            name: displayName,
            customName: stageReq.custom_name || null,
            originalName: stageReq.name,
            color: getStageColor(stageReq.name),
            mandatory_inputs: stageInfo?.mandatory_inputs || [],
            optional_inputs: stageInfo?.optional_inputs || [],
            input_types: stageInfo?.input_types || {},
            output_fields: stageInfo?.output_fields || [],
            output_model_name: stageInfo?.output_model_name || 'VoidOutput',
            description: stageReq.description || stageInfo?.description || '',
            notes: stageReq.notes || '',
            cacheable: stageInfo?.cacheable !== undefined ? stageInfo.cacheable : true,
          },
        });

        stageMap.set(displayName, nodeId);
      });

      // Build edges FIRST (before rendering anything)
      const newEdges = [];

      // Data flow edges
      result.data_flow_edges.forEach((edge) => {
        const sourceNodeId = stageMap.get(edge.source_stage);
        const targetNodeId = stageMap.get(edge.destination_stage);

        if (sourceNodeId && targetNodeId) {
          newEdges.push({
            id: generateEdgeId(),
            source: sourceNodeId,
            target: targetNodeId,
            sourceHandle: 'output',
            targetHandle: edge.input_name,
            type: 'dataFlow',
            data: { inputName: edge.input_name },
            markerEnd: { type: MarkerType.ArrowClosed },
          });
        }
      });

      // Execution dependencies
      result.execution_dependencies.forEach((dep) => {
        const sourceNodeId = stageMap.get(dep.target_stage);
        const targetNodeId = stageMap.get(dep.stage);

        // Map dependency type to handle ID
        const handleMap = {
          'on_success': 'exec-success',
          'on_failure': 'exec-failure',
          'always_after': 'exec-always'
        };
        const targetHandle = handleMap[dep.dependency_type] || 'exec-always';

        // Map dependency type to execution type for edge display
        const executionTypeMap = {
          'on_success': 'success',
          'on_failure': 'failure',
          'always_after': 'always'
        };
        const executionType = executionTypeMap[dep.dependency_type] || 'always';

        if (sourceNodeId && targetNodeId) {
          newEdges.push({
            id: generateEdgeId(),
            source: sourceNodeId,
            target: targetNodeId,
            sourceHandle: 'exec-source',
            targetHandle: targetHandle,
            type: 'execution',
            data: { executionType: executionType },
            markerEnd: { type: MarkerType.ArrowClosed },
          });
        }
      });

      // Apply layout to nodes BEFORE setting them (no blink!)
      const layoutedNodes = applyLayoutToNodes(newNodes, newEdges);

      // Set both nodes and edges at once
      setNodes(layoutedNodes);
      setEdges(newEdges);

      setError(null);

      if (result.warnings && result.warnings.length > 0) {
        setError('Warnings: ' + result.warnings.join(', '));
      }
    } catch (err) {
      setError('Failed to load config: ' + (err?.message || String(err)));
    } finally {
      setLoadingConfig(false);
    }
  }, [stages, setNodes, setEdges, applyLayoutToNodes]);

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
          🏗️ GigaEvo DAG Builder
        </h1>
        <Toolbar
          onExportYAML={exportYAMLConfig}
          onExportPDF={exportPDF}
          onClear={clearCanvas}
          onLoadConfig={handleLoadConfig}
          onUndo={undo}
          onRedo={redo}
          canUndo={historyIndex > 0}
          canRedo={historyIndex < history.length - 1}
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

      {/* Validation Errors Box */}
      {validationErrors.length > 0 && (
        <div
          style={{
            background: '#fef2f2',
            borderBottom: '1px solid #fecaca',
            padding: '12px 20px',
          }}
        >
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '8px',
            color: '#991b1b',
            fontWeight: 600,
            fontSize: '14px'
          }}>
            <span style={{ fontSize: '18px' }}>⚠️</span>
            <span>Validation Errors:</span>
          </div>
          <ul style={{
            margin: 0,
            paddingLeft: '32px',
            color: '#7f1d1d',
            fontSize: '13px'
          }}>
            {validationErrors.map((err, idx) => (
              <li key={idx} style={{ marginBottom: '4px' }}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Validation Success */}
      {validationStatus?.is_valid && nodes.length > 0 && (
        <div
          style={{
            background: '#f0fdf4',
            borderBottom: '1px solid #bbf7d0',
            padding: '8px 20px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            color: '#166534',
            fontWeight: 600,
            fontSize: '14px'
          }}
        >
          <span style={{ fontSize: '18px' }}>✅</span>
          <span>DAG is valid!</span>
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
            maxWidth: 280,
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

          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={handleEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            onNodesDelete={onNodesDelete}
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

            {/* Auto Layout Button Panel */}
            <Panel position="top-left" style={{
              margin: '8px',
              background: 'white',
              borderRadius: '6px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: '1px solid #e5e7eb'
            }}>
              <button
                onClick={() => autoLayout()}
                disabled={nodes.length === 0}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '6px 10px',
                  background: nodes.length === 0 ? '#f8f9fa' : 'linear-gradient(135deg, #20c997 0%, #17a2b8 100%)',
                  color: nodes.length === 0 ? '#6c757d' : 'white',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: nodes.length === 0 ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease',
                  boxShadow: nodes.length === 0 ? 'none' : '0 1px 4px rgba(32, 201, 151, 0.3)',
                }}
                onMouseEnter={(e) => {
                  if (nodes.length > 0) {
                    e.target.style.transform = 'translateY(-1px)';
                    e.target.style.boxShadow = '0 2px 8px rgba(32, 201, 151, 0.4)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (nodes.length > 0) {
                    e.target.style.transform = 'translateY(0)';
                    e.target.style.boxShadow = '0 1px 4px rgba(32, 201, 151, 0.3)';
                  }
                }}
                title="Arrange nodes using topological sort (left to right flow)"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/>
                  <path d="M13 6h3a2 2 0 0 1 2 2v7"/><line x1="6" y1="9" x2="6" y2="15"/>
                  <circle cx="6" cy="18" r="3"/><circle cx="18" cy="6" r="3"/>
                  <path d="M18 9v7a2 2 0 0 1-2 2h-3"/>
                </svg>
                <span>Auto Layout</span>
              </button>
            </Panel>
          </ReactFlow>
        </div>

        {/* Node Details Panel */}
        {selectedNode && (
          <div
            style={{
              flex: '0 0 280px',
              maxWidth: 280,
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

      {/* Config Loader Modal */}
      {showConfigLoader && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000,
          }}
          onClick={() => setShowConfigLoader(false)}
        >
          <div
            style={{
              background: 'white',
              borderRadius: '12px',
              padding: '24px',
              maxWidth: '600px',
              width: '90%',
              maxHeight: '70vh',
              overflowY: 'auto',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 style={{ margin: '0 0 16px 0', fontSize: '20px', fontWeight: '600' }}>
              Load Pipeline Config
            </h2>
            <p style={{ margin: '0 0 20px 0', color: '#666', fontSize: '14px' }}>
              Select a Hydra YAML pipeline configuration to load into the builder
            </p>

            {loadingConfig ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                Loading...
              </div>
            ) : availableConfigs.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                No configuration files found in config/pipeline/
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {availableConfigs.map((config) => (
                  <button
                    key={config.path}
                    onClick={() => handleSelectConfig(config.path)}
                    style={{
                      padding: '16px',
                      background: '#f9fafb',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'all 0.2s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.background = '#f3f4f6';
                      e.target.style.borderColor = '#d1d5db';
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.background = '#f9fafb';
                      e.target.style.borderColor = '#e5e7eb';
                    }}
                  >
                    <div style={{ fontWeight: '600', marginBottom: '4px' }}>
                      {config.display_name}
                    </div>
                    <div style={{ fontSize: '12px', color: '#6b7280' }}>
                      {config.path}
                    </div>
                  </button>
                ))}
              </div>
            )}

            <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowConfigLoader(false)}
                style={{
                  padding: '8px 16px',
                  background: '#6b7280',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '500',
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = '#4b5563';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = '#6b7280';
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Quick Add Modal */}
      {showQuickAdd && (
        <QuickAdd
          stages={stages}
          onAddStage={addStageToCanvas}
          onClose={() => setShowQuickAdd(false)}
          position={quickAddPosition}
        />
      )}

      {/* Toast Notifications */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
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
