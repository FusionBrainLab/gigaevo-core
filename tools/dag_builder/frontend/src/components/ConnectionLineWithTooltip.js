import React, { useState, useEffect, useMemo, useRef } from 'react';
import { getBezierPath, EdgeLabelRenderer } from 'reactflow';

const DEFAULT_NODE_PROXIMITY = 56; // px radius to "snap" to a node center
const DEFAULT_HANDLE_RADIUS = 22;  // px within which a handle is considered "closest"
const TOOLTIP_SHOW_DELAY_MS = 80;  // small delay to avoid flicker

function getPaneRect() {
  const pane = document.querySelector('.react-flow__pane');
  return pane ? pane.getBoundingClientRect() : null;
}

function dist(x1, y1, x2, y2) {
  const dx = x1 - x2;
  const dy = y1 - y2;
  return Math.hypot(dx, dy);
}

function isTargetHandleEl(el) {
  if (!el) return false;
  // Prefer explicit attributes if available (varies by RF version)
  // data-handlepos is usually one of: left/right/top/bottom
  const pos = el.getAttribute('data-handlepos');
  const type = el.getAttribute('data-handle-type'); // often "source" | "target" (if present)
  // Heuristics: inputs are usually left-positioned OR explicitly type="target"
  return type === 'target' || pos === 'left' || el.className.includes('--target');
}

const ConnectionLineWithTooltip = ({
  fromX,
  fromY,
  fromPosition,
  toX,
  toY,
  toPosition,
  connectionLineStyle,
  connectionLineType, // not used, but kept for parity with RF signature
  connectionStatus,
  nodes,
  nodeProximity = DEFAULT_NODE_PROXIMITY,
  handleRadius = DEFAULT_HANDLE_RADIUS,
}) => {
  const [targetHandleInfo, setTargetHandleInfo] = useState(null);
  const showTimerRef = useRef(null);
  const rafRef = useRef(null);
  const paneRectRef = useRef(null);

  const [edgePath, labelX, labelY] = useMemo(
    () =>
      getBezierPath({
        sourceX: fromX,
        sourceY: fromY,
        sourcePosition: fromPosition,
        targetX: toX,
        targetY: toY,
        targetPosition: toPosition,
      }),
    [fromX, fromY, fromPosition, toX, toY, toPosition]
  );

  // Cache pane rect; update on resize/zoom/pan (RF updates classed container sizes)
  useEffect(() => {
    const updatePaneRect = () => {
      paneRectRef.current = getPaneRect();
    };
    updatePaneRect();

    const ro = new ResizeObserver(updatePaneRect);
    const pane = document.querySelector('.react-flow__pane');
    if (pane) ro.observe(pane);

    // Listen to wheel (zoom) and mousemove (pans can change rect due to transforms + scrollbars)
    window.addEventListener('wheel', updatePaneRect, { passive: true });
    window.addEventListener('mousemove', updatePaneRect);

    return () => {
      ro.disconnect();
      window.removeEventListener('wheel', updatePaneRect);
      window.removeEventListener('mousemove', updatePaneRect);
    };
  }, []);

  // Clear timers/raf on unmount
  useEffect(() => {
    return () => {
      if (showTimerRef.current) clearTimeout(showTimerRef.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  useEffect(() => {
    // Guard: allow 0 coords; only null/undefined should early-out
    if (
      connectionStatus !== 'valid' ||
      toX == null ||
      toY == null ||
      !Array.isArray(nodes) ||
      nodes.length === 0
    ) {
      if (showTimerRef.current) clearTimeout(showTimerRef.current);
      setTargetHandleInfo(null);
      return;
    }

    // rAF throttle to avoid heavy DOM reads every mousemove
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const paneRect = paneRectRef.current || getPaneRect();
      if (!paneRect) {
        setTargetHandleInfo(null);
        return;
      }

      // 1) Find the nearest node by center within a proximity radius
      let nearestNode = null;
      let minNodeDist = Infinity;

      for (const node of nodes) {
        const nodeEl = document.querySelector(`[data-id="${CSS.escape(node.id)}"]`);
        if (!nodeEl) continue;
        const rect = nodeEl.getBoundingClientRect();

        const nodeCenterX = rect.left - paneRect.left + rect.width / 2;
        const nodeCenterY = rect.top - paneRect.top + rect.height / 2;

        const d = dist(toX, toY, nodeCenterX, nodeCenterY);
        if (d < minNodeDist) {
          minNodeDist = d;
          nearestNode = { node, rect };
        }
      }

      if (!nearestNode || minNodeDist > nodeProximity) {
        if (showTimerRef.current) clearTimeout(showTimerRef.current);
        setTargetHandleInfo(null);
        return;
      }

      // 2) Among that node’s handles, find the closest *target* handle
      const containerSel = `[data-id="${CSS.escape(nearestNode.node.id)}"]`;
      const handleEls = document.querySelectorAll(
        `${containerSel} .react-flow__handle`
      );
      if (!handleEls.length) {
        if (showTimerRef.current) clearTimeout(showTimerRef.current);
        setTargetHandleInfo(null);
        return;
      }

      let closestHandleEl = null;
      let minHandleDist = Infinity;

      handleEls.forEach((handle) => {
        const hRect = handle.getBoundingClientRect();
        const hx = hRect.left - paneRect.left + hRect.width / 2;
        const hy = hRect.top - paneRect.top + hRect.height / 2;
        const d = dist(toX, toY, hx, hy);

        if (d < minHandleDist) {
          minHandleDist = d;
          closestHandleEl = handle;
        }
      });

      if (!closestHandleEl || minHandleDist > handleRadius || !isTargetHandleEl(closestHandleEl)) {
        if (showTimerRef.current) clearTimeout(showTimerRef.current);
        setTargetHandleInfo(null);
        return;
      }

      const handleId =
        closestHandleEl.getAttribute('data-handleid') ||
        closestHandleEl.getAttribute('data-handle-id') ||
        '';

      const nodeName = nearestNode.node?.data?.name || nearestNode.node?.id || 'Node';

      // tiny delay to avoid tooltip flicker when skimming across handles
      if (showTimerRef.current) clearTimeout(showTimerRef.current);
      showTimerRef.current = setTimeout(() => {
        setTargetHandleInfo({
          inputName: handleId || 'input',
          nodeName,
        });
      }, TOOLTIP_SHOW_DELAY_MS);
    });
  }, [toX, toY, connectionStatus, nodes, nodeProximity, handleRadius]);

  return (
    <g>
      {/* Arrow marker for connection line */}
      <defs>
        <marker
          id="connection-arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill="#b1b1b7" stroke="none" />
        </marker>
      </defs>

      <path
        style={{ ...connectionLineStyle, markerEnd: 'url(#connection-arrowhead)' }}
        className="react-flow__connection-path"
        d={edgePath}
      />

      {/* Tooltip */}
      {targetHandleInfo && toX != null && toY != null && (
        <EdgeLabelRenderer>
          <div
            role="status"
            aria-live="polite"
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${toX}px, ${toY - 20}px)`,
              fontSize: 11,
              fontWeight: 500,
              color: '#333',
              background: 'white',
              padding: '4px 8px',
              borderRadius: 6,
              border: '1px solid #e1e5e9',
              pointerEvents: 'none',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              zIndex: 1000,
              whiteSpace: 'nowrap',
              userSelect: 'none',
            }}
          >
            <div style={{ fontSize: 10, color: '#666', marginBottom: 2 }}>
              {targetHandleInfo.nodeName}
            </div>
            <div style={{ fontWeight: 600 }}>{targetHandleInfo.inputName}</div>
          </div>
        </EdgeLabelRenderer>
      )}
    </g>
  );
};

export default React.memo(ConnectionLineWithTooltip);
