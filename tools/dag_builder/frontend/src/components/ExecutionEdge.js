import React from 'react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath, MarkerType } from 'reactflow';

const ExecutionEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
}) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const executionColors = {
    always: '#6c757d',
    success: '#28a745',
    failure: '#dc3545'
  };

  const executionType = data?.executionType || 'always';
  const edgeColor = executionColors[executionType] || '#6c757d';

  // Calculate edge center for label positioning
  const centerX = (sourceX + targetX) / 2;
  const centerY = (sourceY + targetY) / 2;

  return (
    <>
      {/* Define arrow marker with unique ID to avoid conflicts */}
      <defs>
        <marker
          id={`exec-arrowhead-${id}`}
          markerWidth="12"
          markerHeight="8"
          refX="11"
          refY="4"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <polygon
            points="0 0, 12 4, 0 8"
            fill={edgeColor}
            stroke="none"
          />
        </marker>
      </defs>

      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={`url(#exec-arrowhead-${id})`}
        style={{
          stroke: edgeColor,
          strokeWidth: 3,
          strokeDasharray: '5,5',
          ...style,
        }}
      />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${centerX}px,${centerY}px)`,
            fontSize: 12,
            fontWeight: 'bold',
            color: edgeColor,
            background: 'white',
            padding: '4px 8px',
            borderRadius: '6px',
            border: `2px solid ${edgeColor}`,
            pointerEvents: 'all',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
            minWidth: '20px',
            textAlign: 'center',
            zIndex: 10,
            willChange: 'transform' // Optimize rendering
          }}
          className="nodrag nopan"
        >
          {executionType.charAt(0).toUpperCase()}
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

export default ExecutionEdge;
