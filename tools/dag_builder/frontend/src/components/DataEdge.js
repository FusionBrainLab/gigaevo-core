import React from 'react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath, MarkerType } from 'reactflow';

const DataEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
}) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const edgeColor = '#b1b1b7';

  // Calculate edge center for label positioning
  const centerX = (sourceX + targetX) / 2;
  const centerY = (sourceY + targetY) / 2;

  return (
    <>
      {/* Define arrow marker for data edges */}
      <defs>
        <marker
          id={`data-arrowhead-${id}`}
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
          markerUnits="userSpaceOnUse"
        >
          <polygon
            points="0 0, 10 3.5, 0 7"
            fill={edgeColor}
            stroke="none"
          />
        </marker>
      </defs>

      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={`url(#data-arrowhead-${id})`}
        style={{
          stroke: edgeColor,
          strokeWidth: 2,
          ...style,
        }}
      />

      {/* Optional: Show input name label */}
      {data?.inputName && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${centerX}px,${centerY}px)`,
              fontSize: 12,
              fontWeight: '600',
              color: '#111827',
              background: 'rgba(255, 255, 255, 0.95)',
              padding: '3px 8px',
              borderRadius: '4px',
              border: '1px solid #E5E7EB',
              pointerEvents: 'none',
              boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
              minWidth: '16px',
              textAlign: 'center',
              zIndex: 10,
              willChange: 'transform', // Optimize rendering
            }}
            className="nodrag nopan"
          >
            {data.inputName}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default DataEdge;
