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
              fontSize: 10,
              fontWeight: '500',
              color: '#666',
              background: 'white',
              padding: '2px 6px',
              borderRadius: '4px',
              border: `1px solid ${edgeColor}`,
              pointerEvents: 'none',
              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
              minWidth: '16px',
              textAlign: 'center',
              opacity: 0.8
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
