import React from 'react';
import { Download, Trash2, Save, Eye } from 'lucide-react';

const Toolbar = ({ onExport, onClear, nodeCount, edgeCount }) => {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '12px'
    }}>
      {/* Stats */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        fontSize: '14px',
        color: '#666',
        marginRight: '16px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Eye size={16} />
          {nodeCount} nodes
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: '#666'
          }} />
          {edgeCount} edges
        </div>
      </div>

      {/* Actions */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>

        <button
          onClick={onExport}
          disabled={nodeCount === 0}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            background: nodeCount === 0 ? '#f8f9fa' : '#007bff',
            color: nodeCount === 0 ? '#6c757d' : 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: nodeCount === 0 ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (nodeCount > 0) {
              e.target.style.background = '#0056b3';
            }
          }}
          onMouseLeave={(e) => {
            if (nodeCount > 0) {
              e.target.style.background = '#007bff';
            }
          }}
        >
          <Download size={16} />
          Export Pipeline
        </button>

        <button
          onClick={onClear}
          disabled={nodeCount === 0}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            background: nodeCount === 0 ? '#f8f9fa' : '#dc3545',
            color: nodeCount === 0 ? '#6c757d' : 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: nodeCount === 0 ? 'not-allowed' : 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (nodeCount > 0) {
              e.target.style.background = '#c82333';
            }
          }}
          onMouseLeave={(e) => {
            if (nodeCount > 0) {
              e.target.style.background = '#dc3545';
            }
          }}
        >
          <Trash2 size={16} />
          Clear Canvas
        </button>
      </div>
    </div>
  );
};

export default Toolbar;
