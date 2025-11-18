import React, { useState, useEffect, useRef } from 'react';
import { Trash2, FileText, Upload, HelpCircle, X, Undo, Redo, FileImage } from 'lucide-react';

const Toolbar = ({ onExportYAML, onExportPDF, onClear, onLoadConfig, onUndo, onRedo, canUndo, canRedo, nodeCount, edgeCount }) => {
  const [showLegend, setShowLegend] = useState(false);
  const legendRef = useRef(null);

  // Close legend when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (legendRef.current && !legendRef.current.contains(event.target)) {
        setShowLegend(false);
      }
    };

    if (showLegend) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showLegend]);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      position: 'relative'
    }}>
      {/* Actions */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>

        <button
          onClick={onLoadConfig}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            background: '#17a2b8',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.background = '#138496';
          }}
          onMouseLeave={(e) => {
            e.target.style.background = '#17a2b8';
          }}
        >
          <Upload size={16} />
          Load Config
        </button>

        {/* Undo/Redo buttons */}
        <button
          onClick={onUndo}
          disabled={!canUndo}
          title="Undo (Ctrl/Cmd+Z)"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 12px',
            background: canUndo ? '#6c757d' : '#f8f9fa',
            color: canUndo ? 'white' : '#adb5bd',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: canUndo ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (canUndo) {
              e.target.style.background = '#5a6268';
            }
          }}
          onMouseLeave={(e) => {
            if (canUndo) {
              e.target.style.background = '#6c757d';
            }
          }}
        >
          <Undo size={16} />
        </button>

        <button
          onClick={onRedo}
          disabled={!canRedo}
          title="Redo (Ctrl/Cmd+Shift+Z)"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 12px',
            background: canRedo ? '#6c757d' : '#f8f9fa',
            color: canRedo ? 'white' : '#adb5bd',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: canRedo ? 'pointer' : 'not-allowed',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (canRedo) {
              e.target.style.background = '#5a6268';
            }
          }}
          onMouseLeave={(e) => {
            if (canRedo) {
              e.target.style.background = '#6c757d';
            }
          }}
        >
          <Redo size={16} />
        </button>

        <button
          onClick={onExportYAML}
          disabled={nodeCount === 0}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            background: nodeCount === 0 ? '#f8f9fa' : '#6f42c1',
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
              e.target.style.background = '#5a32a3';
            }
          }}
          onMouseLeave={(e) => {
            if (nodeCount > 0) {
              e.target.style.background = '#6f42c1';
            }
          }}
        >
          <FileText size={16} />
          Export YAML
        </button>

        <button
          onClick={onExportPDF}
          disabled={nodeCount === 0}
          title="Export canvas as PDF"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            background: nodeCount === 0 ? '#f8f9fa' : '#ff6b6b',
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
              e.target.style.background = '#ee5a52';
            }
          }}
          onMouseLeave={(e) => {
            if (nodeCount > 0) {
              e.target.style.background = '#ff6b6b';
            }
          }}
        >
          <FileImage size={16} />
          Export PDF
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

        {/* Legend Button */}
        <button
          onClick={() => setShowLegend(!showLegend)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '8px 16px',
            background: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '500',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.background = '#5a6268';
          }}
          onMouseLeave={(e) => {
            e.target.style.background = '#6c757d';
          }}
          title="Show legend"
        >
          <HelpCircle size={16} />
          Legend
        </button>
      </div>

      {/* Legend Popup */}
      {showLegend && (
        <div
          ref={legendRef}
          style={{
            position: 'absolute',
            top: '100%',
            right: 0,
            marginTop: '8px',
            background: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
            padding: '16px',
            width: '320px',
            zIndex: 1000
          }}
        >
          {/* Header */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px',
            paddingBottom: '8px',
            borderBottom: '1px solid #e5e7eb'
          }}>
            <h3 style={{
              margin: 0,
              fontSize: '16px',
              fontWeight: '600',
              color: '#111827'
            }}>
              Icon Legend
            </h3>
            <button
              onClick={() => setShowLegend(false)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '4px',
                display: 'flex',
                alignItems: 'center',
                color: '#6c757d'
              }}
              aria-label="Close legend"
            >
              <X size={18} />
            </button>
          </div>

          {/* Legend Items */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {/* Cacheable */}
            <div style={{ display: 'flex', alignItems: 'start', gap: '10px' }}>
              <span style={{ fontSize: '18px', flexShrink: 0 }}>💾</span>
              <div>
                <div style={{ fontWeight: '600', fontSize: '13px', color: '#111827' }}>
                  Cacheable Stage
                </div>
                <div style={{ fontSize: '12px', color: '#6c757d', lineHeight: '1.4' }}>
                  Results can be cached for reuse
                </div>
              </div>
            </div>

            {/* Non-cacheable */}
            <div style={{ display: 'flex', alignItems: 'start', gap: '10px' }}>
              <span style={{ fontSize: '18px', flexShrink: 0 }}>🔄</span>
              <div>
                <div style={{ fontWeight: '600', fontSize: '13px', color: '#111827' }}>
                  Non-Cacheable Stage
                </div>
                <div style={{ fontSize: '12px', color: '#6c757d', lineHeight: '1.4' }}>
                  Always recomputes on execution
                </div>
              </div>
            </div>

            {/* Type Colors */}
            <div style={{ display: 'flex', alignItems: 'start', gap: '10px' }}>
              <div style={{
                width: '18px',
                height: '18px',
                borderRadius: '50%',
                background: 'linear-gradient(135deg, #3b82f6 0%, #ec4899 50%, #8b5cf6 100%)',
                flexShrink: 0
              }} />
              <div>
                <div style={{ fontWeight: '600', fontSize: '13px', color: '#111827' }}>
                  Type Colors
                </div>
                <div style={{ fontSize: '12px', color: '#6c757d', lineHeight: '1.4' }}>
                  Matching colors indicate compatible types for connections
                </div>
              </div>
            </div>

            {/* Execution Edges */}
            <div style={{
              marginTop: '8px',
              paddingTop: '12px',
              borderTop: '1px solid #e5e7eb'
            }}>
              <div style={{ fontWeight: '600', fontSize: '13px', color: '#111827', marginBottom: '8px' }}>
                Execution Dependencies
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#28a745', border: '2px solid white', boxShadow: '0 0 0 1px #28a745' }} />
                  <span style={{ color: '#6c757d' }}>On Success</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#dc3545', border: '2px solid white', boxShadow: '0 0 0 1px #dc3545' }} />
                  <span style={{ color: '#6c757d' }}>On Failure</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#6c757d', border: '2px solid white', boxShadow: '0 0 0 1px #6c757d' }} />
                  <span style={{ color: '#6c757d' }}>Always After</span>
                </div>
              </div>
            </div>

            {/* Keyboard Shortcuts */}
            <div style={{
              marginTop: '8px',
              paddingTop: '12px',
              borderTop: '1px solid #e5e7eb'
            }}>
              <div style={{ fontWeight: '600', fontSize: '13px', color: '#111827', marginBottom: '8px' }}>
                Keyboard Shortcuts
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#6c757d' }}>Quick Add</span>
                  <span style={{ fontFamily: 'monospace', color: '#111827', fontWeight: '500' }}>/ or Ctrl+K</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#6c757d' }}>Copy Node</span>
                  <span style={{ fontFamily: 'monospace', color: '#111827', fontWeight: '500' }}>Ctrl+C</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#6c757d' }}>Paste Node</span>
                  <span style={{ fontFamily: 'monospace', color: '#111827', fontWeight: '500' }}>Ctrl+V</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#6c757d' }}>Undo</span>
                  <span style={{ fontFamily: 'monospace', color: '#111827', fontWeight: '500' }}>Ctrl+Z</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#6c757d' }}>Redo</span>
                  <span style={{ fontFamily: 'monospace', color: '#111827', fontWeight: '500' }}>Ctrl+Shift+Z</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#6c757d' }}>Delete Node</span>
                  <span style={{ fontFamily: 'monospace', color: '#111827', fontWeight: '500' }}>Del</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#6c757d' }}>Duplicate Node</span>
                  <span style={{ fontFamily: 'monospace', color: '#111827', fontWeight: '500' }}>Ctrl+D</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Toolbar;
