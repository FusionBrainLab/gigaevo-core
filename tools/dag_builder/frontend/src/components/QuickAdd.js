import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Search, X } from 'lucide-react';

const QuickAdd = ({ stages, onAddStage, onClose, position }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  // Filter stages based on search term
  const filteredStages = useMemo(() => {
    if (!searchTerm.trim()) return stages;

    const term = searchTerm.toLowerCase();
    return stages.filter(stage => {
      const nameMatch = stage.name.toLowerCase().includes(term);
      const descMatch = stage.description?.toLowerCase().includes(term);
      const inputMatch = [...stage.mandatory_inputs, ...stage.optional_inputs]
        .some(input => input.toLowerCase().includes(term));
      const outputMatch = stage.output_model_name?.toLowerCase().includes(term);

      return nameMatch || descMatch || inputMatch || outputMatch;
    });
  }, [stages, searchTerm]);

  // Reset selected index when filtered results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredStages.length]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current) {
      const selectedElement = listRef.current.children[selectedIndex];
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }, [selectedIndex]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, filteredStages.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredStages[selectedIndex]) {
          onAddStage(filteredStages[selectedIndex], position);
          onClose();
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [filteredStages, selectedIndex, onAddStage, onClose, position]);

  const formatStageName = (name) => {
    return name
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .trim();
  };

  return (
    <>
      {/* Backdrop */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          zIndex: 9998,
          backdropFilter: 'blur(2px)',
        }}
        onClick={onClose}
      />

      {/* Dialog */}
      <div
        style={{
          position: 'fixed',
          top: '20%',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '600px',
          maxWidth: '90vw',
          maxHeight: '60vh',
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header with search */}
        <div
          style={{
            padding: '16px',
            borderBottom: '1px solid #e5e7eb',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}
        >
          <Search size={20} color="#6b7280" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search stages... (type to filter)"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              flex: 1,
              border: 'none',
              outline: 'none',
              fontSize: '16px',
              color: '#111827',
              background: 'transparent',
            }}
          />
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px',
              display: 'flex',
              alignItems: 'center',
              color: '#6b7280',
            }}
            title="Close (Esc)"
          >
            <X size={20} />
          </button>
        </div>

        {/* Results list */}
        <div
          ref={listRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '8px',
          }}
        >
          {filteredStages.length === 0 ? (
            <div
              style={{
                padding: '40px 20px',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '14px',
              }}
            >
              No stages found matching "{searchTerm}"
            </div>
          ) : (
            filteredStages.map((stage, index) => (
              <div
                key={stage.name}
                onClick={() => {
                  onAddStage(stage, position);
                  onClose();
                }}
                style={{
                  padding: '12px',
                  margin: '4px 0',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  background: index === selectedIndex ? '#eff6ff' : 'transparent',
                  border: `1px solid ${index === selectedIndex ? '#93c5fd' : 'transparent'}`,
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={() => setSelectedIndex(index)}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '4px',
                  }}
                >
                  <div style={{ fontWeight: '600', fontSize: '14px', color: '#111827' }}>
                    {formatStageName(stage.name)}
                    {stage.cacheable && (
                      <span style={{ marginLeft: '6px', fontSize: '11px' }} title="Cacheable">
                        💾
                      </span>
                    )}
                  </div>
                  {index === selectedIndex && (
                    <div
                      style={{
                        fontSize: '11px',
                        color: '#3b82f6',
                        fontWeight: '500',
                      }}
                    >
                      ↵ Enter
                    </div>
                  )}
                </div>
                <div style={{ fontSize: '12px', color: '#6b7280', lineHeight: '1.4' }}>
                  {stage.description}
                </div>
                <div
                  style={{
                    marginTop: '6px',
                    display: 'flex',
                    gap: '8px',
                    fontSize: '11px',
                    color: '#9ca3af',
                  }}
                >
                  {stage.mandatory_inputs.length > 0 && (
                    <span>
                      In: {stage.mandatory_inputs.slice(0, 3).join(', ')}
                      {stage.mandatory_inputs.length > 3 && '...'}
                    </span>
                  )}
                  {stage.output_model_name && (
                    <span>Out: {stage.output_model_name}</span>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer with hints */}
        <div
          style={{
            padding: '10px 16px',
            borderTop: '1px solid #e5e7eb',
            background: '#f9fafb',
            display: 'flex',
            gap: '16px',
            fontSize: '11px',
            color: '#6b7280',
          }}
        >
          <span>↑↓ Navigate</span>
          <span>↵ Select</span>
          <span>Esc Close</span>
          <div style={{ flex: 1 }} />
          <span>{filteredStages.length} of {stages.length} stages</span>
        </div>
      </div>
    </>
  );
};

export default QuickAdd;
