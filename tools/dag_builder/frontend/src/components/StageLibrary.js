import React, { useRef, useState, useMemo } from 'react';
import { Package } from 'lucide-react';
import { getStageColor, getStageIcon, formatStageName } from '../utils/stageUtils';

// Individual stage card component to handle hover states properly
const StageCard = ({ stage, onAddStage }) => {
  const cardRef = useRef(null);
  const buttonRef = useRef(null);
  const lastClickTime = useRef(0);

  const handleCardMouseEnter = () => {
    if (cardRef.current) {
      // Only glow - no layout shift
      cardRef.current.style.boxShadow = '0 0 0 2px rgba(59, 130, 246, 0.15)';
      cardRef.current.style.borderColor = '#93C5FD';
    }
  };

  const handleCardMouseLeave = () => {
    if (cardRef.current) {
      // Return to invisible shadow (prevents layout shift)
      cardRef.current.style.boxShadow = '0 0 0 0 transparent';
      cardRef.current.style.borderColor = '#e1e5e9';
    }
  };

  const handleButtonMouseEnter = (e) => {
    e.stopPropagation();
    if (buttonRef.current) {
      // Subtle hover without scale (no shaking)
      buttonRef.current.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
      buttonRef.current.style.opacity = '0.9';
    }
  };

  const handleButtonMouseLeave = (e) => {
    e.stopPropagation();
    if (buttonRef.current) {
      buttonRef.current.style.boxShadow = '0 1px 4px rgba(0,0,0,0.2)';
      buttonRef.current.style.opacity = '1';
    }
  };

  return (
    <div
      ref={cardRef}
      style={{
        background: 'white',
        border: '1px solid #e1e5e9',
        borderRadius: '6px',
        padding: '8px',
        paddingRight: '36px', // Make room for the add button
        margin: '4px 0',
        cursor: 'grab',
        transition: 'box-shadow 0.15s ease, border-color 0.15s ease',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: '0 0 0 0 transparent', // Initial transparent shadow prevents layout shift
        userSelect: 'none',
        WebkitUserSelect: 'none',
        MozUserSelect: 'none',
        msUserSelect: 'none'
      }}
      onMouseEnter={handleCardMouseEnter}
      onMouseLeave={handleCardMouseLeave}
      onClick={(e) => {
        // Only trigger if the click wasn't on the add button
        if (e.target.tagName !== 'BUTTON') {
          console.log('Stage container clicked for:', stage.name);
        }
      }}
    >

      {/* Add button */}
      <button
        ref={buttonRef}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();

          // Debounce rapid clicks (prevent multiple stage creation)
          const now = Date.now();
          if (now - lastClickTime.current < 500) {
            console.log('Click ignored - too rapid');
            return;
          }
          lastClickTime.current = now;

          console.log('Add button clicked for:', stage.name);
          if (onAddStage) {
            onAddStage(stage);
          }
        }}
        onMouseDown={(e) => {
          e.stopPropagation();
        }}
        onMouseEnter={handleButtonMouseEnter}
        onMouseLeave={handleButtonMouseLeave}
        style={{
          position: 'absolute',
          top: '6px',
          right: '6px',
          background: '#64748B',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          width: '24px',
          height: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          fontSize: '16px',
          fontWeight: '600',
          lineHeight: '1',
          transition: 'opacity 0.15s ease, box-shadow 0.15s ease',
          boxShadow: '0 1px 4px rgba(0,0,0,0.2)',
          zIndex: 10,
          pointerEvents: 'auto',
          padding: '0',
          margin: '0'
        }}
        title="Click to add to canvas"
      >
        +
      </button>

      {/* Stage info */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '6px'
      }}>
        <span style={{ fontSize: '14px' }}>
          {getStageIcon()}
        </span>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          flex: 1
        }}>
          <div style={{
            fontWeight: '600',
            fontSize: '13px',
            color: '#333',
            lineHeight: '1.2'
          }}>
            {formatStageName(stage.name)}
          </div>
          {stage.cacheable !== undefined && (
            <span
              style={{
                fontSize: 10,
                color: '#6b7280',
              }}
              title={stage.cacheable ? "Cacheable - results can be cached" : "Non-cacheable - always recomputes"}
            >
              {stage.cacheable ? '💾' : '🔄'}
            </span>
          )}
        </div>
      </div>

      <div style={{
        fontSize: '11px',
        color: '#666',
        lineHeight: '1.3',
        marginBottom: '6px'
      }}>
        {stage.description}
      </div>

      {/* Inputs/Outputs */}
      <div style={{ fontSize: '10px' }}>
        {/* Inputs */}
        <div style={{
          display: 'flex',
          gap: '12px',
          marginBottom: '8px'
        }}>
          <div style={{
            flex: 1,
            background: '#f8f9fa',
            padding: '4px 6px',
            borderRadius: '4px',
            border: '1px solid #e9ecef'
          }}>
            <div style={{
              fontWeight: '600',
              color: '#495057',
              marginBottom: '2px'
            }}>
              Mandatory Inputs
            </div>
            <div style={{ color: '#6c757d' }}>
              {stage.mandatory_inputs.length > 0
                ? stage.mandatory_inputs.join(', ')
                : 'None'
              }
            </div>
          </div>

          <div style={{
            flex: 1,
            background: '#f8f9fa',
            padding: '4px 6px',
            borderRadius: '4px',
            border: '1px solid #e9ecef'
          }}>
            <div style={{
              fontWeight: '600',
              color: '#495057',
              marginBottom: '2px'
            }}>
              Optional Inputs
            </div>
            <div style={{ color: '#6c757d' }}>
              {stage.optional_inputs.length > 0
                ? stage.optional_inputs.join(', ')
                : 'None'
              }
            </div>
          </div>
        </div>

        {/* Output */}
        {stage.output_fields && stage.output_fields.length > 0 && stage.output_model_name !== 'VoidOutput' && (
          <div style={{
            background: '#fffbeb',
            padding: '4px 6px',
            borderRadius: '4px',
            border: '1px solid #fde68a'
          }}>
            <div style={{
              fontWeight: '600',
              color: '#92400e',
              marginBottom: '2px'
            }}>
              Output: {stage.output_model_name}
            </div>
            <div style={{ color: '#b45309' }}>
              {stage.output_fields.join(', ')}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const StageLibrary = ({ stages, onAddStage }) => {
  const [searchTerm, setSearchTerm] = useState('');

  // Filter stages based on search term
  const filteredStages = useMemo(() => {
    if (!searchTerm.trim()) {
      return stages;
    }

    const lowerSearch = searchTerm.toLowerCase();
    return stages.filter((stage) => {
      // Search in name, description, inputs, and outputs
      const nameMatch = stage.name.toLowerCase().includes(lowerSearch);
      const descMatch = stage.description?.toLowerCase().includes(lowerSearch) || false;
      const inputMatch = [
        ...stage.mandatory_inputs,
        ...stage.optional_inputs
      ].some(input => input.toLowerCase().includes(lowerSearch));
      const outputMatch = stage.output_model_name?.toLowerCase().includes(lowerSearch) || false;

      return nameMatch || descMatch || inputMatch || outputMatch;
    });
  }, [stages, searchTerm]);

  return (
    <div style={{
      width: '100%',
      background: 'white',
      display: 'flex',
      flexDirection: 'column',
      height: '100%'
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid #e1e5e9',
        background: '#f8f9fa'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '14px',
          fontWeight: '600',
          color: '#333'
        }}>
          <Package size={18} />
          Stage Library
        </div>
        <div style={{
          fontSize: '11px',
          color: '#666',
          marginTop: '2px'
        }}>
          {filteredStages.length} of {stages.length} stages
        </div>
      </div>

      {/* Search */}
      <div style={{
        padding: '10px 12px',
        borderBottom: '1px solid #e1e5e9'
      }}>
        <input
          type="text"
          placeholder="Search stages..."
          value={searchTerm}
          style={{
            width: '100%',
            padding: '8px 12px',
            border: '1px solid #e1e5e9',
            borderRadius: '6px',
            fontSize: '14px',
            outline: 'none'
          }}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {/* Stages List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '8px'
      }}>
        {filteredStages.length > 0 ? (
          filteredStages.map((stage) => (
            <StageCard
              key={stage.name}
              stage={stage}
              onAddStage={onAddStage}
            />
          ))
        ) : (
          <div style={{
            padding: '20px',
            textAlign: 'center',
            color: '#666',
            fontSize: '13px'
          }}>
            No stages found matching "{searchTerm}"
          </div>
        )}
      </div>
    </div>
  );
};

export default StageLibrary;
