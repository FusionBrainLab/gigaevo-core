import React, { useRef } from 'react';
import { Package } from 'lucide-react';
import { getStageColor, getStageIcon, formatStageName } from '../utils/stageUtils';

// Individual stage card component to handle hover states properly
const StageCard = ({ stage, onAddStage }) => {
  const cardRef = useRef(null);
  const buttonRef = useRef(null);
  const lastClickTime = useRef(0);

  const handleCardMouseEnter = () => {
    if (cardRef.current) {
      cardRef.current.style.transform = 'translateY(-2px)';
      cardRef.current.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
      cardRef.current.style.borderColor = getStageColor(stage.name);
    }
  };

  const handleCardMouseLeave = () => {
    if (cardRef.current) {
      cardRef.current.style.transform = 'translateY(0)';
      cardRef.current.style.boxShadow = 'none';
      cardRef.current.style.borderColor = '#e1e5e9';
    }
  };

  const handleButtonMouseEnter = (e) => {
    e.stopPropagation();
    if (buttonRef.current) {
      buttonRef.current.style.transform = 'scale(1.15)';
      buttonRef.current.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
      buttonRef.current.style.background = getStageColor(stage.name);
    }
  };

  const handleButtonMouseLeave = (e) => {
    e.stopPropagation();
    if (buttonRef.current) {
      buttonRef.current.style.transform = 'scale(1)';
      buttonRef.current.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
      buttonRef.current.style.background = getStageColor(stage.name);
    }
  };

  return (
    <div
      ref={cardRef}
      style={{
        background: 'white',
        border: '1px solid #e1e5e9',
        borderRadius: '8px',
        padding: '12px',
        margin: '6px 0',
        cursor: 'grab',
        transition: 'all 0.2s ease',
        position: 'relative',
        overflow: 'hidden',
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
      {/* Color indicator */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '3px',
        background: getStageColor(stage.name)
      }} />
      
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
          top: '8px',
          right: '8px',
          background: getStageColor(stage.name),
          color: 'white',
          border: 'none',
          borderRadius: '50%',
          width: '28px',
          height: '28px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          fontSize: '18px',
          fontWeight: 'bold',
          lineHeight: '1',
          transition: 'all 0.2s ease',
          boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
          zIndex: 10,
          pointerEvents: 'auto',
          padding: '0',
          margin: '0'
        }}
        title="Click to add to canvas"
      >
        <span style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          height: '100%',
          fontSize: '16px',
          fontWeight: 'bold',
          lineHeight: '1',
          margin: '0',
          padding: '0',
          textAlign: 'center',
          verticalAlign: 'middle'
        }}>
          +
        </span>
      </button>

      {/* Stage info */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '8px'
      }}>
        <span style={{ fontSize: '16px' }}>
          {getStageIcon()}
        </span>
        <div>
          <div style={{
            fontWeight: '600',
            fontSize: '14px',
            color: '#333',
            lineHeight: '1.2'
          }}>
            {formatStageName(stage.name)}
          </div>
        </div>
      </div>

      <div style={{
        fontSize: '12px',
        color: '#666',
        lineHeight: '1.4',
        marginBottom: '8px'
      }}>
        {stage.description}
      </div>

      {/* Inputs/Outputs */}
      <div style={{
        display: 'flex',
        gap: '12px',
        fontSize: '11px'
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
    </div>
  );
};

const StageLibrary = ({ stages, onAddStage }) => {

  return (
    <div style={{
      width: '320px',
      background: 'white',
      borderRight: '1px solid #e1e5e9',
      display: 'flex',
      flexDirection: 'column',
      boxShadow: '2px 0 4px rgba(0,0,0,0.1)'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid #e1e5e9',
        background: '#f8f9fa'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '16px',
          fontWeight: '600',
          color: '#333'
        }}>
          <Package size={20} />
          Stage Library
        </div>
        <div style={{
          fontSize: '12px',
          color: '#666',
          marginTop: '4px'
        }}>
          {stages.length} stages available
        </div>
      </div>

      {/* Search */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid #e1e5e9'
      }}>
        <input
          type="text"
          placeholder="Search stages..."
          style={{
            width: '100%',
            padding: '8px 12px',
            border: '1px solid #e1e5e9',
            borderRadius: '6px',
            fontSize: '14px',
            outline: 'none'
          }}
          onChange={(e) => {
            // TODO: Implement search functionality
          }}
        />
      </div>

      {/* Stages List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '8px'
      }}>
        {stages.map((stage) => (
          <StageCard 
            key={stage.name}
            stage={stage}
            onAddStage={onAddStage}
          />
        ))}
      </div>
    </div>
  );
};

export default StageLibrary;
