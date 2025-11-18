import React from 'react';
import { X, Settings, Activity, Code, Zap, Edit3 } from 'lucide-react';
import { getStageColor, formatStageName, getTypeColor } from '../utils/stageUtils';

const NodeDetails = ({ node, onClose, onEdit }) => {
  const stageColor = getStageColor(node.data.originalName || node.data.name);
  const displayName = formatStageName(node.data.customName || node.data.name);

  return (
    <div style={{
      width: '100%',
      background: 'white',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid #e1e5e9',
        background: '#f8f9fa',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '14px',
          fontWeight: '600',
          color: '#333'
        }}>
          <Settings size={18} />
          Node Details
        </div>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <button
            onClick={() => onEdit && onEdit(node)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#007bff'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#e3f2fd';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'none';
            }}
            title="Edit stage properties"
          >
            <Edit3 size={16} />
          </button>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            onMouseEnter={(e) => {
              e.target.style.background = '#e9ecef';
            }}
            onMouseLeave={(e) => {
              e.target.style.background = 'none';
            }}
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        overflowX: 'hidden',
        padding: '20px',
        wordWrap: 'break-word',
        overflowWrap: 'break-word'
      }}>
        {/* Stage Info */}
        <div style={{
          background: stageColor,
          color: 'white',
          padding: '16px',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <div style={{
            fontSize: '18px',
            fontWeight: '600',
            marginBottom: '4px'
          }}>
            {displayName}
          </div>
          <div style={{
            fontSize: '14px',
            opacity: 0.9
          }}>
            {node.data.description}
          </div>
        </div>

        {/* Inputs Section */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '12px'
          }}>
            <Zap size={16} />
            Inputs
          </div>

          {/* Mandatory Inputs */}
          {node.data.mandatory_inputs.length > 0 && (
            <div style={{ marginBottom: '12px' }}>
              <div style={{
                fontSize: '12px',
                fontWeight: '600',
                color: '#dc3545',
                marginBottom: '6px'
              }}>
                Required Inputs:
              </div>
              {node.data.mandatory_inputs.map((input, index) => {
                const inputType = node.data.input_types && node.data.input_types[input];
                const typeColor = getTypeColor(inputType);
                return (
                  <div key={input} style={{
                    background: '#fff5f5',
                    border: '1px solid #fed7d7',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    marginBottom: '4px',
                    fontSize: '12px'
                  }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      marginBottom: inputType ? '4px' : '0'
                    }}>
                      <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: typeColor,
                        flexShrink: 0
                      }} />
                      <span style={{ fontWeight: '600' }}>{input}</span>
                    </div>
                    {inputType && (
                      <div style={{
                        fontSize: '11px',
                        color: typeColor,
                        marginLeft: '16px',
                        fontFamily: 'monospace',
                        fontWeight: '500'
                      }}>
                        {inputType}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Optional Inputs */}
          {node.data.optional_inputs.length > 0 && (
            <div>
              <div style={{
                fontSize: '12px',
                fontWeight: '600',
                color: '#6c757d',
                marginBottom: '6px'
              }}>
                Optional Inputs:
              </div>
              {node.data.optional_inputs.map((input, index) => {
                const inputType = node.data.input_types && node.data.input_types[input];
                const typeColor = getTypeColor(inputType);
                return (
                  <div key={input} style={{
                    background: '#f8f9fa',
                    border: '1px solid #e9ecef',
                    borderRadius: '4px',
                    padding: '8px 12px',
                    marginBottom: '4px',
                    fontSize: '12px',
                    opacity: 0.7
                  }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      marginBottom: inputType ? '4px' : '0'
                    }}>
                      <div style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: typeColor,
                        flexShrink: 0
                      }} />
                      <span style={{ fontWeight: '600' }}>{input}</span>
                    </div>
                    {inputType && (
                      <div style={{
                        fontSize: '11px',
                        color: typeColor,
                        marginLeft: '16px',
                        fontFamily: 'monospace',
                        fontWeight: '500'
                      }}>
                        {inputType}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {node.data.mandatory_inputs.length === 0 && node.data.optional_inputs.length === 0 && (
            <div style={{
              background: '#f8f9fa',
              border: '1px solid #e9ecef',
              borderRadius: '6px',
              padding: '12px',
              fontSize: '12px',
              color: '#6c757d',
              textAlign: 'center'
            }}>
              No inputs required
            </div>
          )}
        </div>

        {/* Outputs Section */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '12px'
          }}>
            <Code size={16} />
            Outputs
          </div>
          <div style={{
            background: '#f0f8ff',
            border: '1px solid #b3d9ff',
            borderRadius: '6px',
            padding: '12px',
            fontSize: '12px'
          }}>
            {(() => {
              const outputType = node.data.output_model_name || 'Output';
              const typeColor = getTypeColor(outputType);
              return (
                <>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    marginBottom: '4px'
                  }}>
                    <div style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: typeColor,
                      flexShrink: 0
                    }} />
                    <span style={{ fontWeight: '600' }}>output</span>
                  </div>
                  <div style={{
                    fontSize: '11px',
                    color: typeColor,
                    marginLeft: '16px',
                    fontFamily: 'monospace',
                    fontWeight: '500'
                  }}>
                    {outputType}
                  </div>
                </>
              );
            })()}
          </div>
        </div>

        {/* Node Properties */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '12px'
          }}>
            <Activity size={16} />
            Properties
          </div>
          <div style={{
            background: '#f8f9fa',
            border: '1px solid #e9ecef',
            borderRadius: '6px',
            padding: '12px'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '12px',
              marginBottom: '6px'
            }}>
              <span style={{ color: '#6c757d' }}>Node ID:</span>
              <span style={{ color: '#333', fontFamily: 'monospace' }}>{node.id}</span>
            </div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '12px',
              marginBottom: '6px'
            }}>
              <span style={{ color: '#6c757d' }}>Position:</span>
              <span style={{ color: '#333', fontFamily: 'monospace' }}>
                ({Math.round(node.position.x)}, {Math.round(node.position.y)})
              </span>
            </div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '12px',
              marginBottom: '6px'
            }}>
              <span style={{ color: '#6c757d' }}>Type:</span>
              <span style={{ color: '#333', fontFamily: 'monospace' }}>{node.type}</span>
            </div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '12px',
              alignItems: 'center'
            }}>
              <span style={{ color: '#6c757d' }}>Cacheable:</span>
              <span style={{ color: '#333', display: 'flex', alignItems: 'center', gap: '4px' }}>
                {node.data.cacheable ? (
                  <>
                    <span>Yes</span>
                    <span style={{ fontSize: 11 }} title="Results can be cached">💾</span>
                  </>
                ) : (
                  <>
                    <span>No</span>
                    <span style={{ fontSize: 11 }} title="Always recomputes">🔄</span>
                  </>
                )}
              </span>
            </div>
          </div>
        </div>

        {/* Export Information */}
        <div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '12px'
          }}>
            <Code size={16} />
            Export Information
          </div>
          <div style={{
            background: '#f8f9fa',
            border: '1px solid #e9ecef',
            borderRadius: '6px',
            padding: '12px',
            fontSize: '12px',
            color: '#333'
          }}>
            <div style={{ marginBottom: '8px' }}>
              <strong>Class:</strong> {node.data.originalName || node.data.name}
            </div>
            <div style={{ marginBottom: '8px' }}>
              <strong>Import:</strong> {node.data.import_path || 'Not specified'}
            </div>
            <div>
              <strong>Description:</strong> {node.data.description || 'No description'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NodeDetails;
