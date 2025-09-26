import React, { useState, useEffect } from 'react';
import { X, Save, Edit3, Tag, FileText, AlertCircle } from 'lucide-react';
import { isStageNameUnique } from '../utils/stageUtils';

const StageEditor = ({ node, nodes, onSave, onClose }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedData, setEditedData] = useState({
    customName: '',
    description: '',
    notes: ''
  });
  const [nameError, setNameError] = useState('');

  // Initialize with node data
  useEffect(() => {
    if (node) {
      setEditedData({
        customName: node.data.customName || '',
        description: node.data.description || '',
        notes: node.data.notes || ''
      });
    }
  }, [node]);

  const validateCustomName = (name) => {
    if (!name || name.trim() === '') {
      setNameError('');
      return true;
    }

    const trimmedName = name.trim();
    const isUnique = isStageNameUnique(trimmedName, nodes, node.id);
    
    if (!isUnique) {
      setNameError('This name is already used by another stage');
      return false;
    }

    setNameError('');
    return true;
  };

  const handleSave = () => {
    // Validate custom name before saving
    if (!validateCustomName(editedData.customName)) {
      return;
    }

    if (onSave) {
      onSave(node.id, editedData);
    }
    setIsEditing(false);
  };

  const handleCancel = () => {
    // Reset to original data
    setEditedData({
      customName: node.data.customName || '',
      description: node.data.description || '',
      notes: node.data.notes || ''
    });
    setIsEditing(false);
  };

  const handleInputChange = (field, value) => {
    setEditedData(prev => ({
      ...prev,
      [field]: value
    }));

    // Validate custom name in real-time
    if (field === 'customName') {
      validateCustomName(value);
    }
  };

  if (!node) return null;

  return (
    <div style={{
      width: '320px',
      background: 'white',
      borderLeft: '1px solid #e1e5e9',
      display: 'flex',
      flexDirection: 'column',
      boxShadow: '-2px 0 4px rgba(0,0,0,0.1)'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid #e1e5e9',
        background: '#f8f9fa',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '16px',
          fontWeight: '600',
          color: '#333'
        }}>
          <Edit3 size={20} />
          Stage Editor
        </div>
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

      {/* Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px'
      }}>
        {/* Original Stage Info */}
        <div style={{
          background: '#f8f9fa',
          border: '1px solid #e9ecef',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '20px'
        }}>
          <div style={{
            fontSize: '14px',
            fontWeight: '600',
            color: '#6c757d',
            marginBottom: '8px'
          }}>
            Original Stage
          </div>
          <div style={{
            fontSize: '16px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '4px'
          }}>
            {node.data.name}
          </div>
          <div style={{
            fontSize: '12px',
            color: '#6c757d'
          }}>
            {node.data.description}
          </div>
        </div>

        {/* Custom Name */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '8px'
          }}>
            <Tag size={16} />
            Custom Display Name
          </div>
          {isEditing ? (
            <div>
              <input
                type="text"
                value={editedData.customName}
                onChange={(e) => handleInputChange('customName', e.target.value)}
                placeholder="Enter custom name..."
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: `1px solid ${nameError ? '#dc3545' : '#e1e5e9'}`,
                  borderRadius: '6px',
                  fontSize: '14px',
                  outline: 'none',
                  transition: 'border-color 0.2s ease'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = nameError ? '#dc3545' : '#007bff';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = nameError ? '#dc3545' : '#e1e5e9';
                }}
              />
              {nameError && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  marginTop: '6px',
                  fontSize: '12px',
                  color: '#dc3545',
                  fontWeight: '500'
                }}>
                  <AlertCircle size={14} />
                  {nameError}
                </div>
              )}
            </div>
          ) : (
            <div style={{
              padding: '8px 12px',
              background: editedData.customName ? '#e8f5e8' : '#f8f9fa',
              border: '1px solid #e1e5e9',
              borderRadius: '6px',
              fontSize: '14px',
              color: editedData.customName ? '#333' : '#6c757d',
              minHeight: '36px',
              display: 'flex',
              alignItems: 'center'
            }}>
              {editedData.customName || 'No custom name set'}
            </div>
          )}
        </div>

        {/* Custom Description */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '8px'
          }}>
            <FileText size={16} />
            Custom Description
          </div>
          {isEditing ? (
            <textarea
              value={editedData.description}
              onChange={(e) => handleInputChange('description', e.target.value)}
              placeholder="Enter custom description..."
              rows={3}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid #e1e5e9',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                resize: 'vertical',
                fontFamily: 'inherit',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => {
                e.target.style.borderColor = '#007bff';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = '#e1e5e9';
              }}
            />
          ) : (
            <div style={{
              padding: '8px 12px',
              background: editedData.description ? '#e8f5e8' : '#f8f9fa',
              border: '1px solid #e1e5e9',
              borderRadius: '6px',
              fontSize: '14px',
              color: editedData.description ? '#333' : '#6c757d',
              minHeight: '60px',
              whiteSpace: 'pre-wrap'
            }}>
              {editedData.description || 'No custom description set'}
            </div>
          )}
        </div>

        {/* Notes */}
        <div style={{ marginBottom: '20px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600',
            color: '#333',
            marginBottom: '8px'
          }}>
            <FileText size={16} />
            Notes
          </div>
          {isEditing ? (
            <textarea
              value={editedData.notes}
              onChange={(e) => handleInputChange('notes', e.target.value)}
              placeholder="Add notes about this stage..."
              rows={4}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid #e1e5e9',
                borderRadius: '6px',
                fontSize: '14px',
                outline: 'none',
                resize: 'vertical',
                fontFamily: 'inherit',
                transition: 'border-color 0.2s ease'
              }}
              onFocus={(e) => {
                e.target.style.borderColor = '#007bff';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = '#e1e5e9';
              }}
            />
          ) : (
            <div style={{
              padding: '8px 12px',
              background: editedData.notes ? '#e8f5e8' : '#f8f9fa',
              border: '1px solid #e1e5e9',
              borderRadius: '6px',
              fontSize: '14px',
              color: editedData.notes ? '#333' : '#6c757d',
              minHeight: '80px',
              whiteSpace: 'pre-wrap'
            }}>
              {editedData.notes || 'No notes added'}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div style={{
          display: 'flex',
          gap: '8px',
          justifyContent: 'flex-end'
        }}>
          {isEditing ? (
            <>
              <button
                onClick={handleCancel}
                style={{
                  padding: '8px 16px',
                  background: '#f8f9fa',
                  color: '#6c757d',
                  border: '1px solid #e1e5e9',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = '#e9ecef';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = '#f8f9fa';
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={!!nameError}
                style={{
                  padding: '8px 16px',
                  background: nameError ? '#6c757d' : '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: nameError ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  opacity: nameError ? 0.6 : 1
                }}
                onMouseEnter={(e) => {
                  if (!nameError) {
                    e.target.style.background = '#218838';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!nameError) {
                    e.target.style.background = '#28a745';
                  }
                }}
              >
                <Save size={16} />
                Save
              </button>
            </>
          ) : (
            <button
              onClick={() => setIsEditing(true)}
              style={{
                padding: '8px 16px',
                background: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '14px',
                fontWeight: '500',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
              onMouseEnter={(e) => {
                e.target.style.background = '#0056b3';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = '#007bff';
              }}
            >
              <Edit3 size={16} />
              Edit
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default StageEditor;
