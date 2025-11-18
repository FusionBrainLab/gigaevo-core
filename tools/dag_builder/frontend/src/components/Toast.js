import React, { useEffect } from 'react';
import { CheckCircle, XCircle, AlertCircle, Info } from 'lucide-react';

const Toast = ({ message, type = 'success', onClose, duration = 3000 }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, duration);

    return () => clearTimeout(timer);
  }, [duration, onClose]);

  const config = {
    success: {
      bg: '#10b981',
      icon: <CheckCircle size={20} />,
    },
    error: {
      bg: '#ef4444',
      icon: <XCircle size={20} />,
    },
    warning: {
      bg: '#f59e0b',
      icon: <AlertCircle size={20} />,
    },
    info: {
      bg: '#3b82f6',
      icon: <Info size={20} />,
    },
  };

  const { bg, icon } = config[type] || config.success;

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        background: bg,
        color: 'white',
        padding: '12px 20px',
        borderRadius: '8px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        zIndex: 10000,
        minWidth: '300px',
        maxWidth: '500px',
        animation: 'slideIn 0.3s ease-out',
      }}
    >
      {icon}
      <span style={{ flex: 1, fontSize: '14px', fontWeight: '500' }}>
        {message}
      </span>
      <style>
        {`
          @keyframes slideIn {
            from {
              transform: translateX(400px);
              opacity: 0;
            }
            to {
              transform: translateX(0);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
};

export default Toast;
