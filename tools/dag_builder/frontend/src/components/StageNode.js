// src/components/StageNode.js
import React, { useMemo, useCallback, useEffect } from 'react';
import { Handle, Position, useReactFlow, useUpdateNodeInternals } from 'reactflow';
import { getStageColor, getStageIcon, formatStageName } from '../utils/stageUtils';

const EXEC_COLORS = { always: '#6c757d', success: '#28a745', failure: '#dc3545' };
const INPUT_COLORS = { mandatory: '#dc3545', optional: '#6c757d' };
const portSpacing = 18;

/* --------------------- HSL helpers (unchanged) --------------------- */
function parseHsl(hsl) {
  const m = /hsl\(\s*([0-9.]+)\s*,\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*\)/i.exec(hsl || '');
  if (!m) return null;
  return { h: Number(m[1]), s: Number(m[2]), l: Number(m[3]) };
}
function hsla({ h, s, l }, a = 1) { return `hsla(${Math.round(h)}, ${Math.round(s)}%, ${Math.round(l)}%, ${a})`; }
function clamp(n, min = 0, max = 100) { return Math.max(min, Math.min(max, n)); }
function shadesFromStageColor(hslStr) {
  const p = parseHsl(hslStr);
  if (!p) return {
    accent: '#64748B',
    accentSoft: 'rgba(100,116,139,0.08)',
    accentBorder: 'rgba(100,116,139,0.35)',
    accentStrong: '#64748B',
  };
  const accent = hsla(p, 1);
  const accentSoft = hsla({ ...p, l: clamp(p.l, 30, 85) }, 0.10);
  const accentBorder = hsla({ ...p, l: clamp(p.l, 30, 65) }, 0.35);
  const accentStrong = hsla({ ...p, l: clamp(p.l - 6, 25, 60) }, 1);
  return { accent, accentSoft, accentBorder, accentStrong };
}

/* --------------------- Main component --------------------- */
function StageNodeBase({ data = {}, isConnectable, id, selected }) {
  const { setNodes, setEdges, getEdges, getNodes } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  // Canonical key for color/icon: use originalName to match StageLibrary colors
  const colorKey = useMemo(() => {
    const n = (data.originalName && String(data.originalName).trim()) || '';
    if (n) return n;
    return ((data.name && String(data.name).trim()) ||
      (data.customName && String(data.customName).trim()) ||
      'Untitled Stage');
  }, [data.originalName, data.name, data.customName]);

  // Pretty display name
  const displayName = useMemo(() => {
    const raw =
      (data.customName && String(data.customName)) ||
      (data.name && String(data.name)) ||
      '';
    const formatted = raw ? formatStageName(raw) : '';
    return (formatted && formatted.trim()) || raw || 'Untitled Stage';
  }, [data.customName, data.name]);

  // Visuals
  const baseStageColor = useMemo(() => getStageColor(colorKey), [colorKey]);
  const palette = useMemo(() => shadesFromStageColor(baseStageColor), [baseStageColor]);
  const stageIcon  = useMemo(() => getStageIcon(colorKey), [colorKey]);

  // Inputs
  const mandatoryInputs = Array.isArray(data.mandatory_inputs) ? data.mandatory_inputs : [];
  const optionalInputs  = Array.isArray(data.optional_inputs)  ? data.optional_inputs  : [];
  const totalInputs = mandatoryInputs.length + optionalInputs.length;

  // Recompute handle layout when counts change
  useEffect(() => { updateNodeInternals(id); }, [id, totalInputs, updateNodeInternals]);

  // Color of exec-source reflects outgoing execution type
  const getOutgoingExecutionColor = useCallback(() => {
    const outgoingExec = getEdges().filter(
      (e) => e.source === id && e.sourceHandle === 'exec-source' && e.type === 'execution'
    );
    if (!outgoingExec.length) return '#9CA3AF';
    const t = outgoingExec[0]?.data?.executionType;
    return EXEC_COLORS[t] || '#9CA3AF';
  }, [getEdges, id]);

  const doDuplicate = useCallback(() => {
    const me = getNodes().find((n) => n.id === id);
    if (!me) return;
    setNodes((nds) => [
      ...nds,
      {
        ...me,
        id: (crypto.randomUUID && crypto.randomUUID()) || `${me.id}-copy-${Date.now()}`,
        position: { x: me.position.x + 40, y: me.position.y + 40 },
        selected: false,
      },
    ]);
  }, [getNodes, id, setNodes]);

  const doDelete = useCallback(() => {
    setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
    setNodes((nds) => nds.filter((n) => n.id !== id));
  }, [id, setEdges, setNodes]);

  return (
    <div
      className="stage-node"
      role="group"
      aria-label={`${displayName} node`}
      style={{
        position: 'relative',
        background: 'white',
        border: `2px solid ${selected ? '#3B82F6' : palette.accentBorder}`,
        borderRadius: 14,
        minWidth: 260,
        maxWidth: 360,
        minHeight: 120,
        overflow: 'visible',
        boxShadow: selected
          ? '0 16px 36px rgba(59,130,246,0.25), 0 2px 0 rgba(59,130,246,0.05) inset'
          : '0 10px 28px rgba(0,0,0,0.12), 0 1px 0 rgba(0,0,0,0.04) inset',
        transition: 'box-shadow 140ms ease, border-color 140ms ease, transform 120ms ease',
        willChange: 'transform, box-shadow',
      }}
    >
      {/* Accent rail */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          left: -2,
          top: -2,
          bottom: -2,
          width: 6,
          borderTopLeftRadius: 12,
          borderBottomLeftRadius: 12,
          background: `linear-gradient(180deg, ${palette.accent} 0%, ${palette.accentStrong} 100%)`,
        }}
      />

      {/* Header */}
      <div
        style={{
          background: palette.accentSoft,
          color: '#111827',
          padding: '10px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          borderTopLeftRadius: 12,
          borderTopRightRadius: 12,
          borderBottom: '1px solid #E5E7EB',
          userSelect: 'none',
        }}
        title={displayName}
      >
        <span aria-hidden style={{ fontSize: 18, lineHeight: 1 }}>{stageIcon}</span>
        <div
          style={{
            flex: 1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            letterSpacing: 0.2,
            fontWeight: 800,
            color: '#111827',
          }}
        >
          {displayName}
        </div>

        {/* Quick actions */}
        <div className="node-actions" style={{ display: 'flex', gap: 6, opacity: 0, transition: 'opacity 140ms ease' }}>
          <IconBtn onClick={doDuplicate} title="Duplicate (⌘/Ctrl+D)" ariaLabel="Duplicate node">⎘</IconBtn>
          <IconBtn onClick={doDelete} title="Delete (Del)" ariaLabel="Delete node" danger>✕</IconBtn>
        </div>
      </div>

      {/* Body */}
      <div style={{ padding: '12px 48px 16px 48px' }}>
        {data.description && (
          <div
            style={{
              fontSize: 12,
              color: '#4B5563',
              lineHeight: 1.45,
              marginBottom: 8,
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
            title={data.description}
          >
            {data.description}
          </div>
        )}

        <div
          style={{
            display: 'flex',
            gap: 8,
            alignItems: 'center',
            fontSize: 11,
            color: '#6B7280',
            flexWrap: 'wrap',
          }}
        >
          {totalInputs > 0 && (
            <Badge>
              Inputs: {totalInputs}
              {mandatoryInputs.length ? ` (${mandatoryInputs.length} req)` : ''}
            </Badge>
          )}
          {mandatoryInputs.slice(0, 3).map((n) => (
            <Badge key={`mi-${n}`} muted title={n}>
              {trimName(n)}
            </Badge>
          ))}
          {mandatoryInputs.length > 3 && <Badge muted>+{mandatoryInputs.length - 3} more</Badge>}
        </div>
      </div>

      {/* ===== Ports ===== */}

      {/* Input rail */}
      {totalInputs > 0 && (
        <div
          style={{
            position: 'absolute',
            left: -6,
            top: 18,
            bottom: 24,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            gap: portSpacing - 12,
            pointerEvents: 'none',
          }}
        >
          {mandatoryInputs.map((name, i) => (
            <MemoPortWithLabel
              key={`m-${name}`}
              id={name}
              label={name}
              type="target"
              color={INPUT_COLORS.mandatory}
              position={Position.Left}
              topOffset={i * portSpacing}
              isConnectable={isConnectable}
              dataType="data"
              dataPortKind="mandatory"
            />
          ))}
          {optionalInputs.map((name, j) => (
            <MemoPortWithLabel
              key={`o-${name}`}
              id={name}
              label={name}
              type="target"
              color={INPUT_COLORS.optional}
              position={Position.Left}
              topOffset={(mandatoryInputs.length + j) * portSpacing}
              isConnectable={isConnectable}
              muted
              dataType="data"
              dataPortKind="optional"
            />
          ))}
        </div>
      )}

      {/* Data output */}
      <Handle
        className="handle-data-output"
        type="source"
        position={Position.Right}
        id="output"
        style={{
          right: -6,
          top: '50%',
          transform: 'translateY(-50%)',
          background: palette.accentStrong,
          border: '2px solid white',
          width: 12,
          height: 12,
        }}
        isConnectable={isConnectable}
        title="Data output"
        data-handle-type="data"
        data-direction="out"
      />

      {/* Execution targets */}
      <MemoExecTarget
        id="exec-success" color={EXEC_COLORS.success} title="On success"
        offsetX={-24} isConnectable={isConnectable}
      />
      <MemoExecTarget
        id="exec-always" color={EXEC_COLORS.always} title="Always"
        offsetX={0} isConnectable={isConnectable}
      />
      <MemoExecTarget
        id="exec-failure" color={EXEC_COLORS.failure} title="On failure"
        offsetX={24} isConnectable={isConnectable}
      />

      {/* Execution source */}
      <Handle
        className="handle-exec-source"
        type="source"
        position={Position.Top}
        id="exec-source"
        style={{
          left: '50%',
          top: -8,
          transform: 'translateX(-50%)',
          width: 16,
          height: 16,
          background: getOutgoingExecutionColor(),
          border: '2px solid white',
          borderRadius: '50%',
          cursor: 'crosshair',
          pointerEvents: 'auto',
          zIndex: 5,
          boxShadow: '0 0 0 2px rgba(0,0,0,0.05)',
        }}
        isConnectable={isConnectable}
        title="Execution output (color shows type)"
        aria-label="Execution output"
        data-handle-type="exec"
        data-direction="out"
      />
    </div>
  );
}

/* --------------------- Subcomponents --------------------- */

const quickBtnStyle = {
  border: '1px solid #E5E7EB',
  background: 'white',
  color: '#374151',
  padding: '2px 6px',
  fontSize: 12,
  borderRadius: 6,
  cursor: 'pointer',
  lineHeight: 1,
  boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
  transition: 'transform 100ms ease, box-shadow 140ms ease, border-color 140ms ease',
  userSelect: 'none',
};

const IconBtn = React.memo(function IconBtn({ onClick, title, ariaLabel, children, danger }) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-label={ariaLabel || title}
      tabIndex={-1}
      style={{
        ...quickBtnStyle,
        color: danger ? '#EF4444' : quickBtnStyle.color,
        borderColor: danger ? '#FCA5A5' : quickBtnStyle.border,
      }}
      onMouseDown={(e) => e.preventDefault()}
      onPointerDown={(e) => e.preventDefault()}
      className="node-action-btn"
    >
      {children}
    </button>
  );
});

function Badge({ children, muted, title }) {
  return (
    <span
      title={title}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '2px 8px',
        borderRadius: 999,
        background: muted ? '#F3F4F6' : '#EEF2FF',
        color: muted ? '#4B5563' : '#4338CA',
        border: `1px solid ${muted ? '#E5E7EB' : '#E0E7FF'}`,
        fontWeight: 700,
        maxWidth: 180,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  );
}

function PortWithLabel({ id, label, type, color, position, topOffset, isConnectable, muted, dataType = 'data', dataPortKind }) {
  return (
    <div style={{ position: 'relative', height: 0 }}>
      <div
        className="port-label"
        style={{
          position: 'absolute',
          left: 14,
          top: topOffset - 6,
          fontSize: 10.5,
          color: muted ? '#6B7280' : '#111827',
          background: 'rgba(255,255,255,0.96)',
          border: '1px solid #E5E7EB',
          padding: '2px 6px',
          borderRadius: 6,
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          pointerEvents: 'none',
          opacity: 0,
          transform: 'translateX(-4px)',
          transition: 'opacity 120ms ease, transform 120ms ease',
        }}
      >
        {label}
      </div>

      <Handle
        type={type}
        position={position}
        id={id}
        style={{
          left: 0,
          top: topOffset,
          background: color,
          border: '2px solid white',
          width: 12,
          height: 12,
          zIndex: 10,
          boxShadow: '0 0 0 2px rgba(0,0,0,0.05)',
          pointerEvents: 'auto',
        }}
        isConnectable={isConnectable}
        title={label}
        aria-label={`${label} input`}
        data-handle-type={dataType}          // <- for connection validation
        data-port-kind={dataPortKind}        // <- mandatory/optional
        data-direction="in"
      />
    </div>
  );
}

const MemoPortWithLabel = React.memo(PortWithLabel);

function ExecTarget({ id, color, title, offsetX, isConnectable }) {
  return (
    <Handle
      type="target"
      position={Position.Bottom}
      id={id}
      style={{
        left: `calc(50% + ${offsetX}px)`,
        bottom: -8,
        transform: 'translateX(-50%)',
        width: 16,
        height: 16,
        background: color,
        border: '2px solid white',
        borderRadius: '50%',
        cursor: 'crosshair',
        pointerEvents: 'auto',
        zIndex: 5,
        boxShadow: '0 0 0 2px rgba(0,0,0,0.05)',
      }}
      isConnectable={isConnectable}
      title={title}
      aria-label={title}
      data-handle-type="exec"                // <- for connection validation
      data-direction="in"
    />
  );
}

const MemoExecTarget = React.memo(ExecTarget);

/* --------------------- Small utils --------------------- */
function trimName(s) {
  if (!s) return '';
  const t = String(s);
  return t.length > 22 ? t.slice(0, 19) + '…' : t;
}

/* --------------------- Hover/animation CSS --------------------- */
const styleEl = typeof document !== 'undefined' ? document.createElement('style') : null;
if (styleEl && !styleEl.dataset.stageNodeCss) {
  styleEl.dataset.stageNodeCss = 'true';
  styleEl.textContent = `
    .stage-node { transform: translateZ(0); }
    .stage-node:hover { transform: translateY(-1px); }
    .stage-node:active { transform: translateY(0); }
    .stage-node:hover .node-actions { opacity: 1; }
    .stage-node:hover .port-label { opacity: 1; transform: translateX(0); }
    .node-action-btn:hover { transform: translateY(-1px); }
    .node-action-btn:active { transform: translateY(0); box-shadow: 0 0 0 rgba(0,0,0,0.06); }

    /* Respect reduced motion */
    @media (prefers-reduced-motion: reduce) {
      .stage-node, .node-action-btn, .port-label, .node-actions { transition: none !important; }
    }
  `;
  document.head.appendChild(styleEl);
}

export default React.memo(StageNodeBase);
