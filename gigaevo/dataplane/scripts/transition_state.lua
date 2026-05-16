-- transition_state.lua — atomic FSM-validated program state transition.
--
-- One Lua call does the full sequence:
--   1. fetch the program blob and decode the current state
--   2. short-circuit on idempotency hit; replay returns the stored blob
--   3. enforce the caller's expected_from (if provided)
--   4. validate (from, to) against the FSM table loaded at startup
--   5. record the idempotency token
--   6. merge the caller's patch fields into the blob
--   7. bump the global epoch and stamp it on the blob
--   8. write the new blob, update status-set membership, emit a
--      status event onto the audit stream
--
-- KEYS layout:
--   KEYS[1] = program blob key        "{prefix}:program:{pid}"
--   KEYS[2] = status events stream    "{prefix}:status_events"
--   KEYS[3] = global epoch counter    "{prefix}:ts"
--   KEYS[4] = FSM transitions hash    "{prefix}:fsm:program_state"
--   KEYS[5] = idempotency hash        "{prefix}:idem:{pid}"
--
-- ARGV layout:
--   ARGV[1] = program_id
--   ARGV[2] = expected_from   — ""=any, else must match prog.state
--   ARGV[3] = to              — target state (must be in FSM[from])
--   ARGV[4] = patch_json      — ""=no patch, else canonical-JSON dict
--                               to merge field-by-field into the blob
--   ARGV[5] = idempotency_tok — per-(pid, from, to, patch) hash from
--                               the Python wrapper; second call with
--                               the same token is a no-op duplicate
--   ARGV[6] = key_prefix      — used to build status-set keys dynamically
--                               (single-node Redis only; cluster mode
--                               would need static KEYS slots per state)
--
-- Returns:
--   { 'ok'        , epoch, blob_json }   — transition applied
--   { 'duplicate' , epoch, blob_json }   — idempotency hit; same outcome
--   { 'stale'     , '0',   detail }      — blob missing OR expected_from mismatch
--   { 'illegal'   , '0',   detail }      — FSM rejects (from, to)
--   { 'invalid'   , '0',   detail }      — patch JSON malformed or not an object

local IDEM_TTL_SECONDS = 300
local STREAM_MAXLEN = 10000
-- Reserved blob fields the caller's patch must not overwrite. ``state``
-- and ``epoch`` are advanced server-side; ``id`` identifies the blob.
local RESERVED_PATCH_FIELDS = {state = true, epoch = true, id = true}

if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('transition_state: program_id must be non-empty')
end
if ARGV[3] == nil or ARGV[3] == '' then
    return redis.error_reply('transition_state: target state must be non-empty')
end
if ARGV[5] == nil or ARGV[5] == '' then
    return redis.error_reply('transition_state: idempotency token must be non-empty')
end

local cur = redis.call('GET', KEYS[1])
if not cur then
    return {'stale', '0', 'program not found: ' .. ARGV[1]}
end

local ok_decode, prog = pcall(cjson.decode, cur)
if not ok_decode or type(prog) ~= 'table' then
    return {'invalid', '0', 'program blob decode failed for ' .. ARGV[1]}
end
local from = prog.state
if type(from) ~= 'string' or from == '' then
    return {'invalid', '0', 'program blob has no string state field for ' .. ARGV[1]}
end

-- Idempotency is checked before ``expected_from`` so a replay of an
-- already-applied call returns ``duplicate`` with the stored blob; if
-- the order were reversed, a successful first call would advance the
-- state and the retry's ``expected_from`` would no longer match. The
-- idempotency hash carries an ``IDEM_TTL_SECONDS`` window.
if redis.call('HEXISTS', KEYS[5], ARGV[5]) == 1 then
    local existing_epoch = tostring(prog.epoch or 0)
    return {'duplicate', existing_epoch, cur}
end

if ARGV[2] ~= '' and ARGV[2] ~= from then
    return {'stale', '0', 'expected_from=' .. ARGV[2] .. ' but observed ' .. from}
end

-- FSM legality: HGET the comma-joined target set for ``from`` and
-- check membership by walking the comma-separated tokens.
local fsm_row = redis.call('HGET', KEYS[4], from)
if not fsm_row then
    return {'illegal', '0', 'no FSM row for from-state ' .. tostring(from)}
end
local legal = false
for state in string.gmatch(fsm_row, '[^,]+') do
    if state == ARGV[3] then
        legal = true
        break
    end
end
if not legal then
    return {'illegal', '0', tostring(from) .. ' -> ' .. ARGV[3] .. ' not in FSM'}
end

-- Decode the patch BEFORE the idempotency HSET. A malformed patch
-- aborts the script as a Lua error; Redis does not roll back the
-- already-executed commands, so persisting the idem token first would
-- make subsequent (legal) replays return 'duplicate' with the
-- pre-transition blob — silently swallowing the call.
local patch = nil
if ARGV[4] ~= '' then
    local decoded
    local ok, err = pcall(function() decoded = cjson.decode(ARGV[4]) end)
    if not ok then
        return {'invalid', '0', 'patch decode failed: ' .. tostring(err)}
    end
    if type(decoded) ~= 'table' then
        return {'invalid', '0', 'patch must be a JSON object'}
    end
    patch = decoded
end

-- Record the idempotency token now so a concurrent retry observes the
-- hit on its HEXISTS above. ``HEXPIRE`` of the field would be cleaner
-- (Redis 7.4+) but EXPIRE on the whole hash matches the deployment
-- floor; the per-replay window is identical to the per-program window
-- because every program owns its own idem hash key.
redis.call('HSET', KEYS[5], ARGV[5], '1')
redis.call('EXPIRE', KEYS[5], IDEM_TTL_SECONDS)

-- Merge the caller's patch fields into the blob. Reserved fields are
-- silently dropped — the script owns ``state`` and ``epoch``, and ``id``
-- pins the blob's identity. A future hardening pass could reject the
-- patch as 'invalid' instead of dropping, but silent-drop is the
-- conservative choice while callers may still emit a legacy ``epoch``
-- key by accident.
if patch ~= nil then
    for k, v in pairs(patch) do
        if not RESERVED_PATCH_FIELDS[k] then
            prog[k] = v
        end
    end
end

prog.state = ARGV[3]
local new_epoch = redis.call('INCR', KEYS[3])
prog.epoch = new_epoch

redis.call('SET', KEYS[1], cjson.encode(prog))
redis.call('SREM', ARGV[6] .. ':status:' .. from, ARGV[1])
redis.call('SADD', ARGV[6] .. ':status:' .. ARGV[3], ARGV[1])
redis.call('XADD', KEYS[2], 'MAXLEN', '~', STREAM_MAXLEN, '*',
           'pid', ARGV[1],
           'from', from,
           'to', ARGV[3],
           'epoch', tostring(new_epoch))

return {'ok', tostring(new_epoch), cjson.encode(prog)}
