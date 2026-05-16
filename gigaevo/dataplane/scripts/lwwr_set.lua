-- lwwr_set.lua — Last-Write-Wins register with HLC tiebreak.
--
-- Stores a single ``(value, hlc)`` pair under one Redis hash. Writes
-- arriving with an HLC strictly newer than the stored HLC replace; ties
-- or older HLCs are rejected silently as 'kept'. The 32-char hex HLC
-- format is lexicographic-comparable (big-endian physical_ns dominates,
-- counter breaks ties, trailing 8-byte pad is always zero) so a string
-- compare suffices for "is newer".
--
-- KEYS layout:
--   KEYS[1] = register hash         e.g. "{prefix}:lwwr:{name}"
--                                   (fields: 'value' (string), 'hlc' (32 hex))
--
-- ARGV layout:
--   ARGV[1] = candidate value       — caller-encoded canonical JSON
--   ARGV[2] = candidate hlc_hex     — 32-char big-endian hex
--
-- Returns:
--   'replaced' — candidate HLC was strictly newer; stored value updated
--   'kept'     — candidate HLC was older or equal; no write
--
-- Errors out on a malformed candidate HLC. Lexicographic comparison
-- is only order-preserving when both operands are equal length and
-- drawn from the same alphabet; a short / non-hex HLC would compare
-- non-monotonically against well-formed stored HLCs and silently let
-- an older write win. The wrapper computes ``hlc.pack_hex()`` which
-- always returns 32 lowercase hex chars; this check is the server-
-- side belt.

local cand_hlc = ARGV[2]
if cand_hlc == nil or #cand_hlc ~= 32 or string.match(cand_hlc, '^[0-9a-f]+$') == nil then
    return redis.error_reply('lwwr_set: candidate hlc must be 32 lowercase hex chars, got ' .. tostring(cand_hlc))
end

local cur_hlc = redis.call('HGET', KEYS[1], 'hlc')
if cur_hlc and cur_hlc >= cand_hlc then
    return 'kept'
end
redis.call('HSET', KEYS[1], 'value', ARGV[1], 'hlc', cand_hlc)
return 'replaced'
