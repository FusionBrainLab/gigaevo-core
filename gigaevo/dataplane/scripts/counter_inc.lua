-- counter_inc.lua — atomic per-actor G-counter increment.
--
-- A G-counter stores one integer per actor. The consensus value is the
-- sum across all actor fields. Increments commute and are idempotent
-- under deduplication, so two engines incrementing the same counter
-- from disjoint actor identities never lose updates and never need
-- coordination beyond this single atomic Lua call.
--
-- KEYS layout:
--   KEYS[1] = counter hash         e.g. "{prefix}:{counter_key}:counts"
--                                  (field = actor_id, value = signed int)
--   KEYS[2] = generation counter   e.g. "{prefix}:{counter_key}:gen"
--                                  (per-counter monotonic, used by Versioned reads)
--   KEYS[3] = global epoch         e.g. "{prefix}:ts"
--                                  (process-wide monotonic; witness on every write)
--
-- ARGV layout:
--   ARGV[1] = actor_id      — the field name under KEYS[1] (string)
--   ARGV[2] = delta         — signed integer; negative deltas decrement
--
-- Returns:
--   { new_per_actor_count, generation, epoch }
--   all elements are ASCII integer strings (so the client can int(...) them).
--
-- Errors out on a non-integer delta / empty actor_id; HINCRBY would
-- error too, but a Lua-side check produces a clearer message and keeps
-- the failure path before any side effect.

if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('counter_inc: actor_id must be non-empty')
end
local delta = tonumber(ARGV[2])
-- Reject nil, NaN, +/-inf, and non-integer values. NaN fails the
-- self-equality check; +/-inf fail the math.huge bounds; fractional
-- values fail the floor-equality check; non-numeric strings make
-- tonumber return nil.
if not delta
   or delta ~= delta
   or delta == math.huge
   or delta == -math.huge
   or delta ~= math.floor(delta) then
    return redis.error_reply('counter_inc: delta must be a finite integer, got ' .. tostring(ARGV[2]))
end

local new_count = redis.call('HINCRBY', KEYS[1], ARGV[1], delta)
local generation = redis.call('INCR', KEYS[2])
local epoch = redis.call('INCR', KEYS[3])
return {tostring(new_count), tostring(generation), tostring(epoch)}
