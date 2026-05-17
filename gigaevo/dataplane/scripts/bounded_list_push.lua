-- bounded_list_push.lua — atomic LPUSH + LTRIM bounded recency list.
--
-- Holds the most-recent N entries of a value stream in a Redis list. The
-- two ops collapse to one EVALSHA round-trip so concurrent writers
-- cannot race between the push and the trim, leaving the list briefly
-- over-cap. Newest-first ordering: index 0 is the freshest entry.
--
-- KEYS layout:
--   KEYS[1] = list key            e.g. "{prefix}:prompt:fitnesses:{prompt_id}"
--
-- ARGV layout:
--   ARGV[1] = value to push       (opaque string; encoded by the wrapper)
--   ARGV[2] = cap                 (positive integer; max entries retained)
--
-- Returns:
--   { new_length, cap }
--   both as ASCII integer strings. ``new_length`` is the list length
--   AFTER the trim, capped at ``cap``.
--
-- Reject cap <= 0; LTRIM 0 -1 would silently no-op the cap.

local cap = tonumber(ARGV[2])
if not cap or cap ~= math.floor(cap) or cap <= 0 then
    return redis.error_reply('bounded_list_push: cap must be a positive integer, got ' .. tostring(ARGV[2]))
end

redis.call('LPUSH', KEYS[1], ARGV[1])
redis.call('LTRIM', KEYS[1], 0, cap - 1)
local new_len = redis.call('LLEN', KEYS[1])
return {tostring(new_len), tostring(cap)}
