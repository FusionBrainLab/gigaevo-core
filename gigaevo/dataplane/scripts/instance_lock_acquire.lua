-- instance_lock_acquire.lua — token-CAS acquisition with TTL.
--
-- Atomic SET NX EX. The caller mints a fresh random token client-side
-- and passes it as ARGV[1]; the server stores the token as the lock
-- key's value so a subsequent renew / release can token-CAS-verify
-- ownership before mutating.
--
-- KEYS layout:
--   KEYS[1] = lock key      e.g. "{prefix}:lock:{name}"
--                           (string; value = caller's lease token)
--
-- ARGV layout:
--   ARGV[1] = lease_token   — client-minted opaque random string
--   ARGV[2] = ttl_ms        — positive integer; millisecond precision
--                             (sub-second TTLs need PX, not EX)
--
-- Returns:
--   1  — lock acquired (the caller now holds it for up to ttl_ms ms)
--   0  — lock currently held by some other token; caller must retry or fail
--
-- Errors out with a script error on invalid TTL / empty token; both are
-- caller bugs and surfaceable as ``DataPlaneError`` rather than silent
-- "couldn't acquire" returns.

local ttl_ms = tonumber(ARGV[2])
if not ttl_ms or ttl_ms < 1 then
    return redis.error_reply('instance_lock_acquire: ttl_ms must be a positive integer, got ' .. tostring(ARGV[2]))
end
if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('instance_lock_acquire: lease_token must be non-empty')
end

local ok = redis.call('SET', KEYS[1], ARGV[1], 'NX', 'PX', ttl_ms)
if ok then
    return 1
end
return 0
