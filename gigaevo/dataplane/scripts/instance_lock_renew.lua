-- instance_lock_renew.lua — token-CAS renewal of a held lock.
--
-- Reads the current lock value and refreshes the TTL only if it matches
-- the caller's token. If the stored value differs (another holder took
-- over after a TTL expiry, or the key was deleted by a release / DEL),
-- the renewal fails — the caller must observe Err(LockLost) and stop
-- behaving as the lock holder.
--
-- KEYS layout:
--   KEYS[1] = lock key
--
-- ARGV layout:
--   ARGV[1] = lease_token   — must equal the stored value to renew
--   ARGV[2] = ttl_ms        — new TTL in milliseconds (matches acquire's PX)
--
-- Returns:
--   1  — renewed; TTL reset to ttl_ms milliseconds
--   0  — token mismatch or key absent; caller should surface LockLost
--
-- Errors out on invalid TTL / empty token — both are caller bugs.

local ttl_ms = tonumber(ARGV[2])
if not ttl_ms or ttl_ms < 1 then
    return redis.error_reply('instance_lock_renew: ttl_ms must be a positive integer, got ' .. tostring(ARGV[2]))
end
if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('instance_lock_renew: lease_token must be non-empty')
end

-- ``GET`` on a missing key returns Redis nil → Lua false; ``false ==
-- ARGV[1]`` is false for any string ARGV[1] so the absent-key branch
-- falls through to ``return 0`` without further work.
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('PEXPIRE', KEYS[1], ttl_ms)
    return 1
end
return 0
