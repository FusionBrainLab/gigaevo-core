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

if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
    return 1
end
return 0
