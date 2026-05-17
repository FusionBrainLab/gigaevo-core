-- instance_lock_release.lua — token-CAS release of a held lock.
--
-- DELs the lock key only if its value matches the caller's token. The
-- CAS prevents the "A's TTL expired, B took over, A blind-DELs B's
-- lock" race.
--
-- KEYS layout:
--   KEYS[1] = lock key
--
-- ARGV layout:
--   ARGV[1] = lease_token
--
-- Returns:
--   1  — released (token matched, key deleted)
--   0  — not our lock; no-op
--
-- An empty token is a caller bug; absent keys naturally fall through.

if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('instance_lock_release: lease_token must be non-empty')
end

if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
