-- instance_lock_release.lua — token-CAS release of a held lock.
--
-- Atomically deletes the lock key only if its stored value matches the
-- caller's token. The token check is the protection against the
-- classic "process A's TTL expired between when A decided to release
-- and when A actually DEL'd, and B has taken over in the meantime"
-- footgun. Without the CAS, A's blind DEL would release B's lock.
--
-- KEYS layout:
--   KEYS[1] = lock key
--
-- ARGV layout:
--   ARGV[1] = lease_token   — must equal the stored value to release
--
-- Returns:
--   1  — released; the key was owned by our token and is now deleted
--   0  — not our lock (token mismatch / key absent); no-op
--
-- An empty token is a caller bug — silently no-op'ing would mask the
-- bug and a release-without-token mutation is something we want to
-- never accept. ``GET`` on an absent key returns Lua false, which
-- compares not-equal to any non-empty string, so the no-match branch
-- is naturally covered.

if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('instance_lock_release: lease_token must be non-empty')
end

if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
