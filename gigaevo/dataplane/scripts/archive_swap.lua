-- archive_swap.lua — atomic MAP-Elites cell swap with score comparison.
--
-- A cell holds at most one program at a time. The candidate wins iff
-- the cell is empty OR its score (strictly greater, or equal with
-- caller-supplied tiebreak bit set) beats the occupant's stored score.
-- The whole compare-and-swap runs as one Lua atomic — no WATCH retry
-- and no cache-vs-Redis divergence window for sibling tasks to race.
--
-- KEYS layout:
--   KEYS[1] = archive hash      "{prefix}:archive"
--                               (field = cell_field, value = program_id)
--   KEYS[2] = reverse hash      "{prefix}:archive:reverse"
--                               (field = program_id, value = cell_field)
--   KEYS[3] = scores hash       "{prefix}:archive:scores"
--                               (field = cell_field, value = float-as-string)
--
-- ARGV layout:
--   ARGV[1] = cell_field        — caller-chosen cell key
--   ARGV[2] = candidate_id      — the program competing for the cell
--   ARGV[3] = candidate_score   — float, Python repr() for round-trip
--   ARGV[4] = tiebreak_bit      — "1" wins ties, "0" loses ties
--
-- Returns:
--   {'inserted', ''}            — cell was empty; candidate now occupies
--   {'swapped',  displaced_id}  — candidate beat occupant; displaced_id evicted
--   {'rejected', occupant_id}   — candidate lost; occupant_id retained

local cur_id = redis.call('HGET', KEYS[1], ARGV[1])
if not cur_id then
    redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
    redis.call('HSET', KEYS[2], ARGV[2], ARGV[1])
    redis.call('HSET', KEYS[3], ARGV[1], ARGV[3])
    return {'inserted', ''}
end

local cur_score_str = redis.call('HGET', KEYS[3], ARGV[1])
local cur_score = tonumber(cur_score_str) or 0.0
local cand_score = tonumber(ARGV[3])

if cand_score < cur_score then
    return {'rejected', cur_id}
end
if cand_score == cur_score and ARGV[4] == '0' then
    return {'rejected', cur_id}
end

-- Swap: cur_id loses, candidate becomes the new occupant.
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[2], cur_id)
redis.call('HSET', KEYS[2], ARGV[2], ARGV[1])
redis.call('HSET', KEYS[3], ARGV[1], ARGV[3])
return {'swapped', cur_id}
