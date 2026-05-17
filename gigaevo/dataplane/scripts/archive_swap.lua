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
--   {'invalid',  detail}        — candidate score parse failed (NaN/Inf/non-numeric)

if ARGV[1] == nil or ARGV[1] == '' then
    return redis.error_reply('archive_swap: cell_field must be non-empty')
end
if ARGV[2] == nil or ARGV[2] == '' then
    return redis.error_reply('archive_swap: candidate_id must be non-empty')
end

local cand_score = tonumber(ARGV[3])
-- Reject non-numeric, NaN (compares false to everything), and ±inf
-- (always-wins sentinel). Server-side belt; the wrapper rejects too.
if not cand_score or cand_score ~= cand_score or cand_score == math.huge or cand_score == -math.huge then
    return {'invalid', 'candidate score not finite: ' .. tostring(ARGV[3])}
end

local cur_id = redis.call('HGET', KEYS[1], ARGV[1])
if not cur_id then
    -- Vacate any prior cell the candidate occupies; preserves the
    -- (program → at-most-one-cell) bijection.
    local prior_cell = redis.call('HGET', KEYS[2], ARGV[2])
    if prior_cell and prior_cell ~= ARGV[1] then
        redis.call('HDEL', KEYS[1], prior_cell)
        redis.call('HDEL', KEYS[3], prior_cell)
    end
    redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
    redis.call('HSET', KEYS[2], ARGV[2], ARGV[1])
    redis.call('HSET', KEYS[3], ARGV[1], ARGV[3])
    return {'inserted', ''}
end

local cur_score_str = redis.call('HGET', KEYS[3], ARGV[1])
local cur_score = tonumber(cur_score_str)
-- Unparseable or NaN stored score ⇒ treat as 0.0 (corrupt rows).
if not cur_score or cur_score ~= cur_score then
    cur_score = 0.0
end

if cand_score < cur_score then
    return {'rejected', cur_id}
end
if cand_score == cur_score and ARGV[4] == '0' then
    return {'rejected', cur_id}
end

-- Swap: cur_id loses, candidate becomes the new occupant. Vacate any
-- prior cell the candidate held to preserve the bijection. The
-- candidate == cur_id case re-points the reverse entry to the same cell.
if ARGV[2] ~= cur_id then
    local prior_cell = redis.call('HGET', KEYS[2], ARGV[2])
    if prior_cell and prior_cell ~= ARGV[1] then
        redis.call('HDEL', KEYS[1], prior_cell)
        redis.call('HDEL', KEYS[3], prior_cell)
    end
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HDEL', KEYS[2], cur_id)
redis.call('HSET', KEYS[2], ARGV[2], ARGV[1])
redis.call('HSET', KEYS[3], ARGV[1], ARGV[3])
return {'swapped', cur_id}
