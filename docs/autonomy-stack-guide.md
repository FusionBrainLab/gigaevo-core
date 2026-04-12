# Research Autonomy Stack — Operational Guide

**Branch**: `worktree-research-autonomy-stack`
**Status**: committed, not yet merged to main

This guide explains what was built, what is new vs. existing, and exactly how to use it.

---

## What exists vs. what is new

### Already existed (unchanged)
- `/run-experiment <task/name>` — full lifecycle skill (design → implement → launch → close)
- `/experiment-design`, `/experiment-implement`, `/experiment-launch`, `/experiment-closeout`
- `/experiment-checkpoint`, `/experiment-diagnose`, `/experiment-retrospective`
- Agents: Elena (ml-research-methodologist), Volkov (reviewer-2-adversary), literature-scout, treatment-verifier, checkpoint-analyst, anomaly-detector

### NEW in this branch

| What | File | Purpose |
|---|---|---|
| `code-archaeologist` agent | `.claude/agents/code-archaeologist.md` | Maps codebase before Elena designs; writes `codebase_map.md` |
| `implementation-aligner` agent | `.claude/agents/implementation-aligner.md` | Checks design↔code after implement; ALIGNED/MISALIGNED verdict |
| `paper-section-writer` agent | `.claude/agents/paper-section-writer.md` | Converts results JSON → paper prose sections |
| `/idea-generate` skill | `.claude/skills/idea-generate/` | Post-retrospective idea generation; updates IDEAS.yaml |
| `/research-scheduler` skill | `.claude/skills/research-scheduler/` | Picks top idea from IDEAS.yaml; starts experiment autonomously |
| `/experiment-paper-draft` skill | `.claude/skills/experiment-paper-draft/` | Builds paper_data.json → full 6-section draft |
| `tools/telegram_notify.py` | `tools/telegram_notify.py` | Send/receive Telegram messages for async gates |
| `tools/resource_manager.py` | `tools/resource_manager.py` | Auto-detect free GPU servers + Redis DBs |
| `experiments/IDEAS.yaml` | `experiments/IDEAS.yaml` | Ranked idea queue with lineage graph (seeded with 8 ideas) |
| `plans/research-autonomy-stack.md` | `plans/` | Architecture doc with SOTA paper citations |
| `docs/setup/telegram-bot.md` | `docs/setup/` | Telegram bot setup instructions |

### MODIFIED in this branch (enhanced, not rewritten)

| Skill/Agent | What changed |
|---|---|
| `experiment-design` | Added Step 2b — calls code-archaeologist before Elena |
| `experiment-implement` | Added Step 10b — implementation-aligner check before smoke test; Step 7 shows resource suggestions |
| `run-experiment` | Gates 1/2/3 now send Telegram and wait for async reply instead of blocking terminal |
| `experiment-retrospective` | Calls `/idea-generate` at end to auto-populate IDEAS.yaml |
| `experiment-closeout` | Steps 13b (paper-draft) + 13c (IDEAS.yaml update + idea-generate) added at end |
| `checkpoint-analyst` | Added goal-drift check — re-reads 01_design.md hypothesis at each checkpoint |

---

## Setup checklist (one-time)

### 1. Merge this branch (when ready)
```bash
cd /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/gigaevo-core-internal
gh pr create --base main --head worktree-research-autonomy-stack \
  --title "feat(autonomy): research autonomy stack"
# then: gh pr merge --merge --delete-branch
```

### 2. Set up Telegram (optional but recommended)

See `docs/setup/telegram-bot.md` for full instructions. Short version:

1. Message `@BotFather` on Telegram → `/newbot` → copy token
2. Message your bot → open `https://api.telegram.org/bot<TOKEN>/getUpdates` → get chat ID
3. Add to `~/.bashrc`:
```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdef..."
export TELEGRAM_CHAT_ID="123456789"
```
4. Test: `gigaevo notify "test"`

**If you skip Telegram**: everything still works — gates fall back to blocking terminal prompts.

### 3. Verify resource manager
```bash
cd /mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/gigaevo-core-internal
gigaevo resources --check
```
Expected output: list of servers with GPU% and list of free/used Redis DBs.

---

## End-to-end flow: what actually happens now

### Scenario A: You have an idea in mind

```
/run-experiment hover/gradient-critique "Does LLM critique feedback break improver stagnation?"
```

**What fires, in order:**

1. **Literature search** — literature-scout searches papers, writes `literature_brief.md`
2. **Codebase recon** (NEW) — code-archaeologist maps which code the treatment touches, writes `codebase_map.md`
3. **Elena designs** — reads literature_brief + codebase_map, writes `01_design.md` with specific file paths in treatment section
4. **Volkov reviews** — adversarial review, auto-resolves minor concerns
5. **Telegram Gate 1** (NEW) — you get a message: "Review 01_design.md. Reply 'approved' or feedback." You reply from your phone.
6. **Implementation** — agent implements the code
7. **Alignment check** (NEW) — implementation-aligner checks design requirements vs. code diff. If MISALIGNED → fixes gaps automatically, re-checks.
8. **Treatment verification** — treatment-verifier traces silent fallbacks
9. **Resource suggestion** (NEW) — resource_manager shows free servers/DBs to use
10. **Smoke test** — 3 generations, verifies treatment observables
11. **Telegram Gate 2** (NEW) — you get a message with config summary. Reply 'approved' to launch.
12. **Launch** — experiment runs. Background: watchdog (hourly Telegram fitness updates), checkpoint cron (4h), anomaly detector (2h)
13. **Completion** — anomaly detector triggers closeout
14. **Paper draft** (NEW) — paper_data.json → 6 paper sections generated automatically
15. **IDEAS.yaml updated** (NEW) — idea marked done, 3-5 new ideas generated and ranked
16. **Telegram Gate 3** (NEW) — you get "Verdict: POSITIVE, +8.5pp. Review 05_results.md. Reply 'approved' to merge." You reply from your phone.
17. **Merge PR** — done.
18. **Retrospective** (optional) — `/experiment-retrospective hover` → synthesizes patterns, generates more ideas

### Scenario B: Fully autonomous mode

```
/research-scheduler --task hover
```

1. Reads IDEAS.yaml, picks highest-ranked queued idea
2. Prints: "Selected hover_001 — Gradient critique. Starting in 10s..."
3. Starts `/run-experiment hover/gradient-critique <hypothesis>` autonomously
4. You still get Telegram gates — system doesn't proceed without your approval
5. After closeout: IDEAS.yaml updated, next idea ready to pick

### Scenario C: Draft a paper from existing results

```
/experiment-paper-draft hover hover/dynamic-topology hover/steady-state-v2
```

Produces in `experiments/hover/`:
- `paper_data.json` — structured experiment data
- `paper_draft.md` — full 6-section draft (~3500 words)
- `paper_review.md` — Reviewer 2 critique + responses

---

## Ideas pool: how to manage IDEAS.yaml

File: `experiments/IDEAS.yaml`

**Add your own idea** — edit the file directly:
```yaml
- id: hover_005
  title: "Your idea title"
  task: hover
  status: queued
  rank: 0.85          # your priority estimate
  hypothesis: >
    What mechanism? Why should it work?
  mechanism: "One sentence: which code + what change"
  expected_effect: "+X-Y pp"
  estimated_cost: "4 runs × 50 gen"
  source: human
  created: "2026-04-09"
  builds_on: []       # idea IDs or experiment names
  contradicts: []
  alternative_to: []
  notes: ""
  literature_refs: []
```

**Pause an idea**: set `status: on_hold`
**Skip an idea**: set `status: rejected`
**Auto-generate new ideas**: `/experiment-retrospective hover`

**See top ideas**:
```bash
PYTHONPATH=. python -c "
import yaml
from pathlib import Path
ideas = yaml.safe_load(Path('experiments/IDEAS.yaml').read_text())['ideas']
queued = sorted([i for i in ideas if i['status']=='queued'], key=lambda x: -x['rank'])
for i in queued[:5]:
    print(f\"  [{i['rank']:.2f}] {i['id']}: {i['title']}\")
"
```

---

## Resource manager: check + assign

**Check what's available:**
```bash
gigaevo resources --check
```

**Get assignment suggestions for 4 runs:**
```bash
gigaevo resources --assign --experiment hover/my-exp --n-runs 4
```

Output (paste into experiment.yaml Step 7):
```
servers:
  - host: INTERNAL_IP
run_db_assignments:
  R1: db=5   # INTERNAL_IP (23% GPU)
  R2: db=6   # INTERNAL_IP (23% GPU)
  R3: db=7   # INTERNAL_IP (41% GPU)
  R4: db=8   # INTERNAL_IP (41% GPU)
```

---

## Paper writing: standalone usage

Generate a paper draft from any completed experiment(s):
```
/experiment-paper-draft hover hover/dynamic-topology
```

Or from multiple experiments at once:
```
/experiment-paper-draft hover hover/dynamic-topology hover/steady-state-v2 hover/feedback-softfit
```

Output in `experiments/hover/`:
- `paper_data.json` — editable structured data
- `paper_draft.md` — assembled draft
- `paper_review.md` — adversarial critique

To convert to LaTeX:
```bash
pandoc experiments/hover/paper_draft.md -o paper.tex
```

---

## What you do vs. what the system does

| Step | You | System |
|---|---|---|
| Start experiment | `/run-experiment` or `/research-scheduler` | — |
| Literature review | — | Automatic (literature-scout) |
| Codebase recon | — | Automatic (code-archaeologist) |
| Experiment design | Review `01_design.md` | Elena + Volkov |
| Gate 1 | Reply "approved" on Telegram (or terminal) | Sends notification, waits |
| Implementation | — | Automatic |
| Alignment check | — | Automatic (implementation-aligner) |
| Gate 2 | Reply "approved" on Telegram | Sends config summary, waits |
| Monitor | (optional) Check Telegram for hourly updates | Watchdog + crons run automatically |
| Gate 3 | Reply "approved" on Telegram | Sends verdict + effect size, waits |
| Paper review | Review `paper_draft.md` + `paper_review.md` | Generates automatically |
| Next idea | — | IDEAS.yaml updated automatically |
| Next experiment | `/research-scheduler` (or pick manually) | — |

---

## Key files to know

```
experiments/
  IDEAS.yaml                    ← ranked idea queue — edit this
  INDEX.md                      ← all experiment results (canonical)
  PATTERNS.md                   ← confirmed/refuted patterns
  <task>/
    CONTEXT.md                  ← task knowledge + baselines
    <name>/
      01_design.md              ← experiment design (human reviews)
      codebase_map.md           ← NEW: code recon output
      literature_brief.md       ← literature search output
      experiment.yaml           ← machine-readable manifest
      05_results.md             ← results (human reviews)
      paper_data.json           ← NEW: structured paper data
      paper_draft.md            ← NEW: generated paper sections
      paper_review.md           ← NEW: adversarial self-critique

tools/
  telegram_notify.py            ← NEW: Telegram gates
  resource_manager.py           ← NEW: GPU/DB auto-detection

.claude/
  agents/
    code-archaeologist.md       ← NEW
    implementation-aligner.md   ← NEW
    paper-section-writer.md     ← NEW
  skills/
    idea-generate/              ← NEW
    research-scheduler/         ← NEW
    experiment-paper-draft/     ← NEW
```
