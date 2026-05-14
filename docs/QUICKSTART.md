# GigaEvo Quick Start Guide

This guide gets you from zero to running evolution in **5 minutes**.

## Prerequisites

- Python 3.12+
- Redis server running
- OpenRouter API key (or other LLM provider)

## Step 1: Install (30 seconds)

```bash
# Clone and install
pip install -e .

# Create .env file
echo "OPENAI_API_KEY=sk-or-v1-your-key-here" > .env
```

## Step 2: Start Redis (10 seconds)

```bash
# In a separate terminal
redis-server
```

## Step 3: Run Your First Evolution (5 seconds to start)

```bash
# Run the heilbron problem (triangle packing)
python run.py problem.name=heilbron max_mutants=5
```

You should see:
```
[INFO] GigaEvo Evolution Experiment
[INFO] Problem: heilbron
[INFO] Loading initial programs...
[INFO] Loaded 5 initial programs
[INFO] Starting evolution...
```

**Congratulations!** Evolution is running. 🎉

## What's Happening?

1. **Initial Programs**: 5 seed programs loaded from `problems/heilbron/initial_programs/`
2. **Evaluation**: Each program is evaluated (runs its `entrypoint()` function)
3. **Mutation**: LLM mutates the best programs to create new ones
4. **Selection**: Programs that improve fitness are kept in the archive
5. **Repeat**: The steady-state engine continuously mutates and ingests until
   the configured stopper (here `max_mutants=5`) fires

## Step 4: Inspect Results (while evolution runs)

Open a new terminal:

```bash
# Show current run status (fitness, gen count, invalidity, etc.)
# Preferred: use --experiment mode (auto-discovers runs from experiment.yaml)
gigaevo status -e <task>/<name>
# Or specify a single run directly:
gigaevo status -r heilbron@0:run-1

# Show top N programs
gigaevo top -r heilbron@0:run-1 -n 5

# Export results to CSV
gigaevo export csv -r heilbron@0:run-1
```

## Step 5: View Evolution Logs

```bash
# Logs are in outputs/YYYY-MM-DD/HH-MM-SS/
tail -f outputs/*/*/evolution_*.log
```

## Step 6: Analyze Results

After evolution completes:

```bash
# Export to CSV
gigaevo export csv -r heilbron@0:run-1

# Compare fitness curves across runs (pass multiple --run flags; -o is required)
gigaevo plot comparison -r heilbron@0:run-1 -r heilbron@1:run-2 -o plots/

# View top programs
gigaevo top -r heilbron@0:run-1 -n 10
```

## Understanding the Output

### Console Output

```
[INFO] Step 1/5: Initializing components... ✓
[INFO] Step 2/5: Checking Redis database... ✓
[INFO] Step 3/5: Loading initial programs... (5 programs) ✓
[INFO] Step 4/5: Starting evolution... ✓
[INFO] Step 5/5: Running until completion...

[INFO] [SteadyState] Start | producer_sema=N buffer_sema=N (max_in_flight=N) ...
[INFO] [EvolutionEngine] Init | strategy=..., acceptor=..., stopper=MaxMutantsStopper
[INFO] [SteadyState] Dispatcher / Ingestor running — continuous mutation + ingest
```

### Key Metrics to Watch

- **Added**: Programs accepted into the archive (good!)
- **Rejected**: Programs that didn't improve any cell (normal)
- **Fitness**: The main objective value (higher is better for heilbron)

## Common First-Time Issues

### Issue: "Redis database is not empty"

**Solution:**
```bash
# Flush the database (kills exec_runner workers first):
gigaevo flush --db 0 --confirm
# Or use a different database:
python run.py problem.name=heilbron redis.db=1
```

### Issue: "No programs reaching DONE state"

**Cause**: Programs might be failing validation (state machine:
`QUEUED → RUNNING → DONE` or `→ DISCARDED`).

**Solution:**
```bash
# Check invalidity rate
gigaevo status -r <prefix>@<db>:<label>

# View top programs and their fitness
gigaevo top -r <prefix>@<db>:<label> -n 10
```

### Issue: Evolution seems slow

**Cause**: LLM API calls take time.

**What's normal**:
- Initial evaluation: ~30 seconds per program
- Mutation creation: ~10-30 seconds per mutant
- Generation cycle: ~2-5 minutes

**Speed it up**:
- Use faster LLM models
- Increase `max_in_flight` (concurrent mutation + ingest tasks; see
  `config/constants/evolution.yaml`) — beware of LLM rate limits
- Increase `max_concurrent_dags` (DAG runner parallelism)

## Next Steps

### 1. Create Your Own Problem

```bash
# Copy the heilbron template
cp -r problems/heilbron problems/my_problem

# Edit the key files:
# - problems/my_problem/validate.py      (fitness function; can return (metrics_dict, artifact) for mutation context)
# - problems/my_problem/metrics.yaml     (metric definitions)
# - problems/my_problem/initial_programs/ (seed programs)
# - problems/my_problem/task_description.txt (LLM instructions)
```

### 2. Customize Evolution

```bash
# Try multi-island evolution
python run.py experiment=multi_island_complexity problem.name=heilbron

# Use different LLM models
python run.py experiment=multi_llm_exploration problem.name=heilbron

# Adjust parameters
python run.py problem.name=heilbron \
    max_mutants=20 \
    max_in_flight=15 \
    model_name=anthropic/claude-3.5-sonnet
```

### 3. Read the Documentation

- **Architecture**: `docs/ARCHITECTURE.md` - Understand the system design
- **DAG System**: `docs/DAG_SYSTEM.md` - Learn about pipelines
- **Evolution Strategies**: `docs/EVOLUTION_STRATEGIES.md` - Learn about MAP-Elites
- **Contributing**: `docs/CONTRIBUTING.md` - Development guidelines

### 4. Explore Examples

```bash
# View all available experiments
ls config/experiment/

# View all available problems
ls problems/

# View available LLM configurations
ls config/llm/
```

## Quick Reference Commands

```bash
# Run evolution
python run.py problem.name=<problem>

# Run with config override
python run.py problem.name=<problem> max_generations=10

# Use different experiment
python run.py experiment=<experiment> problem.name=<problem>

# Preview config (no execution)
python run.py problem.name=<problem> --cfg job

# Check run status (--experiment mode preferred for managed experiments)
gigaevo status -e <task>/<name>
# Or for a single run:
gigaevo status -r <prefix>@<db>:<label>

# View top programs
gigaevo top -r <prefix>@<db>:<label> -n 10

# Export results to CSV
gigaevo export csv -r <prefix>@<db>:<label>

# Compare fitness curves across runs (-o output dir required)
gigaevo plot comparison -r <prefix>@<db>:A -r <prefix>@<db>:B -o plots/

# Flush Redis (kills exec_runners first — never use redis-cli FLUSHDB directly)
gigaevo flush --db 0 --confirm

# View logs
tail -f outputs/*/evolution_*.log
```

## Getting Help

1. **Check logs**: Most issues are explained in the logs
2. **Check run status**: `gigaevo status -r <prefix>@<db>:<label>`
3. **Read architecture doc**: `docs/ARCHITECTURE.md` explains the system
4. **Check examples**: Look at existing problems in `problems/`

## What You Just Learned

✅ How to run evolution
✅ How to inspect evolution state
✅ How to debug common issues
✅ Where to find logs and results

## Recommended Learning Path

1. **Day 1**: Run existing problems, inspect results
2. **Day 2**: Read `docs/ARCHITECTURE.md`, understand the flow
3. **Day 3**: Create your own simple problem
4. **Day 4**: Customize pipeline (add custom stages)
5. **Day 5**: Experiment with multi-island evolution

**Happy Evolving!** 🚀
