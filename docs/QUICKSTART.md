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
python run.py problem.name=heilbron max_generations=5
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
5. **Repeat**: Continues for 5 generations

## Step 4: Inspect Results (while evolution runs)

Open a new terminal:

```bash
# Show current state
python tools/inspect.py --summary

# Check if evolution is stuck
python tools/inspect.py --stuck-check

# View a specific program
python tools/inspect.py --program <program-id>

# See island contents
python tools/inspect.py --island main_island
```

## Step 5: View Evolution Logs

```bash
# Logs are in outputs/
tail -f outputs/2025-11-11/*/evolution_*.log
```

## Step 6: Analyze Results

After evolution completes:

```bash
# Export to CSV
python tools/redis2pd.py --output results.csv

# View best program
redis-cli GET "program:<best-program-id>" | jq .
```

## Understanding the Output

### Console Output

```
[INFO] Step 1/5: Initializing components... ✓
[INFO] Step 2/5: Checking Redis database... ✓
[INFO] Step 3/5: Loading initial programs... (5 programs) ✓
[INFO] Step 4/5: Starting evolution... ✓
[INFO] Step 5/5: Running until completion...

[INFO] [EvolutionEngine] Phase 1: Idle confirmed
[INFO] [EvolutionEngine] Phase 2: Created 10 mutant(s)
[INFO] [EvolutionEngine] Phase 3: Mutant DAGs finished
[INFO] [EvolutionEngine] Phase 4: Ingestion done (added=3, rejected=7)
[INFO] [EvolutionEngine] Phase 5: Refreshed 8 program(s)
```

### Key Metrics to Watch

- **Added**: Programs accepted into the archive (good!)
- **Rejected**: Programs that didn't improve any cell (normal)
- **Fitness**: The main objective value (higher is better for heilbron)

## Common First-Time Issues

### Issue: "Redis database is not empty"

**Solution:**
```bash
redis-cli FLUSHDB
# Or use a different database:
python run.py problem.name=heilbron redis.db=1
```

### Issue: "No programs in EVOLVING state"

**Cause**: Programs might be failing validation.

**Solution:**
```bash
# Check what's failing
python tools/inspect.py --stuck-check

# View a specific program's errors
python tools/inspect.py --program <program-id>
```

### Issue: Evolution seems slow

**Cause**: LLM API calls take time.

**What's normal**:
- Initial evaluation: ~30 seconds per program
- Mutation creation: ~10-30 seconds per mutant
- Generation cycle: ~2-5 minutes

**Speed it up**:
- Reduce `max_mutations_per_generation` in config
- Use faster LLM models
- Increase `max_concurrent_dags` (but beware of rate limits)

## Next Steps

### 1. Create Your Own Problem

```bash
# Copy the heilbron template
cp -r problems/heilbron problems/my_problem

# Edit the key files:
# - problems/my_problem/validate.py      (your fitness function)
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
    max_generations=20 \
    max_mutations_per_generation=15 \
    model_name=anthropic/claude-3.5-sonnet
```

### 3. Read the Documentation

- **Architecture**: `ARCHITECTURE.md` - Understand the system design
- **DAG System**: `DAG_SYSTEM.md` - Learn about pipelines
- **Evolution Strategies**: `EVOLUTION_STRATEGIES.md` - Learn about MAP-Elites
- **Contributing**: `CONTRIBUTING.md` - Development guidelines

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

# Inspect evolution state
python tools/inspect.py --summary
python tools/inspect.py --stuck-check

# Export results
python tools/redis2pd.py --output results.csv

# Clear Redis
redis-cli FLUSHDB

# View logs
tail -f outputs/*/evolution_*.log

# List Redis keys
redis-cli KEYS "*" | head -20
```

## Getting Help

1. **Check logs**: Most issues are explained in the logs
2. **Use inspection tool**: `python tools/inspect.py --stuck-check`
3. **Read architecture doc**: `ARCHITECTURE.md` explains the system
4. **Check examples**: Look at existing problems in `problems/`

## What You Just Learned

✅ How to run evolution
✅ How to inspect evolution state
✅ How to debug common issues
✅ Where to find logs and results

## Recommended Learning Path

1. **Day 1**: Run existing problems, inspect results
2. **Day 2**: Read `ARCHITECTURE.md`, understand the flow
3. **Day 3**: Create your own simple problem
4. **Day 4**: Customize pipeline (add custom stages)
5. **Day 5**: Experiment with multi-island evolution

**Happy Evolving!** 🚀
