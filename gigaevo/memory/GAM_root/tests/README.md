# Tests Directory

This directory contains comprehensive tests for the TTL (Time-To-Live) feature.

## Test Files

### Unit Tests

1. **test_ttl_memory.py** - Unit tests for TTLMemoryStore
   - Basic add/load functionality
   - Duplicate prevention
   - Statistics calculation
   - TTL configuration (days/hours/minutes/seconds)
   - Auto-cleanup and manual cleanup
   - Persistence across sessions
   - Backward compatibility with legacy format
   - Mixed expired/valid entries

2. **test_ttl_page.py** - Unit tests for TTLPageStore
   - Basic add/load functionality
   - Timestamp tracking in meta
   - Statistics calculation
   - Auto-cleanup and manual cleanup
   - get() method for index retrieval
   - Persistence across sessions
   - Backward compatibility
   - Mixed expired/valid pages

### Integration Tests

3. **test_ttl_before_after.py** - Before/After comparison demonstration
   - Shows unbounded growth problem (Before TTL)
   - Shows controlled growth solution (After TTL)
   - Demonstrates TTL cleanup in action
   - Comparison summary with metrics

## Running Tests

### Run Before/After Comparison (Visual Demo)

```bash
source /home/zsheriff/my-dev/Nuha_tool_context_framework/agent_mem/bin/activate
cd /home/zsheriff/my-dev/Nuha_tool_context_framework/clone_gam/general-agentic-memory
python3 tests/test_ttl_before_after.py
```

### Run Unit Tests (Requires pytest)

```bash
# Install pytest if needed
pip install pytest

# Run all TTL tests
pytest tests/test_ttl_memory.py -v
pytest tests/test_ttl_page.py -v

# Run all tests
pytest tests/ -v
```

## Test Coverage

- ✅ Basic functionality (add, load, save)
- ✅ TTL configuration (flexible timing)
- ✅ Auto-cleanup on load
- ✅ Manual cleanup
- ✅ Statistics tracking
- ✅ Persistence
- ✅ Backward compatibility
- ✅ Edge cases (empty, expired, mixed)
- ✅ Before/after metrics

## Expected Results

All tests should pass, demonstrating:
- TTL stores work correctly
- Automatic cleanup functions
- Statistics are accurate
- Backward compatibility maintained
- Improvement over non-TTL stores
