#!/usr/bin/env python3
"""
TTL Feature Validation - Direct Module Import Test
Bypasses gam/__init__.py to avoid Java dependency
"""

import sys
import os
import tempfile
import shutil
import time
import importlib.util

# Direct module loading without going through __init__.py
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get base path
base_path = '/home/zsheriff/my-dev/Nuha_tool_context_framework/clone_gam/general-agentic-memory/gam/schemas'

# Load modules directly
print("Loading TTL modules...")
ttl_memory = load_module('ttl_memory', f'{base_path}/ttl_memory.py')
ttl_page = load_module('ttl_page', f'{base_path}/ttl_page.py')
page_mod = load_module('page', f'{base_path}/page.py')

print("✅ Modules loaded successfully!\n")

print('╔' + '=' * 78 + '╗')
print('║' + ' TTL FEATURE COMPREHENSIVE VALIDATION TEST SUITE '.center(78) + '║')
print('╚' + '=' * 78 + '╝')
print()

results = []
tmpdir = tempfile.mkdtemp(prefix='ttl_validation_')

def run_test(name, test_func):
    """Run a test and record result"""
    try:
        test_func()
        print(f'✅ PASS: {name}')
        results.append((name, True, None))
        return True
    except Exception as e:
        print(f'❌ FAIL: {name} - {e}')
        results.append((name, False, str(e)))
        return False

# TTLMemoryStore Tests
print('\n📦 TTLMemoryStore Tests')
print('-' * 80)

def test_basic_add_load():
    store = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_1', ttl_days=30)
    store.add('Abstract 1')
    store.add('Abstract 2')
    store.add('Abstract 3')
    state = store.load()
    assert len(state.abstracts) == 3, f'Expected 3, got {len(state.abstracts)}'

run_test('Basic add and load', test_basic_add_load)

def test_duplicates():
    store = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_2', ttl_days=30)
    store.add('Same')
    store.add('Same')
    store.add('Different')
    state = store.load()
    assert len(state.abstracts) == 2, 'Duplicates not prevented'

run_test('Duplicate prevention', test_duplicates)

def test_stats():
    store = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_3', ttl_days=30)
    store.add('A1')
    store.add('A2')
    stats = store.get_stats()
    assert stats['total'] == 2
    assert stats['valid'] == 2
    assert stats['ttl_enabled'] == True

run_test('Statistics calculation', test_stats)

def test_ttl_disabled():
    store = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_4')  # No TTL params
    store.add('A1')
    stats = store.get_stats()
    assert stats['ttl_enabled'] == False

run_test('TTL disabled mode', test_ttl_disabled)

def test_ttl_days():
    store = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_5', ttl_days=7)
    assert store.get_stats()['ttl_seconds'] == 7*86400

run_test('Flexible TTL: days', test_ttl_days)

def test_ttl_hours():
    store = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_6', ttl_hours=12)
    assert store.get_stats()['ttl_seconds'] == 12*3600

run_test('Flexible TTL: hours', test_ttl_hours)

def test_ttl_combined():
    store = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_7', ttl_days=1, ttl_hours=6)
    assert store.get_stats()['ttl_seconds'] == 86400 + 21600

run_test('Flexible TTL: combined', test_ttl_combined)

def test_manual_cleanup():
    store = ttl_memory.TTLMemoryStore(tmpdir+'_8', ttl_seconds=1, enable_auto_cleanup=False)
    store.add('E1')
    store.add('E2')
    time.sleep(1.5)
    removed = store.cleanup_expired()
    assert removed == 2, f'Expected 2 removed, got {removed}'

run_test('Manual cleanup', test_manual_cleanup)

def test_persistence():
    # Session 1
    store1 = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_9', ttl_days=30)
    store1.add('Persistent 1')
    store1.add('Persistent 2')
    
    # Session 2
    store2 = ttl_memory.TTLMemoryStore(dir_path=tmpdir+'_9', ttl_days=30)
    state = store2.load()
    assert len(state.abstracts) == 2

run_test('Persistence across sessions', test_persistence)

# TTLPageStore Tests
print('\n📄 TTLPageStore Tests')
print('-' * 80)

def test_page_add_load():
    store = ttl_page.TTLPageStore(dir_path=tmpdir+'_p1', ttl_days=30)
    store.add(page_mod.Page(header='H1', content='C1'))
    store.add(page_mod.Page(header='H2', content='C2'))
    pages = store.load()
    assert len(pages) == 2, f'Expected 2 pages, got {len(pages)}'

run_test('Page add and load', test_page_add_load)

def test_timestamp():
    store = ttl_page.TTLPageStore(dir_path=tmpdir+'_p2', ttl_days=30)
    store.add(page_mod.Page(header='H1', content='C1'))
    pages = store.load()
    assert 'timestamp' in pages[0].meta, 'No timestamp in meta'

run_test('Timestamp in meta', test_timestamp)

def test_page_stats():
    store = ttl_page.TTLPageStore(dir_path=tmpdir+'_p3', ttl_days=30)
    store.add(page_mod.Page(header='H1', content='C1'))
    store.add(page_mod.Page(header='H2', content='C2'))
    stats = store.get_stats()
    assert stats['total'] == 2
    assert stats['valid'] == 2

run_test('Page statistics', test_page_stats)

def test_page_get():
    store = ttl_page.TTLPageStore(dir_path=tmpdir+'_p4', ttl_days=30)
    store.add(page_mod.Page(header='H1', content='C1'))
    store.add(page_mod.Page(header='H2', content='C2'))
    p0 = store.get(0)
    p1 = store.get(1)
    assert p0 and p0.header == 'H1'
    assert p1 and p1.header == 'H2'

run_test('Page get() method', test_page_get)

# Cleanup
shutil.rmtree(tmpdir, ignore_errors=True)

# Results Summary
print()
print('=' * 80)
print('TEST RESULTS SUMMARY')
print('=' * 80)

passed = sum(1 for _, success, _ in results if success)
failed = sum(1 for _, success, _ in results if not success)
total = len(results)

print(f'Total Tests: {total}')
print(f'Passed: {passed} ✅')
print(f'Failed: {failed} ❌')
print(f'Success Rate: {passed/total*100:.1f}%')

if failed > 0:
    print('\nFailed Tests:')
    for name, success, error in results:
        if not success:
            print(f'  - {name}: {error}')

print('=' * 80)

if passed == total:
    print('\n🎉 ALL TESTS PASSED! TTL FEATURE FULLY VALIDATED!')
    exit(0)
else:
    print(f'\n⚠️  {failed} test(s) failed')
    exit(1)
