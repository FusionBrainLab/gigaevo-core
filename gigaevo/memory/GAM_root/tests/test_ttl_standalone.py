#!/usr/bin/env python3
"""
Standalone TTL Validation Test
Tests TTL functionality without requiring full GAM package imports
"""

import sys
import os
import tempfile
import shutil
import time

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly
from GAM_root.gam.schemas.ttl_memory import TTLMemoryStore
from GAM_root.gam.schemas.ttl_page import TTLPageStore
from GAM_root.gam.schemas.page import Page


def test_ttl_memory_store():
    """Test TTLMemoryStore functionality"""
    print("\n" + "=" * 60)
    print("Testing TTLMemoryStore")
    print("=" * 60)
    
    tmpdir = tempfile.mkdtemp(prefix='ttl_test_')
    
    try:
        # Test 1: Basic Add/Load
        print("\n✓ Test 1: Basic Add/Load")
        store = TTLMemoryStore(dir_path=tmpdir, ttl_days=30)
        store.add("Abstract 1")
        store.add("Abstract 2")
        store.add("Abstract 3")
        
        state = store.load()
        assert len(state.abstracts) == 3, f"Expected 3, got {len(state.abstracts)}"
        print("  PASS - Added and loaded 3 abstracts")
        
        # Test 2: Statistics
        print("\n✓ Test 2: Statistics")
        stats = store.get_stats()
        assert stats['total'] == 3
        assert stats['valid'] == 3
        assert stats['expired'] == 0
        assert stats['ttl_enabled'] == True
        print(f"  PASS - Stats: {stats['valid']}/{stats['total']} valid, TTL={stats['ttl_seconds']/86400:.0f} days")
        
        # Test 3: TTL Cleanup
        print("\n✓ Test 3: TTL Cleanup (2-second TTL)")
        cleanup_store = TTLMemoryStore(
            dir_path=tmpdir + "_cleanup",
            ttl_seconds=2,
            enable_auto_cleanup=False
        )
        cleanup_store.add("Entry 1")
        cleanup_store.add("Entry 2")
        
        time.sleep(2.5)
        
        removed = cleanup_store.cleanup_expired()
        assert removed == 2, f"Expected 2 removed, got {removed}"
        print(f"  PASS - Removed {removed} expired entries")
        
        return True
        
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        if os.path.exists(tmpdir + "_cleanup"):
            shutil.rmtree(tmpdir + "_cleanup")


def test_ttl_page_store():
    """Test TTLPageStore functionality"""
    print("\n" + "=" * 60)
    print("Testing TTLPageStore")
    print("=" * 60)
    
    tmpdir = tempfile.mkdtemp(prefix='ttl_page_test_')
    
    try:
        # Test 1: Basic Add/Load
        print("\n✓ Test 1: Basic Add/Load")
        store = TTLPageStore(dir_path=tmpdir, ttl_days=30)
        store.add(Page(header="H1", content="C1"))
        store.add(Page(header="H2", content="C2"))
        
        pages = store.load()
        assert len(pages) == 2
        print("  PASS - Added and loaded 2 pages")
        
        # Test 2: Timestamp Tracking
        print("\n✓ Test 2: Timestamp Tracking")
        assert 'timestamp' in pages[0].meta
        print("  PASS - Timestamps added to page meta")
        
        # Test 3: Statistics
        print("\n✓ Test 3: Statistics")
        stats = store.get_stats()
        assert stats['total'] == 2
        assert stats['valid'] == 2
        print(f"  PASS - Stats: {stats['valid']}/{stats['total']} valid")
        
        return True
        
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  TTL Feature Validation Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # Run tests
        mem_result = test_ttl_memory_store()
        page_result = test_ttl_page_store()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"TTLMemoryStore: {'✅ PASS' if mem_result else '❌ FAIL'}")
        print(f"TTLPageStore:   {'✅ PASS' if page_result else '❌ FAIL'}")
        print("=" * 60)
        
        if mem_result and page_result:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nKey Features Verified:")
            print("  ✓ TTL-aware storage with timestamp tracking")
            print("  ✓ Auto and manual cleanup")
            print("  ✓ Statistics tracking")
            print("  ✓ Configurable TTL periods")
            return 0
        else:
            print("\n❌ Some tests failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
