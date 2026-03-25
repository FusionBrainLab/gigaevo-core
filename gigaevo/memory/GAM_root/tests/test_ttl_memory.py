#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for TTLMemoryStore

Tests TTL functionality including timestamp tracking, auto-cleanup,
manual cleanup, statistics, and backward compatibility.
"""

import pytest
import tempfile
import shutil
import os
import time
from datetime import datetime, timedelta, timezone

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GAM_root.gam.schemas.ttl_memory import TTLMemoryStore, TTLMemoryEntry, TTLMemoryState
from GAM_root.gam.schemas.memory import MemoryState


class TestTTLMemoryStore:
    """Unit tests for TTLMemoryStore"""
    
    def setup_method(self):
        """Create temporary directory for each test"""
        self.tmpdir = tempfile.mkdtemp(prefix='ttl_memory_test_')
    
    def teardown_method(self):
        """Clean up temporary directory after each test"""
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
    
    def test_basic_add_and_load(self):
        """Test basic add and load functionality"""
        store = TTLMemoryStore(dir_path=self.tmpdir, ttl_days=30)
        
        # Add abstracts
        store.add("Abstract 1")
        store.add("Abstract 2")
        store.add("Abstract 3")
        
        # Load and verify
        state = store.load()
        assert len(state.abstracts) == 3
        assert "Abstract 1" in state.abstracts
        assert "Abstract 2" in state.abstracts
        assert "Abstract 3" in state.abstracts
    
    def test_duplicate_prevention(self):
        """Test that duplicate abstracts are not added"""
        store = TTLMemoryStore(dir_path=self.tmpdir, ttl_days=30)
        
        store.add("Same abstract")
        store.add("Same abstract")  # Duplicate
        store.add("Different abstract")
        
        state = store.load()
        assert len(state.abstracts) == 2  # Only 2, not 3
    
    def test_empty_abstract_ignored(self):
        """Test that empty abstracts are ignored"""
        store = TTLMemoryStore(dir_path=self.tmpdir, ttl_days=30)
        
        store.add("")
        store.add("Valid abstract")
        store.add(None)  # Should not crash
        
        state = store.load()
        assert len(state.abstracts) == 1
    
    def test_stats_calculation(self):
        """Test statistics calculation"""
        store = TTLMemoryStore(dir_path=self.tmpdir, ttl_days=30)
        
        store.add("Abstract 1")
        store.add("Abstract 2")
        
        stats = store.get_stats()
        
        assert stats['total'] == 2
        assert stats['valid'] == 2
        assert stats['expired'] == 0
        assert stats['ttl_enabled'] is True
        assert stats['ttl_seconds'] == 30 * 86400  # 30 days in seconds
    
    def test_ttl_disabled_mode(self):
        """Test that TTL can be disabled"""
        # No TTL parameters = disabled
        store = TTLMemoryStore(dir_path=self.tmpdir)
        
        store.add("Abstract 1")
        
        stats = store.get_stats()
        
        assert stats['ttl_enabled'] is False
        assert stats['total'] == 1
        assert stats['valid'] == 1
        assert stats['expired'] == 0
    
    def test_flexible_ttl_config(self):
        """Test flexible TTL configuration options"""
        # Test days
        store1 = TTLMemoryStore(dir_path=self.tmpdir + "_1", ttl_days=7)
        assert store1.get_stats()['ttl_seconds'] == 7 * 86400
        
        # Test hours
        store2 = TTLMemoryStore(dir_path=self.tmpdir + "_2", ttl_hours=12)
        assert store2.get_stats()['ttl_seconds'] == 12 * 3600
        
        # Test minutes
        store3 = TTLMemoryStore(dir_path=self.tmpdir + "_3", ttl_minutes=30)
        assert store3.get_stats()['ttl_seconds'] == 30 * 60
        
        # Test seconds (overrides others)
        store4 = TTLMemoryStore(dir_path=self.tmpdir + "_4", ttl_seconds=1000)
        assert store4.get_stats()['ttl_seconds'] == 1000
        
        # Test combination
        store5 = TTLMemoryStore(dir_path=self.tmpdir + "_5", ttl_days=1, ttl_hours=6)
        assert store5.get_stats()['ttl_seconds'] == (1 * 86400) + (6 * 3600)
    
    def test_manual_cleanup(self):
        """Test manual cleanup of expired entries"""
        # Create store with very short TTL (2 seconds)
        store = TTLMemoryStore(
            dir_path=self.tmpdir, 
            ttl_seconds=2,
            enable_auto_cleanup=False  # Disable auto-cleanup for this test
        )
        
        # Add entries
        store.add("Entry 1")
        store.add("Entry 2")
        
        # Verify they exist
        stats_before = store.get_stats()
        assert stats_before['total'] == 2
        assert stats_before['valid'] == 2
        assert stats_before['expired'] == 0
        
        # Wait for expiration
        time.sleep(2.5)
        
        # Check stats (should show expired but not removed yet)
        stats_after_wait = store.get_stats()
        assert stats_after_wait['total'] == 2
        assert stats_after_wait['valid'] == 0
        assert stats_after_wait['expired'] == 2
        
        # Manual cleanup
        removed = store.cleanup_expired()
        assert removed == 2
        
        # Verify cleanup
        stats_final = store.get_stats()
        assert stats_final['total'] == 0
        assert stats_final['valid'] == 0
        assert stats_final['expired'] == 0
    
    def test_auto_cleanup_on_load(self):
        """Test automatic cleanup when loading"""
        # Create store with short TTL and auto-cleanup enabled
        store = TTLMemoryStore(
            dir_path=self.tmpdir,
            ttl_seconds=1,
            enable_auto_cleanup=True
        )
        
        # Add entries
        store.add("Entry 1")
        store.add("Entry 2")
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Load should trigger auto-cleanup
        state = store.load()
        
        # Should be empty after auto-cleanup
        assert len(state.abstracts) == 0
    
    def test_persistence_across_sessions(self):
        """Test that data persists across sessions"""
        # Session 1: Create and add
        store1 = TTLMemoryStore(dir_path=self.tmpdir, ttl_days=30)
        store1.add("Persistent abstract 1")
        store1.add("Persistent abstract 2")
        
        # Session 2: Load in new instance
        store2 = TTLMemoryStore(dir_path=self.tmpdir, ttl_days=30)
        state = store2.load()
        
        assert len(state.abstracts) == 2
        assert "Persistent abstract 1" in state.abstracts
        assert "Persistent abstract 2" in state.abstracts
    
    def test_backward_compatibility_legacy_format(self):
        """Test loading legacy format without timestamps"""
        # Manually create legacy format file
        legacy_file = os.path.join(self.tmpdir, "ttl_memory_state.json")
        os.makedirs(self.tmpdir, exist_ok=True)
        
        import json
        legacy_data = {
            "abstracts": [
                "Legacy abstract 1",
                "Legacy abstract 2"
            ]
        }
        
        with open(legacy_file, 'w') as f:
            json.dump(legacy_data, f)
        
        # Load with TTL store
        store = TTLMemoryStore(dir_path=self.tmpdir, ttl_days=30)
        state = store.load()
        
        # Should successfully load legacy data
        assert len(state.abstracts) == 2
        assert "Legacy abstract 1" in state.abstracts
        assert "Legacy abstract 2" in state.abstracts
    
    def test_in_memory_only_mode(self):
        """Test in-memory only mode (no persistence)"""
        store = TTLMemoryStore(ttl_days=30)  # No dir_path
        
        store.add("Ephemeral abstract")
        
        state = store.load()
        assert len(state.abstracts) == 1
        
        # Should not create any files
        # (This is verified by not providing dir_path)
    
    def test_mixed_expired_and_valid(self):
        """Test handling mix of expired and valid entries"""
        store = TTLMemoryStore(
            dir_path=self.tmpdir,
            ttl_seconds=2,
            enable_auto_cleanup=False
        )
        
        # Add first batch
        store.add("Old entry 1")
        store.add("Old entry 2")
        
        # Wait for partial expiration
        time.sleep(2.5)
        
        # Add new batch (not expired)
        store.add("New entry 1")
        store.add("New entry 2")
        
        # Check stats
        stats = store.get_stats()
        assert stats['total'] == 4
        assert stats['valid'] == 2  # Only new ones
        assert stats['expired'] == 2  # Only old ones
        
        # Cleanup
        removed = store.cleanup_expired()
        assert removed == 2
        
        # Verify only new entries remain
        state = store.load()
        assert len(state.abstracts) == 2
        assert "New entry 1" in state.abstracts
        assert "New entry 2" in state.abstracts
        assert "Old entry 1" not in state.abstracts
        assert "Old entry 2" not in state.abstracts


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
