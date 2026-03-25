#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for TTLPageStore

Tests TTL functionality for page storage including timestamp tracking,
auto-cleanup, manual cleanup, and backward compatibility.
"""

import pytest
import tempfile
import shutil
import os
import time
from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GAM_root.gam.schemas.ttl_page import TTLPageStore
from GAM_root.gam.schemas.page import Page


class TestTTLPageStore:
    """Unit tests for TTLPageStore"""
    
    def setup_method(self):
        """Create temporary directory for each test"""
        self.tmpdir = tempfile.mkdtemp(prefix='ttl_page_test_')
    
    def teardown_method(self):
        """Clean up temporary directory after each test"""
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
    
    def test_basic_add_and_load(self):
        """Test basic add and load functionality"""
        store = TTLPageStore(dir_path=self.tmpdir, ttl_days=30)
        
        # Add pages
        store.add(Page(header="Header 1", content="Content 1"))
        store.add(Page(header="Header 2", content="Content 2"))
        store.add(Page(header="Header 3", content="Content 3"))
        
        # Load and verify
        pages = store.load()
        assert len(pages) == 3
        assert pages[0].header == "Header 1"
        assert pages[1].header == "Header 2"
        assert pages[2].header == "Header 3"
    
    def test_timestamp_added_to_meta(self):
        """Test that timestamp is automatically added to page meta"""
        store = TTLPageStore(dir_path=self.tmpdir, ttl_days=30)
        
        # Create page without timestamp
        page = Page(header="Test", content="Content")
        store.add(page)
        
        # Load and check timestamp exists
        pages = store.load()
        assert len(pages) == 1
        assert 'timestamp' in pages[0].meta
        
        # Verify timestamp is valid ISO format
        timestamp_str = pages[0].meta['timestamp']
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))  # Should not raise
    
    def test_stats_calculation(self):
        """Test statistics calculation"""
        store = TTLPageStore(dir_path=self.tmpdir, ttl_days=30)
        
        store.add(Page(header="H1", content="C1"))
        store.add(Page(header="H2", content="C2"))
        
        stats = store.get_stats()
        
        assert stats['total'] == 2
        assert stats['valid'] == 2
        assert stats['expired'] == 0
        assert stats['ttl_enabled'] is True
        assert stats['ttl_seconds'] == 30 * 86400
    
    def test_ttl_disabled_mode(self):
        """Test that TTL can be disabled"""
        store = TTLPageStore(dir_path=self.tmpdir)  # No TTL params
        
        store.add(Page(header="H1", content="C1"))
        
        stats = store.get_stats()
        assert stats['ttl_enabled'] is False
        assert stats['total'] == 1
    
    def test_manual_cleanup(self):
        """Test manual cleanup of expired pages"""
        store = TTLPageStore(
            dir_path=self.tmpdir,
            ttl_seconds=2,
            enable_auto_cleanup=False
        )
        
        # Add pages
        store.add(Page(header="H1", content="C1"))
        store.add(Page(header="H2", content="C2"))
        
        # Wait for expiration
        time.sleep(2.5)
        
        # Manual cleanup
        removed = store.cleanup_expired()
        assert removed == 2
        
        # Verify cleanup
        pages = store.load()
        assert len(pages) == 0
    
    def test_auto_cleanup_on_load(self):
        """Test automatic cleanup when loading"""
        store = TTLPageStore(
            dir_path=self.tmpdir,
            ttl_seconds=1,
            enable_auto_cleanup=True
        )
        
        # Add pages
        store.add(Page(header="H1", content="C1"))
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Load triggers auto-cleanup
        pages = store.load()
        assert len(pages) == 0
    
    def test_get_method(self):
        """Test get method for retrieving page by index"""
        store = TTLPageStore(dir_path=self.tmpdir, ttl_days=30)
        
        store.add(Page(header="H1", content="C1"))
        store.add(Page(header="H2", content="C2"))
        store.add(Page(header="H3", content="C3"))
        
        # Test valid indices
        page0 = store.get(0)
        assert page0 is not None
        assert page0.header == "H1"
        
        page2 = store.get(2)
        assert page2 is not None
        assert page2.header == "H3"
        
        # Test invalid indices
        assert store.get(-1) is None
        assert store.get(10) is None
    
    def test_persistence_across_sessions(self):
        """Test that pages persist across sessions"""
        # Session 1
        store1 = TTLPageStore(dir_path=self.tmpdir, ttl_days=30)
        store1.add(Page(header="Persistent", content="Data"))
        
        # Session 2
        store2 = TTLPageStore(dir_path=self.tmpdir, ttl_days=30)
        pages = store2.load()
        
        assert len(pages) == 1
        assert pages[0].header == "Persistent"
    
    def test_backward_compatibility_without_timestamp(self):
        """Test loading pages without timestamps (legacy format)"""
        # Manually create legacy format file
        legacy_file = os.path.join(self.tmpdir, "ttl_pages.json")
        os.makedirs(self.tmpdir, exist_ok=True)
        
        import json
        legacy_data = [
            {
                "header": "Legacy header",
                "content": "Legacy content",
                "meta": {}  # No timestamp
            }
        ]
        
        with open(legacy_file, 'w') as f:
            json.dump(legacy_data, f)
        
        # Load with TTL store
        store = TTLPageStore(dir_path=self.tmpdir, ttl_days=30)
        pages = store.load()
        
        # Should successfully load and add timestamp
        assert len(pages) == 1
        assert pages[0].header == "Legacy header"
        assert 'timestamp' in pages[0].meta  # Timestamp added automatically
    
    def test_mixed_expired_and_valid(self):
        """Test handling mix of expired and valid pages"""
        store = TTLPageStore(
            dir_path=self.tmpdir,
            ttl_seconds=2,
            enable_auto_cleanup=False
        )
        
        # Add old pages
        store.add(Page(header="Old1", content="C1"))
        store.add(Page(header="Old2", content="C2"))
        
        # Wait
        time.sleep(2.5)
        
        # Add new pages
        store.add(Page(header="New1", content="C3"))
        store.add(Page(header="New2", content="C4"))
        
        # Cleanup
        removed = store.cleanup_expired()
        assert removed == 2
        
        # Verify only new pages remain
        pages = store.load()
        assert len(pages) == 2
        headers = [p.header for p in pages]
        assert "New1" in headers
        assert "New2" in headers
        assert "Old1" not in headers


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_tests()
