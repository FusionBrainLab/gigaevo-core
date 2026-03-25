# -*- coding: utf-8 -*-
"""
TTL Page Store Module

Provides time-to-live (TTL) functionality for page storage with automatic
expiration of old pages. Extends base page functionality with timestamp
tracking and configurable cleanup.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .page import Page


class TTLPageStore:
    """
    TTL-aware page store with automatic expiration.
    
    Features:
    - Automatic timestamp tracking in page meta
    - Configurable TTL period
    - Auto-cleanup on load (optional)
    - Manual cleanup method
    - Statistics tracking
    - Backward compatible with pages without timestamps
    
    Usage:
        # 30-day TTL with auto-cleanup
        store = TTLPageStore(dir_path="./data", ttl_days=30)
        
        # Disable auto-cleanup
        store = TTLPageStore(dir_path="./data", ttl_days=30, enable_auto_cleanup=False)
        
        # Disable TTL entirely (works like InMemoryPageStore)
        store = TTLPageStore(dir_path="./data", ttl_seconds=None)
    """
    
    def __init__(
        self,
        dir_path: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        ttl_days: Optional[int] = None,
        ttl_hours: Optional[int] = None,
        ttl_minutes: Optional[int] = None,
        enable_auto_cleanup: bool = True
    ) -> None:
        """
        Initialize TTL page store.
        
        Args:
            dir_path: Directory for persistent storage (None = in-memory only)
            ttl_seconds: TTL in seconds (overrides days/hours/minutes)
            ttl_days: TTL in days
            ttl_hours: TTL in hours
            ttl_minutes: TTL in minutes
            enable_auto_cleanup: Whether to auto-cleanup on load
            
        Note: If none of the TTL parameters are set, TTL is disabled.
        """
        self._dir_path = Path(dir_path) if dir_path else None
        self._enable_auto_cleanup = enable_auto_cleanup
        
        # Calculate total TTL in seconds
        if ttl_seconds is not None:
            self._ttl_seconds = ttl_seconds
        elif any([ttl_days, ttl_hours, ttl_minutes]):
            self._ttl_seconds = 0
            if ttl_days:
                self._ttl_seconds += ttl_days * 86400
            if ttl_hours:
                self._ttl_seconds += ttl_hours * 3600
            if ttl_minutes:
                self._ttl_seconds += ttl_minutes * 60
        else:
            self._ttl_seconds = None  # TTL disabled
        
        # Initialize pages list
        self._pages: List[Page] = []
        
        if self._dir_path:
            self._pages_file = self._dir_path / "ttl_pages.json"
            if self._pages_file.exists():
                self._pages = self._load_from_disk()
                if self._enable_auto_cleanup and self._ttl_seconds is not None:
                    self.cleanup_expired()
    
    def _load_from_disk(self) -> List[Page]:
        """Load pages from disk"""
        try:
            with open(self._pages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                pages = []
                for page_data in data:
                    # Ensure meta dict exists
                    if 'meta' not in page_data:
                        page_data['meta'] = {}
                    
                    # Add timestamp if missing (backward compatibility)
                    if 'timestamp' not in page_data['meta']:
                        page_data['meta']['timestamp'] = datetime.now(timezone.utc).isoformat()
                    
                    pages.append(Page(**page_data))
                return pages
            elif isinstance(data, dict) and 'pages' in data:
                # Handle wrapped format
                pages = []
                for page_data in data['pages']:
                    if 'meta' not in page_data:
                        page_data['meta'] = {}
                    if 'timestamp' not in page_data['meta']:
                        page_data['meta']['timestamp'] = datetime.now(timezone.utc).isoformat()
                    pages.append(Page(**page_data))
                return pages
            
            return []
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load TTL pages from {self._pages_file}: {e}")
            return []
    
    def _save_to_disk(self) -> None:
        """Save pages to disk"""
        if self._dir_path:
            self._dir_path.mkdir(parents=True, exist_ok=True)
            try:
                pages_data = [page.model_dump() for page in self._pages]
                with open(self._pages_file, 'w', encoding='utf-8') as f:
                    json.dump(pages_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save TTL pages to {self._pages_file}: {e}")
    
    def add(self, page: Page) -> None:
        """
        Add page with timestamp in meta.
        
        Args:
            page: Page object to add
        """
        # Ensure meta exists
        if page.meta is None:
            page.meta = {}
        
        # Add timestamp
        page.meta['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        self._pages.append(page)
        
        if self._dir_path:
            self._save_to_disk()
    
    def load(self) -> List[Page]:
        """
        Load pages, optionally cleaning expired entries.
        
        Returns:
            List of valid (non-expired) pages
        """
        # Auto-cleanup if enabled
        if self._enable_auto_cleanup and self._ttl_seconds is not None:
            self.cleanup_expired()
        
        return self._pages
    
    def cleanup_expired(self) -> int:
        """
        Remove expired pages based on TTL.
        
        Returns:
            Number of pages removed
        """
        if self._ttl_seconds is None:
            return 0  # TTL disabled
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._ttl_seconds)
        
        original_count = len(self._pages)
        
        # Filter to keep only non-expired pages
        valid_pages = []
        for page in self._pages:
            # Get timestamp from meta
            timestamp_str = page.meta.get('timestamp') if page.meta else None
            
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp > cutoff:
                        valid_pages.append(page)
                except (ValueError, AttributeError):
                    # Invalid timestamp, keep page (backward compat)
                    valid_pages.append(page)
            else:
                # No timestamp, keep page (backward compat)
                valid_pages.append(page)
        
        self._pages = valid_pages
        removed_count = original_count - len(self._pages)
        
        if removed_count > 0:
            print(f"TTLPageStore: Cleaned up {removed_count} expired pages")
            if self._dir_path:
                self._save_to_disk()
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about page store.
        
        Returns:
            Dictionary with total, valid, expired counts
        """
        total = len(self._pages)
        
        if self._ttl_seconds is None:
            return {
                'total': total,
                'valid': total,
                'expired': 0,
                'ttl_enabled': False
            }
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._ttl_seconds)
        
        expired_count = 0
        for page in self._pages:
            timestamp_str = page.meta.get('timestamp') if page.meta else None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp <= cutoff:
                        expired_count += 1
                except (ValueError, AttributeError):
                    pass
        
        return {
            'total': total,
            'valid': total - expired_count,
            'expired': expired_count,
            'ttl_enabled': True,
            'ttl_seconds': self._ttl_seconds
        }
    
    def save(self, pages: List[Page]) -> None:
        """
        Save pages (for compatibility with PageStore protocol).
        
        Args:
            pages: List of Page objects
        """
        # Add timestamps to all pages
        for page in pages:
            if page.meta is None:
                page.meta = {}
            if 'timestamp' not in page.meta:
                page.meta['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        self._pages = pages
        
        if self._dir_path:
            self._save_to_disk()
    
    def get(self, index: int) -> Optional[Page]:
        """
        Get page by index.
        
        Args:
            index: Page index
            
        Returns:
            Page if exists, None otherwise
        """
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None
