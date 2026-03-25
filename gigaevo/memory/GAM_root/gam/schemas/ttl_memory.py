# -*- coding: utf-8 -*-
"""
TTL Memory Store Module

Provides time-to-live (TTL) functionality for memory storage with automatic
expiration of old abstracts. Extends base memory functionality with timestamp
tracking and configurable cleanup.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


class TTLMemoryEntry(BaseModel):
    """Memory entry with timestamp for TTL tracking"""
    content: str = Field(..., description="Abstract content")
    timestamp: str = Field(..., description="ISO format timestamp")


class TTLMemoryState(BaseModel):
    """Memory state with TTL-aware entries"""
    entries: List[TTLMemoryEntry] = Field(default_factory=list, description="List of memory entries with timestamps")
    
    def to_abstracts(self) -> List[str]:
        """Convert to simple list of abstracts for compatibility"""
        return [entry.content for entry in self.entries]


class TTLMemoryStore:
    """
    TTL-aware memory store with automatic expiration.
    
    Features:
    - Automatic timestamp tracking on add
    - Configurable TTL period
    - Auto-cleanup on load (optional)
    - Manual cleanup method
    - Statistics tracking
    - Backward compatible data format
    
    Usage:
        # 30-day TTL with auto-cleanup
        store = TTLMemoryStore(dir_path="./data", ttl_days=30)
        
        # Disable auto-cleanup
        store = TTLMemoryStore(dir_path="./data", ttl_days=30, enable_auto_cleanup=False)
        
        # Disable TTL entirely (works like InMemoryMemoryStore)
        store = TTLMemoryStore(dir_path="./data", ttl_seconds=None)
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
        Initialize TTL memory store.
        
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
        
        # Initialize state
        self._state = TTLMemoryState()
        
        if self._dir_path:
            self._memory_file = self._dir_path / "ttl_memory_state.json"
            if self._memory_file.exists():
                self._state = self._load_from_disk()
                if self._enable_auto_cleanup and self._ttl_seconds is not None:
                    self.cleanup_expired()
    
    def _load_from_disk(self) -> TTLMemoryState:
        """Load state from disk, handling both new and legacy formats"""
        try:
            with open(self._memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle new format (with entries)
            if isinstance(data, dict) and 'entries' in data:
                return TTLMemoryState(**data)
            
            # Handle legacy format (just abstracts list) - backward compatibility
            elif isinstance(data, dict) and 'abstracts' in data:
                # Convert legacy format to TTL format (no timestamps = never expire)
                entries = [
                    TTLMemoryEntry(
                        content=abstract,
                        timestamp=datetime.now(timezone.utc).isoformat()
                    )
                    for abstract in data['abstracts']
                ]
                return TTLMemoryState(entries=entries)
            
            # Handle direct list format
            elif isinstance(data, list):
                entries = [
                    TTLMemoryEntry(
                        content=item if isinstance(item, str) else item.get('content', ''),
                        timestamp=item.get('timestamp', datetime.now(timezone.utc).isoformat()) 
                        if isinstance(item, dict) else datetime.now(timezone.utc).isoformat()
                    )
                    for item in data
                ]
                return TTLMemoryState(entries=entries)
            
            return TTLMemoryState()
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to load TTL memory state from {self._memory_file}: {e}")
            return TTLMemoryState()
    
    def _save_to_disk(self) -> None:
        """Save current state to disk"""
        if self._dir_path:
            self._dir_path.mkdir(parents=True, exist_ok=True)
            try:
                with open(self._memory_file, 'w', encoding='utf-8') as f:
                    json.dump(self._state.model_dump(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save TTL memory state to {self._memory_file}: {e}")
    
    def add(self, abstract: str) -> None:
        """
        Add abstract with current timestamp.
        
        Args:
            abstract: Abstract text to add
        """
        if not abstract:
            return
        
        # Check for duplicates
        existing_contents = {entry.content for entry in self._state.entries}
        if abstract in existing_contents:
            return
        
        # Add with timestamp
        entry = TTLMemoryEntry(
            content=abstract,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        self._state.entries.append(entry)
        
        if self._dir_path:
            self._save_to_disk()
    
    def load(self) -> Any:
        """
        Load memory state, optionally cleaning expired entries.
        
        Returns:
            MemoryState compatible object (has .abstracts attribute)
        """
        # Auto-cleanup if enabled
        if self._enable_auto_cleanup and self._ttl_seconds is not None:
            self.cleanup_expired()
        
        # Return compatible object
        from .memory import MemoryState
        return MemoryState(abstracts=self._state.to_abstracts())
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries based on TTL.
        
        Returns:
            Number of entries removed
        """
        if self._ttl_seconds is None:
            return 0  # TTL disabled
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._ttl_seconds)
        
        original_count = len(self._state.entries)
        
        # Filter to keep only non-expired entries
        self._state.entries = [
            entry for entry in self._state.entries
            if datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00')) > cutoff
        ]
        
        removed_count = original_count - len(self._state.entries)
        
        if removed_count > 0:
            print(f"TTLMemoryStore: Cleaned up {removed_count} expired entries")
            if self._dir_path:
                self._save_to_disk()
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory store.
        
        Returns:
            Dictionary with total, valid, expired counts
        """
        total = len(self._state.entries)
        
        if self._ttl_seconds is None:
            return {
                'total': total,
                'valid': total,
                'expired': 0,
                'ttl_enabled': False
            }
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._ttl_seconds)
        
        expired_count = sum(
            1 for entry in self._state.entries
            if datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00')) <= cutoff
        )
        
        return {
            'total': total,
            'valid': total - expired_count,
            'expired': expired_count,
            'ttl_enabled': True,
            'ttl_seconds': self._ttl_seconds
        }
    
    def save(self, state: Any) -> None:
        """
        Save memory state (for compatibility with MemoryStore protocol).
        
        Args:
            state: MemoryState object with abstracts
        """
        # Convert MemoryState to TTLMemoryState
        self._state.entries = [
            TTLMemoryEntry(
                content=abstract,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            for abstract in state.abstracts
        ]
        
        if self._dir_path:
            self._save_to_disk()
