#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTL Usage Example

Demonstrates how to use TTL (Time-To-Live) functionality to automatically
clean up old memory entries and pages in long-running applications.
"""

import os
import time
from datetime import datetime, timedelta

from gam import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    TTLMemoryStore,
    TTLPageStore,
    IndexRetriever,
    IndexRetrieverConfig,
)


def basic_ttl_example():
    """Basic TTL usage with 30-day expiration"""
    print("=== Basic TTL Example ===\n")
    
    # 1. Configure generator
    gen_config = OpenAIGeneratorConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
        max_tokens=256
    )
    generator = OpenAIGenerator.from_config(gen_config)
    
    # 2. Create TTL stores with 30-day expiration
    memory_store = TTLMemoryStore(
        dir_path="./ttl_demo_data",
        ttl_days=30,
        enable_auto_cleanup=True
    )
    page_store = TTLPageStore(
        dir_path="./ttl_demo_data",
        ttl_days=30,
        enable_auto_cleanup=True
    )
    
    print(f"✅ Created TTL stores with 30-day expiration")
    print(f"   Auto-cleanup: Enabled\n")
    
    # 3. Create MemoryAgent with TTL stores
    memory_agent = MemoryAgent(
        generator=generator,
        memory_store=memory_store,
        page_store=page_store
    )
    
    # 4. Add some memories
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "TensorFlow is an open-source machine learning framework developed by Google.",
        "Docker enables containerization of applications for consistent deployments.",
    ]
    
    print(f"Adding {len(documents)} documents...")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. Memorizing...")
        memory_agent.memorize(doc)
    
    # 5. Check statistics
    mem_stats = memory_store.get_stats()
    page_stats = page_store.get_stats()
    
    print(f"\n📊 Statistics:")
    print(f"   Memory Store: {mem_stats['valid']}/{mem_stats['total']} valid")
    print(f"   Page Store: {page_stats['valid']}/{page_stats['total']} valid")
    print(f"   TTL: {mem_stats['ttl_seconds']/86400:.0f} days\n")
    
    return memory_store, page_store


def manual_cleanup_example():
    """Demonstrate manual cleanup"""
    print("\n=== Manual Cleanup Example ===\n")
    
    # Create store with very short TTL (1 minute) and auto-cleanup disabled
    memory_store = TTLMemoryStore(
        dir_path="./ttl_manual_demo",
        ttl_minutes=1,
        enable_auto_cleanup=False  # Manual cleanup only
    )
    
    print("✅ Created store with 1-minute TTL (auto-cleanup disabled)")
    
    # Add some test data
    memory_store.add("Test entry 1")
    memory_store.add("Test entry 2")
    memory_store.add("Test entry 3")
    
    print(f"   Added 3 entries\n")
    
    # Wait a bit
    print("⏳ Waiting 65 seconds for entries to expire...")
    time.sleep(65)
    
    # Check stats before cleanup
    stats_before = memory_store.get_stats()
    print(f"\n📊 Before cleanup:")
    print(f"   Total: {stats_before['total']}")
    print(f"   Valid: {stats_before['valid']}")
    print(f"   Expired: {stats_before['expired']}")
    
    # Manual cleanup
    removed = memory_store.cleanup_expired()
    
    # Check stats after cleanup
    stats_after = memory_store.get_stats()
    print(f"\n🧹 After cleanup:")
    print(f"   Removed: {removed} entries")
    print(f"   Remaining: {stats_after['total']}")


def ttl_disabled_example():
    """TTL disabled works like regular store"""
    print("\n=== TTL Disabled Example ===\n")
    
    # No TTL parameters = TTL disabled
    memory_store = TTLMemoryStore(dir_path="./ttl_disabled_demo")
    
    memory_store.add("Entry 1")
    memory_store.add("Entry 2")
    
    stats = memory_store.get_stats()
    
    print(f"✅ Created store without TTL")
    print(f"📊 Statistics:")
    print(f"   Total: {stats['total']}")
    print(f"   TTL Enabled: {stats['ttl_enabled']}")
    print(f"   (Works like regular InMemoryMemoryStore)")


def flexible_ttl_example():
    """Different TTL configurations"""
    print("\n=== Flexible TTL Configuration ===\n")
    
    # Various TTL configurations
    configs = [
        ("1 day", {"ttl_days": 1}),
        ("12 hours", {"ttl_hours": 12}),
        ("30 minutes", {"ttl_minutes": 30}),
        ("7 days + 6 hours", {"ttl_days": 7, "ttl_hours": 6}),
        ("2592000 seconds (30 days)", {"ttl_seconds": 2592000}),
    ]
    
    print("Available TTL configurations:")
    for label, config in configs:
        store = TTLMemoryStore(dir_path=f"./ttl_demo_{label.replace(' ', '_')}", **config)
        stats = store.get_stats()
        ttl_days = stats['ttl_seconds'] / 86400
        print(f"   ✓ {label:30s} = {ttl_days:6.2f} days")


def backward_compatibility_example():
    """Demonstrate backward compatibility"""
    print("\n=== Backward Compatibility Example ===\n")
    
    # 1. Create data with regular store
    from gam import InMemoryMemoryStore
    
    print("1. Creating data with InMemoryMemoryStore...")
    regular_store = InMemoryMemoryStore(dir_path="./ttl_compat_demo")
    regular_store.add"Old abstract 1")
    regular_store.add("Old abstract 2")
    print(f"   Added 2 abstracts\n")
    
    # 2. Load with TTL store (should work!)
    print("2. Loading with TTLMemoryStore...")
    ttl_store = TTLMemoryStore(
        dir_path="./ttl_compat_demo",
        ttl_days=30
    )
    
    state = ttl_store.load()
    print(f"   ✅ Successfully loaded {len(state.abstracts)} abstracts")
    print(f"   (Old data without timestamps preserved)")


def main():
    """Run all examples"""
    print("=" * 60)
    print("GAM TTL (Time-To-Live) Examples")
    print("=" * 60)
    print()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set. Skipping full example.")
        print("   Running TTL-only examples...\n")
        
        manual_cleanup_example()
        ttl_disabled_example()
        flexible_ttl_example()
        backward_compatibility_example()
        
        print("\n" + "=" * 60)
        print("✅ TTL Examples Complete!")
        print("=" * 60)
        print("\nTo run full example with MemoryAgent:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # Run all examples
        basic_ttl_example()
        manual_cleanup_example()
        ttl_disabled_example()
        flexible_ttl_example()
        backward_compatibility_example()
        
        print("\n" + "=" * 60)
        print("✅ All TTL Examples Complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("  • TTL stores automatically clean up old data")
        print("  • Auto-cleanup runs on load() by default")
        print("  • Manual cleanup available: cleanup_expired()")
        print("  • get_stats() shows total/valid/expired counts")
        print("  • TTL disabled (no params) = regular store behavior")
        print("  • Fully backward compatible with existing data")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
