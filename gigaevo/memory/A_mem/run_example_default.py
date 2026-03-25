import os

from agentic_memory.memory_system import AgenticMemorySystem
from agent.agent_class import LLMService

# Configure OpenRouter-backed LLMService (replace placeholders with your values)
openrouter_key = os.getenv("OPENROUTER_API_KEY") or ""
llm_service = LLMService(
    service="openrouter",
    model_name="qwen/qwen3-235b-a22b",
    api_key=openrouter_key,
    temperature=0,
    max_tokens=0,
)

# Initialize the memory system 🚀
memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',  # Embedding model for ChromaDB
    llm_backend="custom",           # Use external LLMService
    llm_service=llm_service
)

# Add Memories ➕
# Simple addition
#memory_id = memory_system.add_note("Deep learning neural networks")

# Addition with metadata
memory_id = memory_system.add_note(
    content="Machine learning project notes",
    tags=["ml", "project"],
    category="Research",
    timestamp="202503021500"  # YYYYMMDDHHmm format
)

# Read (Retrieve) Memories 📖
# Get memory by ID
memory = memory_system.read(memory_id)
print(f"Content: {memory.content}")
print(f"Tags: {memory.tags}")
print(f"Context: {memory.context}")
print(f"Keywords: {memory.keywords}")

# Search memories
results = memory_system.search_agentic("neural networks", k=5)
for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Tags: {result['tags']}")
    print("---")

# Update Memories 🔄
memory_system.update(memory_id, content="info about toy story characters")

# Delete Memories ❌
#memory_system.delete(memory_id)

# Memory Evolution 🧬
# The system automatically evolves memories by:
# 1. Finding semantic relationships using ChromaDB
# 2. Updating metadata and context
# 3. Creating connections between related memories
# This happens automatically when adding or updating memories!
