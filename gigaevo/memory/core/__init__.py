"""Modular efficacy core: reputation, admission, auction, and the read/write
pipelines as composable units, every threshold a constructor parameter for
Hydra composition.

Auction draw order (theta then baseline, per candidate) is a seed-exact
contract pinned in tests/memory/test_core_efficacy.py. The renderer adds a
``mechanism:`` line and the budgeter enforces ``max_cards`` post-auction.
"""

from gigaevo.memory.core.admitter import PermissiveAdmitter, SignBasedAdmitter
from gigaevo.memory.core.auctioneer import (
    AuctionBid,
    AuctionCandidate,
    EVThompsonAuctioneer,
    ThompsonAuctioneer,
)
from gigaevo.memory.core.budgeter import TopBidBudgeter, TopThetaBudgeter
from gigaevo.memory.core.card_selector import LLMCardSelector
from gigaevo.memory.core.deduplicator import LLMDeduplicator, NullDeduplicator
from gigaevo.memory.core.events import (
    DEFAULT_MEMORY_EVENTS_FILENAME,
    MEMORY_EVENT_SCHEMA_VERSION,
    MemoryEventRecord,
    emit_memory_event,
    memory_event_context,
    new_memory_decision_id,
    resolve_memory_event_path,
)
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.excluder import LineageExcluder, NullExcluder
from gigaevo.memory.core.protocols import (
    Auctioneer,
    Budgeter,
    CardExcluder,
    CardRenderer,
    CardRetriever,
    CardShortlister,
    Deduplicator,
    Evictor,
    MemoryAdmitter,
    ReputationModel,
)
from gigaevo.memory.core.random_drop import RandomDropExcluder
from gigaevo.memory.core.read_pipeline import MemoryReadPipeline
from gigaevo.memory.core.renderer import EfficacyCardRenderer
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.retriever import GamRetriever
from gigaevo.memory.core.selection import MemorySelection
from gigaevo.memory.core.write_ledger import WriteLedger, WriteLedgerRecord
from gigaevo.memory.core.write_pipeline import MemoryWritePipeline

__all__ = [
    "AuctionBid",
    "AuctionCandidate",
    "Auctioneer",
    "BetaBinomialReputation",
    "Budgeter",
    "CardExcluder",
    "CardRenderer",
    "CardRetriever",
    "CardShortlister",
    "Deduplicator",
    "EfficacyCardRenderer",
    "Evictor",
    "DEFAULT_MEMORY_EVENTS_FILENAME",
    "GamRetriever",
    "HarmEvictor",
    "LLMCardSelector",
    "LLMDeduplicator",
    "LineageExcluder",
    "MemoryAdmitter",
    "MEMORY_EVENT_SCHEMA_VERSION",
    "MemoryEventRecord",
    "MemoryReadPipeline",
    "MemorySelection",
    "MemoryWritePipeline",
    "NullDeduplicator",
    "NullExcluder",
    "PermissiveAdmitter",
    "RandomDropExcluder",
    "ReputationModel",
    "SignBasedAdmitter",
    "EVThompsonAuctioneer",
    "ThompsonAuctioneer",
    "TopBidBudgeter",
    "TopThetaBudgeter",
    "WriteLedger",
    "WriteLedgerRecord",
    "emit_memory_event",
    "memory_event_context",
    "new_memory_decision_id",
    "resolve_memory_event_path",
]
