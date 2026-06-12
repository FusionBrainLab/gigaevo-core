"""Modular efficacy core: reputation, admission, auction, and the read/write
pipelines as composable units, every threshold a constructor parameter for
Hydra composition.

Auction draw order (theta then baseline, per candidate) is a seed-exact
contract pinned in tests/memory/test_core_efficacy.py. The renderer adds a
``mechanism:`` line and the budgeter enforces ``max_cards`` post-auction.
"""

from gigaevo.memory.core.admitter import (
    PermissiveAdmitter,
    SignBasedAdmitter,
    TieredAdmitter,
)
from gigaevo.memory.core.auctioneer import (
    AuctionBid,
    AuctionCandidate,
    ThompsonAuctioneer,
)
from gigaevo.memory.core.budgeter import TopThetaBudgeter
from gigaevo.memory.core.card_selector import LLMCardSelector
from gigaevo.memory.core.deduplicator import LLMDeduplicator, NullDeduplicator
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.protocols import (
    Auctioneer,
    Budgeter,
    CardRenderer,
    CardRetriever,
    CardShortlister,
    Deduplicator,
    Evictor,
    MemoryAdmitter,
    ReputationModel,
)
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
    "CardRenderer",
    "CardRetriever",
    "CardShortlister",
    "Deduplicator",
    "EfficacyCardRenderer",
    "Evictor",
    "GamRetriever",
    "HarmEvictor",
    "LLMCardSelector",
    "LLMDeduplicator",
    "MemoryAdmitter",
    "MemoryReadPipeline",
    "MemorySelection",
    "MemoryWritePipeline",
    "NullDeduplicator",
    "PermissiveAdmitter",
    "ReputationModel",
    "SignBasedAdmitter",
    "ThompsonAuctioneer",
    "TieredAdmitter",
    "TopThetaBudgeter",
    "WriteLedger",
    "WriteLedgerRecord",
]
