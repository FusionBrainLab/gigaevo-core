"""
IdeaBank: stores and manages Idea objects for an IdeaTracker session.

Also contains CARD_STRUCTURE_v4_FINAL §2 helpers (canonical keys, packed-grammar
parser, verification predicates, mechanism-gate decision tree).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any
from uuid import uuid4

from gigaevo.memory.ideas_tracker.models import (
    AnalysisResult,
    ClassificationChunk,
    Idea,
    IdeaExplanation,
    IdeaUpdate,
)

# ---------------------------------------------------------------------------
# CARD_STRUCTURE_v4_FINAL §2: Canonical-key derivation + packed grammar
# ---------------------------------------------------------------------------

_ESTIMATOR_ALIAS = {
    "rf": "randomforest",
    "xgb": "xgboost",
    "lr": "linear",
    "lgbm": "lightgbm",
    "cb": "catboost",
}

_VERB_WHITELIST = {"ADD", "REMOVE", "UPDATE", "SWAP", "USE"}

_MECHANISM_ALIASES = {
    "l2_leaf_reg": ["l2 regularization", "l2_reg", "l2"],
    "depth": ["tree depth", "max_depth"],
    "early_stopping_rounds": ["early stopping", "early_stopping"],
    "learning_rate": ["lr", "step size"],
}

_ML_LEXICON = frozenset(
    {
        "converge",
        "convergence",
        "overfit",
        "overfits",
        "overfitting",
        "regularise",
        "regularize",
        "regularisation",
        "regularization",
        "distribution",
        "distributions",
        "variance",
        "gradient",
        "gradients",
        "generalise",
        "generalize",
        "generalisation",
        "generalization",
        "sparse",
        "sparsity",
        "dense",
        "density",
        "noise",
        "noisy",
        "signal",
        "bias",
        "residual",
        "residuals",
        "interaction",
        "interactions",
        "monotone",
        "monotonic",
        "robust",
        "robustness",
        "stochastic",
        "smooth",
        "smooths",
        "smoothing",
        "shrink",
        "shrinks",
        "shrinkage",
    }
)

_IDENT_RE = re.compile(r"\b[a-z_][a-z0-9_]*\b")


def normalize_canonical_value(v: Any) -> str:
    if v is None:
        return "_"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        x = round(float(v), 3)
        s = f"{x:.3g}"
        if s.endswith(".0"):
            s = s[:-2]
        return s
    if isinstance(v, str):
        s = v.strip().lower()
        return _ESTIMATOR_ALIAS.get(s, s)
    return hashlib.sha1(repr(v).encode()).hexdigest()[:8]


def derive_canonical_key(verb: str, target: str, old: Any, new: Any) -> str:
    verb_norm = verb.strip().upper()
    target_norm = target.strip().lower()
    return f"{verb_norm}:{target_norm}:{normalize_canonical_value(old)}:{normalize_canonical_value(new)}"


_PACKED_RE = re.compile(
    r"^(?P<unverified>UNVERIFIED_)?(?P<verb>ADD|REMOVE|UPDATE|SWAP|USE)\s+"
    r"(?P<rest>.+?)"
    r":\s*"
    r"(?P<mechanism>.+?)"
    r";\s*support=(?P<support>\d+)"
    r";\s*Δbest=(?P<delta_best>[+-]?[\d.]+)"
    r";\s*co=\[(?P<co>[^\]]*)\]\s*$"
)


def parse_packed_description(desc: str) -> dict[str, Any]:
    if desc.count(":") != 1:
        raise ValueError(
            f"single-`:` invariant violated: description contains "
            f"{desc.count(':')} colons; expected exactly 1"
        )
    m = _PACKED_RE.match(desc.strip())
    if not m:
        first_word = (desc.strip().split() or [""])[0]
        first_word_clean = first_word.removeprefix("UNVERIFIED_")
        if first_word_clean not in _VERB_WHITELIST:
            raise ValueError(
                f"unknown verb {first_word_clean!r}; expected one of {_VERB_WHITELIST}"
            )
        raise ValueError(f"description does not match packed grammar: {desc!r}")

    rest = m.group("rest").strip()
    parenthetical_unverified = False
    if rest.endswith("(UNVERIFIED)"):
        parenthetical_unverified = True
        rest = rest[: -len("(UNVERIFIED)")].strip()

    old: str | None = None
    new: str | None = None
    target: str = rest
    if "→" in rest:
        target, _, new_part = rest.partition(" ")
        if "→" in new_part:
            old, _, new = new_part.partition("→")
            old = old.strip()
            new = new.strip()
    elif " = " in rest:
        target, _, new = rest.partition(" = ")
        target = target.strip()
        new = new.strip()
    elif " " in rest and m.group("verb") in {"UPDATE", "SWAP"}:
        target = rest.split(" ", 1)[0]
    else:
        target = rest

    co_raw = m.group("co").strip()
    co = [t.strip() for t in co_raw.split(",") if t.strip()] if co_raw else []

    return {
        "verified": m.group("unverified") is None and not parenthetical_unverified,
        "verb": m.group("verb"),
        "target": target,
        "old": old,
        "new": new,
        "mechanism": m.group("mechanism").strip(),
        "support": int(m.group("support")),
        "delta_best": float(m.group("delta_best")),
        "co": co,
    }


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    method: str  # "ast_diff" | "regex_diff" | "absent"


def _value_present(code: str, target: str, value: Any) -> bool:
    if value is None:
        return False
    val_str = str(value).strip()
    pattern = (
        rf"['\"]?\b{re.escape(target)}\b['\"]?\s*[:=]\s*"
        rf"['\"]?{re.escape(val_str)}"
    )
    return bool(re.search(pattern, code))


def _identifier_present(code: str, identifier: str) -> bool:
    return bool(re.search(rf"\b{re.escape(identifier)}\b", code))


def _substring_present(code: str, fragment: str) -> bool:
    return fragment in code


def _ast_value_for_kwarg(code: str, target: str) -> Any | None:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == target:
            if isinstance(node.value, ast.Constant):
                return node.value.value
    return None


def _values_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def verify_lever(
    parent_code: str,
    child_code: str,
    verb: str,
    target: str,
    old: Any,
    new: Any,
) -> VerificationResult:
    verb_u = verb.strip().upper()
    if verb_u in {"UPDATE", "SWAP"}:
        parent_val = _ast_value_for_kwarg(parent_code, target)
        child_val = _ast_value_for_kwarg(child_code, target)
        if (
            parent_val is not None
            and child_val is not None
            and _values_equal(parent_val, old)
            and _values_equal(child_val, new)
        ):
            return VerificationResult(verified=True, method="ast_diff")
        if _value_present(parent_code, target, old) and _value_present(
            child_code, target, new
        ):
            return VerificationResult(verified=True, method="regex_diff")
        if verb_u == "SWAP" and old and new:
            if _substring_present(parent_code, str(old)) and _substring_present(
                child_code, str(new)
            ):
                return VerificationResult(verified=True, method="regex_diff")
        return VerificationResult(verified=False, method="regex_diff")
    if verb_u == "ADD":
        if (not _identifier_present(parent_code, target)) and _identifier_present(
            child_code, target
        ):
            return VerificationResult(verified=True, method="regex_diff")
        return VerificationResult(verified=False, method="regex_diff")
    if verb_u == "REMOVE":
        if _identifier_present(parent_code, target) and not _identifier_present(
            child_code, target
        ):
            return VerificationResult(verified=True, method="regex_diff")
        return VerificationResult(verified=False, method="regex_diff")
    return VerificationResult(verified=False, method="absent")


def mechanism_mentions_target(mechanism: str, target: str) -> bool:
    mech_lower = mechanism.lower()
    if target.lower() in mech_lower:
        return True
    for alias in _MECHANISM_ALIASES.get(target.lower(), []):
        if alias.lower() in mech_lower:
            return True
    return False


def changed_tokens(parent_code: str, child_code: str) -> set[str]:
    p = set(_IDENT_RE.findall(parent_code.lower()))
    c = set(_IDENT_RE.findall(child_code.lower()))
    return (p - c) | (c - p)


_NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def mechanism_grounded_in_diff(
    mechanism: str,
    target: str,
    parent_code: str,
    child_code: str,
    *,
    extra_changed_tokens: set[str] | None = None,
) -> tuple[bool, str]:
    mech_lower = mechanism.lower()
    mech_tokens = set(_IDENT_RE.findall(mech_lower))
    mech_tokens |= set(_NUMERIC_RE.findall(mech_lower))
    changed = changed_tokens(parent_code, child_code)
    if extra_changed_tokens:
        changed |= {
            str(t).strip().lower() for t in extra_changed_tokens if t is not None
        }
    changed.discard(target.lower())
    if changed & mech_tokens:
        return (True, "code")
    if len(_ML_LEXICON & mech_tokens) >= 2:
        return (True, "lexicon")
    return (False, "none")


def decide_verification(
    parent_code: str,
    child_code: str,
    verb: str,
    target: str,
    old: Any,
    new: Any,
    mechanism: str,
) -> dict[str, Any]:
    lever = verify_lever(parent_code, child_code, verb, target, old, new)
    if not lever.verified:
        return {
            "verb_prefix": "UNVERIFIED_",
            "keywords": ["verified:false"],
            "parent_diff_verified": False,
            "verification_method": lever.method,
        }
    extra_tokens: set[str] = set()
    if (
        verb.strip().upper() in {"UPDATE", "SWAP"}
        and old is not None
        and new is not None
    ):
        extra_tokens = {str(old).strip(), str(new).strip()}
    target_ok = mechanism_mentions_target(mechanism, target)
    passed, strength = mechanism_grounded_in_diff(
        mechanism,
        target,
        parent_code,
        child_code,
        extra_changed_tokens=extra_tokens,
    )
    if target_ok and passed and strength == "code":
        return {
            "verb_prefix": "",
            "keywords": ["verified:true"],
            "parent_diff_verified": True,
            "verification_method": lever.method,
        }
    return {
        "verb_prefix": "UNVERIFIED_",
        "keywords": ["verified:true", "mechanism_unverified:true"],
        "parent_diff_verified": True,
        "verification_method": lever.method,
    }


def _coerce_value(s: str | None) -> Any:
    if s is None or s == "" or s == "_":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s.strip()


def enrich_with_verification(
    description: str,
    parent_code: str,
    child_code: str,
) -> dict[str, Any]:
    """Run the v4-FINAL verification gate on a packed-grammar description.

    Returns dict with keys: description (possibly UNVERIFIED_-prefixed),
    keywords (list[str] of verification tags), parent_diff_verified (bool).

    Falls back to passthrough when description does not match packed grammar,
    or when parent_code is empty (insufficient context for verification).
    """
    if not parent_code or not child_code:
        return {
            "description": description,
            "keywords": [],
            "parent_diff_verified": False,
        }

    try:
        parsed = parse_packed_description(description)
    except (ValueError, KeyError):
        return {
            "description": description,
            "keywords": [],
            "parent_diff_verified": False,
        }

    verdict = decide_verification(
        parent_code=parent_code,
        child_code=child_code,
        verb=parsed["verb"],
        target=parsed["target"],
        old=_coerce_value(parsed.get("old")),
        new=_coerce_value(parsed.get("new")),
        mechanism=parsed["mechanism"],
    )

    new_desc = description
    verb_prefix = verdict["verb_prefix"]
    if verb_prefix and not new_desc.startswith(verb_prefix):
        new_desc = verb_prefix + new_desc.lstrip()

    return {
        "description": new_desc,
        "keywords": list(verdict["keywords"]),
        "parent_diff_verified": bool(verdict["parent_diff_verified"]),
    }


def _canonical_keyword(idea: Idea) -> str | None:
    for kw in idea.keywords or []:
        if isinstance(kw, str) and kw.startswith("canonical:"):
            return kw
    return None


# ---------------------------------------------------------------------------
# IdeaBank
# ---------------------------------------------------------------------------


class IdeaBank:
    """
    Stores and manages Idea objects for an IdeaTracker session.

    Provides add/get/update/enrich operations and produces chunked
    representations for LLM classification calls (ClassifyingAnalyzer).
    All mutations return a new Idea via model_copy — the internal list
    is updated in place but ideas themselves are treated as immutable.
    """

    def __init__(self, chunk_size: int = 5) -> None:
        self._ideas: list[Idea] = []
        self._id_index: dict[str, int] = {}  # O(1) id → list-index lookup
        self._chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add(self, idea: Idea) -> None:
        """Append an Idea. Merges into existing if canonical-key collides; else reassigns id on UUID collision."""
        new_canonical = _canonical_keyword(idea)
        if new_canonical is not None:
            for i, existing in enumerate(self._ideas):
                if _canonical_keyword(existing) == new_canonical:
                    merged_programs = list(
                        dict.fromkeys(list(existing.programs) + list(idea.programs))
                    )
                    archive_entry = {
                        f"{existing.id}-canonical-merge": {
                            "description": idea.description,
                            "programs": list(idea.programs),
                        }
                    }
                    merged_entries = list(existing.explanation.entries) + list(
                        idea.explanation.entries
                    )
                    merged = existing.model_copy(
                        update={
                            "programs": merged_programs,
                            "aliases": list(existing.aliases) + [archive_entry],
                            "explanation": IdeaExplanation(
                                entries=merged_entries,
                                summary=existing.explanation.summary,
                            ),
                        }
                    )
                    self._ideas[i] = merged
                    return
        if idea.id in self._id_index:
            idea = idea.model_copy(update={"id": str(uuid4())})
        self._id_index[idea.id] = len(self._ideas)
        self._ideas.append(idea)

    def apply(self, result: AnalysisResult) -> None:
        """Add all new_ideas and apply all updates from an AnalysisResult."""
        for idea in result.new_ideas:
            self.add(idea)
        for upd in result.updates:
            self.update(upd)

    def update(self, upd: IdeaUpdate) -> bool:
        """Apply an IdeaUpdate to an existing Idea. Returns False if not found."""
        idx = self._index(upd.idea_id)
        if idx is None:
            return False
        idea = self._ideas[idx]
        patches: dict[str, Any] = {}

        if upd.programs:
            merged_programs = list(dict.fromkeys(idea.programs + upd.programs))
            patches["programs"] = merged_programs

        if upd.generation and upd.generation > idea.last_generation:
            patches["last_generation"] = upd.generation

        if upd.new_description is not None:
            archive_entry = {
                f"{upd.idea_id}-update": {
                    "description": idea.description,
                    "programs": list(idea.programs),
                    "explanations": list(idea.explanation.entries),
                }
            }
            patches["aliases"] = idea.aliases + [archive_entry]
            patches["description"] = upd.new_description

        new_entries = idea.explanation.entries + (
            [upd.motivation] if upd.motivation else []
        )
        patches["explanation"] = IdeaExplanation(
            entries=new_entries,
            summary=idea.explanation.summary,
        )

        self._ideas[idx] = idea.model_copy(update=patches)
        return True

    def enrich(
        self,
        idea_id: str,
        *,
        keywords: list[str],
        summary: str,
        task_summary: str,
    ) -> bool:
        """Set keywords, explanation summary, and task_description_summary. Returns False if not found."""
        idx = self._index(idea_id)
        if idx is None:
            return False
        idea = self._ideas[idx]
        self._ideas[idx] = idea.model_copy(
            update={
                "keywords": keywords,
                "explanation": IdeaExplanation(
                    entries=idea.explanation.entries,
                    summary=summary,
                ),
                "task_description_summary": task_summary,
            }
        )
        return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, idea_id: str) -> Idea | None:
        """Return the Idea with the given id, or None."""
        idx = self._index(idea_id)
        return self._ideas[idx] if idx is not None else None

    def all_ideas(self) -> list[Idea]:
        """Return all ideas in insertion order."""
        return list(self._ideas)

    def classification_chunks(self) -> list[ClassificationChunk]:
        """
        Return ideas grouped into fixed-size chunks for LLM classification.

        Each chunk contains a formatted text block and short-id mappings,
        matching the format expected by ClassifyingAnalyzer._classify_against_bank.
        """
        if not self._ideas:
            return []
        chunks: list[ClassificationChunk] = []
        for i in range(0, len(self._ideas), self._chunk_size):
            batch = self._ideas[i : i + self._chunk_size]
            short_ids = [
                {
                    "id": idea.id,
                    "short_id": idea.id.split("-")[0],
                    "description": idea.description,
                }
                for idea in batch
            ]
            text = "".join(
                f"[{s['short_id']}]: {s['description']} \n " for s in short_ids
            )
            chunks.append(ClassificationChunk(text=text, short_ids=short_ids))
        return chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index(self, idea_id: str) -> int | None:
        return self._id_index.get(idea_id)
