"""MultiWOZ Knowledge Base — load domain DBs and query entities."""

from __future__ import annotations

import json
import os
from typing import Any


class MultiWOZKB:
    """In-memory knowledge base for all MultiWOZ domains.

    Each domain DB is a list of entity dicts loaded from
    ``data/data/{domain}_db.json``.  The KB supports constraint-based
    filtering with case-insensitive value matching.

    Usage::

        kb = MultiWOZKB("data/eval/multiwoz21/data/data")
        results = kb.query("hotel", {"area": "centre", "pricerange": "cheap"})
        for hotel in results:
            print(hotel["name"], hotel["phone"])
    """

    # ── Per-domain field name mappings ──────────────────────────
    # Normalise field names across domains so the query interface is
    # consistent (e.g. "pricerange" <-> "price range", "type").
    _NORMALIZE: dict[str, dict[str, str]] = {
        "hotel": {
            "pricerange": "price range",
        },
        "restaurant": {
            "pricerange": "price range",
        },
        "attraction": {
            "pricerange": "price range",
            "entrance fee": "entrance fee",
        },
        "train": {
            "arriveBy": "arrive by",
            "leaveAt": "leave at",
            "trainID": "train id",
        },
        "taxi": {},    # DB is corrupt, handled gracefully
        "hospital": {},
        "police": {},
    }

    # Which DB keys are slot-like (exclude ids, coordinates, text blurbs)
    _META_KEYS = {"id", "location", "takesbookings", "introduction", "openhours", "price"}

    def __init__(self, db_dir: str):
        self.db_dir = db_dir
        self._dbs: dict[str, list[dict]] = {}
        self._load_all()

    def _load_all(self) -> None:
        for domain in ["attraction", "hospital", "hotel", "police",
                       "restaurant", "taxi", "train"]:
            path = os.path.join(self.db_dir, f"{domain}_db.json")
            if not os.path.exists(path):
                self._dbs[domain] = []
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._dbs[domain] = self._normalize_entity(domain, data)
                else:
                    self._dbs[domain] = []
            except (json.JSONDecodeError, Exception):
                self._dbs[domain] = []

    def _normalize_entity(self, domain: str, entities: list[dict]) -> list[dict]:
        """Apply field-name normalization."""
        mapping = self._NORMALIZE.get(domain, {})
        if not mapping:
            return entities
        out = []
        for e in entities:
            normed = dict(e)
            for old, new in mapping.items():
                if old in normed:
                    normed[new] = normed.pop(old)
            out.append(normed)
        return out

    def query(
        self,
        domain: str,
        constraints: dict[str, str] | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Query entities in a domain matching all constraints.

        Args:
            domain: Domain name (``"hotel"``, ``"restaurant"``, etc.).
            constraints: ``{slot_name: desired_value}``.  Matching is
                case-insensitive substring.  Pass ``None`` or ``{}`` for
                all entities.
            max_results: Maximum number of entities to return.

        Returns:
            List of matching entity dicts (slot_name -> value).
        """
        entities = self._dbs.get(domain, [])
        if not entities:
            return []

        constraints = constraints or {}
        results: list[dict] = []

        for entity in entities:
            match = True
            for key, want in constraints.items():
                have = entity.get(key)
                if have is None:
                    match = False
                    break
                # Case-insensitive string match
                if str(want).lower() not in str(have).lower():
                    match = False
                    break
            if match:
                # Only return relevant slot keys (strip metadata)
                clean = {
                    k: v for k, v in entity.items()
                    if k not in self._META_KEYS
                }
                results.append(clean)
                if len(results) >= max_results:
                    break

        return results

    def query_formatted(
        self,
        domain: str,
        constraints: dict[str, str] | None = None,
        max_results: int = 5,
    ) -> str:
        """Query and return a human-readable string for the LLM.

        Args:
            domain: Domain name.
            constraints: Filter constraints.
            max_results: Max entities.

        Returns:
            Formatted string, e.g.::

                Found 3 hotel(s):
                1. name: Ashley Hotel, area: centre, price range: cheap, ...
        """
        results = self.query(domain, constraints, max_results)
        if not results:
            constraints_desc = ", ".join(f"{k}={v}" for k, v in (constraints or {}).items())
            return f"No {domain} found matching: {constraints_desc or 'none'}"

        lines = [f"Found {len(results)} {domain}(s):"]
        for i, entity in enumerate(results, 1):
            slot_str = ", ".join(f"{k}: {v}" for k, v in sorted(entity.items()))
            lines.append(f"  {i}. {slot_str}")
        return "\n".join(lines)

    @property
    def domains(self) -> list[str]:
        """List domains with loaded data."""
        return sorted(self._dbs.keys())

    def domain_size(self, domain: str) -> int:
        """Number of entities in a domain."""
        return len(self._dbs.get(domain, []))
