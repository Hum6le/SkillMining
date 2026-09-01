"""Versioned accept/rollback management for ABCD online ASI libraries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

from .candidate_induction import ASISkillCandidate
from .library import ASILibrary, render_asi_library
from .online_validation import ASIOnlineValidationResult


@dataclass(frozen=True)
class ASILibraryUpdate:
    """A staged library version awaiting held-out evaluation."""

    version: int
    version_dir: str
    candidate_names: list[str]
    ast_before: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "version_dir": self.version_dir,
            "candidate_names": self.candidate_names,
            "ast_before": self.ast_before,
            "status": self.status,
        }


def candidate_to_library_record(candidate: ASISkillCandidate) -> dict[str, Any]:
    """Convert a validated candidate to the frozen-library record schema."""
    return {
        "episode_id": candidate.episode_id,
        "name": candidate.skill_name,
        "parameters": list(candidate.parameters),
        "action_start_index": candidate.action_start_index,
        "action_end_index": candidate.action_end_index,
        "action_template": [
            {
                "action": str(action["action"]),
                "arguments": [str(value) for value in (action.get("parameter_names") or [])],
            }
            for action in candidate.primitive_actions
        ],
    }


class ASIOnlineLibraryManager:
    """Manage immutable candidate versions and one accepted current library."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.current_dir = self.root / "current"
        self.history_path = self.root / "update_history.jsonl"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.current_dir.mkdir(parents=True, exist_ok=True)
        if not (self.current_dir / "ASI_ACTIONS.md").is_file():
            library = render_asi_library([])
            self._write_library(self.current_dir, library)

    def _next_version(self) -> int:
        versions = []
        for path in self.versions_dir.glob("version_*"):
            try:
                versions.append(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(versions, default=0) + 1

    @staticmethod
    def _write_library(directory: Path, library: ASILibrary) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "ASI_ACTIONS.md").write_text(library.rendered_text, encoding="utf-8")
        (directory / "library.json").write_text(
            json.dumps(library.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def current_library_path(self) -> Path:
        return self.current_dir / "ASI_ACTIONS.md"

    def current_records(self) -> list[dict[str, Any]]:
        path = self.current_dir / "library.json"
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("functions", [])
        return list(records) if isinstance(records, list) else []

    def stage(
        self,
        candidates: list[ASISkillCandidate],
        validation: ASIOnlineValidationResult,
        *,
        ast_before: float | None = None,
    ) -> ASILibraryUpdate:
        """Create a candidate version without changing the active library."""
        accepted_names = set(validation.accepted_candidates)
        selected = [candidate for candidate in candidates if candidate.skill_name in accepted_names]
        if not selected:
            raise ValueError("cannot stage an online library update without validated candidates")
        version = self._next_version()
        version_dir = self.versions_dir / f"version_{version:04d}"
        records = self.current_records() + [candidate_to_library_record(candidate) for candidate in selected]
        library = render_asi_library(records)
        self._write_library(version_dir, library)
        update = ASILibraryUpdate(
            version=version,
            version_dir=str(version_dir),
            candidate_names=[candidate.skill_name for candidate in selected],
            ast_before=ast_before,
            status="staged",
        )
        (version_dir / "update.json").write_text(
            json.dumps(update.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._append_history({**update.to_dict()})
        return update

    def accept(self, update: ASILibraryUpdate, *, ast_after: float | None = None) -> Path:
        """Atomically promote a staged version to the active library."""
        source = Path(update.version_dir)
        if not (source / "ASI_ACTIONS.md").is_file():
            raise FileNotFoundError(f"staged ASI library is missing: {source}")
        self.current_dir.mkdir(parents=True, exist_ok=True)
        for name in ("ASI_ACTIONS.md", "library.json"):
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.current_dir, prefix=f".{name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write((source / name).read_bytes())
            temporary.replace(self.current_dir / name)
        self._append_history({
            **update.to_dict(),
            "status": "accepted",
            "ast_after": ast_after,
        })
        return self.current_library_path()

    def rollback(self, update: ASILibraryUpdate, *, ast_after: float | None = None, reason: str = "heldout_regression") -> None:
        """Keep the active library unchanged and record the rejected version."""
        self._append_history({
            **update.to_dict(),
            "status": "rolled_back",
            "ast_after": ast_after,
            "rollback_reason": reason,
        })

    def _append_history(self, record: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
