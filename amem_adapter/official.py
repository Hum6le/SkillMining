"""Inject project transport and persistence into unmodified official A-MEM.

Note construction, evolution, neighbour expansion, cosine search and prompts
are executed by the upstream robust implementation. No substitute retriever.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import ast
import json
from pathlib import Path
import sqlite3
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/a-mem-official"
COMMIT = "b31a1dddf56f87ec762cce3539418602d7e2ffb1"
SOURCE_HASHES = {
    "memory_layer.py": "9b49b3ca85044e5d8133f4971f71859b4c871dc66a1ae115643968e0c446743e",
    "memory_layer_robust.py": "8bb1757e301434af89688f3283fbb395a784d7e2975214581851333ed13cb731",
    "llm_text_parsers.py": "44f661c5ca2879a6daa571d5559741251d26c9dc9dfd9fa70c8a25411782f588",
    "LICENSE": "dc109f631e462ed61faa782e465b9dca947b95b500baff6de873a179dae2c714",
}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance(directory=UPSTREAM):
    actual = {name: file_sha(Path(directory) / name) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise ValueError("Official A-MEM source differs from pinned commit; use a new explicit protocol")
    return {"repository": "https://github.com/WujiangXu/A-mem", "commit": COMMIT,
            "entry": "memory_layer_robust.RobustAgenticMemorySystem", "files": actual}


def load_official(directory=UPSTREAM):
    provenance(directory)
    for name in ("memory_layer", "memory_layer_robust", "llm_text_parsers"):
        loaded = sys.modules.get(name)
        if loaded and Path(loaded.__file__).resolve().parent != Path(directory).resolve():
            raise RuntimeError(f"Conflicting module name: {name}")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    # The pinned upstream has an unused `from ast import Str`, removed in
    # Python 3.14. Scope the compatibility alias to imports; source stays exact.
    with patch.object(ast, "Str", getattr(ast, "Str", ast.Constant), create=True):
        return importlib.import_module("memory_layer"), importlib.import_module("memory_layer_robust")


class SharedController:
    """Every request/retry goes through the existing SGD/workflow request gate."""
    def __init__(self, model, journal=None):
        self.model = model
        self.journal = Path(journal) if journal else None
        self.lock = threading.Lock()
        self.errors = 0

    def get_completion(self, prompt, temperature=.7):
        from eval_tod.sgd_llm import sgd_chat_with_retry
        if prompt.startswith("Analyze the following content"):
            stage = "analysis"
        elif prompt.startswith("You are an AI memory evolution"):
            stage = "evolution_decision"
        elif prompt.startswith("Given the new memory and its neighbors"):
            stage = "strengthen"
        elif prompt.startswith("Given the new memory and its neighbor memories"):
            stage = "update_neighbors"
        else:
            stage = "query"
        messages = [{"role": "system", "content": "Follow the format specified in the prompt exactly. Do not add extra commentary."},
                    {"role": "user", "content": prompt}]
        try:
            reply = sgd_chat_with_retry(messages, model=self.model, temperature=temperature,
                max_tokens=1000, call_tag="sgd_amem_" + stage)
            if not reply.strip():
                raise RuntimeError("Empty A-MEM completion")
        except Exception:
            with self.lock:
                self.errors += 1
            raise
        if self.journal:
            with self.lock:
                with self.journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"stage": stage, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt": prompt, "response": reply}, ensure_ascii=False) + "\n")
        return reply


class OfficialMemory:
    """Native algorithm with an injected, pinned CPU encoder and LLM transport."""
    def __init__(self, encoder, controller, *, refresh_index=False, upstream_dir=UPSTREAM):
        import numpy as np
        self.np = np
        self.base, self.robust = load_official(upstream_dir)
        self.controller = controller
        self.lock = threading.RLock()
        self.touched = set()
        self.base.FAILURES.clear()

        def retriever_factory(_model_name="all-MiniLM-L6-v2"):
            with patch.object(self.base, "SentenceTransformer", lambda *_a, **_kw: encoder):
                return self.base.SimpleEmbeddingRetriever(_model_name)

        self.retriever_factory = retriever_factory
        with patch.object(self.robust, "SimpleEmbeddingRetriever", retriever_factory), \
             patch.object(self.robust, "RobustLLMController", lambda *_a, **_kw: SimpleNamespace(llm=controller)):
            self.system = self.robust.RobustAgenticMemorySystem(refresh_index=refresh_index)
        # Observe exactly the neighbours the native algorithm actually reads.
        original_find = self.system.find_related_memories
        def observed_find(query, k=5):
            text, indices = original_find(query, k)
            self.touched.update(int(i) for i in indices)
            return text, indices
        self.system.find_related_memories = observed_find
        original_consolidate = self.system.consolidate_memories
        def pinned_consolidate():
            with patch.object(self.robust, "SimpleEmbeddingRetriever", retriever_factory):
                original_consolidate()
            self.touched.update(range(len(self.system.memories)))
        self.system.consolidate_memories = pinned_consolidate

    def add(self, content, sample_id):
        self.touched.clear()
        errors_before = self.controller.errors
        failures_before = Counter(self.base.FAILURES)
        index = len(self.system.memories)
        note_id = "sgd-" + hashlib.sha256(sample_id.encode()).hexdigest()[:24]
        if note_id in self.system.memories:
            raise ValueError("Duplicate TRAIN memory ID")
        # SGD has no real recording timestamp. Never inject a fabricated date
        # into the memory text shown to the policy's date/slot binder.
        result = self.system.add_note(content, time="TRAIN_TIME_UNSPECIFIED", id=note_id)
        # Upstream fallbacks remain observable and cannot silently become a clean run.
        failures = dict(Counter(self.base.FAILURES) - failures_before)
        if self.controller.errors != errors_before or failures.get("analyze") or failures.get("evolve"):
            raise RuntimeError(f"Official A-MEM absorbed a failed operation: {failures}; checkpoint not advanced")
        self.touched.add(index)
        self.validate()
        return result, failures

    def validate(self):
        notes = list(self.system.memories.values())
        retriever = self.system.retriever
        if len(notes) != len(retriever.corpus) or (notes and len(retriever.embeddings) != len(notes)):
            raise ValueError("A-MEM note/vector index misalignment")
        if notes and not self.np.isfinite(retriever.embeddings).all():
            raise ValueError("Invalid A-MEM embeddings")
        if any(not self.base.valid_link(i, len(notes)) for note in notes for i in note.links):
            raise ValueError("Invalid A-MEM neighbour index")

    def retrieve(self, query, k=10):
        if k < 1:
            raise ValueError("retrieval k must be positive")
        with self.lock:
            # The upstream method retains its native order, duplicates and neighbour expansion.
            return self.system.find_related_memories_raw(query, k=k)

    def state_hash(self):
        h = hashlib.sha256()
        for note in self.system.memories.values():
            h.update(json.dumps(vars(note), sort_keys=True, ensure_ascii=False).encode())
        h.update(json.dumps(self.system.retriever.corpus, ensure_ascii=False).encode())
        if self.system.retriever.embeddings is not None:
            h.update(self.system.retriever.embeddings.tobytes())
        h.update(str(self.system.evo_cnt).encode())
        return h.hexdigest()


class MemoryCheckpoint:
    """One SQLite transaction per TRAIN note, including evolved neighbours.

    Existing embeddings are persisted verbatim, including paper-default stale
    indices until the native consolidation interval. No quadratic JSON rewrites.
    """
    def __init__(self, path, manifest_hash):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS notes (position INTEGER PRIMARY KEY, id TEXT UNIQUE, note TEXT, document TEXT, embedding BLOB, dtype TEXT, dimension INTEGER)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
        existing = self.get("manifest_hash")
        if existing is not None and existing != manifest_hash:
            self.close()
            raise ValueError("A-MEM checkpoint protocol/data mismatch")
        self.connection.execute("INSERT OR IGNORE INTO state VALUES (?,?)", ("manifest_hash", json.dumps(manifest_hash)))
        self.connection.commit()

    def get(self, key, default=None):
        row = self.connection.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def restore(self, memory):
        import numpy as np
        rows = self.connection.execute("SELECT position,id,note,document,embedding,dtype,dimension FROM notes ORDER BY position").fetchall()
        documents, embeddings = [], []
        for i, row in enumerate(rows):
            pos, ident, note, document, blob, dtype, dimension = row
            if pos != i:
                raise ValueError("Non-contiguous memory positions")
            obj = memory.robust.RobustMemoryNote(**json.loads(note))
            if obj.id != ident:
                raise ValueError("Memory ID mismatch")
            memory.system.memories[ident] = obj
            documents.append(document)
            vector = np.frombuffer(blob, dtype=dtype).copy()
            if len(vector) != dimension:
                raise ValueError("Corrupt memory vector")
            embeddings.append(vector)
        ret = memory.system.retriever
        ret.corpus = documents
        ret.embeddings = np.stack(embeddings) if embeddings else None
        ret.document_ids = {doc: i for i, doc in enumerate(documents)}
        memory.system.evo_cnt = self.get("evo_cnt", 0)
        memory.validate()
        if len(rows) != self.get("completed_notes", 0):
            raise ValueError("Non-atomic A-MEM checkpoint")
        return len(rows)

    def commit(self, memory, completed_notes):
        if completed_notes != len(memory.system.memories):
            raise ValueError("Checkpoint count differs from native memory count")
        notes = list(memory.system.memories.values())
        ret = memory.system.retriever
        with self.connection:
            for i in sorted(memory.touched):
                vector = ret.embeddings[i]
                self.connection.execute("INSERT OR REPLACE INTO notes VALUES (?,?,?,?,?,?,?)",
                    (i, notes[i].id, json.dumps(vars(notes[i]), ensure_ascii=False), ret.corpus[i],
                     vector.tobytes(), str(vector.dtype), len(vector)))
            for key, value in {"completed_notes": completed_notes, "evo_cnt": memory.system.evo_cnt}.items():
                self.connection.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def close(self):
        self.connection.close()
