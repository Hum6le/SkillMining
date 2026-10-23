"""Code-only company update, preserving native llm.py/config and local data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    from amem_adapter.official import provenance, SOURCE_HASHES
    official = provenance()
    if output.exists():
        raise FileExistsError(output)
    files = {Path(name) for name in ["workflow_llm.py", "workflow_request_pool.py", "workflow_http_transport.py",
        "requirements.txt", "requirements-sgd-server.txt", "requirements-amem-official.txt",
        "docs/SGD_AMEM_OFFICIAL.md"]}
    for name in ["amem_adapter", "eval_tod", "skill_mining", "awm", "Trace2Skill/skill_evolver",
                 "Trace2Skill/src", "scripts", "skill_disco", "expel_adapter", "reflexion_adapter"]:
        files.update(p.relative_to(ROOT) for p in (ROOT / name).rglob("*") if p.is_file()
            and p.suffix in {".py", ".txt"} and "__pycache__" not in p.parts and p.name != "config.py")
    files.add(Path("eval_tod/judge/config.py"))
    files.update(Path("third_party/a-mem-official") / name for name in SOURCE_HASHES)
    manifest = {"code_only": True, "data_included": False, "credentials_included": False,
        "embedding_weights_included": False, "official": official,
        "preserved": ["llm.py", "llm_new.py", "config.py", "configs/sgd_workflows.json"], "files": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative in sorted(files):
            payload = (ROOT / relative).read_bytes()
            if re.search(rb"sk-[A-Za-z0-9_-]{24,}", payload):
                raise ValueError(f"Possible credential in {relative}")
            manifest["files"][relative.as_posix()] = hashlib.sha256(payload).hexdigest()
            info = tarfile.TarInfo(relative.as_posix())
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("SGD_AMEM_OFFICIAL_CODE_MANIFEST.json")
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    return {"archive": str(output), "files": len(files), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "bytes": output.stat().st_size}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output)))
