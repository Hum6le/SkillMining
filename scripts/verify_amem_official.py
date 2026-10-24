"""Verify the pinned official source without importing ML models or calling APIs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amem_adapter.official import provenance, UPSTREAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=UPSTREAM)
    args = parser.parse_args()
    result = provenance(args.directory)
    print("Official A-MEM source OK")
    print(json.dumps(result, ensure_ascii=False, indent=2))
