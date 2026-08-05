#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_awm_case_report.sh RUN_DIR [visualize options]" >&2
    exit 2
fi

RUN_DIR="$1"
shift
python scripts/visualize_awm_cases.py --run-dir "$RUN_DIR" --n-turns 10 --train-turns 6 "$@"
