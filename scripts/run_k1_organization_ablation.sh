#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export K1_ENTRYPOINT=run_k1_organization_ablation.py
exec bash "$SCRIPT_DIR/run_k1_context_ablation.sh" "$@"
