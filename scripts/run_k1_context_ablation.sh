#!/usr/bin/env bash
# K1 context experiment wrapper. Empty K1_CONDA_ENV uses the active env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
CONDA_ENV="${K1_CONDA_ENV-skillmining310}"
PYTHON_BIN="${K1_PYTHON_BIN-python}"
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                        "$HOME/anaconda3/etc/profile.d/conda.sh" \
                        /data1/anaconda3/etc/profile.d/conda.sh \
                        /opt/conda/etc/profile.d/conda.sh; do
            if [[ -f "$conda_sh" ]]; then
                source "$conda_sh"
                break
            fi
        done
    fi
    command -v conda >/dev/null 2>&1 || {
        echo "conda not found; set K1_CONDA_ENV= to use the active environment" >&2
        exit 1
    }
    conda_base="$(conda info --base)"
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_k1_context_ablation.py" "$@"
