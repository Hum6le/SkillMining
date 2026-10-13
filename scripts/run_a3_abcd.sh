#!/usr/bin/env bash
# A3 environment wrapper. Set A3_CONDA_ENV= to use the active environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
CONDA_ENV="${A3_CONDA_ENV-skillmining310}"
PYTHON_BIN="${A3_PYTHON_BIN-python}"
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
        echo "conda not found; set A3_CONDA_ENV= to use the active environment" >&2
        exit 1
    }
    conda_base="$(conda info --base)"
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_a3_abcd.py" "$@"
