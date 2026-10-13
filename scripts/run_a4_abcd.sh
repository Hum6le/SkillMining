#!/usr/bin/env bash
# Run A4 in the same project environment used by A2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${A4_CONDA_ENV-skillmining310}"
PYTHON_BIN="${A4_PYTHON_BIN-python}"
if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        for conda_sh in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                        "$HOME/anaconda3/etc/profile.d/conda.sh" \
                        /data1/anaconda3/etc/profile.d/conda.sh \
                        /data1/anaconda/etc/profile.d/conda.sh \
                        /opt/conda/etc/profile.d/conda.sh; do
            if [[ -f "$conda_sh" ]]; then
                # shellcheck source=/dev/null
                source "$conda_sh"
                break
            fi
        done
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda not found; set A4_CONDA_ENV= to use the current environment" >&2
        exit 1
    fi
    conda_base="$(conda info --base)"
    if [[ -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$conda_base/etc/profile.d/conda.sh"
    fi
    conda activate "$CONDA_ENV"
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
}
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_a4_abcd.py" "$@"
