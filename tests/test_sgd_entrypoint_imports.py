import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("entrypoint", ["run_full_sgd_experiments.py", "run_sgd_domain_eval.py"])
def test_file_entrypoint_ignores_foreign_scripts_package(tmp_path, entrypoint):
    foreign = tmp_path / "scripts"
    foreign.mkdir()
    (foreign / "__init__.py").write_text(
        "raise RuntimeError('foreign scripts package was imported')\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, str(root / "scripts" / entrypoint), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--ours-mining-method" in result.stdout
