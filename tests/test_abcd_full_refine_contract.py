from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_full_refine_defaults_match_standalone_and_failures_are_checked():
    full = (ROOT / "scripts/run_full_backbone_online_refine.sh").read_text(encoding="utf-8")
    standalone = (ROOT / "scripts/run_backbone_online_refine.py").read_text(encoding="utf-8")
    assert 'SUBFLOW_MODE="serial"' in full
    assert 'REBUILD_SPLITS=0' in full
    assert 'HYBRID_MAP_BATCH_SIZE="8"' in full
    assert '"--hybrid-map-batch-size", type=int, default=8' in standalone
    assert 'run_serial || SERIAL_FAILURE=1' in full
    assert '"$RUN_ROOT/serial_failed.txt" 2>/dev/null; then' in full
