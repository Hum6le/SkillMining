"""Show workflow transport admissions and completion latency, without prompts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workflow_request_pool import WorkflowRequestPool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/sgd_workflows.json'))
    parser.add_argument('--window', type=float, default=60)
    args = parser.parse_args()
    if args.window <= 0: parser.error('Window must be positive')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    database = Path(config.get('rate_database', 'outputs/workflow_rate.sqlite'))
    if not database.is_absolute(): database = Path(__file__).resolve().parent.parent/database
    if not database.is_file():
        raise FileNotFoundError(f'No rate database at {database.resolve()}; check the running batch configuration')
    result = {'_diagnostics': {'database': str(database.resolve()),
        'rate_limiter_file': str(Path(sys.modules['workflow_request_pool'].__file__).resolve()),
        'admission_clock': 'client_before_http_not_server_receive_time',
        'note': 'Updating files or reading this status does not reload running worker processes.'}}
    for name, method in config['methods'].items():
        pool = WorkflowRequestPool(database, method['workflow_ids'], config.get('qps_per_workflow', 2),
                                   min_interval_seconds=config.get('min_interval_seconds'),
                                   max_inflight=config.get('max_inflight',10),
                                   max_active_workflows=config.get('max_active_workflows',5))
        result[name] = pool.summary(args.window)
    result['_global_model'] = pool.global_summary()
    print(json.dumps(result, indent=2, ensure_ascii=False))
