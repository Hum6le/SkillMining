"""Create a portable SGD code/data archive without credentials or experiment caches."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ['run_full_sgd_experiments.py', 'run_sgd_domain_eval.py', 'run_sgd_trace2skill_pilot.py',
           'run_multiwoz_trace2skill_pilot.py', 'llm_usage_utils.py', 'launch_sgd_server_batch.py',
           'prepare_sgd_10pct_protocol.py', 'sample_sgd_by_domain.py', 'package_sgd_server.py',
           'workflow_pool_status.py', 'benchmark_workflow_pool.py']


def build(output, splits, llm_backend='openai'):
    if llm_backend not in {'openai', 'workflow'}: raise ValueError('Unknown LLM backend')
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {Path('llm.py'), Path('requirements.txt'), Path('requirements-sgd-server.txt'),
             Path('configs/sgd_server.env.example'), Path('docs/SGD_10PCT_SERVER_PLAN.md'),
             Path('data/eval/sgd/data.zip'), Path('workflow_llm.py'), Path('workflow_request_pool.py'),
             Path('workflow_http_transport.py'),
             Path('configs/sgd_workflows.json.example'), Path('docs/SGD_WORKFLOW_PARALLEL.md')}
    files.update(Path('scripts')/name for name in SCRIPTS)
    for directory in ['eval_tod', 'skill_mining', 'awm', 'Trace2Skill/skill_evolver', 'Trace2Skill/src']:
        files.update(p.relative_to(ROOT) for p in (ROOT/directory).rglob('*')
                     if p.is_file() and p.suffix in {'.py', '.txt'}
                     and (p.name != 'config.py' or p.relative_to(ROOT).as_posix() == 'eval_tod/judge/config.py')
                     and '__pycache__' not in p.parts)
    splits = splits.resolve()
    # Profile lives at a stable relative location after extraction, independent
    # of the local Windows path recorded by older source sampling scripts.
    entries = {str(p).replace('\\', '/'): ROOT/p for p in files}
    if llm_backend == 'workflow': entries['llm.py'] = ROOT/'workflow_llm.py'
    for path in splits.rglob('*'):
        if path.is_file():
            entries['data/eval/sgd/protocol_10pct_seed42_seen_schema_v1/'+str(path.relative_to(splits)).replace('\\', '/')] = path
    manifest = {'bundle_format': 'sgd_server_v1', 'files': {}, 'llm_backend': llm_backend,
                'company_dependencies_required': ['config.py'] if llm_backend == 'workflow' else [],
                'pooled_transport': 'single_http_attempt_with_shared_per_id_admission' if llm_backend == 'workflow' else None,
                'credentials_included': False, 'outputs_or_caches_included': False}
    with tarfile.open(output, 'w:gz') as tar:
        for relative, path in sorted(entries.items()):
            if not path.is_file(): raise FileNotFoundError(path)
            data = path.read_bytes()
            if path.suffix in {'.py', '.txt'} and re.search(rb'sk-[A-Za-z0-9_-]{24,}', data):
                raise ValueError(f'Possible hardcoded credential; inspect file before packaging: {relative}')
            info = tarfile.TarInfo('Skill_Baseline_SGD/'+relative)
            info.size = len(data); info.mode = 0o644; info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
            manifest['files'][relative] = hashlib.sha256(data).hexdigest()
        encoded = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
        info = tarfile.TarInfo('Skill_Baseline_SGD/BUNDLE_MANIFEST.json')
        info.size = len(encoded); info.mode = 0o644; info.mtime = 0
        tar.addfile(info, io.BytesIO(encoded))
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(checksum+'  '+output.name+'\n', encoding='ascii')
    return {'archive': str(output), 'files': len(entries), 'size_MB': round(output.stat().st_size/2**20, 2),
            'sha256': checksum, 'credentials_included': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits-dir', type=Path, default=ROOT/'data/eval/sgd/protocol_10pct_seed42_seen_schema_v1')
    parser.add_argument('--llm-backend', choices=['openai', 'workflow'], default='openai')
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.splits_dir, args.llm_backend)))
