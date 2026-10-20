#!/usr/bin/env python3
"""Show local batch status without printing environment variables or secrets."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    args=p.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    state=read(args.output_dir/'batch_state.json')
    overrides=read(args.output_dir/'method_run_overrides.json')
    def method_dir(method):
        return Path(overrides[method]['output_dir']) if method in overrides else args.output_dir/str(method)
    print(f"Batch: {state.get('status','not started')}  PID: {state.get('pid','--')}")
    scale=read(args.output_dir/'reflexion_scale_up.json')
    if scale:
        print(f"Reflexion scale-up: {scale.get('status')}; target workers={scale.get('target_workers')}")
    for index,lane in enumerate(state.get('lanes',[]),1):
        method=lane.get('current_method')
        progress=read(method_dir(method)/'progress.json') if method else {}
        aggregate=read(method_dir(method)/'aggregate.json') if method else {}
        print(f"Lane {index}: {' -> '.join(lane['methods'])}; current={method}; {lane['status']}")
        if progress:
            print(f"  {progress['family']} / {progress['phase']}: {progress['completed']}/{progress['total']}; "
                  f"complete families={len(aggregate.get('completed_families',[]))}/9; updated={progress['updated_at']}")
            calls=tokens=failed=0
            # Request journals include live worker cases before their parent
            # has merged a completed dialogue. Parent merges do not duplicate
            # these journal rows, so counting every journal is safe.
            for file in (method_dir(method)/progress['family']).rglob('llm_requests.jsonl'):
                try:
                    lines=file.read_text(encoding='utf-8').splitlines()
                except OSError:
                    continue
                for line in lines:
                    try:
                        bucket=json.loads(line)['total']
                    except (ValueError,KeyError):
                        continue
                    calls+=bucket['calls']; tokens+=bucket['total_tokens']; failed+=bucket['failed_calls']
            print(f'  live family requests={calls}; tokens={tokens:,}; failed requests={failed}')
    for method in ['react','reflexion','skill_disco','expel']:
        aggregate=read(method_dir(method)/'aggregate.json')
        if aggregate.get('evaluation_complete'):
            print(f"{method}: action={aggregate['action_set_exact']:.2%}, "
                  f"slot|action={aggregate['slot_exact_given_action']:.2%}, "
                  f"joint={aggregate['turn_joint_ast']:.2%}; case_errors={aggregate['case_errors']}")
    print('Comparison:',args.output_dir/'comparison.md')


if __name__=='__main__':
    main()
