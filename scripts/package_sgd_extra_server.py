"""Build a code-only update for the extra SGD company baselines."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def build(output):
    if output.exists():raise FileExistsError(output)
    files=set()
    for name in ['workflow_llm.py','workflow_request_pool.py',
                 'workflow_http_transport.py','requirements.txt','requirements-sgd-server.txt']:
        files.add(Path(name))
    for name in ['eval_tod','skill_mining','awm','Trace2Skill/skill_evolver','Trace2Skill/src',
                 'skill_disco','expel_adapter','reflexion_adapter','scripts']:
        files.update(p.relative_to(ROOT) for p in (ROOT/name).rglob('*')
                     if p.is_file() and p.suffix in {'.py','.txt'} and '__pycache__' not in p.parts
                     and p.name!='config.py')
    files.add(Path('eval_tod/judge/config.py'))
    files.add(Path('docs/SGD_EXTRA_SERVER_20PCT.md'))
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest={'code_only':True,'data_included':False,'credentials_included':False,
              'company_files_preserved':['llm.py','config.py','configs/sgd_workflows.json'],'files':{}}
    with tarfile.open(output,'w:gz') as archive:
        for relative in sorted(files):
            path=ROOT/relative;data=path.read_bytes()
            if re.search(rb'sk-[A-Za-z0-9_-]{24,}',data):
                raise ValueError(f'Possible embedded key in {relative}; package stopped')
            manifest['files'][relative.as_posix()]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(relative.as_posix());info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
        data=json.dumps(manifest,indent=2).encode()
        info=tarfile.TarInfo('SGD_EXTRA_CODE_MANIFEST.json');info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
    return {'archive':str(output),'files':len(files),'size_mb':round(output.stat().st_size/2**20,2),
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'data_included':False,
            'credentials_included':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(p.parse_args().output)))
