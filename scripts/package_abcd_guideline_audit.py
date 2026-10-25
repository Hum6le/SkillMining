"""Package audit tools and three public ABCD references; never package clients/secrets/traces."""
import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'scripts/extract_skill_knowledge.py', 'scripts/benchmark_abcd_skill_guidelines.py',
    'scripts/package_abcd_guideline_audit.py', 'skill_mining/knowledge_extraction.py',
    'scripts/recover_abcd_guideline_audit.py',
    'scripts/repair_abcd_guideline_extraction.py', 'tests/test_guideline_extraction_repair.py',
    'scripts/run_abcd_guideline_batch.py', 'tests/test_abcd_guideline_batch.py',
    'scripts/repair_guideline_citations.py', 'tests/test_guideline_citation_repair.py',
    'scripts/repair_guideline_coverage_ids.py', 'tests/test_guideline_coverage_id_repair.py',
    'skill_mining/guideline_coverage.py', 'docs/ABCD_GUIDELINE_SKILL_AUDIT.md',
    'docs/SKILL_KNOWLEDGE_EXTRACTION.md', 'tests/test_knowledge_extraction.py',
    'tests/test_abcd_guideline_coverage.py', 'data/eval/abcd/data/guidelines.json',
    'data/eval/abcd/data/ontology.json', 'data/eval/abcd/data/kb.json', 'data/eval/abcd/LICENSE',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-file', type=Path, default=ROOT / 'artifacts' / ('abcd_guideline_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.tar.gz'))
    args = parser.parse_args()
    if args.output_file.exists():
        raise FileExistsError(args.output_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'files': [{'path': name, 'sha256': hashlib.sha256((ROOT / name).read_bytes()).hexdigest()} for name in FILES],
        'scope': 'Only audit code/tests/docs and public official reference JSONs. No llm.py, config, .env, dialogue corpus or experiment artifacts.'}
    with tarfile.open(args.output_file, 'w:gz') as archive:
        for name in FILES:
            archive.add(ROOT / name, arcname=name, recursive=False)
        body = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('ABCD_GUIDELINE_AUDIT_MANIFEST.json')
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    print(json.dumps({'archive': str(args.output_file), 'bytes': args.output_file.stat().st_size, 'files': len(FILES)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
