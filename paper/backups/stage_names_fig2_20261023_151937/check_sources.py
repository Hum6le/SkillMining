"""Check changed LaTeX inputs and unchanged experiment table values."""
from pathlib import Path
from collections import Counter
import json
import re
import shutil

ROOT = Path(r"D:\paper\Skill_Baseline\paper\overleaf_sync")
BACKUP = ROOT.parent / "backups/stage_names_fig2_20261023_151937"
texts = {}

def uncomment(text):
    return "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0]
                     for line in text.splitlines())

def collect(path):
    if path in texts:
        return
    assert path.is_file(), path
    texts[path] = uncomment(path.read_text(encoding="utf-8"))
    for name in re.findall(r"\\input\{([^}]+)\}", texts[path]):
        child = ROOT / name
        collect(child if child.suffix else child.with_suffix(".tex"))

collect(ROOT / "main.tex")
all_text = "\n".join(texts.values())
labels = re.findall(r"\\label\{([^}]+)\}", all_text)
assert all(n == 1 for n in Counter(labels).values())
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", all_text))
assert not refs - set(labels), refs - set(labels)
figures = []
for path, text in texts.items():
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if kind == "begin":
            stack.append(env)
        else:
            assert stack and stack.pop() == env, path
    assert not stack, path
    depth = 0
    for brace in re.findall(r"(?<!\\)[{}]", text):
        depth += 1 if brace == "{" else -1
        assert depth >= 0, path
    assert depth == 0, path
    for image in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        assert (ROOT / image).is_file(), image
        figures.append(image)
assert figures[1] == "figures/WWW 27 fig2.pdf", figures
bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",
                         (ROOT / "references.bib").read_text(encoding="utf-8")))
cites = {k.strip() for group in re.findall(
    r"\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}", all_text) for k in group.split(",")}
assert not cites - bib_keys, cites - bib_keys
method = texts[ROOT / "sections/3_methodology.tex"]
assert len(re.findall(r"\\subsection\{", method)) == 4
assert "Offline" not in method and "Online" not in method
assert "Mining Stage" in method and "Refine Stage" in method
assert not re.search(r"^%%", method, flags=re.M)
before = (BACKUP / "sections/4_experiments.tex").read_text(encoding="utf-8")
after = (ROOT / "sections/4_experiments.tex").read_text(encoding="utf-8")
def table_numbers(text):
    tables = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, flags=re.S)
    return [re.findall(r"\d+(?:\.\d+)?", row.split("&", 1)[1])
            for table in tables for row in table.splitlines() if "&" in row]
assert table_numbers(before) == table_numbers(after)
assert "AWM (online)" in after
compilers = {name: shutil.which(name)
             for name in ("latexmk", "pdflatex", "xelatex", "tectonic")}
report = {"active_files": [str(p.relative_to(ROOT)) for p in texts],
          "figures": figures, "undefined_references": [],
          "braces_and_environments": "passed", "citations": len(cites),
          "experiment_table_values": "unchanged", "compiler_paths": compilers}
(BACKUP / "source_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True))
