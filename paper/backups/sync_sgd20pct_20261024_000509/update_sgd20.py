"""Update active SGD results while preparing a patch excluding other local edits."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "overleaf_sync"
ROOT = REPO.parent.parent
REPORT = ROOT / "outputs/company_sgd_20pct_reported_20261022/reported_results.json"
evidence = json.loads((HERE / "update_evidence.json").read_text(encoding="utf-8"))
assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == evidence["source_sha256"]
values = evidence["table_percentages_SR_Action_Slot"]
paths = [
    "sections/0_abstract_v1.tex",
    "sections/1_introduction_v1.tex",
    "sections/4_experiments.tex",
    "sections/5_conclusion.tex",
]


def replace_once(text, before, after):
    assert text.count(before) == 1, before[:160]
    return text.replace(before, after, 1)


def update(name, text):
    if name.endswith("0_abstract_v1.tex"):
        return replace_once(
            text,
            "outperforming the strongest baseline by 4.89 percentage points on ABCD and nearly doubling that of the strongest baseline on SGD.",
            "outperforming the strongest baseline by 4.89 and 6.53 percentage points on ABCD and SGD, respectively.",
        )
    if name.endswith("1_introduction_v1.tex"):
        return replace_once(
            text,
            "On SGD, it achieves 28.85\\% success, compared with 14.59\\% for AWM and 9.35\\% for Trace2Skill.",
            "On SGD, it achieves 28.15\\% success, compared with 21.61\\% for Trace2Skill and 15.04\\% for AWM.",
        )
    if name.endswith("5_conclusion.tex"):
        text = replace_once(
            text,
            "On the nine-family, seen-schema 1/10 SGD validation set, it achieves\n"
            "28.85\\% success, compared with 14.59\\% for AWM and 9.35\\% for Trace2Skill.",
            "On the nine-family, seen-schema 1/5 SGD validation set, it achieves\n"
            "28.15\\% success, compared with 21.61\\% for Trace2Skill and 15.04\\% for AWM.",
        )
        return replace_once(
            text, "validation subset with six\nreported baselines;", "validation subset with seven\nreported baselines;"
        )

    assert name.endswith("4_experiments.tex")
    text = replace_once(text, "sample whole dialogues at 10\\%", "sample whole dialogues at 20\\%")
    text = replace_once(
        text,
        "It contains 8,269 training decision turns from 1,201 unique dialogues and 1,241 validation decision turns from 203 unique dialogues; the reported held-out results use this validation set.",
        "The resulting split contains 16,591 training decision turns from 2,354 unique dialogues and 2,480 validation decision turns from 394 unique dialogues; we use the validation split for held-out evaluation.",
    )
    text = replace_once(
        text,
        "1/10 SGD & 1,404 & 9 & 29,654 & 9,510 & 21.1",
        "1/5 SGD & 2,748 & 9 & 58,172 & 19,071 & 21.2",
    )
    text = replace_once(
        text,
        "The four newly completed SGD baselines use Qwen3.6-Flash with the same action-head/payload executor, model settings, evaluation samples, and gold policies.\n"
        "%\n"
        "The earlier SGD results for AWM, Trace2Skill, and SkillMiner are retained from company-run aggregates; their backbone configuration remains to be cross-checked against the new local runs.",
        "The 1/5 SGD comparison uses the supplied company-run summaries; run-level model configurations and evaluated coverage remain to be cross-checked.",
    )

    start = text.index("\\paragraph{SGD.}")
    end = text.index("\\begin{table*}[t]", start)
    replacement = r"""\paragraph{SGD.}
{\color{red} On 1/5 SGD, SkillMiner achieves 28.15\% turn-level success, compared with 21.61\% for Trace2Skill and 15.04\% for AWM, gains of 6.53 and 13.10 percentage points, respectively.}
%
Its action-set accuracy is 53.51\%, and its conditional slot accuracy is 52.60\%, the highest among the reported methods on both measures.
%
Among the remaining baselines, Reflexion obtains 9.39\% turn-level success, followed by ExpeL at 8.15\%, SKILL-DISCO at 7.22\%, ReAct at 3.39\%, and A-MEM at 1.21\%.
%
ExpeL exceeds ReAct by 4.76 percentage points in turn-level success.
%
These reported results concern the nine-family, seen-schema validation protocol.

% SGD values are author-reported company 20% summaries:
% outputs/company_sgd_20pct_reported_20261022/reported_results.json.
% The dataset split contains 2,480 validation policy targets; the summaries
% do not include run-level counts, completed/missing families, or errors.
% Reflexion's evaluated coverage/aggregation is not yet verified against
% this split. Preserve supplied metrics; do not infer an evaluated count.
% SR=turn_joint_ast; Act.=action_set_exact;
% Slot=slot_exact_given_action. A-MEM values were supplied to two decimals.
% Improvements are calculated before rounding the displayed percentages.
% Historical 10% runs and Homes-only diagnostics are not included here.
% ABCD results and ablation values are unchanged.
"""
    text = text[:start] + replacement + text[end:]
    text = replace_once(
        text,
        "  correct actions. SGD results use 1,241 validation policy turns across\n  nine seen-schema families.",
        "  correct actions. The SGD validation split contains 2,480 policy turns\n  across nine seen-schema families.",
    )
    text = replace_once(
        text,
        "All SkillMiner variants use TraceCover. Dashes indicate unavailable results.",
        "SkillMiner denotes the full two-stage framework.",
    )
    display_names = {
        "ReAct": "react",
        "Reflexion": "reflexion",
        "AWM (online)": "awm",
        "ExpeL": "expel",
        "A-MEM": "a_mem",
        "Trace2Skill": "trace2skill",
        "SKILL-DISCO": "skill_disco",
    }
    for display, method in display_names.items():
        pattern = rf"(?m)^    {re.escape(display)} & [^\n]+$"
        matches = list(re.finditer(pattern, text))
        assert len(matches) == 1, display
        line = matches[0].group()
        assert line.endswith(r" \\")
        cells = line[:-3].split(" & ")
        assert len(cells) == 7, cells
        revised = " & ".join(cells[:4] + values[method]) + r" \\"
        text = text[: matches[0].start()] + revised + text[matches[0].end() :]
    text = replace_once(
        text,
        r"& \textbf{60.09} & \textbf{28.85} & \textbf{51.49} & \textbf{56.03}",
        r"& \textbf{60.09} & \textbf{28.15} & \textbf{53.51} & \textbf{52.60}",
    )
    return text.replace("1/10 SGD", "1/5 SGD")


assert not subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=REPO).strip()
snapshot = HERE / "after_sync"
expected_dir = HERE / "expected_staged"
patches = []
for name in paths:
    target = REPO / name
    original = target.read_bytes()
    original_path = snapshot / name
    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_bytes(original)
    head = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=REPO)
    (original_path.parent / (original_path.name + ".HEAD")).write_bytes(head)
    head_text = head.decode("utf-8")
    current_text = original.decode("utf-8").replace("\r\n", "\n")
    updated_head = update(name, head_text)
    updated_current = update(name, current_text)
    expected = expected_dir / name
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(updated_head.encode("utf-8"))
    newline = "\r\n" if b"\r\n" in original else "\n"
    target.write_bytes(updated_current.replace("\n", newline).encode("utf-8"))
    patches.append(
        "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in difflib.unified_diff(
                head_text.splitlines(keepends=True),
                updated_head.splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
    )
(HERE / "sgd20_only.patch").write_bytes("".join(patches).encode("utf-8"))
print(json.dumps({"changed_paths": paths, "sgd_table_percentages": values}))
