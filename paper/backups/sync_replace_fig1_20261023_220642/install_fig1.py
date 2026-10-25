"""Install the prepared figure and prepare a patch limited to its description."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "overleaf_sync"
INTRO = "sections/1_introduction_v1.tex"
changes = [
    (
        "  % Trim order: left, bottom, right, top; values crop the PDF margins.\n"
        "  \\includegraphics[width=\\columnwidth,trim=68bp 0bp 102bp 0bp,clip]{figures/WWW 27 fig1.pdf}",
        "  % The PDF is cropped horizontally; retain its vector text and graphics.\n"
        "  \\includegraphics[width=\\columnwidth]{figures/WWW 27 fig1.pdf}",
    ),
    (
        "  \\caption{From dialogue cases to a reusable skill. The illustrated\n"
        "  requests share account, order, and status checks, then follow different\n"
        "  branches. Supporting rules and evidence remain accessible as resources.",
        "  \\caption{From a dialogue case to a reusable skill. Shared account and\n"
        "  purchase checks precede a refund or fee-waiver branch, depending on\n"
        "  shipment status. Supporting dialogue evidence remains accessible as resources.",
    ),
    (
        "Figure~\\ref{fig:tod-skill-example} illustrates this challenge through two shipping-fee enquiries. ",
        "Figure~\\ref{fig:tod-skill-example} illustrates this challenge through a shipping-fee enquiry with two possible continuations.",
    ),
    (
        "The two dialogues share the same steps, namely account validation and purchase checks, but diverge at the shipment state:",
        "Both continuations follow shared account verification and purchase checks, but depend on the shipment state:",
    ),
    (
        "The right-hand side of the figure places each dialogue as a particular path along the spine and through its branches, with dialogue resources retaining the evidence for when each branch applies.",
        "The right-hand side organizes the shared steps and alternative continuations as a reusable skill, with dialogue resources retaining the evidence for when each branch applies.",
    ),
]


def revise(data):
    newline = b"\r\n" if b"\r\n" in data else b"\n"
    for before, after in changes:
        before = before.encode().replace(b"\n", newline)
        after = after.encode().replace(b"\n", newline)
        assert data.count(before) == 1, before[:100]
        data = data.replace(before, after, 1)
    return data


snapshot = json.loads((HERE / "backup_info.json").read_text(encoding="utf-8"))
current = (REPO / INTRO).read_bytes()
assert hashlib.sha256(current).hexdigest() == snapshot["files"][INTRO]
head_intro = subprocess.check_output(["git", "show", f"HEAD:{INTRO}"], cwd=REPO)
staged_intro = revise(head_intro)
worktree_intro = revise(current)

patch = "".join(
    difflib.unified_diff(
        head_intro.decode().splitlines(keepends=True),
        staged_intro.decode().splitlines(keepends=True),
        fromfile=f"a/{INTRO}",
        tofile=f"b/{INTRO}",
    )
)
(HERE / "figure_only.patch").write_bytes(patch.encode())
(HERE / "expected_staged_introduction.tex").write_bytes(staged_intro)
(REPO / INTRO).write_bytes(worktree_intro)
shutil.copy2(HERE / "fig1_cropped.pdf", REPO / "figures/WWW 27 fig1.pdf")
print(patch)
