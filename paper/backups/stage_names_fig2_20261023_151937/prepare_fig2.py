"""Prepare the updated vector figure for the manuscript."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil
import fitz

ROOT = Path(r"D:\paper\Skill_Baseline")
BACKUP = ROOT / "paper/backups/stage_names_fig2_20261023_151937"
SOURCE = Path("D:/paper/\u7ed8\u56fe/WWW 27 fig2.pdf")
DEST = ROOT / "paper/overleaf_sync/figures/WWW 27 fig2.pdf"
CANDIDATE = BACKUP / "prepared_fig2.pdf"
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
doc = fitz.open(SOURCE)
assert len(doc) == 1
page = doc[0]

def displayed_images(pdf_page):
    return Counter(
        (item["digest"].hex(), tuple(round(x, 3) for x in item["bbox"]))
        for item in pdf_page.get_image_info(hashes=True)
    )

original_images = displayed_images(page)
spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b
         for line in b["lines"] for s in line["spans"]]
mining_rect = fitz.Rect(770, 14, 895, 39)
refine_rect = fitz.Rect(83, 287, 238, 312.5)
batch_labels = [s for s in spans if s["text"] in ("Trace2", "Trace3")
                and 490 < s["bbox"][0] < 530 and s["bbox"][1] > 470]
final_labels = [s for s in spans if s["text"] == "K* = (T*, R*)"]
assert len(batch_labels) == 2 and len(final_labels) == 1
final_label = final_labels[0]
for rect in [mining_rect, refine_rect, fitz.Rect(final_label["bbox"])] + [
        fitz.Rect(s["bbox"]) for s in batch_labels]:
    page.add_redact_annot(rect, fill=None, cross_out=False)
page.apply_redactions(images=0, graphics=0, text=0)

fonts = {
    "StageHeading": Path(r"C:\Windows\Fonts\courbd.ttf"),
    "BatchLabel": Path(r"C:\Windows\Fonts\ARLRDBD.TTF"),
    "SkillFormula": Path(r"C:\Windows\Fonts\cambria.ttc"),
}
metrics = {}
for name, path in fonts.items():
    assert path.is_file(), path
    page.insert_font(fontname=name, fontfile=str(path))
    metrics[name] = fitz.Font(fontfile=str(path))

# PowerPoint exports these headings as vector glyphs plus invisible text.
# Cover the glyphs with the existing solid panel colors, then add vector text.
page.draw_rect(mining_rect, color=None, fill=(213/255, 246/255, 253/255))
page.draw_rect(refine_rect, color=None, fill=(1, 209/255, 209/255))
page.insert_text((772.51, 33.60), "Mining Stage", fontname="StageHeading",
                 fontsize=15.96, color=(0, 0, 0))
page.insert_text((84.432, 306.26), "Refine Stage", fontname="StageHeading",
                 fontsize=15.96, color=(0, 0, 0))
for span in batch_labels:
    page.insert_text(span["origin"], {"Trace2": "Trace3", "Trace3": "Trace2"}[span["text"]],
                     fontname="BatchLabel", fontsize=span["size"], color=(0, 0, 0))

# Keep the final artifact notation consistent with the methodology.
x, baseline = final_label["origin"]
for text, superscript in [("K", False), ("final", True), (" = (T", False),
                          ("final", True), (", R", False), ("final", True), (" )", False)]:
    size = 6.6 if superscript else 12
    page.insert_text((x, baseline - (4.5 if superscript else 0)), text,
                     fontname="SkillFormula", fontsize=size,
                     color=(17/255, 17/255, 17/255))
    x += metrics["SkillFormula"].text_length(text, fontsize=size)
assert x < 180
doc.set_metadata({"title": "SkillMiner: Mining Stage and Refine Stage",
                  "author": "", "subject": "", "keywords": "",
                  "creator": "", "producer": "", "creationDate": "",
                  "modDate": "", "trapped": ""})
if hasattr(doc, "del_xml_metadata"):
    doc.del_xml_metadata()
assert not CANDIDATE.exists(), "Retain the previous prepared candidate before rerunning."
doc.save(CANDIDATE, garbage=4, deflate=True)
doc.close()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
with fitz.open(CANDIDATE) as result:
    page = result[0]
    assert displayed_images(page) == original_images
    text = page.get_text().replace("\u00a0", " ")
    assert "Mining Stage" in text and "Refine Stage" in text
    assert "K* = (T*, R*)" not in text
    assert not result.metadata.get("author")
    page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(10, 6, 952, 536),
                    alpha=False).save(str(BACKUP / "paper_fig2_preview.png"))
    (BACKUP / "paper_fig2_text.txt").write_text(text, encoding="utf-8")
if DEST.exists():
    shutil.copy2(DEST, BACKUP / "first_paper_candidate.pdf")
shutil.copy2(CANDIDATE, DEST)
report = {"source": str(SOURCE), "destination": str(DEST),
          "source_sha256": source_hash,
          "destination_sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
          "all_displayed_images_unchanged": True,
          "displayed_image_count": sum(original_images.values()),
          "updates": ["Mining Stage and Refine Stage headings",
                      "B3 trace IDs agree with displayed outcomes",
                      "Final skill superscript matches manuscript",
                      "Anonymous manuscript metadata"],
          "latex_trim_bp": [10, 4, 8, 6]}
(BACKUP / "fig2_preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=True))
