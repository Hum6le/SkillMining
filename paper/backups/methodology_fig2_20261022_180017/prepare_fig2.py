"""Prepare the paper's vector PDF copy; keep the PowerPoint export unchanged."""
from pathlib import Path
import hashlib
import json
from collections import Counter
import fitz

ROOT = Path(r'D:\paper\Skill_Baseline')
BACKUP = ROOT / 'paper/backups/methodology_fig2_20261022_180017'
SOURCE = ROOT / 'figures/WWW 27 fig 2.pdf'
DEST = ROOT / 'paper/overleaf_sync/figures/WWW 27 fig 2.pdf'
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
doc = fitz.open(SOURCE)
assert len(doc) == 1
page = doc[0]
original_images = len(page.get_images(full=True))
original_visible_images = Counter(
    (entry['digest'].hex(), tuple(round(v, 3) for v in entry['bbox']))
    for entry in page.get_image_info(hashes=True))
spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b
         for line in b['lines'] for s in line['spans']]

def find_text(text):
    matches = [s for s in spans if s['text'] == text]
    assert len(matches) == 1, (text, len(matches))
    return matches[0]

formula_rect = fitz.Rect(462.5, 167.5, 584.1, 181.3)
heading_rect = fitz.Rect(74.5, 289.0, 298.0, 313.0)
diagnosis = find_text('Contrasitive')
final_label = find_text('K* = (T*, R*)')
# The final illustrative batch had its trace IDs opposite to their outcomes.
batch_labels = [s for s in spans
                if s['text'] in ('Trace2', 'Trace3')
                and 490 < s['bbox'][0] < 600 and s['bbox'][1] > 450]
assert len(batch_labels) == 2, [(s['text'], s['bbox']) for s in batch_labels]
for rect in [formula_rect, heading_rect, fitz.Rect(diagnosis['bbox']),
             fitz.Rect(final_label['bbox'])] + [fitz.Rect(s['bbox']) for s in batch_labels]:
    page.add_redact_annot(rect, fill=None, cross_out=False)
# Preserve all underlying vector shapes and all embedded icons.
page.apply_redactions(images=0, graphics=0, text=0)

fonts = {
    'FixCambria': Path(r'C:\Windows\Fonts\cambria.ttc'),
    'FixRounded': Path(r'C:\Windows\Fonts\ARLRDBD.TTF'),
    'FixCourier': Path(r'C:\Windows\Fonts\courbd.ttf'),
}
metrics = {}
for name, path in fonts.items():
    assert path.is_file()
    page.insert_font(fontname=name, fontfile=str(path))
    metrics[name] = fitz.Font(fontfile=str(path))

formula = 'min  C(G) \u2212 \u03b2 W(G)'
size = 13.32
x = (formula_rect.x0 + formula_rect.x1
     - metrics['FixCambria'].text_length(formula, fontsize=size)) / 2
page.insert_text((x, 178.03), formula, fontname='FixCambria', fontsize=size,
                 color=(17/255, 17/255, 17/255))

# The original heading consists of vector glyphs plus invisible text.
# Cover only the heading area with its exact existing panel fill.
page.draw_rect(heading_rect, color=None, fill=(1.0, 0.82, 0.82), overlay=True)
page.insert_text((75.36, 306.84), 'Online Skill Refinement',
                 fontname='FixCourier', fontsize=15.96, color=(0, 0, 0))
page.insert_text(diagnosis['origin'], 'Contrastive', fontname='FixRounded',
                 fontsize=diagnosis['size'], color=(1, 1, 1))
for span in batch_labels:
    replacement = {'Trace2': 'Trace3', 'Trace3': 'Trace2'}[span['text']]
    page.insert_text(span['origin'], replacement, fontname='FixRounded',
                     fontsize=span['size'], color=(0, 0, 0))

# Match the methodology's final-skill notation, without implying optimality.
x, baseline = final_label['origin']
for text, superscript in [('K', False), ('final', True), (' = (T', False),
                          ('final', True), (', R', False), ('final', True), (')', False)]:
    size = 6.6 if superscript else 12.0
    page.insert_text((x, baseline - (4.5 if superscript else 0)), text,
                     fontname='FixCambria', fontsize=size,
                     color=(17/255, 17/255, 17/255))
    x += metrics['FixCambria'].text_length(text, fontsize=size)
assert x < 175, x

# Remove identifying PowerPoint metadata from the anonymous manuscript asset.
doc.set_metadata({'title': 'Overview of SkillMiner', 'author': '', 'subject': '',
                  'keywords': '', 'creator': '', 'producer': '',
                  'creationDate': '', 'modDate': '', 'trapped': ''})
if hasattr(doc, 'del_xml_metadata'):
    doc.del_xml_metadata()
assert not DEST.exists(), 'Do not overwrite an existing figure without review.'
doc.save(DEST, garbage=4, deflate=True)
doc.close()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash

with fitz.open(DEST) as result:
    page = result[0]
    extracted = page.get_text()
    normalized_text = extracted.replace('\u00a0', ' ')
    assert 'tree' not in extracted
    assert 'Contrasitive' not in extracted
    assert 'Online Skill Refinement' in normalized_text
    assert 'Contrastive' in extracted
    assert not result.metadata.get('author')
    visible_images = Counter(
        (entry['digest'].hex(), tuple(round(v, 3) for v in entry['bbox']))
        for entry in page.get_image_info(hashes=True))
    assert visible_images == original_visible_images
    crop = fitz.Rect(10, 6, 952, 536)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop, alpha=False).save(
        BACKUP / 'fig2_paper_preview.png')
    report = {
        'source': str(SOURCE), 'destination': str(DEST),
        'source_sha256': source_hash,
        'destination_sha256': hashlib.sha256(DEST.read_bytes()).hexdigest(),
        'page_size': list(page.rect), 'source_image_resource_count': original_images,
        'displayed_image_count': sum(visible_images.values()),
        'all_displayed_images_unchanged': True,
        'latex_trim_bp': [10, 4, 8, 6],
        'corrections': ['W_tree(G) -> W(G)', 'Onine -> Online',
                        'Contrasitive -> Contrastive',
                        'Final skill superscripts: star -> final',
                        'Swap final batch trace IDs to match shown outcomes'],
        'metadata_author_removed': True,
    }
    (BACKUP / 'fig2_preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (BACKUP / 'fig2_paper_text.txt').write_text(extracted, encoding='utf-8')
print(json.dumps(report, ensure_ascii=True))
