"""Consolidate each slide into one title and one black body text box."""
from pathlib import Path
from collections import Counter
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.pptx_deps'))
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement

OUT = Path(__file__).resolve().parent
SOURCE = OUT / 'SkillMiner_导师汇报_20261014.pptx'
TARGET = OUT / 'SkillMiner_导师汇报_20261014_统一正文黑色版.pptx'
prs = Presentation(SOURCE)
original_text, original_notes = [], []

def largest_font(shape):
    return max((r.font.size.pt if r.font.size else 0
                for p in shape.text_frame.paragraphs for r in p.runs), default=0)

for index, slide in enumerate(prs.slides):
    shapes = [s for s in slide.shapes if s.has_text_frame and s.text.strip()]
    original_text.append(Counter(p.text for s in shapes
                                 for p in s.text_frame.paragraphs if p.text.strip()))
    original_notes.append(slide.notes_slide.notes_text_frame.text)
    title = max(shapes, key=largest_font)
    title.name = 'Title'
    title.left, title.top = Inches(.78), Inches(.95)
    title.width, title.height = Inches(11.85), Inches(1.0)
    for p in title.text_frame.paragraphs:
        for r in p.runs: r.font.color.rgb = RGBColor(0, 0, 0)

    rest = [s for s in shapes if s != title]
    # Keep the reading order, placing the former right-hand insertion note
    # after the main content. Footer text is also included in the same box.
    normal = sorted([s for s in rest if s.top < Inches(6.95)
                     and s.left < Inches(8.5)], key=lambda s: (s.top, s.left))
    reserve = sorted([s for s in rest if s.top < Inches(6.95)
                      and s.left >= Inches(8.5)], key=lambda s: (s.top, s.left))
    footer = sorted([s for s in rest if s.top >= Inches(6.95)], key=lambda s: s.left)
    content = []
    for shape in normal + reserve + footer:
        for para in shape.text_frame.paragraphs:
            if not para.text.strip(): continue
            runs = [(r.text, r.font.size.pt if r.font.size else 22,
                     bool(r.font.bold), r.font.name or 'Microsoft YaHei')
                    for r in para.runs]
            content.append((para.text, runs))

    for shape in rest:
        element = shape._element
        element.getparent().remove(element)

    body = slide.shapes.add_textbox(Inches(.78), Inches(2.12), Inches(11.85), Inches(5.03))
    body.name = 'Body'
    tf = body.text_frame
    tf.clear(); tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(.02)
    tf.margin_top = tf.margin_bottom = Inches(.01)
    for j, (value, runs) in enumerate(content):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        size = max((r[1] for r in runs), default=22)
        p.line_spacing = 1.12
        p.space_before = Pt(0)
        p.space_after = Pt(4 if size >= 24 else 10 if size >= 20 else 5)
        if not runs: runs = [(value,22,False,'Microsoft YaHei')]
        for value, size, bold, font in runs:
            r = p.add_run(); r.text = value
            r.font.name = font; r.font.size = Pt(size); r.font.bold = bold
            r.font.color.rgb = RGBColor(0,0,0)
            ea = OxmlElement('a:ea'); ea.set('typeface',font)
            r._r.get_or_add_rPr().append(ea)

prs.save(TARGET)
result = Presentation(TARGET)
for i, slide in enumerate(result.slides):
    boxes = [s for s in slide.shapes if s.has_text_frame and s.text.strip()]
    assert len(boxes) == 2, (i+1,len(boxes))
    assert {s.name for s in boxes} == {'Title','Body'}
    current = Counter(p.text for s in boxes for p in s.text_frame.paragraphs if p.text.strip())
    assert current == original_text[i], f'text changed on slide {i+1}'
    assert slide.notes_slide.notes_text_frame.text == original_notes[i]
    for shape in boxes:
        for p in shape.text_frame.paragraphs:
            for r in p.runs:
                assert r.font.color.rgb == RGBColor(0,0,0)
print('Verified: all 16 slides have exactly one title and one body text box;')
print('all visible text is black; content and speaker notes are preserved.')
print(TARGET)
