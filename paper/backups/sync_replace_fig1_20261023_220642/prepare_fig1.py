"""Crop Fig. 1 horizontally without rasterizing its PDF content."""
from pathlib import Path
import hashlib
import json
from collections import Counter

import fitz
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "new_fig1_original.pdf"
OUTPUT = HERE / "fig1_cropped.pdf"


def image_signatures(page):
    return Counter(
        (item["width"], item["height"], item["digest"].hex())
        for item in page.get_image_info(hashes=True)
    )


with fitz.open(SOURCE) as original:
    assert len(original) == 1
    page = original[0]
    original_text = page.get_text()
    original_images = image_signatures(page)
    original_drawings = len(page.get_drawings())
    original_pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    crop = fitz.Rect(93.5, 0, 941.5, 540)
    page.set_cropbox(crop)
    original.set_metadata({})
    original.save(OUTPUT, garbage=4, deflate=True)

with fitz.open(OUTPUT) as cropped:
    page = cropped[0]
    assert page.get_text() == original_text
    assert image_signatures(page) == original_images
    assert len(page.get_drawings()) == original_drawings
    assert not cropped.metadata.get("author")
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    before = Image.frombytes(
        "RGB", (original_pixmap.width, original_pixmap.height), original_pixmap.samples
    ).crop((187, 0, 1883, 1080))
    after = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    assert before.size == after.size
    difference = ImageChops.difference(before, after)
    max_channel_delta = max(high for low, high in difference.getextrema())
    # A translated PDF viewport can round a few antialiased pixels differently.
    assert max_channel_delta <= 1
    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(
        HERE / "fig1_cropped_preview.png"
    )
    result = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "page_count": len(cropped),
        "original_width_bp": 960,
        "cropped_width_bp": page.rect.width,
        "height_bp": page.rect.height,
        "left_trim_bp": 93.5,
        "right_trim_bp": 18.5,
        "text_preserved": True,
        "embedded_image_instances_preserved": sum(original_images.values()),
        "vector_drawing_objects_preserved": original_drawings,
        "max_render_channel_delta_at_144dpi": max_channel_delta,
        "render_equivalent_within_one_channel_level": True,
    }
    (HERE / "figure_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))
