"""Build samples/mixed_handwriting.pdf: printed text + table + real IAM handwriting lines (body and margins).
Ground truth for every handwritten line goes to samples/mixed_handwriting.truth.json (used for CER)."""
import json
from pathlib import Path

import pymupdf as fitz
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
iam = pd.read_parquet(ROOT / "samples/raw/iam_test.parquet")
lines = iam.sample(24, random_state=7).to_dict("records")

PRINTED = [
    "Field report 7: Water quality survey of the northern reservoir.",
    "Samples were collected at five stations between March and May. Turbidity rose after each storm, "
    "and nitrate levels at station C exceeded the guideline twice. The table below lists the mean values.",
    "Handwritten observations from the field team are reproduced below and in the margins.",
]
TABLE = [["Station", "Turbidity (NTU)", "Nitrate (mg/L)", "pH"],
         ["A", "3.2", "4.1", "7.4"], ["B", "5.8", "6.0", "7.1"], ["C", "9.6", "11.3", "6.8"],
         ["D", "2.1", "3.7", "7.6"], ["E", "4.4", "5.2", "7.2"]]

doc, truth, i = fitz.open(), [], 0
for p in range(3):
    page = doc.new_page(width=612, height=792)
    y = 60
    page.insert_text((90, y + 14), f"{PRINTED[0]} (page {p + 1})", fontsize=13, fontname="hebo")
    y += 40
    for t in PRINTED[1:]:
        page.insert_textbox(fitz.Rect(90, y, 480, y + 60), t, fontsize=10.5)
        y += 55
    if p == 0:  # one ruled table
        for r, row in enumerate(TABLE):
            for c, cell in enumerate(row):
                rect = fitz.Rect(90 + c * 97, y + r * 18, 90 + (c + 1) * 97, y + (r + 1) * 18)
                page.draw_rect(rect, width=0.6)
                page.insert_text((rect.x0 + 3, rect.y1 - 5), cell, fontsize=9, fontname="hebo" if r == 0 else "helv")
        y += 18 * len(TABLE) + 20
    for slot in range(8):
        rec = lines[i]; i += 1
        img = rec["image"]["bytes"]
        w, h = fitz.Pixmap(img).width, fitz.Pixmap(img).height
        margin = slot >= 6  # last two lines of each page go in the right margin
        width = 112 if margin else 390
        rect = (fitz.Rect(488, 120 + (slot - 6) * 120, 488 + width, 120 + (slot - 6) * 120 + width * h / w) if margin
                else fitz.Rect(90, y, 90 + width, y + width * h / w))
        if not margin:
            y = rect.y1 + 8
        page.insert_image(rect, stream=img)
        truth.append({"page": p + 1, "bbox": list(rect), "margin": margin, "text": rec["text"]})

out = ROOT / "samples/mixed_handwriting.pdf"
doc.save(out)
(ROOT / "samples/mixed_handwriting.truth.json").write_text(json.dumps(truth, indent=1))
assert len(truth) == 24 and fitz.open(out).page_count == 3
print("wrote", out, len(truth), "handwritten lines")
