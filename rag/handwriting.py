"""Handwriting: low-confidence OCR lines and line-structured pictures -> local TrOCR.
TrOCR's own confidence decides whether its reading replaces RapidOCR's. Updates page checkpoints in place (idempotent)."""
import json

import cv2
import numpy as np
import pymupdf
from PIL import Image

from rag import config
from rag.models import read_handwriting
from rag.store import write_json

SCALE = 3  # render at 216 dpi
CHARTS = {"bar_chart", "line_chart", "pie_chart", "flow_chart", "scatter_plot", "chemistry_structure", "map",
          "logo", "qr_code", "bar_code", "icon", "screenshot", "natural_image", "engineering_drawing", "geographical_map",
          "remote_sensing", "heatmap", "box_plot", "stratigraphic_chart", "cad_drawing", "electrical_diagram", "3d_chart"}


def split_lines(img: Image.Image) -> list[Image.Image]:
    """Horizontal projection: rows with ink form lines; each line is trimmed to its ink columns."""
    g = np.asarray(img.convert("L"))
    if g.std() < 5:  # blank crop
        return []
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = g <= min(t, 200)
    rows = ink.mean(axis=1) > 0.005
    lines, start, gap = [], None, 0
    for y, on in enumerate(list(rows) + [False] * 4):
        if on:
            start = y if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 3:
                end = y - gap + 1
                if end - start >= 8:
                    lines.append((start, end))
                start, gap = None, 0
    boxes = []
    for a, b in lines:
        cols = np.where(ink[a:b].any(axis=0))[0]
        if len(cols) >= 10:
            boxes.append((max(cols[0] - 6, 0), max(a - 6, 0), min(cols[-1] + 6, g.shape[1]), min(b + 6, g.shape[0])))
    if not boxes:
        return []
    med_h = float(np.median([y1 - y0 for _, y0, _, y1 in boxes]))
    # drop stray dots/descenders: both much shorter than a line and narrow
    return [img.crop(b).convert("RGB") for b in boxes
            if not (b[3] - b[1] < 0.5 * med_h and b[2] - b[0] < 0.1 * g.shape[1])]


def _crop(img, bbox, pad=3):
    x0, y0, x1, y1 = (int(v * SCALE) for v in bbox)
    p = pad * SCALE
    return img.crop((max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, img.width), min(y1 + p, img.height))).convert("RGB")


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _lines(r, img) -> list[dict]:
    """Low-confidence OCR lines of a text item, re-read by TrOCR; kept only where TrOCR is confident."""
    low = [c for c in r.get("ocr_cells", []) if c["conf"] < config.OCR_CONFIDENCE_MIN]
    if not low:
        return []
    reads = read_handwriting([_crop(img, c["bbox"]) for c in low])
    return [{**c, "hw": t.strip(), "hw_conf": round(p, 3)} for c, (t, p) in zip(low, reads)
            if t.strip() and p >= config.HANDWRITING_CONFIDENCE_MIN]


def _process(r, img) -> list[dict]:
    """Returns the record(s) that replace r."""
    r["hw_checked"] = True
    if r["type"] == "figure":
        if r.get("picture_class") in CHARTS or r.get("caption"):  # a captioned picture is a figure, not a note
            return [r]
        lines = split_lines(_crop(img, r["bbox"], pad=0))
        if not lines or any(l.width < 3 * l.height for l in lines):
            return [r]
        reads = read_handwriting(lines)
        conf = sum(p for _, p in reads) / len(reads)
        if conf < config.HANDWRITING_CONFIDENCE_MIN:
            return [r]
        r.update(type="text", handwritten=True, source="trocr", confidence=round(conf, 3), figure_text=r["text"],
                 text="\n".join(t.strip() for t, _ in reads if t.strip()))
        return [r]
    hw = _lines(r, img)
    if not hw:
        return [r]
    rest = r["text"]
    for c in hw:
        rest = rest.replace(c["text"], " ", 1)  # shortcut: drops the first occurrence, upgrade to cell-position matching if printed words vanish
    new = {"id": r["id"] + "h", "page": r["page"], "type": "text", "label": r["label"], "bbox": _union([c["bbox"] for c in hw]),
           "text": "\n".join(c["hw"] for c in hw), "ocr_text": "\n".join(c["text"] for c in hw), "handwritten": True,
           "source": "trocr", "confidence": min(c["hw_conf"] for c in hw), "hw_checked": True}
    if not rest.strip():
        return [new]
    # mixed item (printed text + handwritten lines): split so the stitcher sees each part's own position
    keep = [c for c in r["ocr_cells"] if c not in hw and c["text"] in rest]
    r["text"] = " ".join(rest.split())
    r["ocr_cells"] = keep
    hb, mid = new["bbox"], (r["bbox"][0] + r["bbox"][2]) / 2
    if hb[0] > mid:  # handwriting sat to the right of the printed text: shrink the printed box
        r["bbox"][2] = hb[0] - 8
    elif hb[2] < mid:
        r["bbox"][0] = hb[2] + 8
    if not any(c["text"] for c in keep) and r.get("confidence") is not None:
        r["confidence"] = None  # what's left is text-layer text
    return [r, new]


def _overlap(a, b) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _uncovered(page, recs, img, pno) -> list[dict]:
    """Embedded images no item covers (OCR dropped them): try them as handwriting."""
    area = page.rect.width * page.rect.height
    out = []
    for k, info in enumerate(page.get_image_info()):
        b = [round(v, 1) for v in info["bbox"]]
        size = (b[2] - b[0]) * (b[3] - b[1])
        if size <= 0 or size > 0.8 * area or sum(_overlap(b, r["bbox"]) for r in recs) > 0.3 * size:
            continue
        lines = split_lines(_crop(img, b, pad=0))
        if not lines or any(l.width < 3 * l.height for l in lines):
            continue
        reads = read_handwriting(lines)
        conf = sum(p for _, p in reads) / len(reads)
        if conf >= config.HANDWRITING_CONFIDENCE_MIN:
            out.append({"id": f"p{pno}_u{k}", "page": pno, "type": "text", "label": "uncovered_image", "bbox": b,
                        "text": "\n".join(t.strip() for t, _ in reads if t.strip()), "handwritten": True,
                        "source": "trocr", "confidence": round(conf, 3), "hw_checked": True})
    return out


def _dedupe(recs) -> list[dict]:
    """OCR fragments and a picture can cover the same handwritten line: keep the largest reading."""
    area = lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])
    hw = sorted((r for r in recs if r.get("handwritten")), key=area, reverse=True)
    drop = set()
    for n, big in enumerate(hw):
        for small in hw[n + 1:]:
            if id(small) not in drop and id(big) not in drop and _overlap(big["bbox"], small["bbox"]) > 0.5 * area(small):
                drop.add(id(small))
    return [r for r in recs if id(r) not in drop]


def run(pdf, doc_dir) -> int:
    done = 0
    with pymupdf.open(pdf) as d:
        for f in sorted((doc_dir / "pages").glob("*.json")):
            recs = json.loads(f.read_text(encoding="utf-8"))
            pno = int(f.stem)
            page = d[pno - 1]
            todo = [r for r in recs if not r.get("hw_checked") and (r.get("ocr_cells") or r["type"] == "figure")]
            if not todo and not page.get_images():
                continue
            pix = page.get_pixmap(matrix=pymupdf.Matrix(SCALE, SCALE))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out = []
            for r in recs:
                if r in todo:
                    new = _process(r, img)
                    done += sum(x.get("handwritten", False) for x in new)
                    out += new
                else:
                    out.append(r)
            extra = _uncovered(page, out, img, pno)
            done += len(extra)
            write_json(f, _dedupe(out + extra))
    return done


if __name__ == "__main__":  # self-check: two dark bars on white -> two lines
    a = np.full((100, 300), 255, np.uint8)
    a[10:30, 20:280] = 0
    a[60:85, 20:200] = 0
    ls = split_lines(Image.fromarray(a))
    assert len(ls) == 2 and ls[1].size[0] < ls[0].size[0], [l.size for l in ls]
    # mixed paragraph: printed text + one low-confidence OCR line -> split in two, one occurrence removed, box shrunk
    read_handwriting = lambda imgs: [("note line", 0.9)] * len(imgs)  # noqa: F811  fake TrOCR
    blank = Image.new("RGB", (612 * SCALE, 792 * SCALE), "white")
    r = {"id": "p1_0", "page": 1, "type": "text", "label": "text", "bbox": [90, 100, 600, 140], "confidence": 0.8,
         "text": "the fish and the cat the",
         "ocr_cells": [{"bbox": [90, 100, 480, 140], "text": "the fish and the cat", "conf": 0.99},
                       {"bbox": [490, 105, 600, 120], "text": "the", "conf": 0.5}]}
    kept, note = _process(r, blank)
    assert kept["text"] == "fish and the cat the" and kept["bbox"][2] == 482 and len(kept["ocr_cells"]) == 1, kept
    assert note["text"] == "note line" and note["handwritten"] and note["bbox"] == [490, 105, 600, 120], note
    assert _crop(blank, [-5, -5, 20, 20]).size == (69, 69)  # clamped at the page edge
    print("ok")
