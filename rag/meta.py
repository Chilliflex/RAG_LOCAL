"""Document metadata, built at the end of ingest into doc["meta"]: file, pages, content, visuals, structure, links,
quality, processing. Count questions ("how many images?") are answered from it without a model."""
from collections import Counter
from pathlib import Path

import pymupdf

from rag import config
from rag.outline import describe

TEXT = {"text", "list", "margin_note", "footnote", "heading", "caption"}


def build(doc: dict, pdf: Path, triage: list[dict], timings: dict, api_cost_usd: float | None = None) -> dict:
    with pymupdf.open(pdf) as d:
        m = d.metadata or {}
        encrypted = d.is_encrypted
    items = doc["items"]
    ocr_pages = {i["page"] for i in items if i.get("confidence") is not None or i.get("handwritten")}
    page_types = {t["page"]: "scanned" if t["scanned"] else "mixed" if t["page"] in ocr_pages else "digital" for t in triage}
    pages = Counter(page_types.values())
    text = [i for i in items if i["type"] in TEXT]
    figures = [i for i in items if i["type"] == "figure"]
    confs = [i["confidence"] for i in items if i.get("confidence") is not None]
    words = sum(len(i["text"].split()) for i in items)
    return {
        "file": {"name": doc["file"], "bytes": pdf.stat().st_size, "title": doc["title"], "author": m.get("author") or None,
                 "created": m.get("creationDate") or None, "modified": m.get("modDate") or None, "producer": m.get("producer") or None,
                 "creator": m.get("creator") or None, "format": m.get("format") or None, "encrypted": encrypted},
        "pages": {"total": doc["pages"], "digital": pages["digital"], "scanned": pages["scanned"], "mixed": pages["mixed"],
                  "with_content": len({i["page"] for i in items}), "types": page_types},
        "content": {"printed_items": sum(1 for i in text if i.get("confidence") is None and not i.get("handwritten")),
                    "ocr_items": sum(1 for i in text if i.get("confidence") is not None and not i.get("handwritten")),
                    "handwritten_items": sum(bool(i.get("handwritten")) for i in items),
                    "margin_notes": sum(i["type"] == "margin_note" for i in items),
                    "words": words, "tokens_estimate": int(words * 1.3), "language": None},  # shortcut: no language detection, add a detector when non-English PDFs matter
        "visuals": {"embedded_images": sum(t["images"] for t in triage), "figures": len(figures),
                    "figure_types": dict(Counter(i.get("picture_class") or "unknown" for i in figures)),
                    "tables": sum(i["type"] == "table" for i in items), "formulas": sum(i["type"] == "formula" for i in items),
                    "code_blocks": sum(i["type"] == "code" for i in items)},
        "structure": {**doc["outline"], "summary": describe(doc["outline"])},
        "links": {"urls": len(doc["urls"]), "references": len(doc["references"]),
                  "references_with_doi": sum(bool(r.get("doi")) for r in doc["references"]),
                  "references_with_arxiv": sum(bool(r.get("arxiv")) for r in doc["references"]), "cross_references": len(doc["xrefs"])},
        "quality": {"avg_ocr_confidence": round(sum(confs) / len(confs), 3) if confs else None,
                    "low_confidence_items": sum(c < config.OCR_CONFIDENCE_MIN for c in confs)},
        "processing": {"mode": config.PROVIDER, "parsed_with": doc.get("parsed_with", []),
                       "seconds": round(sum(t["seconds"] for t in timings.values()), 1), "api_cost_usd": api_cost_usd},
    }


def count(doc: dict, noun: str) -> str | None:
    """One line for a count question about images, diagrams, scanned pages, handwriting or words; None if no meta."""
    meta = doc.get("meta")
    if not meta:
        return None
    v, p, c = meta["visuals"], meta["pages"], meta["content"]
    if noun in ("image", "picture", "photo"):
        return f"{v['figures']} figures extracted ({v['embedded_images']} embedded images in the file); types: {_types(v)}"
    if noun in ("diagram", "chart", "flowchart"):
        kinds = {k: n for k, n in v["figure_types"].items() if noun in k.replace("_", "") or (noun == "diagram" and ("flow" in k or "diagram" in k))}
        return f"{sum(kinds.values())} {noun}s among {v['figures']} figures" + (f" ({_types(kinds)})" if kinds else "") + \
            ("; figure types are only classified in local mode" if v["figure_types"] == {"unknown": v["figures"]} and v["figures"] else "")
    if noun == "scanned":
        return f"{p['scanned']} scanned pages, {p['mixed']} mixed and {p['digital']} digital of {p['total']}"
    if noun in ("handwritten", "handwriting"):
        return f"{c['handwritten_items']} handwritten passages ({c['margin_notes']} margin notes)"
    if noun == "word":
        return f"about {c['words']:,} words ({c['tokens_estimate']:,} tokens)"
    return None


def _types(v) -> str:
    types = v.get("figure_types", v) if isinstance(v, dict) else v
    return ", ".join(f"{k} {n}" for k, n in sorted(types.items(), key=lambda kv: -kv[1])) or "none"


if __name__ == "__main__":  # self-check on a tiny generated PDF
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "t.pdf"
        with pymupdf.open() as d:
            d.new_page().insert_text((72, 72), "hello world")
            d.set_metadata({"author": "Ann"}); d.save(pdf)
        doc = {"file": "t.pdf", "title": "T", "pages": 2, "urls": [], "references": [{"doi": "10.1/x"}], "xrefs": [],
               "outline": {"source": "headings", "levels": {"1": 1}, "total": 1, "bookmarked": 0, "unbookmarked": [], "approximate": True},
               "items": [{"page": 1, "type": "text", "text": "hello world", "confidence": None},
                         {"page": 2, "type": "text", "text": "scan", "confidence": 0.9, "handwritten": True},
                         {"page": 2, "type": "figure", "text": "", "picture_class": "bar_chart"}]}
        tri = [{"page": 1, "chars": 11, "images": 0, "scanned": False}, {"page": 2, "chars": 0, "images": 1, "scanned": True}]
        m = build(doc, pdf, tri, {"parse": {"seconds": 1.5}}, None)
        assert m["file"]["author"] == "Ann" and m["pages"]["scanned"] == 1 and m["pages"]["digital"] == 1 and m["content"]["words"] == 3
        assert m["visuals"]["figure_types"] == {"bar_chart": 1} and m["links"]["references_with_doi"] == 1 and m["quality"]["avg_ocr_confidence"] == 0.9
        doc["meta"] = m
        assert count(doc, "chart") == "1 charts among 1 figures (bar_chart 1)" and count(doc, "scanned").startswith("1 scanned pages, 0 mixed and 1 digital")
        assert count(doc, "word") == "about 3 words (3 tokens)" and count({"meta": None}, "word") is None
    print("ok")
