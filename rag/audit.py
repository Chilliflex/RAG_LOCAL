"""Completeness audit: did we parse everything? Writes data/<doc_id>/audit.json."""
import re

from rag import config
from rag.stitch import norm_title
from rag.store import write_json


def run(doc: dict, triage: list[dict], toc: list) -> dict:
    parsed_pages = {i["page"] for i in doc["items"]}
    chars = {}
    for i in doc["items"]:
        chars[i["page"]] = chars.get(i["page"], 0) + len(i["text"])
    gaps = []
    missing = [p for p in range(1, doc["pages"] + 1) if p not in parsed_pages and triage[p - 1]["chars"] + triage[p - 1]["images"] > 0]
    if missing:
        gaps.append(f"{len(missing)} non-empty pages produced no items: {missing[:20]}")
    low_text = [t["page"] for t in triage if not t["scanned"] and t["chars"] > 200 and chars.get(t["page"], 0) < 0.5 * t["chars"]]
    if low_text:
        gaps.append(f"{len(low_text)} digital pages parsed < 50% of text-layer chars: {low_text[:20]}")
    titles = {norm_title(s["title"]) for s in doc["sections"]}
    toc_missing = [t[1] for t in toc if norm_title(t[1]) not in titles]
    if toc and len(toc_missing) > 0.2 * len(toc):
        gaps.append(f"{len(toc_missing)}/{len(toc)} TOC entries not in section tree, e.g. {toc_missing[:5]}")
    text = " ".join(i["text"] for i in doc["items"] if i["type"] not in ("table", "figure"))
    counts = {k: sum(i["type"] == k for i in doc["items"]) for k in ("table", "figure", "formula", "margin_note")}
    counts["handwritten"] = sum(bool(i.get("handwritten")) for i in doc["items"])
    for kind, rx in (("table", r"\b(?:Table|TABLE|Tab\.)\s*(\d+)"), ("figure", r"\b(?:Figure|FIGURE|Fig\.)\s*(\d+)")):
        mentioned = set(re.findall(rx, text))
        captioned = {m.group(1) for i in doc["items"] if i["type"] == kind and (m := re.match(rx, i.get("caption") or ""))}
        missing = sorted(mentioned - captioned, key=int)
        if missing:
            gaps.append(f"{kind}s mentioned in the text but not extracted with a caption: {missing}")
    low = [{"page": i["page"], "type": i["type"], "confidence": round(i["confidence"], 2), "text": i["text"][:80]}
           for i in doc["items"] if i.get("confidence") is not None and i["confidence"] < config.OCR_CONFIDENCE_MIN]
    notes = []
    if cloud := [i for i in doc["items"] if i.get("source") == "mistral"]:
        notes.append(f"{len(cloud)} items read by Mistral OCR: margin notes are not detected and OCR confidence is not reported in API mode")
        blank = [i["page"] for i in cloud if i["type"] == "figure" and not i.get("description")]
        if blank:
            gaps.append(f"{len(blank)} figures have no description: pages {blank[:20]}")
    report = {"doc_id": doc["doc_id"], "pages": doc["pages"], "pages_with_items": len(parsed_pages),
              "scanned_pages": sum(t["scanned"] for t in triage), "sections": len(doc["sections"]), "toc_entries": len(toc),
              "references": len(doc["references"]), "urls": len(doc["urls"]), "xrefs": len(doc["xrefs"]),
              "counts": counts, "low_confidence": low, "notes": notes, "gaps": gaps, "complete": not gaps}
    write_json(config.DATA / doc["doc_id"] / "audit.json", report)
    return report

