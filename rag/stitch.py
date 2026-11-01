"""Per-page items -> one Document: drop headers/footers, attach margin notes, merge cross-page
paragraphs/tables, then the section tree (rag.outline: bookmarks > contents page > headings)."""
import re

from rag import outline

BODY = {"text", "list"}
END = re.compile(r"[.!?:;\"”’)\]]\s*$")


def _margin_notes(items: list[dict], page_width: float) -> list[dict]:
    """Items fully outside the main text column become margin notes, placed right after the body item they sit beside."""
    wide = [i for i in items if i["type"] in BODY and i["bbox"][2] - i["bbox"][0] >= 0.3 * page_width]
    if not wide:
        return items
    # text block = outermost wide columns (works for 1- and 2-column layouts)
    left, right = min(i["bbox"][0] for i in wide), max(i["bbox"][2] for i in wide)
    notes = [i for i in items if i["type"] in BODY and i["bbox"][2] - i["bbox"][0] < 0.3 * page_width
             and (i["bbox"][2] < left - 2 or i["bbox"][0] > right + 2)]
    rest = [i for i in items if i not in notes]
    for n in notes:
        n["type"] = "margin_note"
        cy = (n["bbox"][1] + n["bbox"][3]) / 2
        anchors = [i for i in rest if i["type"] not in ("margin_note", "heading")]
        if anchors:
            a = min(anchors, key=lambda i: 0 if i["bbox"][1] <= cy <= i["bbox"][3] else min(abs(cy - i["bbox"][1]), abs(cy - i["bbox"][3])))
            n["attached_to"] = a["id"]
            rest.insert(rest.index(a) + 1 + sum(1 for x in rest if x.get("attached_to") == a["id"]), n)
        else:
            rest.append(n)
    return rest


CAPTION = re.compile(r"^\s*(table|tab\.|figure|fig\.)\s*\d+", re.I)


def _attach_captions(items: list[dict]) -> list[dict]:
    """Orphan 'Table N:' / 'Fig. N:' text goes to the nearest uncaptioned table/figure on its page."""
    drop = set()
    for c in items:
        m = CAPTION.match(c["text"]) if c["type"] in ("caption", "text") else None
        if not m:
            continue
        kind = "table" if m.group(1).lower().startswith("tab") else "figure"
        cands = [i for i in items if i["page"] == c["page"] and i["type"] == kind and not i.get("caption")]
        if cands:
            t = min(cands, key=lambda i: min(abs(i["bbox"][1] - c["bbox"][3]), abs(c["bbox"][1] - i["bbox"][3])))
            t["caption"] = c["text"]
            t["text"] = c["text"] + "\n" + t["text"]
            drop.add(id(c))
    for i in items:
        if i["type"] == "caption" and id(i) not in drop:
            i["type"] = "text"  # unmatched caption is still content
        if i["type"] in ("table", "figure") and (m := CAPTION.match(i.get("caption") or "")):
            i["type"] = "table" if m.group(1).lower().startswith("tab") else "figure"  # the document's own label wins
    return [i for i in items if id(i) not in drop]


def _table_cols(md: str) -> int:
    first = md.strip().splitlines()[0] if md.strip() else ""
    return first.count("|") - 1


def _merge_cross_page(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        prev = next((x for x in reversed(out) if x["type"] not in ("margin_note", "figure")), None)
        if prev and it["page"] == prev["page_end"] + 1 and _is_page_start(it, items) and _is_page_end(prev, items):
            if prev["type"] in BODY and it["type"] in BODY and not END.search(prev["text"]) and it["text"][:1].islower():
                prev["text"] += " " + it["text"]; prev["page_end"] = it["page"]
                continue
            if prev["type"] == it["type"] == "table" and _table_cols(prev["text"]) == _table_cols(it["text"]) > 0 and not it.get("caption"):
                rows = it["text"].strip().splitlines()
                body = rows[2:] if len(rows) > 2 and set(rows[1].replace("|", "").strip()) <= set("-: ") else rows
                prev["text"] += "\n" + "\n".join(body); prev["page_end"] = it["page"]
                continue
        out.append(it)
    return out


def _is_page_start(it, items):
    return it is next(x for x in items if x["page"] == it["page"] and x["type"] not in ("margin_note",))


def _is_page_end(it, items):
    last = [x for x in items if x["page"] == it["page_end"] and x["type"] not in ("margin_note", "footnote", "figure")]
    return bool(last) and it is last[-1]


def build(doc_id: str, title: str, pages: int, items: list[dict], page_widths: dict, toc: list) -> dict:
    items = [i for i in items if i["type"] not in ("page_header", "page_footer")]
    for i in items:
        i.setdefault("page_end", i["page"])
        if i["type"] == "table":
            i["text"] = re.sub(r" {2,}", " ", i["text"])  # Docling pads columns; padding costs embedding tokens
    by_page = {}
    for i in items:
        by_page.setdefault(i["page"], []).append(i)
    items = [x for p in sorted(by_page) for x in _margin_notes(by_page[p], page_widths.get(p, 612))]
    items = _attach_captions(items)
    items = _merge_cross_page(items)
    items, sections, summary = outline.build(items, toc, title, pages)
    return {"doc_id": doc_id, "title": title, "pages": pages, "items": items, "sections": sections, "outline": summary}


if __name__ == "__main__":  # self-check
    def it(id, page, t, text, bbox=(72, 100, 540, 120)):
        return {"id": id, "page": page, "type": t, "text": text, "bbox": list(bbox)}
    its = [it("h1", 1, "heading", "1 Introduction"), it("a", 1, "text", "First body para."),
           it("b", 1, "text", "This sentence continues"), it("m", 1, "text", "note!", (560, 100, 600, 110)),
           it("f", 1, "page_footer", "Page 1"), it("c", 2, "text", "onto the next page."),
           it("h2", 2, "heading", "1.1 Background"), it("t1", 2, "table", "| a | b |\n|---|---|\n| 1 | 2 |"),
           it("t2", 3, "table", "| a | b |\n|---|---|\n| 3 | 4 |"), it("h3", 3, "heading", "2 Method")]
    its[2]["bbox"] = [72, 130, 540, 150]
    d = build("x", "T", 3, its, {}, [])
    ids = [i["id"] for i in d["items"]]
    assert "f" not in ids and d["items"][ids.index("m")]["type"] == "margin_note"
    assert [s["title"] for s in d["sections"]] == ["1 Introduction", "1.1 Background", "2 Method"]
    assert d["sections"][1]["path"] == ["1 Introduction", "1.1 Background"]
    assert next(i for i in d["items"] if i["id"] == "b")["text"] == "This sentence continues onto the next page." and "c" not in ids
    t = next(i for i in d["items"] if i["id"] == "t1")
    assert t["text"].endswith("| 3 | 4 |") and "t2" not in ids, t["text"]
    # markdown input (API mode): full-width reading-order boxes, heading levels from '#', no margin notes
    def slot(id, page, n, t, text, **kw):
        return {"id": id, "page": page, "type": t, "text": text, "bbox": [61.2, n * 100, 550.8, n * 100 + 90], "source": "mistral", **kw}
    md = [slot("h1", 1, 0, "heading", "Methods", level=1), slot("a", 1, 1, "text", "This sentence continues"),
          slot("c", 2, 0, "text", "onto page two."), slot("t1", 2, 1, "table", "| a | b |\n|---|---|\n| 1 | 2 |"),
          slot("cap", 2, 2, "text", "Table 1: Sizes"), slot("h2", 2, 3, "heading", "Setup", level=2),
          slot("t2", 3, 0, "table", "| a | b |\n|---|---|\n| 3 | 4 |"), slot("s", 3, 1, "text", "short")]
    d = build("x", "T", 3, md, {1: 612, 2: 612, 3: 612}, [])
    ids = [i["id"] for i in d["items"]]
    assert [(s["title"], s["level"]) for s in d["sections"]] == [("Methods", 1), ("Setup", 2)] and d["sections"][1]["path"] == ["Methods", "Setup"]
    assert next(i for i in d["items"] if i["id"] == "a")["text"] == "This sentence continues onto page two." and "c" not in ids
    assert next(i for i in d["items"] if i["id"] == "t1")["caption"] == "Table 1: Sizes" and "cap" not in ids
    assert "t2" in ids and not any(i["type"] == "margin_note" for i in d["items"])  # t2 follows a heading change, so it stays its own table
    print("ok")
