"""Structure-first chunking over the Document JSON. Tables, formulas, figures and code stay atomic."""
import uuid

from rag import config

ATOMIC = {"table", "formula", "figure", "code"}


def _tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _chunk(doc, seq, section, items, ctype):
    pages = sorted({i["page"] for i in items})
    path = " > ".join(section["path"]) if section else ""
    text = "\n".join(_render(i) for i in items)
    return {
        "uid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['doc_id']}/{seq}")),
        "doc_id": doc["doc_id"], "doc_title": doc["title"], "seq": seq, "type": ctype,
        "section_id": section["id"] if section else None, "section": path,
        "pages": pages, "item_ids": [i["id"] for i in items], "text": text,
        "embed_text": f"[{doc['title']}][{path}][p{pages[0]}-{pages[-1]}][{ctype}]\n{text}",
    }


def _render(item) -> str:
    t = item["text"]
    hw = ", handwritten" if item.get("handwritten") else ""
    if item["type"] == "margin_note":
        return f"[Margin note{hw}] {t}"
    if hw:
        return f"[Handwritten] {t}"
    if item["type"] == "figure":
        return f"[Figure] {t}"
    return t


def _split_table(item) -> list[dict]:
    """Big tables -> row groups; each repeats caption + header row so it reads on its own."""
    if _tokens(item["text"]) <= 1.5 * config.CHUNK_TOKENS:
        return [item]
    lines = item["text"].splitlines()
    start = next((n for n, l in enumerate(lines) if l.lstrip().startswith("|")), len(lines))
    caption, rows = "\n".join(lines[:start]), lines[start:]
    head = rows[:2] if len(rows) > 2 and set(rows[1].replace("|", "").strip()) <= set("-: ") else rows[:1]
    body, parts, buf = rows[len(head):], [], []
    for r in body:
        if buf and _tokens("\n".join(buf + [r])) > config.CHUNK_TOKENS:
            parts.append(buf); buf = []
        buf.append(r)
    if buf:
        parts.append(buf)
    return [{**item, "text": "\n".join([caption + (" (continued)" if n else "")] * bool(caption) + head + p)}
            for n, p in enumerate(parts)]


def build(doc: dict) -> list[dict]:
    sections = {s["id"]: s for s in doc["sections"]}
    chunks, buf, cur = [], [], None

    def flush():
        nonlocal buf
        if buf:
            chunks.append(_chunk(doc, len(chunks), sections.get(cur), buf, "text"))
        buf = []

    for item in doc["items"]:
        if not item["text"].strip():
            continue
        if item["section_id"] != cur:
            flush()
            cur = item["section_id"]
        if item["type"] in ATOMIC:
            flush()
            for part in _split_table(item) if item["type"] == "table" else [item]:
                chunks.append(_chunk(doc, len(chunks), sections.get(cur), [part], item["type"]))
            continue
        # shortcut: size-based split on item boundaries, no semantic split in heading-less sections; upgrade if eval shows mixed-topic chunks
        if buf and _tokens("\n".join(i["text"] for i in buf + [item])) > config.CHUNK_TOKENS:
            flush()
        buf.append(item)
    flush()

    for s in doc["sections"]:  # summary nodes
        if s.get("summary"):
            chunks.append(_chunk(doc, len(chunks), s, [{"id": s["id"], "type": "summary", "text": s["summary"],
                                                        "page": s["pages"][0]}], "summary"))
    if doc.get("summary"):
        chunks.append(_chunk(doc, len(chunks), None, [{"id": "doc", "type": "summary", "text": doc["summary"], "page": 1}],
                             "doc_summary"))
    return chunks


if __name__ == "__main__":  # self-check
    d = {"doc_id": "x", "title": "T", "summary": "",
         "sections": [{"id": "s1", "title": "A", "path": ["A"], "pages": [1, 1], "summary": "sum"}],
         "items": [{"id": f"i{n}", "type": "text", "text": "word " * 100, "page": 1, "section_id": "s1"} for n in range(5)]
         + [{"id": "t", "type": "table", "text": "| a |", "page": 1, "section_id": "s1"}]}
    c = build(d)
    assert [x["type"] for x in c] == ["text", "text", "text", "table", "summary"], [x["type"] for x in c]
    assert c[0]["embed_text"].startswith("[T][A][p1-1][text]")
    big = {"id": "b", "type": "table", "page": 2, "section_id": "s1",
           "text": "Table 9: Big\n| h1 | h2 |\n|---|---|\n" + "\n".join(f"| {'w ' * 20}| {n} |" for n in range(60))}
    parts = _split_table(big)
    assert len(parts) > 1 and all(p["text"].count("| h1 | h2 |") == 1 and p["text"].startswith("Table 9") for p in parts)
    assert sum(p["text"].count("\n| w") for p in parts) == 60
    print("ok")
