"""References, URLs and the cross-reference graph. Regex only, no online lookups (offline by design)."""
import re

URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'\]\[)(,;]+[^\s<>\"'\]\[)(,;.:]", re.I)
DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>,;]+[^\s\"<>,;.]")
ARXIV = re.compile(r"\barXiv:\s?\d{4}\.\d{4,5}(?:v\d+)?", re.I)
REF_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|works cited|literature cited)\s*$", re.I)
NUMBERED_REF = re.compile(r"^\s*\[\d+\]\s+\S")
MENTION = re.compile(r"\b(Table|Tab\.|Figure|Fig\.|Equation|Eq\.|Section|Sec\.|§)\s*\(?(\d+(?:\.\d+)*)\)?", re.I)
KIND = {"table": "table", "tab.": "table", "figure": "figure", "fig.": "figure", "equation": "formula", "eq.": "formula",
        "section": "section", "sec.": "section", "§": "section"}


def urls(items: list[dict], link_uris: list[dict]) -> list[dict]:
    """link_uris: [{"page", "url"}] from PDF link annotations."""
    seen, out = set(), []
    for u in link_uris + [{"page": i["page"], "url": m.group(0)} for i in items for m in URL.finditer(i["text"])]:
        key = u["url"].rstrip("/").lower()
        if key not in seen and not key.startswith(("mailto:", "#")):
            seen.add(key); out.append(u)
    return out


def references(items: list[dict], sections: list[dict]) -> list[dict]:
    ref_secs = {s["id"] for s in sections if REF_HEADING.match(s["title"])}
    refs = [i for i in items if i["section_id"] in ref_secs and i["type"] in ("text", "list") and len(i["text"]) > 20]
    if not refs:
        refs = [i for i in items if NUMBERED_REF.match(i["text"])]
    return [{"page": i["page"], "item_id": i["id"], "text": i["text"],
             "doi": (DOI.search(i["text"]) or [None])[0], "arxiv": (ARXIV.search(i["text"]) or [None])[0]} for i in refs]


def _number(item) -> str | None:
    """Number of a captioned table/figure, or an equation tag like (4)."""
    if item["type"] in ("table", "figure"):
        m = re.match(r"\s*(?:Table|Tab\.|Figure|Fig\.)\s*(\d+(?:\.\d+)*)", item.get("caption") or "", re.I)
        return m.group(1) if m else None
    if item["type"] == "formula":
        m = re.search(r"\\tag\{(\d+(?:\.\d+)*)\}|\((\d+(?:\.\d+)*)\)\s*$", item["text"])
        return (m.group(1) or m.group(2)) if m else None
    return None


def xrefs(items: list[dict], sections: list[dict]) -> list[dict]:
    targets = {}
    for i in items:
        if (n := _number(i)):
            targets.setdefault((i["type"], n), i["id"])
    for s in sections:
        if (m := re.match(r"\s*(\d+(?:\.\d+)*)\.?\s", s["title"])):
            first = next((i["id"] for i in items if i["section_id"] == s["id"]), None)
            if first:
                targets.setdefault(("section", m.group(1)), first)
    edges, seen = [], set()
    for i in items:
        if i["type"] in ("table", "figure", "formula"):
            continue
        for m in MENTION.finditer(i["text"]):
            to = targets.get((KIND[m.group(1).lower()], m.group(2)))
            if to and to != i["id"] and (i["id"], to) not in seen:
                seen.add((i["id"], to))
                edges.append({"from": i["id"], "to": to, "label": m.group(0)})
    return edges


if __name__ == "__main__":  # self-check
    its = [{"id": "a", "type": "text", "page": 1, "section_id": "s1", "text": "See Table 2 and Eq. (3), https://x.org/a."},
           {"id": "t", "type": "table", "page": 2, "section_id": "s1", "text": "|x|", "caption": "Table 2: Results"},
           {"id": "f", "type": "formula", "page": 2, "section_id": "s1", "text": "E = mc^2 \\tag{3}"},
           {"id": "r", "type": "text", "page": 3, "section_id": "s2", "text": "[1] A. Author. A paper. doi 10.1000/xyz.12."}]
    secs = [{"id": "s1", "title": "1 Intro"}, {"id": "s2", "title": "References"}]
    assert [e["to"] for e in xrefs(its, secs)] == ["t", "f"]
    assert urls(its, [])[0]["url"] == "https://x.org/a"
    assert references(its, secs)[0]["doi"] == "10.1000/xyz.12"
    print("ok")
