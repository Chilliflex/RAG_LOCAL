"""Book-style outline: the section tree from PDF bookmarks, else a printed contents page, else detected headings.
Headings are levelled by their numbering (1.2.3, II., III-A, A., Chapter 3), bookmarks or nesting; the document title,
empty headings sitting between figures and repeated figure text are not sections."""
import re

ROMAN = r"(?=[IVXLC])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
NUMBER = re.compile(
    rf"^\s*(?:(?P<word>chapter|part|lesson|unit|appendix|section)\s+(?P<wn>\d+|{ROMAN}|[A-Z])\b[.:]?"
    rf"|(?P<dec>\d+(?:\.\d+)*)\.?|(?P<roman>{ROMAN})(?:[.\-](?P<rl>[A-Z]))?[.:]?|(?P<letter>[A-Z])[.)]|(?P<paren>\d+)\))\s+(?P<rest>\S.*)$", re.I)
MATTER = re.compile(r"^\s*(abstract|summary|preface|foreword|introduction|acknowledg\w*|references|bibliography|works cited|appendix\w*|glossary|index|conclusions?|coda|epilogue)\s*$", re.I)
CONTENTS = re.compile(r"^\s*(table of )?contents\s*$", re.I)
CONTENTS_LINE = re.compile(r"^\s*(?P<title>.+?)\s*(?:\.{3,}|…+|\s{2,})\s*(?P<page>\d{1,4})\s*$")
BODY = {"text", "list", "table", "formula", "code"}
MAX_LEVEL = 4


def norm_title(t: str) -> str:
    """'3.2 Attention', 'III-A Attention' and 'Attention' compare equal."""
    m = NUMBER.match(t)
    return re.sub(r"[^a-z0-9]", "", (m.group("rest") if m else t).lower())


def number_level(text: str, roman_style: bool) -> int | None:
    """Depth implied by the heading's numbering, or None when it has none."""
    m = NUMBER.match(text)
    if not m:
        return None
    if m.group("word"):
        return 1
    if m.group("dec"):
        return m.group("dec").count(".") + 1
    if m.group("roman"):
        return 2 if m.group("rl") else 1
    if m.group("letter"):
        return 2 if roman_style else 1
    return 3 if roman_style else 2  # "1)"


def looks_like_heading(item: dict, nxt: dict | None, body_size: float) -> bool:
    """A short bold line, larger or equal to body text, with no end punctuation, followed by body text (digital pages only)."""
    f = item.get("font")
    if not f or item["type"] != "text" or not nxt or nxt["type"] not in ("text", "list"):
        return False
    t = item["text"].strip()
    return f["bold"] >= 0.9 and f["size"] >= body_size - 0.5 and len(t.split()) <= 12 and not re.search(r"[.,;:]$", t)


def contents_page(items: list[dict]) -> list:
    """Printed 'Contents' section -> [[level, title, page], ...]; [] when there is no usable one."""
    start = next((n for n, i in enumerate(items) if i["type"] == "heading" and CONTENTS.match(i["text"])), None)
    if start is None:
        return []
    toc = []
    for i in items[start + 1:]:
        if i["type"] == "heading" or i["page"] > items[start]["page"] + 3:
            break
        for line in i["text"].splitlines():
            m = CONTENTS_LINE.match(line)
            if m:
                toc.append([number_level(m.group("title"), False) or 1, m.group("title").strip(), int(m.group("page"))])
    return toc if len(toc) >= 3 else []


def build(items: list[dict], toc: list, title: str, n_pages: int) -> tuple[list[dict], list[dict], dict]:
    """-> (items with section_id, sections, outline summary). items may gain synthetic bookmark headings and lose false ones."""
    source = "bookmarks" if toc else "headings"
    if not toc and (toc := contents_page(items)):
        source = "contents"
    doc_title = norm_title(title)
    sizes = [i["font"]["size"] for i in items if i.get("font") and i["type"] == "text"]
    body_size = sorted(sizes)[len(sizes) // 2] if sizes else 0
    # promote bold short lines Docling left as text; the document title repeated as a running heading is not a section
    for n, it in enumerate(items):
        nxt = items[n + 1] if n + 1 < len(items) else None
        if it["type"] == "text" and looks_like_heading(it, nxt, body_size):
            it["type"], it["promoted"] = "heading", True
        elif it["type"] == "heading" and norm_title(it["text"]) == doc_title:
            it["type"], it["label"] = "text", "title"
    # each bookmark takes the nearest unmatched heading with the same title; on the same page, a heading that sits over a figure
    # (Docling labels figure titles as headings) loses, then one whose numbering matches the bookmark wins
    heads: dict[str, list] = {}
    for n, it in enumerate(items):
        if it["type"] == "heading":
            heads.setdefault(norm_title(it["text"]), []).append((it, n + 1 < len(items) and items[n + 1]["type"] == "figure"))
    unmatched = []
    for lv, t, pg in toc:
        cands = [(h, fig) for h, fig in heads.get(norm_title(t), []) if "bookmark_level" not in h and abs(h["page"] - pg) <= 2]
        if cands:
            numbered = bool(NUMBER.match(t))
            min(cands, key=lambda c: (abs(c[0]["page"] - pg), c[1], bool(NUMBER.match(c[0]["text"])) != numbered))[0]["bookmark_level"] = lv
        else:
            unmatched.append((lv, t, pg))
    # figure titles Docling labels as headings: unnumbered, unbookmarked, and a figure comes right after them (bookmarks make this safe;
    # without bookmarks only when nothing but figures sits under them before the next heading)
    for n, it in enumerate(items):
        nxt = items[n + 1] if n + 1 < len(items) else None
        if it["type"] == "heading" and "bookmark_level" not in it and not NUMBER.match(it["text"]) and nxt and nxt["type"] == "figure":
            until = next((k for k in range(n + 1, len(items)) if items[k]["type"] == "heading"), len(items))
            if toc or not any(x["type"] in BODY for x in items[n + 1:until]):
                it["type"], it["label"] = "text", "figure_text"
    # unmatched bookmarks become headings at the start of their page
    extra = []
    for k, c in enumerate(unmatched):
        pos = next((n for n, i in enumerate(items) if i["page"] >= c[2]), len(items))
        w = items[pos]["bbox"][2] if pos < len(items) else 540
        extra.append((pos, {"id": f"p{c[2]}_toc{k}", "page": c[2], "page_end": c[2], "type": "heading", "text": c[1], "bbox": [60, 0, w, 10],
                            "source": "bookmark", "label": "section_header", "confidence": None, "bookmark_level": c[0], "synthetic": True}))
    for pos, it in sorted(extra, key=lambda e: (-e[0], -e[1]["page"])):  # same slot: later pages first, so they end up in page order
        items.insert(pos, it)
    # the tree
    roman_style = any(it["type"] == "heading" and (m := NUMBER.match(it["text"])) and m.group("roman") and not m.group("rl") for it in items)
    root = {"id": "s0", "title": title, "level": 0, "path": [title], "ancestors": [], "pages": [1, n_pages]}
    sections, stack, cur = [root], [root], root
    for it in items:
        if it["type"] == "heading":
            if "bookmark_level" in it:
                lv = it["bookmark_level"]
            elif (nl := number_level(it["text"], roman_style)):
                lv = nl
            elif MATTER.match(it["text"]):
                lv = 1
            elif toc or it.get("promoted"):
                lv = min(stack[-1]["level"] + 1, MAX_LEVEL)
            else:
                lv = it.get("level") or 1
            while len(stack) > 1 and stack[-1]["level"] >= lv:
                stack.pop()
            cur = {"id": f"s{len(sections)}", "title": it["text"].strip(), "level": lv, "parent": stack[-1]["id"],
                   "ancestors": [s["id"] for s in stack], "path": [s["title"] for s in stack[1:]] + [it["text"].strip()],
                   "pages": [it["page"], it["page"]], "bookmarked": "bookmark_level" in it}
            sections.append(cur); stack.append(cur)
        it["section_id"] = cur["id"]
        for s in stack:
            s["pages"][1] = max(s["pages"][1], it.get("page_end", it["page"]))
    if len(sections) > 1 and not any(i["section_id"] == "s0" for i in items):
        sections = sections[1:]
    real = [s for s in sections if s["level"] > 0]
    levels = {}
    for s in real:
        levels[s["level"]] = levels.get(s["level"], 0) + 1
    summary = {"source": source, "levels": {str(k): v for k, v in sorted(levels.items())}, "total": len(real),
               "bookmarked": sum(s.get("bookmarked", False) for s in real),
               "unbookmarked": [s["title"] for s in real if toc and not s.get("bookmarked")],
               "approximate": source == "headings"}
    return items, sections, summary


def describe(outline: dict) -> str:
    """'3 sections, 12 subsections, 5 sub-subsections (20 total; source: PDF bookmarks)'."""
    names = {"1": "sections", "2": "subsections", "3": "sub-subsections", "4": "level-4 headings"}
    parts = [f"{n} {names[lv]}" for lv, n in outline["levels"].items()]
    src = {"bookmarks": "PDF bookmarks", "contents": "printed contents page", "headings": "detected headings, approximate"}[outline["source"]]
    s = f"{', '.join(parts) or 'no sections'} ({outline['total']} total; source: {src})"
    if outline["unbookmarked"]:
        s += f"; {len(outline['unbookmarked'])} further headings found in the text are not bookmarked, e.g. {', '.join(outline['unbookmarked'][:3])}"
    return s


if __name__ == "__main__":  # self-check
    assert norm_title("III-A Data") == norm_title("A. Data") == "data" and norm_title("Chapter 3: Intro") == "intro"
    assert norm_title("A Survey of LLMs") == "asurveyofllms"  # a bare capital is not numbering
    assert [number_level(t, False) for t in ("1 Intro", "1.2 Back", "1.2.3 Deep", "Chapter 3 Water", "II. Methods", "III-A Data", "Intro")] == [1, 2, 3, 1, 1, 2, None]
    assert number_level("A. Data", True) == 2 and number_level("A. Data", False) == 1 and number_level("In the beginning", True) is None

    def it(id, page, t, text, **kw):
        return {"id": id, "page": page, "type": t, "text": text, "bbox": [72, 100, 540, 120], "label": t, **kw}
    # IEEE style with bookmarks: letters nest under roman; unbookmarked REFERENCES is level 1; title is not a section;
    # a bookmark with no matching heading becomes a synthetic heading; figure text between figures is dropped
    its = [it("t", 1, "heading", "Cloud Cover Nowcasting"), it("h1", 1, "heading", "I. INTRODUCTION"), it("a", 1, "text", "Body."),
           it("h2", 2, "heading", "II. METHODOLOGY"), it("h3", 2, "heading", "A. Data"), it("b", 2, "text", "Body."),
           it("fx", 3, "heading", "Input-Input Layer5"), it("f1", 3, "figure", "Figure 2"), it("h4", 4, "heading", "REFERENCES"), it("r", 4, "text", "[1] ref")]
    toc = [[1, "I Introduction", 1], [1, "II Methodology", 2], [2, "II-A Data", 2], [2, "II-B Protocol", 3]]
    items, secs, out = build(its, toc, "Cloud Cover Nowcasting", 4)
    assert [(s["title"], s["level"], s["bookmarked"]) for s in secs if s["level"]] == \
        [("I. INTRODUCTION", 1, True), ("II. METHODOLOGY", 1, True), ("A. Data", 2, True), ("II-B Protocol", 2, True), ("REFERENCES", 1, False)], secs
    assert next(i for i in items if i["id"] == "t")["type"] == "text" and next(i for i in items if i["id"] == "fx")["type"] == "text"
    # an unnumbered bookmark whose title appears twice on a page: the figure title loses to the real heading, which is then demoted
    its2 = [it("g", 4, "heading", "Scaled Attention"), it("f", 4, "figure", "Figure 2"), it("p", 4, "text", "Body."), it("h", 4, "heading", "3.2.1 Scaled Attention"), it("q", 4, "text", "More.")]
    items2, secs2, _ = build(its2, [[3, "Scaled Attention", 4]], "T", 4)
    assert [(s["title"], s["bookmarked"]) for s in secs2 if s["level"]] == [("3.2.1 Scaled Attention", True)] and next(i for i in items2 if i["id"] == "g")["label"] == "figure_text", secs2
    assert out == {"source": "bookmarks", "levels": {"1": 3, "2": 2}, "total": 5, "bookmarked": 4, "unbookmarked": ["REFERENCES"], "approximate": False}, out
    assert describe(out).startswith("3 sections, 2 subsections (5 total; source: PDF bookmarks); 1 further")
    # no bookmarks: numbering decides, run-in bold line becomes a level-3 heading, a bold line ending in ':' does not
    its = [it("h1", 1, "heading", "1 Intro"), it("h2", 1, "heading", "1.1 Back"), it("p", 1, "text", "Para."),
           it("b", 1, "text", "Knowledge Recency", font={"size": 10, "bold": 1.0}), it("q", 1, "text", "Para two.", font={"size": 10, "bold": 0}),
           it("n", 1, "text", "Note:", font={"size": 10, "bold": 1.0}), it("r", 1, "text", "A note.", font={"size": 10, "bold": 0}), it("h3", 2, "heading", "2 Method")]
    items, secs, out = build(its, [], "T", 2)
    assert [(s["title"], s["level"]) for s in secs if s["level"]] == [("1 Intro", 1), ("1.1 Back", 2), ("Knowledge Recency", 3), ("2 Method", 1)], secs
    assert out["source"] == "headings" and out["approximate"] and out["levels"] == {"1": 2, "2": 1, "3": 1}
    # printed contents page
    its = [it("c", 1, "heading", "Contents"), it("l", 1, "text", "1 Intro ........ 3\n1.1 Back ....... 4\n2 Method ...... 9"), it("h", 3, "heading", "1 Intro"), it("x", 3, "text", "Body.")]
    assert contents_page(its) == [[1, "1 Intro", 3], [2, "1.1 Back", 4], [1, "2 Method", 9]]
    _, secs, out = build(its, [], "T", 9)
    assert out["source"] == "contents" and out["total"] == 4 and [s["title"] for s in secs if s["level"]] == ["Contents", "1 Intro", "1.1 Back", "2 Method"], secs
    print("ok")
