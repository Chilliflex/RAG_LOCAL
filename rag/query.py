"""Router + retrieval + one local LLM answer with page citations."""
import json
import re
import time

from rag import config, index, meta, models, outline, store
from rag.models import count_tokens, generate, reranker

# Plain "Passage n (page N)" headers + a worked example: the 3B model copied page lists from richer headers
SYSTEM = ("Answer the question using only the passages below. Give the answer in one or two sentences, "
          "then the page it came from in parentheses, for example: The model uses 6 layers (p. 3). "
          "If the passages do not contain the answer, reply exactly: Not found in the documents.")
NOT_FOUND = "Not found in the documents."

LOOKUP = re.compile(r"\b(list|how many|count|which|what are|total|number of)\b.*\b(tables?|figures?|images?|pictures?|photos?|diagrams?|charts?|flowcharts?"
                    r"|urls?|links?|websites?|references?|citations?|sections?|subsections?|chapters?|headings?|pages?|formulas?|equations?)\b", re.I)
# content nouns that also name a count: only count words may route them, or "which handwritten line..." becomes a count
COUNT = re.compile(r"\b(how many|count|total|number of)\b.*\b(words?|scanned|handwrit\w*)\b", re.I)
WHOLE = re.compile(r"\b(summar\w*|overview|overall|main (points|ideas|findings|contributions|themes)|whole (document|paper|book)|what is (this|the) (document|paper|book|report) about|key takeaways)\b", re.I)
SECTION = re.compile(r"\b(section|chapter)\s+([\w.]+)|\bin the (introduction|conclusion|abstract|methods?|results|discussion|related work)\b", re.I)
MULTI = re.compile(r"\b(compare|comparison|differ\w*|versus|vs\.?|relationship between|both|how does .+ affect)\b", re.I)


def route(q: str) -> str:
    for name, rx in (("lookup", LOOKUP), ("lookup", COUNT), ("whole_doc", WHOLE), ("section", SECTION), ("multi_hop", MULTI)):
        if rx.search(q):
            return name
    return "fact"


def ask(q: str, doc_ids: list[str] | None = None, depth: str | None = None) -> dict:
    t = time.time()
    models.last_provider = config.PROVIDER
    depth, budget, k = config.depth_settings(depth)
    r = _ask(q, doc_ids, budget, k)
    r["provider"] = models.last_provider  # "local" in cloud mode means Gemini failed and the local model answered
    r["depth"] = depth
    r.setdefault("chunks", [])  # lookup, whole-document and section answers don't come from retrieved chunks
    with open(config.DATA / "queries.jsonl", "a", encoding="utf-8") as f:  # per-question latency/quality log
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": q, "route": r["route"], "provider": r["provider"], "depth": depth,
                            "seconds": round(time.time() - t, 1), "sources": len(r["sources"]),
                            "top_score": max((s.get("score") or 0 for s in r["sources"]), default=None),
                            "not_found": r["answer"] == NOT_FOUND}, ensure_ascii=False) + "\n")
    return r


def _ask(q: str, doc_ids: list[str] | None, budget: int | None = None, k: int = config.TOP_K) -> dict:
    known = store.list_ids()
    docs = [store.load(d) for d in (doc_ids or known) if d in known]
    if not docs:
        return {"answer": "No documents ingested.", "route": None, "sources": []}
    kind = route(q)
    if kind == "lookup" and (ans := _lookup(q, docs)):
        return {"answer": ans, "route": kind, "sources": []}
    if kind == "whole_doc":
        return _whole_doc(q, docs)
    if kind == "section" and (sec := _find_section(q, docs)):
        doc, s = sec
        passages = [_section_passage(doc, s)]
        return _answer(q, passages, kind, budget)
    queries = _decompose(q) if kind == "multi_hop" else [q]
    passages = []
    for sub in queries:
        passages += retrieve(sub, [d["doc_id"] for d in docs], k)
    seen, uniq = set(), []
    for p in sorted(passages, key=lambda p: -p["score"]):
        if p["uid"] not in seen:
            seen.add(p["uid"]); uniq.append(p)
    uniq = uniq[:k + 2]
    if not uniq or max(p["score"] for p in uniq) < config.RELEVANCE_MIN:
        return {"answer": NOT_FOUND, "route": kind, "sources": [], "below_relevance": bool(uniq),
                "chunks": [_chunk_view(n, p) for n, p in enumerate(uniq, 1)]}
    uniq = [p for p in uniq if p["score"] >= config.SCORE_RATIO * uniq[0]["score"]]  # far weaker chunks only distract
    return _answer(q, _expand(uniq, docs), kind, budget)


def retrieve(q: str, doc_ids: list[str], k: int = config.TOP_K) -> list[dict]:
    cands = index.search(q, doc_ids, limit=max(20, 2 * k))
    if not cands:
        return []
    scores = reranker(models.rerank_device()).compute_score([[q, c["text"]] for c in cands], normalize=True)
    for c, s in zip(cands, scores if isinstance(scores, list) else [scores]):
        c["score"] = float(s)
    return sorted(cands, key=lambda c: -c["score"])[:k]


def _expand(passages, docs):
    """Add the item just before/after the passage in its section and up to 2 tables/figures/equations it mentions."""
    by_id = {d["doc_id"]: d for d in docs}
    out = []
    for p in passages:
        doc = by_id[p["doc_id"]]
        order = {i["id"]: n for n, i in enumerate(doc["items"])}
        items = {i["id"]: i for i in doc["items"]}
        pos = [order[i] for i in p["item_ids"] if i in order]
        before = after = ""
        if pos and p["type"] == "text":
            a, b = min(pos) - 1, max(pos) + 1
            if a >= 0 and doc["items"][a]["section_id"] == p["section_id"]:
                before = doc["items"][a]["text"][-400:] + "\n"
            if b < len(doc["items"]) and doc["items"][b]["section_id"] == p["section_id"]:
                after = "\n" + doc["items"][b]["text"][:400]
        linked = [items[e["to"]] for e in doc.get("xrefs", [])
                  if e["from"] in p["item_ids"] and e["to"] in items and e["to"] not in p["item_ids"]][:2]
        text = before + p["text"] + after + "".join(f"\n[Linked {i['type']} p{i['page']}] {i['text']}" for i in linked)
        added = [{"label": "Text before", "text": before.strip()}] * bool(before) + [{"label": "Text after", "text": after.strip()}] * bool(after) \
            + [{"label": f"Linked {i['type']}, p. {i['page']}", "text": i["text"]} for i in linked]
        out.append({**p, "text": text, "pages": sorted(set(p["pages"]) | {i["page"] for i in linked}),
                    "chunk_text": p["text"], "chunk_pages": p["pages"], "added": added})
    return out


def _chunk_view(rank, p) -> dict:
    """One retrieved chunk for the UI: its own text and pages, what expansion added, and whether the model saw it."""
    return {"rank": rank, "score": p.get("score"), "pages": p.get("chunk_pages", p["pages"]), "section": p.get("section"), "type": p["type"],
            "text": p.get("chunk_text", p["text"]), "added": p.get("added", []), "sent": p.get("sent", False), "truncated": p.get("truncated", False)}


def _context(passages, budget_tokens=None) -> str:
    """Passages up to the depth's budget, never past what the window leaves after the answer and instructions."""
    cap = models.context_window() - config.ANSWER_TOKENS - 600
    budget_tokens = min(budget_tokens or cap, cap)
    parts, used = [], 0
    for n, p in enumerate(passages, 1):
        block = f"Passage {n} (page {', '.join(map(str, p['pages']))}):\n{p['text']}"
        cost = count_tokens(block)
        if used + cost > budget_tokens:
            if parts:
                break
            block = block[: int(len(block) * budget_tokens / cost)]  # first passage alone is too big: cut it
            cost = budget_tokens
            p["truncated"] = True
        parts.append(block); used += cost
        p["sent"] = True
    return "\n\n".join(parts)


def _answer(q, passages, kind, budget=None):
    ans = _cite(generate(SYSTEM, f"{_context(passages, budget)}\n\nQuestion: {q}", max_new_tokens=config.ANSWER_TOKENS), passages)
    return {"answer": ans, "route": kind,
            "chunks": [_chunk_view(n, p) for n, p in enumerate(passages, 1)] if kind != "section" else [],
            "sources": [{"doc_id": p["doc_id"], "pages": p["pages"], "section": p["section"], "score": p.get("score"),
                         "type": p["type"], "text": p["text"][:300]} for p in passages]}


def _cite(ans: str, passages) -> str:
    """Normalise '(page 5)' / '(Passage 2)' to '(p. N)'; add the top passage's page if the model gave none."""
    if re.match(r"\W*not\s+found\b", ans, re.I):
        return NOT_FOUND
    pages = lambda n: ", ".join(map(str, passages[n - 1]["pages"])) if 0 < n <= len(passages) else str(n)
    ans = re.sub(r"\(\s*passage\s+(\d+)\s*\)", lambda m: f"(p. {pages(int(m.group(1)))})", ans, flags=re.I)
    ans = re.sub(r"\(\s*pages?\s+([\d,\s\-–]+)\)", r"(p. \1)", ans, flags=re.I)
    if NOT_FOUND.lower().rstrip(".") not in ans.lower() and not re.search(r"\(p+\.\s*\d", ans):
        ans += f" (p. {pages(1)})"
    return ans


def _lookup(q, docs):
    ql, lines = q.lower(), []
    for d in docs:
        items = d["items"]
        pick = None
        if re.search(r"tables?", ql):
            pick = [i for i in items if i["type"] == "table"]
        elif re.search(r"figures?", ql):
            pick = [i for i in items if i["type"] == "figure"]
        elif re.search(r"formulas?|equations?", ql):
            pick = [i for i in items if i["type"] == "formula"]
        if pick is not None:
            lines.append(f"{d['title']}: {len(pick)}")
            lines += [f"  - p{i['page']}: {(i.get('caption') or i['text'])[:100]}" for i in pick]
        elif re.search(r"urls?|links?|websites?", ql):
            lines.append(f"{d['title']}: {len(d['urls'])} URLs")
            lines += [f"  - p{u['page']}: {u['url']}" for u in d["urls"]]
        elif re.search(r"references?|citations?", ql):
            lines.append(f"{d['title']}: {len(d['references'])} references")
            lines += [f"  - {r['text'][:150]}" for r in d["references"]]
        elif re.search(r"sections?|chapters?|headings?", ql):
            secs = [s for s in d["sections"] if s["level"] > 0]
            lines.append(f"{d['title']}: {outline.describe(d['outline']) if d.get('outline') else f'{len(secs)} sections'}")
            lines += [f"  {'  ' * (s['level'] - 1)}- {s['title']} (p{s['pages'][0]}-{s['pages'][1]})" for s in secs]
        elif (m := re.search(r"\b(image|picture|photo|diagram|chart|flowchart|scanned|handwrit|word)", ql)):
            noun = {"handwrit": "handwritten"}.get(m.group(1), m.group(1))
            if (c := meta.count(d, noun)):
                lines.append(f"{d['title']}: {c}")
        elif re.search(r"pages?", ql):
            lines.append(f"{d['title']}: {d['pages']} pages")
    return "\n".join(lines) or None


def _find_section(q, docs):
    m = SECTION.search(q)
    key = (m.group(2) or m.group(3) or "").lower().rstrip(".")
    for d in docs:
        for s in d["sections"]:
            t = s["title"].lower()
            if t.startswith(key) or re.match(rf"^{re.escape(key)}\b", t) or key in t.split():
                return d, s
    return None


def _section_passage(doc, s):
    ids = {s["id"]} | {x["id"] for x in doc["sections"] if s["id"] in x.get("ancestors", [])}
    text = "\n".join(i["text"] for i in doc["items"] if i["section_id"] in ids)
    return {"uid": s["id"], "doc_id": doc["doc_id"], "doc_title": doc["title"], "section": " > ".join(s["path"]),
            "pages": list(range(s["pages"][0], s["pages"][1] + 1)), "type": "section", "item_ids": [],
            "text": text[:9000]}


def _decompose(q) -> list[str]:
    # shortcut: one decomposition round (≤3 sub-questions), not iterative multi-hop; upgrade if eval shows chained questions failing
    out = generate("Split the question into at most 3 short standalone search questions. Reply with a JSON list of strings only.",
                   q, max_new_tokens=150)
    try:
        subs = [s for s in json.loads(out[out.index("["):out.rindex("]") + 1]) if isinstance(s, str)][:3]
    except ValueError:
        subs = []
    return [q] + subs


def _whole_doc(q, docs):
    """Map-reduce over section summaries: notes per batch, then one combined answer."""
    notes, srcs = [], []
    for d in docs:
        summ = [s for s in d["sections"] if s.get("summary")]
        if not summ:  # short doc without summaries: use the start of the document
            text = "\n".join(i["text"] for i in d["items"])[:9000]
            summ = [{"title": d["title"], "pages": [1, d["pages"]], "summary": text}]
        batch, batches = [], []
        for s in summ:
            batch.append(f"{s['title']} (pages {s['pages'][0]}-{s['pages'][1]}): {s['summary']}")
            if len(" ".join(batch).split()) > 1500:
                batches.append(batch); batch = []
        if batch:
            batches.append(batch)
        for b in batches:
            notes.append(generate("Extract the points relevant to the question from these section summaries. Keep page numbers as (p. N).",
                                  f"Summaries of '{d['title']}':\n" + "\n".join(b) + f"\n\nQuestion: {q}", max_new_tokens=300))
        srcs.append({"doc_id": d["doc_id"], "pages": [1, d["pages"]], "section": "whole document", "type": "summary"})
    ans = generate(SYSTEM, "\n\n".join(f"Passage {n}:\n{t}" for n, t in enumerate(notes, 1)) + f"\n\nQuestion: {q}")
    return {"answer": ans, "route": "whole_doc", "sources": srcs}


if __name__ == "__main__":  # self-check of the chunk view, with fake retrieval and LLM: no models needed
    from unittest import mock

    item = lambda i, t: {"id": i, "type": "text", "page": 1, "section_id": "s", "text": t}
    doc = {"doc_id": "d", "title": "T", "sections": [], "xrefs": [{"from": "b", "to": "t"}],
           "items": [item("a", "before text"), item("b", "core"), item("c", "after text"), {**item("t", "| x |"), "type": "table", "page": 3, "section_id": "z"}]}
    hit = lambda u, score, text: {"uid": u, "doc_id": "d", "pages": [1], "item_ids": ["b"], "type": "text", "section_id": "s", "section": "S", "text": text, "score": score}
    patches = {"count_tokens": lambda t: len(t.split()), "generate": lambda s, u, **k: "It is core (p. 1).", "retrieve": None}
    with mock.patch.dict(globals(), patches), mock.patch("rag.store.list_ids", return_value=["d"]), mock.patch("rag.store.load", return_value=doc), \
            mock.patch("rag.models.context_window", return_value=8192):
        globals()["retrieve"] = lambda q, ids, k: [hit("1", 0.9, "core"), hit("2", 0.5, "word " * 3000), hit("3", 0.05, "noise")]
        r = _ask("What is it?", None, 2500, 6)
        first, second = r["chunks"]  # the 0.05 chunk is below SCORE_RATIO x 0.9 and is not sent at all
        assert first["rank"] == 1 and first["score"] == 0.9 and first["text"] == "core" and first["pages"] == [1] and first["sent"] and not first["truncated"], first
        assert [a["label"] for a in first["added"]] == ["Text before", "Text after", "Linked table, p. 3"], first["added"]
        assert not second["sent"] and r["sources"][0]["pages"] == [1, 3]  # over the budget; sources still include the linked page
        globals()["retrieve"] = lambda q, ids, k: [hit("1", 0.5, "word " * 7000)]
        only = _ask("What is it?", None, None, 6)["chunks"][0]
        assert only["sent"] and only["truncated"], only  # budget None = the window: 8192 - answer - 600; the passage is cut to fit
        globals()["retrieve"] = lambda q, ids, k: [hit("1", 0.001, "weak")]
        low = _ask("What is it?", None, 2500, 6)
        assert low["answer"] == NOT_FOUND and low["below_relevance"] and low["chunks"][0]["text"] == "weak" and not low["chunks"][0]["sent"], low
        globals()["retrieve"] = lambda q, ids, k: []
        assert _ask("What is it?", None, 2500, 6)["chunks"] == []
    for q in ("how many total sections are there in this pdf from 1st to last page like sections and subsections", "how many sections and subsections are there?",
              "what is the total number of sections?", "total sections in this pdf?", "how many subsections are there?", "what are the sections of this document?",
              "how many images are there?", "how many diagrams does it have?", "how many scanned pages?"):
        assert route(q) == "lookup", q
    assert route("What was the nitrate level at station C?") == "fact" and route("Which station had the highest turbidity?") == "fact"
    assert route("Which name appears in the handwritten line about victimising someone?") == "fact" and route("how many handwritten passages?") == "lookup"
    assert route("What did the handwritten note say about her wrists?") == "fact" and route("count the words") == "lookup"
    print("ok")
