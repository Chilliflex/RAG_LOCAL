"""Router + retrieval + one local LLM answer with page citations."""
import json
import re
import time

from rag import config, index, models, store
from rag.models import count_tokens, generate, reranker

# Plain "Passage n (page N)" headers + a worked example: the 3B model copied page lists from richer headers
SYSTEM = ("Answer the question using only the passages below. Give the answer in one or two sentences, "
          "then the page it came from in parentheses, for example: The model uses 6 layers (p. 3). "
          "If the passages do not contain the answer, reply exactly: Not found in the documents.")
NOT_FOUND = "Not found in the documents."

LOOKUP = re.compile(r"\b(list|how many|count|which)\b.*\b(tables?|figures?|urls?|links?|websites?|references?|citations?|sections?|chapters?|pages?|formulas?|equations?)\b", re.I)
WHOLE = re.compile(r"\b(summar\w*|overview|overall|main (points|ideas|findings|contributions|themes)|whole (document|paper|book)|what is (this|the) (document|paper|book|report) about|key takeaways)\b", re.I)
SECTION = re.compile(r"\b(section|chapter)\s+([\w.]+)|\bin the (introduction|conclusion|abstract|methods?|results|discussion|related work)\b", re.I)
MULTI = re.compile(r"\b(compare|comparison|differ\w*|versus|vs\.?|relationship between|both|how does .+ affect)\b", re.I)


def route(q: str) -> str:
    for name, rx in (("lookup", LOOKUP), ("whole_doc", WHOLE), ("section", SECTION), ("multi_hop", MULTI)):
        if rx.search(q):
            return name
    return "fact"


def ask(q: str, doc_ids: list[str] | None = None) -> dict:
    t = time.time()
    models.last_provider = config.PROVIDER
    r = _ask(q, doc_ids)
    r["provider"] = models.last_provider  # "local" in cloud mode means Gemini failed and the local model answered
    with open(config.DATA / "queries.jsonl", "a", encoding="utf-8") as f:  # per-question latency/quality log
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": q, "route": r["route"], "provider": r["provider"],
                            "seconds": round(time.time() - t, 1), "sources": len(r["sources"]),
                            "top_score": max((s.get("score") or 0 for s in r["sources"]), default=None),
                            "not_found": r["answer"] == NOT_FOUND}, ensure_ascii=False) + "\n")
    return r


def _ask(q: str, doc_ids: list[str] | None) -> dict:
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
        return _answer(q, passages, kind)
    queries = _decompose(q) if kind == "multi_hop" else [q]
    passages = []
    for sub in queries:
        passages += retrieve(sub, [d["doc_id"] for d in docs])
    seen, uniq = set(), []
    for p in sorted(passages, key=lambda p: -p["score"]):
        if p["uid"] not in seen:
            seen.add(p["uid"]); uniq.append(p)
    uniq = uniq[:config.TOP_K + 2]
    if not uniq or max(p["score"] for p in uniq) < config.RELEVANCE_MIN:
        return {"answer": NOT_FOUND, "route": kind, "sources": []}
    return _answer(q, _expand(uniq, docs), kind)


def retrieve(q: str, doc_ids: list[str], k: int = config.TOP_K) -> list[dict]:
    cands = index.search(q, doc_ids, limit=20)
    if not cands:
        return []
    scores = reranker(config.DEVICE).compute_score([[q, c["text"]] for c in cands], normalize=True)
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
        out.append({**p, "text": text, "pages": sorted(set(p["pages"]) | {i["page"] for i in linked})})
    return out


def _context(passages, budget_tokens=None) -> str:
    """Local: 2500 context + instructions + 512 answer stays inside LLM_CTX 8192. Cloud: 12000."""
    budget_tokens = budget_tokens or config.CONTEXT_TOKENS
    parts, used = [], 0
    for n, p in enumerate(passages, 1):
        block = f"Passage {n} (page {', '.join(map(str, p['pages']))}):\n{p['text']}"
        cost = count_tokens(block)
        if used + cost > budget_tokens:
            if parts:
                break
            block = block[: int(len(block) * budget_tokens / cost)]  # first passage alone is too big: cut it
            cost = budget_tokens
        parts.append(block); used += cost
    return "\n\n".join(parts)


def _answer(q, passages, kind):
    ans = _cite(generate(SYSTEM, f"{_context(passages)}\n\nQuestion: {q}"), passages)
    return {"answer": ans, "route": kind,
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
        elif re.search(r"sections?|chapters?", ql):
            lines.append(f"{d['title']}: {len(d['sections'])} sections")
            lines += [f"  {'  ' * (s['level'] - 1)}- {s['title']} (p{s['pages'][0]})" for s in d["sections"]]
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
