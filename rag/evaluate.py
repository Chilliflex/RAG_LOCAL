"""Eval on a golden set: recall@5 (a gold page in the top-5 retrieved chunks), answer accuracy, handwriting CER."""
import json
import re
import time

from rag import config, store
from rag.ingest import doc_id_for


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()


def cer(ref: str, hyp: str) -> float:
    """Character error rate = Levenshtein distance / reference length."""
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def handwriting_cer(truth_path, doc_id) -> dict:
    """Match each ground-truth line to the extracted item with the largest bbox overlap on the same page."""
    truth = json.loads(truth_path.read_text())
    items = store.load(doc_id)["items"]
    rows = []
    for t in truth:
        tb = t["bbox"]
        best, best_ov = None, 0.0
        for i in items:
            if i["page"] != t["page"]:
                continue
            b = i["bbox"]
            ov = max(0, min(tb[2], b[2]) - max(tb[0], b[0])) * max(0, min(tb[3], b[3]) - max(tb[1], b[1]))
            if ov > best_ov:
                best, best_ov = i, ov
        hyp = best["text"] if best else ""
        # an item may hold several lines; score against its best-matching line
        line = min(hyp.splitlines() or [""], key=lambda h: cer(t["text"], h))
        rows.append({"page": t["page"], "margin": t["margin"], "truth": t["text"], "got": line, "cer": round(cer(t["text"], line), 3)})
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {"cer": mean([r["cer"] for r in rows]), "cer_body": mean([r["cer"] for r in rows if not r["margin"]]),
            "cer_margin": mean([r["cer"] for r in rows if r["margin"]]), "lines": rows}


def run(golden_path, retrieval_only=False) -> dict:
    from rag.query import ask, retrieve, route
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    ids = {f: doc_id_for(config.ROOT / "samples" / f) for f in {g["file"] for g in golden}}
    rows = []
    for g in golden:
        did = ids[g["file"]]
        row = {"question": g["question"], "route": route(g["question"])}
        if g.get("pages") or g["answer"] == ["Not found"]:
            top = retrieve(g["question"], [did], k=5)
            if g.get("pages"):
                row["recall@5"] = any(set(c["pages"]) & set(g["pages"]) for c in top)
            row["retrieved_pages"] = [c["pages"] for c in top]
            row["top_score"] = round(top[0]["score"], 4) if top else None
        if not retrieval_only:
            t = time.time()
            r = ask(g["question"], [did])
            row["seconds"] = round(time.time() - t, 1)
            row["answer"] = r["answer"]
            # shortcut: correct = answer contains a gold key phrase; upgrade to a judged score if phrasing varies a lot
            row["correct"] = any(_norm(a) in _norm(r["answer"]) for a in g["answer"])
        rows.append(row)
    rec = [r["recall@5"] for r in rows if "recall@5" in r]
    acc = [r["correct"] for r in rows if "correct" in r]
    out = {"questions": len(rows), "recall@5": round(sum(rec) / len(rec), 3) if rec else None,
           "accuracy": round(sum(acc) / len(acc), 3) if acc else None}
    truth = config.ROOT / "samples/mixed_handwriting.truth.json"
    mixed = config.ROOT / "samples/mixed_handwriting.pdf"
    if truth.exists() and mixed.exists() and store.path(doc_id_for(mixed)).exists():
        hw = handwriting_cer(truth, doc_id_for(mixed))
        out["handwriting_cer"] = {k: hw[k] for k in ("cer", "cer_body", "cer_margin")}
        rows.append({"handwriting_lines": hw["lines"]})
    path = config.DATA / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({**out, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    out["details"] = str(path)
    return out


if __name__ == "__main__":  # self-check
    assert cer("hello", "hello") == 0 and cer("hello", "hallo") == 0.2 and cer("ab", "") == 1
    print("ok")
