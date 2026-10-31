"""Section + document summaries with the local LLM, cached per section in data/<doc_id>/summaries.json (resumable)."""
import json

from rag.models import generate
from rag.store import write_json

SYS = "You write factual summaries of document sections. 2-4 sentences. Keep key numbers, names and terms. No preamble."
MIN_WORDS = 60
MAX_WORDS = 1800  # keeps prompt + output inside the 4 GB VRAM budget


def run(doc: dict, doc_dir, on_progress=None):
    cache_path = doc_dir / "summaries.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    own = {}
    for i in doc["items"]:
        own.setdefault(i["section_id"], []).append(i["text"])
    todo = [s for s in doc["sections"] if len(" ".join(own.get(s["id"], [])).split()) >= MIN_WORDS]
    key = lambda s: f"{s['title']}|{s['pages'][0]}"  # stable across re-stitching (section ids are positional)
    for n, s in enumerate(todo, 1):
        if key(s) not in cache:
            words = " ".join(own[s["id"]]).split()[:MAX_WORDS]
            cache[key(s)] = generate(SYS, f"Section: {' > '.join(s['path'])} (pages {s['pages'][0]}-{s['pages'][1]})\n\n{' '.join(words)}",
                                      max_new_tokens=160)
            write_json(cache_path, cache)
        if on_progress:
            on_progress(n, len(todo))
    for s in doc["sections"]:
        if key(s) in cache:
            s["summary"] = cache[key(s)]
    if "_doc" not in cache:
        parts = [f"{s['title']}: {s['summary']}" for s in doc["sections"] if s.get("summary")]
        body = " ".join(" ".join(parts).split()[:MAX_WORDS]) or " ".join(" ".join(own.get("s0", [])).split()[:MAX_WORDS])
        cache["_doc"] = generate("Summarise the whole document in one paragraph of 4-6 sentences. No preamble.",
                                 f"Document: {doc['title']}\n\nSection summaries:\n{body}", max_new_tokens=250) if body else ""
        write_json(cache_path, cache)
    doc["summary"] = cache["_doc"]
