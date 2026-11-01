"""CLI: python -m rag ingest <pdf>... | ask "<question>" [--doc ID] | docs | eval [golden.json] | serve | doctor"""
import argparse
import json
from pathlib import Path

from rag import config


def main():
    ap = argparse.ArgumentParser(prog="python -m rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("pdfs", nargs="+", type=Path); p.add_argument("--force", action="store_true", help="re-parse from scratch")
    p.add_argument("--rebuild", action="store_true", help="keep parsed pages and summaries, redo stitching/index")
    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("--doc", action="append")
    p.add_argument("--depth", help="local: fast | balanced | maximum; cloud: standard | thorough | maximum")
    sub.add_parser("docs")
    p = sub.add_parser("eval"); p.add_argument("golden", nargs="?", type=Path, default=config.ROOT / "samples/golden.json")
    p.add_argument("--retrieval-only", action="store_true")
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000)
    sub.add_parser("doctor", help="check the requirements of both modes")
    a = ap.parse_args()

    try:
        _run(a)
    except (ValueError, ConnectionError) as e:  # an API refused or unreachable with nothing to fall back on: say so, no traceback
        raise SystemExit(f"error: {e}")


def _run(a):
    if a.cmd == "ingest":
        from rag.ingest import ingest
        for pdf in a.pdfs:
            print(f"ingesting {pdf}", flush=True)
            r = ingest(pdf, force=a.force, rebuild=a.rebuild)
            print(json.dumps({k: r[k] for k in ("doc_id", "pages", "sections", "chunks", "counts", "notes", "gaps", "complete") if k in r}, indent=1))
    elif a.cmd == "ask":
        from rag.query import ask
        r = ask(a.question, a.doc, a.depth)
        print(f"[{r['route']}, depth {r['depth']}{', answered by the local model' if r['provider'] != config.PROVIDER else ''}]\n{r['answer']}\n")
        for s in r["sources"]:
            print(f"  - {s['doc_id']} p{','.join(map(str, s['pages']))} {s['section']} ({s['type']}, score {s.get('score') or 0:.2f})")
    elif a.cmd == "docs":
        from rag import store
        for d in store.list_ids():
            doc = store.load(d)
            print(f"{d}  {doc['pages']:>4}p  {doc['title'][:70]}  ({doc['file']})")
    elif a.cmd == "eval":
        from rag.evaluate import run
        print(json.dumps(run(a.golden, a.retrieval_only), indent=1))
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("rag.api:app", host="127.0.0.1", port=a.port)
    elif a.cmd == "doctor":
        from rag.doctor import render, run
        print(render(run()))


if __name__ == "__main__":
    main()
