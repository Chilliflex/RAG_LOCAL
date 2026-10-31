"""Ingest one PDF: register -> triage -> parse -> handwriting -> stitch -> enrich -> summaries -> chunk -> index -> audit.
Every stage is resumable; models are freed between stages (4 GB GPU)."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from rag import audit, chunk, config, enrich, handwriting, index, models, parse, stitch, store, summarize


def doc_id_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def doc_id_for(pdf: Path) -> str:
    return doc_id_of(Path(pdf).read_bytes())


def status(doc_id: str) -> dict:
    p = config.DATA / doc_id / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _set_status(doc_id, **kw):
    store.write_json(config.DATA / doc_id / "status.json", {**status(doc_id), **kw, "updated": time.time()})


def ingest(pdf: Path, force: bool = False, rebuild: bool = False) -> dict:
    pdf = Path(pdf)
    doc_id = doc_id_for(pdf)
    doc_dir = config.DATA / doc_id
    if rebuild:  # keep page checkpoints + summary cache, redo stitch onward
        store.path(doc_id).unlink(missing_ok=True)
    if store.path(doc_id).exists() and not force:
        return {"doc_id": doc_id, "skipped": True, **json.loads((doc_dir / "audit.json").read_text())}
    if force and doc_dir.exists():
        shutil.rmtree(doc_dir)
    doc_dir.mkdir(parents=True, exist_ok=True)
    src = doc_dir / "source.pdf"
    if not src.exists():
        shutil.copyfile(pdf, src)
    timings = {}
    models.free()  # the server may still hold the LLM and reranker from a question

    def stage(name, fn, *a, **kw):
        _set_status(doc_id, stage=name, file=pdf.name)
        t = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out = fn(*a, **kw)
        models.free()
        timings[name] = {"seconds": round(time.time() - t, 1),
                         "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0}
        print(f"  {name}: {timings[name]}", flush=True)
        return out

    tri, toc, links, widths = stage("triage", parse.triage, src)
    n = len(tri)
    stage("parse", parse.docling_pages, src, doc_dir, n,
          on_progress=lambda done, total: _set_status(doc_id, stage="parse", done=done, total=total))
    stage("handwriting", handwriting.run, src, doc_dir)
    items = parse.load_pages(doc_dir)
    doc = stitch.build(doc_id, _title(items, pdf), n, items, widths, toc)
    doc["file"] = pdf.name
    doc["urls"] = enrich.urls(doc["items"], links)
    doc["references"] = enrich.references(doc["items"], doc["sections"])
    doc["xrefs"] = enrich.xrefs(doc["items"], doc["sections"])
    stage("summaries", summarize.run, doc, doc_dir,
          on_progress=lambda done, total: _set_status(doc_id, stage="summaries", done=done, total=total))
    chunks = chunk.build(doc)
    stage("index", index.add, chunks)
    report = audit.run(doc, tri, toc)
    report["chunks"] = len(chunks)
    (doc_dir / "timings.json").write_text(json.dumps(timings, indent=1))
    store.save(doc)  # last: document.json existing means "ingested"
    _set_status(doc_id, stage="done", complete=report["complete"])
    return {"doc_id": doc_id, **report}


def _title(items, pdf):
    first = [i for i in items if i["page"] == 1]
    for i in first:
        if i["label"] == "title":
            return i["text"].strip()
    return next((i["text"].strip() for i in first if i["type"] == "heading"), pdf.stem)
