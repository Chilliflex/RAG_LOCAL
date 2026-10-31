# Multimodal PDF RAG: project plan

Built and measured complete on 2026-10-30. This file describes the pipeline and the local (offline) mode. The second mode, "API, fast" (Mistral OCR + Gemini), the mode UI, fallbacks and the two-mode evaluation are in [PLAN_FINAL.md](PLAN_FINAL.md).

Goal: accurate RAG over (A) dense mixed PDFs (handwriting, scans, tables, formulas, diagrams/flows, images, paper/site references) and (B) 100–200+ page PDFs with some of the same content. One pipeline; triage switches modules on. **Fully local and offline**: no API keys, no network calls at runtime. Internet only for the one-time package/model download; then `HF_HUB_OFFLINE=1`, telemetry off.

## Hardware constraints
GTX 1650 4 GB VRAM, 16 GB RAM, ~10 GB free disk. Run models one stage at a time. Put `HF_HOME` on a drive with space (models ≈ 10 GB).
Device setting: `cuda` (NVIDIA) | `mps` (Apple Silicon) | `cpu` (integrated GPUs incl. Ryzen iGPU).

## Stack (Python 3.11)
- Triage/raw text: PyMuPDF
- Layout, tables (TableFormer), OCR (RapidOCR), picture descriptions (SmolVLM-256M): Docling. Formula→LaTeX optional (`FORMULA_LATEX=1`, ~30 s/formula on a GTX 1650); default keeps the formula's text-layer text.
- Handwriting: TrOCR-base-handwritten on (a) OCR lines with RapidOCR confidence < 0.95, (b) pictures made of text lines, (c) embedded images no item covers. TrOCR's reading is kept only if its own confidence ≥ 0.6. No Mermaid conversion.
- References and URLs: regex only (no online lookup).
- Embeddings: BGE-M3 (dense + sparse + multi-vector)
- Vector store: Qdrant embedded mode (qdrant-client, on disk). Structured store: `data/<doc_id>/document.json` (section tree, references, URLs, table/figure registry, cross-reference edges); no SQLite needed at this scale.
- Reranker: bge-reranker-base or jina-reranker-v2-base-multilingual
- Local LLM (answers, section summaries, multi-hop decomposition): Qwen2.5-3B-Instruct Q4_K_M GGUF via llama.cpp (bitsandbytes was ~10× slower at prefill on this GPU); page citations. Router is rule-based (regex), no LLM call.
- Service: FastAPI + simple web page

## Pipeline
1. Register: file hash, skip if already ingested.
2. Triage per page: digital vs scanned, region labels, OCR confidence, page count > ~50 → long-doc modules.
3. Parse per region with the matching tool. Low-confidence OCR lines / handwriting → crop to local TrOCR. Page batches of 10–20 with per-page checkpoints (`data/<doc_id>/pages/NNNN.json`).
4. Stitch: strip headers/footers, merge cross-page paragraphs/tables, attach margin notes by bbox, build section tree (bookmarks + headings + TOC). Output: unified Document JSON (type, text, page, bbox, section path, confidence).
5. Enrich: references, URLs, cross-reference graph, section + document summaries (local LLM).
6. Chunk: structure-first; tables/formulas/figures atomic; semantic split only in heading-less sections; parent–child; free header `[Doc][Section][Pages][Type]`; summary nodes indexed.
7. Embed + index.
8. Completeness audit → `audit.json` (pages parsed, char-count diff vs text layer, TOC vs tree, table/figure counts, reference count, low-confidence list).
9. Query: router (lookup→SQL | fact | section | whole-doc→map-reduce over summaries | multi-hop ≤ 2–3 rounds). Retrieval: dense cosine (HNSW) + sparse dot product → RRF → reranker (MaxSim skipped: the reranker re-scores the same candidates) → top 4–6 → expand (parent, neighbours, linked fig/eq) → relevance gate → one local LLM call with citations.

## Phases
| Phase | Scope | Done when |
|---|---|---|
| 0 Setup | venv, torch CUDA, HF_HOME, offline env, model download, 3 sample PDFs, 30-question golden set | samples + golden set exist, GPU visible to torch |
| 1 Core | triage → Docling → Document JSON → chunking → BGE-M3 → Qdrant hybrid + RRF → rerank → answer; CLI `ingest` / `ask` | correct page-cited answers on a digital PDF |
| 2 Format A | OCR confidence, tables, LaTeX, crop escalation to TrOCR/Florence-2, diagram captions, margin notes, references/URLs | handwriting CER measured; tables/formulas correct on samples |
| 3 Format B | batching + resume, stitching, section tree, summaries, cross-refs, parent–child, router incl. map-reduce + multi-hop | 200-page sample: audit has no gaps; whole-doc questions work |
| 4 Quality | audit report, eval (recall@5, accuracy), threshold tuning, per-stage timing/VRAM log | metrics tracked, thresholds set |
| 5 Service | FastAPI `/upload`, `/ask`, simple UI with progress, audit status, citations | upload + ask from the browser |
