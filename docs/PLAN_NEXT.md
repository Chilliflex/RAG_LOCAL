# Plan: book-style index, full metadata, hardware-based token limits

Status: all four phases built and evaluated on 2026-10-31 (`rag/outline.py`, `rag/meta.py`, depth and window logic in `config`/`models`/`query`, `GET /doc/{id}`, details panel and depth selector in the UI). Results: survey 115/115 bookmarks matched (10 real unbookmarked headings listed separately), Attention 22/22 (+Abstract, References), Berthomier 14/14 with `A.` subsections at level 2; all 9 count wordings route to lookup; metadata spot-checked (tables 4, formulas 6, figures 8, 40 references on the Attention paper; words 6,622 vs 6,095 in the raw text layer because figure descriptions and table markup count); steady-state peak 2.5 GB with 0.82 GB free on the GTX 1650 (reranker on the CPU, 8k window); eval 50/53, recall@5 1.0, CER 0.031, same as before the change. Two ideas were tried and dropped: the score-ratio chunk filter (cost 5 answers, default now 0) and reloading the reranker per question (peak VRAM crept up to 3.5 GB; the reranker now stays resident or moves to the CPU). Follows [PLAN_FINAL.md](PLAN_FINAL.md). Fixes three problems found in testing: wrong section counts, too little stored information per PDF, and token limits fixed in code regardless of the machine.

## 1. Book-style document index

A contents tree built during ingest, with chapters, sections and subsections, page ranges, counts per level and the source used:

1. PDF bookmarks (authoritative when present).
2. A printed contents page.
3. Detected headings scored on numbering (`Chapter 1`, `1.1.1`, `II.`, `A.`), bold or larger text, short length, space above and what follows; duplicates of the title and text inside figures are dropped.

Problems it fixes: survey 126 vs 115 bookmarks; Attention paper title counted twice and figure text as headings; Berthomier `II.`/`A.` levels lost; 4 of 6 count wordings routed to chunk search; one combined total instead of counts per level.

Gate: counts match bookmarks on PDFs that have them (differences explained); all 6 tested wordings route to lookup; `rag.stitch` and `rag.outline` self-checks pass.

## 2. Full document metadata

`doc["meta"]`: file size, author, dates, producer, encrypted; page types (digital/scanned/mixed) and a per-page map; printed/OCR/handwritten item counts, words, token estimate; embedded images and figure types; counts per outline level and source; DOI/arXiv totals; average OCR confidence; mode and API cost per document. Shown in a details panel in the UI; count questions ("how many images / diagrams / scanned pages") answered from it with no model call.

Gate: values match a manual check on 3 PDFs; image, diagram and scanned-page counts answered directly.

## 3. Hardware-based token limits

Local: free VRAM (or RAM on CPU) measured when the LLM loads; window chosen so 0.5-1 GB stays free (about 16k on a 4 GB GTX 1650); the reranker is unloaded before the LLM answers. Depth setting Fast / Balanced / Maximum controls how much text is sent (speed), separately from the window (memory). API: Standard / Thorough / Maximum with estimated cost; VRAM irrelevant except for the local fallback. Both: longer answers allowed (1024 local, 2048 API); chunk count varies with score, adopted only if it beats fixed counts; limits and detected hardware shown in the UI.

Gate: peak VRAM stays within the reserve on the 4 GB GPU with no out-of-memory during an answer; local answers still pass.

## 4. Re-evaluate and document

Gate: no drop on the 33/33 baseline and the 17/20 harder set; README and project_Structure.md updated.

Not included: live API testing (needs keys), a "find everything" route, testing on larger GPUs, the local handwriting "Not found" misses.
