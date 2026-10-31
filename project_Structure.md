# Project structure

A developer reference for this codebase: what every file and function does, what goes in and out, which model or library it uses, and where to make a change.

Line numbers are correct as of 2026-10-31. If one has drifted, search the file for the function name. Run commands from the project root with `.venv\Scripts\python`. Related documents: [README.md](README.md) (setup and use), [PLAN.md](docs/PLAN.md) (pipeline design), [PLAN_FINAL.md](docs/PLAN_FINAL.md) (API mode, phases and gates).

**Contents:** [1. Big picture](#1-big-picture) · [2. Folder map](#2-folder-map) · [3. Call flow](#3-call-flow) · [4. Data formats](#4-data-formats) · [5. Function reference](#5-function-reference) · [6. HTTP API](#6-http-api) · [7. Settings and thresholds](#7-settings-and-thresholds) · [8. Where to change things](#8-where-to-change-things) · [9. Testing](#9-testing) · [10. Known limits](#10-known-limits)

---

## 1. Big picture

The app turns a PDF into a searchable structured document, then answers questions from it with page citations. It has two modes, selected by `config.PROVIDER`:

| Stage | Local mode (`local`) | API mode (`cloud`) | Code |
|---|---|---|---|
| Triage | PyMuPDF | PyMuPDF | `parse.triage` |
| Layout, text, tables, OCR | Docling: Heron layout, TableFormer, RapidOCR | Mistral OCR | `parse.docling_pages` / `parse_mistral.pages` |
| Formulas | Text-layer text; LaTeX if `FORMULA_LATEX=1` | LaTeX from Mistral | `parse._items` / `parse_mistral.blocks` |
| Handwriting | TrOCR-base-handwritten | Mistral OCR | `handwriting.run` / `parse_mistral.pages` |
| Figure descriptions | SmolVLM-256M (inside Docling) | Gemini 2.5 Flash | `parse._converter` / `cloud.describe_image` |
| Stitching, references, cross-references | Plain Python and regex | Same | `stitch.build`, `enrich.*` |
| Summaries, answers, sub-questions | Qwen2.5-3B Q4_K_M GGUF, llama.cpp | Gemini 2.5 Flash | `models.generate` |
| Chunking | Plain Python | Same | `chunk.build` |
| Embeddings | BGE-M3 dense + sparse | Same, local | `index.encode` |
| Vector store | Qdrant embedded | Same, local | `index.*` |
| Reranker | bge-reranker-base | Same, local | `query.retrieve` |

Rules that shape the code:

- **Local mode makes no network calls.** Cloud code sits in `cloud.py` and `parse_mistral.py`, and is imported only inside `PROVIDER == "cloud"` branches. `doctor.py` is the one exception: it imports `cloud` to run its checks when you ask for them.
- **One heavy model at a time.** The target machine has a 4 GB GPU, so `ingest.stage` calls `models.free()` after every stage.
- **Each mode has its own data folder**, `data/` or `data-cloud/`, because the parsers produce different content.
- **Everything is resumable.** Pages and summaries are checkpointed to disk. `document.json` is written last and means "ingested".

---

## 2. Folder map

```text
RAG/
├── rag/                        The application package
│   ├── __main__.py             CLI: ingest, ask, docs, eval, serve, doctor
│   ├── config.py               Settings, paths, thresholds, mode switching
│   ├── store.py                document.json read/write, atomic JSON writes
│   │
│   ├── api.py                  FastAPI web service
│   ├── static/index.html       The whole web UI (HTML, CSS, JS in one file)
│   ├── doctor.py               Requirement checks for both modes
│   │
│   ├── ingest.py               Ingest orchestrator: runs every stage in order
│   ├── parse.py                Triage + Docling parsing (local)
│   ├── parse_mistral.py        Mistral markdown to items (API)
│   ├── handwriting.py          TrOCR pass over page checkpoints (local)
│   ├── stitch.py               Pages to one document: merges, captions, section tree
│   ├── enrich.py               References, URLs, cross-references (regex)
│   ├── summarize.py            Section and document summaries
│   ├── chunk.py                Document to searchable chunks
│   ├── index.py                Qdrant + BGE-M3: store and search
│   ├── audit.py                Completeness check after ingest
│   │
│   ├── query.py                Router, retrieval, expansion, answers
│   ├── models.py               Local model loaders; generate() routes by mode
│   ├── cloud.py                Mistral and Gemini HTTP calls, usage log, cost
│   └── evaluate.py             Golden-set evaluation, handwriting error rate
│
├── scripts/
│   ├── download_models.py      One-time model download (--api-only for API mode)
│   ├── compare_modes.py        Runs both modes, writes data/eval_compare.json
│   └── make_mixed_sample.py    Builds the handwriting sample PDF and its truth file
│
├── langflow/                   Separate, simpler API-only flow for Langflow/Flowgen
│   ├── ingest.flowspec.json    Flow: PDF to Mistral OCR to Qdrant Cloud
│   ├── chat.flowspec.json      Flow: question to Qdrant Cloud to Gemini
│   ├── mistral_ocr.py          Custom Langflow component for Mistral OCR
│   └── check_flows.py          Offline test of the component and both flows
│
├── samples/                    Test PDFs, golden.json (53 questions), handwriting truth
├── docs/                       Plans and this map
├── .claude/                    Ponytail coding rules (not used at runtime)
├── CLAUDE.md                   Rules for coding agents working in this repo
├── requirements.txt            Python dependencies (torch and llama-cpp install separately)
│
├── .env                        Your settings (git-ignored; may hold API keys)
├── models/                     Downloaded models, ~12 GB (git-ignored)
├── data/                       Local mode documents; settings.json lives here too (git-ignored)
└── data-cloud/                 API mode documents and usage log (git-ignored)
```

Runtime files for one document, in `data/<doc_id>/` or `data-cloud/<doc_id>/`:

```text
<doc_id>/                 doc_id = first 16 hex characters of the PDF's SHA-256
├── source.pdf            Copy of the upload
├── status.json           Current stage and progress, polled by the UI
├── pages/0001.json       Checkpoint: the items of one page
├── images/               Figure crops (API mode only)
├── summaries.json        Summary cache
├── document.json         Final structured document; its existence means "ingested"
├── audit.json            Completeness report
└── timings.json          Seconds and peak VRAM per stage
```

Shared files in each data folder: `qdrant/` (vector store), `uploads/`, `queries.jsonl` (question log), `eval_*.json`. `data-cloud/` also has `api_usage.jsonl`. `data/` also has `settings.json` and `eval_compare.json`, whichever mode is active.

---

## 3. Call flow

### Ingest

```text
UI: POST /upload → api.upload → thread: api._ingest          CLI: __main__._run
                                     └────────► ingest.ingest(pdf) ◄──┘
  1 parse.triage                     PyMuPDF: scanned flag, TOC, links, widths
  2 parse (one of):
      local → parse.docling_pages → _converter, _items, _ocr_cells   (Docling)
              handwriting.run → _process, _lines, _uncovered, _dedupe (TrOCR)
      cloud → ingest._cloud_parse → cloud.require_keys
              parse_mistral.pages → _chunks → cloud.ocr              (Mistral)
                                  → items → blocks, _describe        (Gemini)
              on failure → the local path above for pages still missing
  3 parse.load_pages                 all pages/*.json into one list
  4 stitch.build                     _margin_notes, _attach_captions,
                                     _merge_cross_page, section tree
  5 enrich.urls, references, xrefs   regex
  6 summarize.run → models.generate  Qwen or Gemini
  7 chunk.build                      chunks + summary nodes
  8 index.add → encode               BGE-M3 → Qdrant
  9 audit.run                        audit.json
 10 store.save                       document.json, written last
```

### Question

```text
UI: POST /ask → api.ask                                     CLI: __main__._run
                     └───────► query.ask → query._ask ◄──────┘
  route(q) picks one:
    lookup    → _lookup                      from document.json, no model
    whole_doc → _whole_doc                   map-reduce over section summaries
    section   → _find_section, _section_passage → _answer
    multi_hop → _decompose, then as fact
    fact      → retrieve → index.search      BGE-M3 → Qdrant dense + sparse → RRF
                         → reranker          bge-reranker-base, keep TOP_K
              → RELEVANCE_MIN gate           below it: "Not found in the documents."
              → _expand                      neighbours + linked table/figure/formula
              → _answer → _context           token budget
                        → models.generate    Qwen or Gemini
                        → _cite              normalises to (p. N)
```

---

## 4. Data formats

Most modules pass plain dicts, so these formats are the real interfaces between them.

**Item**: one block of content. Produced by `parse._items` and `parse_mistral.items`, consumed by everything after them.

| Field | Type | Meaning |
|---|---|---|
| `id` | str | `p12_3`: item 3 of page 12. TrOCR adds `p12_3h` and `p12_u0` |
| `page`, `page_end` | int | Start page; end page after cross-page merging (`page_end` is set by `stitch.build`) |
| `type` | str | `text`, `list`, `heading`, `table`, `figure`, `formula`, `code`, `footnote`, `caption`, `page_header`, `page_footer`, `margin_note` |
| `text` | str | Content. Tables are markdown; figures are caption + description |
| `bbox` | [x0, y0, x1, y1] | Position in PDF points, top-left origin. **API mode: an evenly spaced, full-width reading-order slot, not a real position** |
| `source` | str | `docling`, `trocr` or `mistral` |
| `label` | str | Original parser label, such as `section_header` |
| `confidence` | float or null | Mean OCR confidence; null for text-layer text and all Mistral items |
| `section_id` | str | Added by `stitch.build` |

Optional fields:

| Field | Set for | Meaning |
|---|---|---|
| `level` | Headings | Heading depth |
| `caption` | Tables, figures | Caption text |
| `picture_class` | Figures, local mode | Classifier label |
| `ocr_cells` | Text, local mode | OCR lines: `{bbox, text, conf}` |
| `handwritten`, `hw_checked`, `ocr_text`, `figure_text` | Local mode | TrOCR state and the text it replaced |
| `attached_to` | Margin notes | ID of the neighbouring item |
| `image`, `image_bbox`, `description` | Figures, API mode | Crop path, real position on the page, Gemini's description |

**Document** (`document.json`):

```json
{"doc_id": "...", "title": "...", "pages": 180, "file": "report.pdf", "parsed_with": ["docling", "trocr"],
 "summary": "whole-document summary",
 "items": ["item, ..."],
 "sections": [{"id": "s2", "title": "1 Intro", "level": 1, "parent": "s0", "ancestors": ["s0"],
               "path": ["1 Intro"], "pages": [3, 9], "summary": "..."}],
 "urls": [{"page": 4, "url": "https://..."}],
 "references": [{"page": 170, "item_id": "p170_2", "text": "[1] ...", "doi": null, "arxiv": null}],
 "xrefs": [{"from": "p5_1", "to": "p6_0", "label": "Table 1"}]}
```

**Chunk**: built by `chunk._chunk`. Stored in Qdrant as the point payload, without `embed_text`.

```json
{"uid": "uuid5 of doc_id/seq", "doc_id": "...", "doc_title": "...", "seq": 12,
 "type": "text | table | formula | figure | code | summary | doc_summary",
 "section_id": "s4", "section": "3 Model > 3.2 Attention", "pages": [4, 5], "item_ids": ["p4_1"],
 "text": "what the LLM reads", "embed_text": "[title][section][p4-5][text]\nwhat gets embedded"}
```

**Answer**: returned by `query.ask` and `POST /ask`.

```json
{"answer": "... (p. 5)", "route": "fact", "provider": "local",
 "sources": [{"doc_id": "...", "pages": [5], "section": "...", "score": 0.91, "type": "text", "text": "first 300 characters"}]}
```

`provider` is `"local"` in API mode when Gemini failed and Qwen answered. In a `whole_doc` answer each source has only `doc_id`, `pages`, `section` and `type`.

**Other files:**

| File | Contents |
|---|---|
| `status.json` | `stage` (`queued`, `triage`, `parse`, `handwriting`, `summaries`, `index`, `done`, `error`), `file`, `done`, `total`, `complete`, `error`, `fallback`, `updated` |
| `audit.json` | `doc_id`, `pages`, `pages_with_items`, `scanned_pages`, `sections`, `toc_entries`, `references`, `urls`, `xrefs`, `counts`, `low_confidence`, `notes`, `gaps`, `complete`, `chunks` |
| `settings.json` | `provider`, `mistral_api_key`, `google_api_key` |
| `api_usage.jsonl` | One line per call: `time`, `op`, `seconds`, then `pages` and `bytes`, or `in_tokens` and `out_tokens`, or `error` |
| `queries.jsonl` | `time`, `question`, `route`, `provider`, `seconds`, `sources`, `top_score`, `not_found` |
| `samples/golden.json` | List of `{file, question, answer: [accepted phrases], pages, category}`. `category` is optional |

---

## 5. Function reference

Names starting with `_` are private to their module. "Self-check" means `python -m rag.<module>` runs assertions without models or keys, except where noted.

### Shared foundations

#### [rag/config.py](rag/config.py): settings and mode

Runs once on import: loads `.env`, resolves `HF_HOME`, then sets module-level constants (see [section 7](#7-settings-and-thresholds)). Read values as `config.DATA`, not `from config import DATA`, because mode switching replaces them at runtime.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`load_settings`](rag/config.py#L16) | none | dict | Reads `data/settings.json`, or `{}` if it doesn't exist |
| [`save_settings`](rag/config.py#L20) | `**kw` | none | Merges values into `settings.json` and restricts its permissions. Not atomic |
| [`_derived`](rag/config.py#L27) | `provider` | `(data_dir, context_tokens)` | Values that depend on the mode. Raises `ValueError` on an unknown provider |
| [`set_mode`](rag/config.py#L35) | `provider`, `**keys` | none | Switches mode in this process and persists it. Empty keys keep the stored ones. Callers must also call `index.reset()` |

#### [rag/store.py](rag/store.py): document storage

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`write_json`](rag/store.py#L8) | `p`, `obj` | none | Atomic write through a temp file, so a crash can't leave half a file |
| [`path`](rag/store.py#L16) | `doc_id` | Path | Path of `document.json` |
| [`save`](rag/store.py#L20) | `doc` | none | Writes `document.json` |
| [`load`](rag/store.py#L24) | `doc_id` | dict | Reads `document.json` |
| [`list_ids`](rag/store.py#L28) | none | list[str] | Every ingested document in the active mode |

#### [rag/models.py](rag/models.py): local models and LLM routing

Self-check: `generate` falls back correctly when Gemini fails.

| Function | Input | Output | Uses | Purpose |
|---|---|---|---|---|
| [`free`](rag/models.py#L12) | none | none | torch, gc | Closes the LLM, clears all model caches, empties VRAM. Called between stages |
| [`embedder`](rag/models.py#L23) | `device` | `BGEM3FlagModel` | FlagEmbedding, BGE-M3 | Cached loader; fp16 on CUDA |
| [`reranker`](rag/models.py#L29) | `device` | `FlagReranker` | FlagEmbedding, bge-reranker-base | Cached loader |
| [`llm`](rag/models.py#L35) | none | `Llama` | llama-cpp-python, Qwen GGUF | Cached loader. `n_ctx` = `LLM_CTX`; all layers on GPU when `DEVICE=cuda` |
| [`generate`](rag/models.py#L46) | `system`, `user`, `max_new_tokens=512` | str | Gemini or Qwen | **The one LLM entry point.** Cloud calls `cloud.generate`; if that fails and the GGUF exists, Qwen answers and `last_provider` becomes `"local"`. Temperature 0 |
| [`count_tokens`](rag/models.py#L62) | `text` | int | Qwen tokenizer or estimate | Used for the prompt budget |
| [`trocr`](rag/models.py#L70) | none | `(processor, model)` | transformers | Cached TrOCR loader |
| [`read_handwriting`](rag/models.py#L77) | list of single-line PIL images | list of `(text, confidence)` | TrOCR | Batches of 8. Confidence is the mean token probability |

Module variable: `last_provider`, reset by `query.ask` at the start of each question.

#### [rag/cloud.py](rag/cloud.py): Mistral and Gemini over HTTPS

Uses `httpx` directly, with no vendor SDK. Self-check: mocked retries, errors, usage log, keys never logged.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`require_keys`](rag/cloud.py#L20) | none | none | Raises `ValueError` naming any missing key |
| [`_log`](rag/cloud.py#L26) | `op`, `seconds`, `**kw` | none | Appends to `api_usage.jsonl`. Never receive text or keys here |
| [`_post`](rag/cloud.py#L33) | `op`, `url`, `body`, `headers`, `timeout` | `(json, seconds)` | 3 attempts with 2 s and 4 s backoff on 429, 5xx and network errors. Exhausted retries raise `ConnectionError`; other non-200 responses raise `ValueError` |
| [`_mistral_headers`](rag/cloud.py#L54) | none | dict | Bearer auth header |
| [`ocr`](rag/cloud.py#L58) | PDF bytes | Mistral response | One OCR call: base64 `data:` URL, images included, headers and footers split out. 600 s timeout |
| [`_gemini`](rag/cloud.py#L68) | `op`, `parts`, `system`, `max_tokens` | str | `generateContent` with temperature 0 and thinking budget 0. 60 s timeout. Raises `ValueError` on an empty or blocked reply |
| [`generate`](rag/cloud.py#L82) | `system`, `user`, `max_new_tokens` | str | Text generation with Gemini |
| [`describe_image`](rag/cloud.py#L86) | image bytes, `mime` | str | Figure description using the `DESCRIBE` prompt, max 200 tokens |
| [`count_tokens`](rag/cloud.py#L90) | `text` | int | Estimate: characters / 4 + 1 |
| [`check_mistral`](rag/cloud.py#L94) | none | `(ok, detail)` | Tests the key with `GET /models`. Free |
| [`check_google`](rag/cloud.py#L104) | none | `(ok, detail)` | Tests the key with `countTokens` |
| [`cost`](rag/cloud.py#L115) | usage-log entries | dict with `calls`, `ocr_pages`, token totals, `usd` | Prices entries using `PRICES` |
| [`online`](rag/cloud.py#L124) | none | bool | Whether `api.mistral.ai` can be reached |

#### [rag/__main__.py](rag/__main__.py): command line

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`main`](rag/__main__.py#L9) | `sys.argv` | none | Defines the subcommands. Turns `ValueError` and `ConnectionError` into a one-line error |
| [`_run`](rag/__main__.py#L28) | parsed args | none | Runs one subcommand. To add a command, add it to both functions |

### Ingest pipeline, in order

#### [rag/ingest.py](rag/ingest.py): orchestrator

Self-check: API-down fallback and the missing-models error.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`doc_id_of`](rag/ingest.py#L14) | bytes | str | First 16 hex characters of SHA-256 |
| [`doc_id_for`](rag/ingest.py#L18) | PDF path | str | Same, for a file |
| [`status`](rag/ingest.py#L22) | `doc_id` | dict | Reads `status.json` |
| [`_set_status`](rag/ingest.py#L27) | `doc_id`, `**kw` | none | Merges values into `status.json` |
| [`ingest`](rag/ingest.py#L31) | `pdf`, `force=False`, `rebuild=False` | audit report + `doc_id`, or `skipped: true` | **Runs the whole pipeline.** `force` deletes everything and starts over. `rebuild` keeps pages and summaries and redoes stitching onward |
| [`stage`](rag/ingest.py#L48) (nested) | `name`, `fn`, args | `fn`'s result | Sets status, times the stage, records peak VRAM, frees models |
| [`_cloud_parse`](rag/ingest.py#L87) | `doc_id`, `src`, `doc_dir`, `n`, `progress` | none | Mistral parse. On `ConnectionError` or `ValueError`, parses the remaining pages locally and records `fallback`. Raises a readable error if local models are missing |
| [`_title`](rag/ingest.py#L103) | items, `pdf` | str | First title or heading on page 1, else the file name |

#### [rag/parse.py](rag/parse.py): triage and local parsing

| Function | Input | Output | Uses | Purpose |
|---|---|---|---|---|
| [`triage`](rag/parse.py#L16) | PDF path | `(pages, toc, links, widths)` | PyMuPDF | Per page: `chars`, `images`, `scanned` (under 50 characters and has an image). Runs in both modes |
| [`_converter`](rag/parse.py#L30) | none | `DocumentConverter` | Docling | Pipeline settings: RapidOCR, TableFormer, figure classifier, SmolVLM. Batch sizes hold peak VRAM inside 4 GB |
| [`_ocr_cells`](rag/parse.py#L47) | `cells`, `bbox`, `height` | list of `{bbox, text, conf}` | none | OCR lines whose centre is inside a box. Text-layer text is excluded |
| [`_items`](rag/parse.py#L63) | Docling result, PDF path | `{page: [items]}` | Docling, PyMuPDF | Converts Docling's document to item records. Tables export to markdown |
| [`docling_pages`](rag/parse.py#L110) | `pdf`, `doc_dir`, `n_pages`, `on_progress` | none, writes `pages/` | Docling | Parses `PAGE_BATCH` pages at a time, skipping checkpointed pages |
| [`load_pages`](rag/parse.py#L133) | `doc_dir` | list of items | none | Concatenates all checkpoints in page order |

Constant: `LABELS` maps Docling labels to item types.

#### [rag/parse_mistral.py](rag/parse_mistral.py): API mode parsing

Self-check: a fixed markdown fixture through to chunks, request splitting and resume, with no network.

| Function | Input | Output | Uses | Purpose |
|---|---|---|---|---|
| [`_describe`](rag/parse_mistral.py#L19) | image bytes, `mime` | str | Gemini | Thin wrapper; the self-check replaces it |
| [`blocks`](rag/parse_mistral.py#L24) | markdown | ordered blocks | regex | Line parser for headings, text, lists, tables, `$$` formulas, code fences and images. **Change this if Mistral's markdown changes** |
| [`items`](rag/parse_mistral.py#L70) | OCR page, `pno`, `doc_dir` | item records | Gemini per figure | Adds item fields and reading-order boxes, saves crops to `images/`, describes figures |
| [`_chunks`](rag/parse_mistral.py#L98) | `pdf`, `todo` pages | yields `(page numbers, PDF bytes)` | PyMuPDF | Splits work to fit `MAX_PAGES` and `MAX_BYTES` |
| [`part`](rag/parse_mistral.py#L101) (nested) | page list | yields `(pages, bytes)` | PyMuPDF | Builds a sub-PDF and halves it until under `MAX_BYTES` |
| [`pages`](rag/parse_mistral.py#L115) | `pdf`, `doc_dir`, `n_pages`, `on_progress` | none, writes `pages/` | Mistral | Sends unprocessed pages and maps each response index back to the real page number |

#### [rag/handwriting.py](rag/handwriting.py): TrOCR pass (local only)

Updates page checkpoints in place. Safe to re-run because of `hw_checked`. Self-check: line splitting and mixed printed/handwritten items, with a fake TrOCR.

| Function | Input | Output | Uses | Purpose |
|---|---|---|---|---|
| [`split_lines`](rag/handwriting.py#L20) | PIL image | list of line images | OpenCV, NumPy | Horizontal projection: finds rows of ink and trims each line |
| [`_crop`](rag/handwriting.py#L53) | page image, `bbox`, `pad` | PIL image | PIL | Crops at `SCALE` and clamps to the page |
| [`_union`](rag/handwriting.py#L59) | boxes | box | none | Box covering all the boxes |
| [`_lines`](rag/handwriting.py#L63) | item, page image | accepted lines | TrOCR | Re-reads lines under `OCR_CONFIDENCE_MIN`, keeps readings at or above `HANDWRITING_CONFIDENCE_MIN` |
| [`_process`](rag/handwriting.py#L73) | item, page image | replacement items | TrOCR | Figures made of text lines become text; mixed items split into printed and handwritten parts |
| [`_overlap`](rag/handwriting.py#L114) | two boxes | area | none | Intersection area |
| [`_uncovered`](rag/handwriting.py#L118) | page, items, image, `pno` | new items | PyMuPDF, TrOCR | Tries embedded images that no item covers |
| [`_dedupe`](rag/handwriting.py#L139) | items | items | none | Keeps the largest reading when handwritten items overlap by over 50% |
| [`run`](rag/handwriting.py#L151) | `pdf`, `doc_dir` | count of handwritten items | PyMuPDF at 216 dpi | Walks all page checkpoints |

Constant: `CHARTS`, picture classes that are never treated as handwriting.

#### [rag/stitch.py](rag/stitch.py): pages into one document

Plain Python. Self-check covers Docling-style and markdown-style input.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`norm_title`](rag/stitch.py#L10) | title | str | Strips numbering and punctuation so `3.2 Attention` equals `Attention` |
| [`_margin_notes`](rag/stitch.py#L15) | one page's items, `page_width` | items | Narrow body items outside the main text column become `margin_note`, placed after the item beside them |
| [`_attach_captions`](rag/stitch.py#L41) | items | items | Moves `Table N:` / `Fig. N:` text to the nearest uncaptioned table or figure on the page. The document's own label decides the type |
| [`_table_cols`](rag/stitch.py#L63) | markdown table | int | Column count of the first row |
| [`_merge_cross_page`](rag/stitch.py#L68) | items | items | Joins paragraphs (no end punctuation, next starts lowercase) and tables (same column count, no new caption) across consecutive pages |
| [`_is_page_start`](rag/stitch.py#L85) | item, items | bool | First on its page, ignoring margin notes |
| [`_is_page_end`](rag/stitch.py#L89) | item, items | bool | Last on its page, ignoring margin notes, footnotes and figures |
| [`_level`](rag/stitch.py#L94) | heading, `toc_levels` | int | Depth: bookmark, then numbering, then parser level |
| [`build`](rag/stitch.py#L102) | `doc_id`, `title`, `pages`, `items`, `page_widths`, `toc` | document without urls, references, xrefs | **Main entry.** Runs the steps above and builds the section tree |
| [`it`](rag/stitch.py#L136), [`slot`](rag/stitch.py#L153) | test values | item | Self-check helpers only |

#### [rag/enrich.py](rag/enrich.py): references and links

Regex only. Self-check included.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`urls`](rag/enrich.py#L14) | items, link annotations | `[{page, url}]` | Merges PDF links and URLs in text, de-duplicated |
| [`references`](rag/enrich.py#L24) | items, sections | `[{page, item_id, text, doi, arxiv}]` | Items under a References heading, else items starting `[N]` |
| [`_number`](rag/enrich.py#L33) | item | str or None | Number of a table, figure or equation |
| [`xrefs`](rag/enrich.py#L44) | items, sections | `[{from, to, label}]` | Links "see Table 2" and "Eq. (3)" mentions to their targets |

#### [rag/summarize.py](rag/summarize.py): summaries

| Function | Input | Output | Uses | Purpose |
|---|---|---|---|---|
| [`run`](rag/summarize.py#L12) | `doc`, `doc_dir`, `on_progress` | none, changes `doc` | `models.generate` | Summarizes each section of at least `MIN_WORDS` (60), using up to `MAX_WORDS` (1800), then the whole document. Cached in `summaries.json` per section title and start page |

#### [rag/chunk.py](rag/chunk.py): chunking

Self-check included.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`_tokens`](rag/chunk.py#L9) | text | int | Estimate: words × 1.3 |
| [`_chunk`](rag/chunk.py#L13) | `doc`, `seq`, `section`, `items`, `ctype` | chunk | Builds the record and the `embed_text` label. The `uid` is deterministic |
| [`_render`](rag/chunk.py#L26) | item | str | Adds `[Margin note]`, `[Handwritten]` or `[Figure]` so the LLM knows the source |
| [`_split_table`](rag/chunk.py#L38) | table item | items | Splits tables over 1.5 × `CHUNK_TOKENS` by rows, repeating caption and header |
| [`build`](rag/chunk.py#L57) | document | chunks | **Main entry.** Never crosses a section; `ATOMIC` types stay whole; text splits at item boundaries near `CHUNK_TOKENS`; adds summary chunks |
| [`flush`](rag/chunk.py#L61) (nested) | none | none | Emits the buffered text as a chunk |

#### [rag/index.py](rag/index.py): vector store

Uses Qdrant embedded on disk, which only one process can open at a time.

| Function | Input | Output | Uses | Purpose |
|---|---|---|---|---|
| [`client`](rag/index.py#L14) | none | `QdrantClient` | qdrant-client | Cached; opens the active mode's `qdrant/` folder |
| [`reset`](rag/index.py#L20) | none | none | | Closes the store after a mode switch |
| [`_ensure`](rag/index.py#L29) | none | none | | Creates the `chunks` collection: dense 1024-dim cosine + sparse |
| [`_sparse`](rag/index.py#L38) | BGE-M3 weights | `SparseVector` | | Format conversion |
| [`encode`](rag/index.py#L42) | texts, `device` | dense and sparse vectors | BGE-M3 | Batch 8, max length 512 |
| [`add`](rag/index.py#L46) | chunks | none | BGE-M3, Qdrant | Deletes the document's old vectors, then upserts in batches of 64 |
| [`delete_doc`](rag/index.py#L59) | `doc_id` | none | Qdrant | Removes one document's vectors |
| [`search`](rag/index.py#L64) | `query`, `doc_ids`, `limit=30`, `device="cpu"` | chunk payloads | BGE-M3, Qdrant | Dense + sparse prefetch, merged by Reciprocal Rank Fusion. The query embeds on CPU to save VRAM for the LLM |

#### [rag/audit.py](rag/audit.py): completeness check

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`run`](rag/audit.py#L9) | `doc`, triage, `toc` | report; writes `audit.json` | Flags: empty pages, digital pages under 50% of text-layer characters, over 20% of bookmarks missing, mentioned tables or figures not extracted, Mistral figures with no description |

### Question answering

#### [rag/query.py](rag/query.py): router, retrieval, answers

No self-check; covered by `rag eval`.

| Function | Input | Output | Uses | Purpose |
|---|---|---|---|---|
| [`route`](rag/query.py#L21) | question | `lookup`, `whole_doc`, `section`, `multi_hop` or `fact` | regex | First match wins, in that order |
| [`ask`](rag/query.py#L28) | `q`, `doc_ids=None` | Answer | | **Public entry.** Adds `provider`, appends to `queries.jsonl` |
| [`_ask`](rag/query.py#L41) | `q`, `doc_ids` | Answer | | Dispatches by route. Applies the `RELEVANCE_MIN` gate |
| [`retrieve`](rag/query.py#L69) | `q`, `doc_ids`, `k=TOP_K` | scored chunks | BGE-M3, Qdrant, bge-reranker-base | Hybrid search for 20 candidates, rerank, keep `k` |
| [`_expand`](rag/query.py#L79) | passages, docs | passages | | Adds up to 400 characters from each neighbour and up to 2 linked items |
| [`_context`](rag/query.py#L102) | passages, `budget_tokens=None` | prompt text | `count_tokens` | Stops at `CONTEXT_TOKENS`; cuts the first passage if it alone is too long |
| [`_answer`](rag/query.py#L118) | `q`, passages, `kind` | Answer | `models.generate` | The single answer call |
| [`_cite`](rag/query.py#L125) | answer, passages | str | regex | Normalizes citations to `(p. N)`; adds one if missing |
| [`_lookup`](rag/query.py#L137) | `q`, docs | str or None | | Lists or counts tables, figures, formulas, URLs, references, sections, pages. No model |
| [`_find_section`](rag/query.py#L165) | `q`, docs | `(doc, section)` or None | | Matches a section by number or title word |
| [`_section_passage`](rag/query.py#L176) | `doc`, section | passage | | Section plus subsections, first 9000 characters |
| [`_decompose`](rag/query.py#L184) | `q` | list of questions | `models.generate` | The original question plus up to 3 sub-questions |
| [`_whole_doc`](rag/query.py#L195) | `q`, docs | Answer | `models.generate` | Batches of section summaries to notes, then one combined answer |

Constants: `SYSTEM` (answer prompt), `NOT_FOUND`, and the regexes `LOOKUP`, `WHOLE`, `SECTION`, `MULTI`.

#### [rag/evaluate.py](rag/evaluate.py): evaluation

Self-check covers `cer` only.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`_norm`](rag/evaluate.py#L10) | str | str | Lowercases and strips punctuation for phrase matching |
| [`cer`](rag/evaluate.py#L14) | `ref`, `hyp` | float | Levenshtein distance / reference length |
| [`handwriting_cer`](rag/evaluate.py#L25) | truth path, `doc_id` | error rates, per-line rows | Matches truth lines to items by box overlap. Local mode only |
| [`run`](rag/evaluate.py#L49) | golden path, `retrieval_only` | `provider`, `questions`, `recall@5`, `accuracy`, `by_category`, `details` | An answer is correct if it contains an accepted phrase. Writes `eval_*.json` |

### Service and UI

#### [rag/api.py](rag/api.py): FastAPI

Binds to 127.0.0.1. Self-check: foreign host and origin refused, non-PDF refused, mode switching, keys never echoed. Endpoint contracts are in [section 6](#6-http-api).

| Function | Purpose |
|---|---|
| [`same_origin`](rag/api.py#L29) | Middleware: refuses POSTs from other origins with 403 |
| [`page`](rag/api.py#L37) | Serves `static/index.html` |
| [`upload`](rag/api.py#L42) | Validates, saves and starts an ingest thread |
| [`_ingest`](rag/api.py#L60) | Thread body: holds the `GPU` lock, records failures in `status.json` |
| [`get_status`](rag/api.py#L70) | Status plus audit; validates the ID format |
| [`doctor`](rag/api.py#L81) | Calls `doctor.run` |
| [`Mode`](rag/api.py#L86), [`Ask`](rag/api.py#L124) | Request bodies (pydantic) |
| [`get_mode`](rag/api.py#L93) | Mode and whether keys exist, never their values |
| [`set_mode`](rag/api.py#L101) | Validates, switches mode, closes the store. 409 while busy |
| [`docs`](rag/api.py#L115) | Lists documents |
| [`ask`](rag/api.py#L130) | Holds the `GPU` lock (409 if busy), caps questions at 2000 characters, returns API failures as 503 |

#### [rag/doctor.py](rag/doctor.py): requirement checks

Self-check: mocked network; key values never appear in the report.

| Function | Input | Output | Purpose |
|---|---|---|---|
| [`_check`](rag/doctor.py#L21) | `name`, `ok`, `detail` | check dict | One line of the report |
| [`_models`](rag/doctor.py#L25) | model list | checks | Tests each model folder exists and isn't empty |
| [`run`](rag/doctor.py#L34) | none | `provider`, `local`, `api`, `apis`, `local_models`, `api_models` | Measures Python, GPU, VRAM, RAM, disk, models, internet and keys |
| [`render`](rag/doctor.py#L63) | report | str | Terminal output |

Constants: `LOCAL_MODELS`, `API_MODELS`, `APIS`. **Edit `APIS` to change the prices and links shown in the UI**, and `cloud.PRICES` to change cost calculations.

#### [rag/static/index.html](rag/static/index.html): web UI

One file with no build step and no external scripts or fonts. Colours are CSS variables on `:root`, and the dark theme is defined twice: once for the OS setting and once for `data-theme="dark"`. **Change both when editing dark colours.**

| JavaScript | Purpose |
|---|---|
| `$`, `esc` | `getElementById`; HTML escaping. **Pass all server text through `esc`** |
| `applyTheme(t)` | Applies System, Light or Dark and remembers it in `localStorage` |
| `loadMode()` | `GET /mode`, then shows the cards or the main page |
| `check(c)` | Renders one requirement line |
| `showReq(p)` | `GET /doctor`, renders the requirements panel and key fields, wires `POST /mode` |
| `loadDocs()` | `GET /docs`, renders the document list |
| `poll(id)` | `GET /status/{id}` every 2 s until `done` or `error` |
| `$("upload").onclick` | `POST /upload`, then polls |
| `$("ask").onclick` | `POST /ask`, renders the answer and sources. Ctrl+Enter also sends |

Constants: `NAMES`, `LOCAL_METHODS`, `API_METHODS` (text on the panels). The comparison on the two mode cards is plain HTML.

### Scripts

| File | Functions | Purpose |
|---|---|---|
| [scripts/download_models.py](scripts/download_models.py) | Top-level script | Downloads models from Hugging Face into `models/`, plus the Docling bundle. `--api-only` fetches only BGE-M3 and the reranker |
| [scripts/compare_modes.py](scripts/compare_modes.py) | [`run`](scripts/compare_modes.py#L20) (`provider` to results, including cost), [`main`](scripts/compare_modes.py#L38) | Runs `rag ingest` and `rag eval` per mode in subprocesses, prints a table, writes `data/eval_compare.json`. `--skip-local` skips local |
| [scripts/make_mixed_sample.py](scripts/make_mixed_sample.py) | Top-level script | Builds `samples/mixed_handwriting.pdf` and its truth file from IAM handwriting data |

### Langflow

A separate, simpler API-only version. It doesn't import `rag/`, and it uses Qdrant Cloud and `mistral-embed` in place of the local models.

| File | Contents |
|---|---|
| [langflow/mistral_ocr.py](langflow/mistral_ocr.py) | [`pages_to_rows`](langflow/mistral_ocr.py#L34) (OCR pages to `{text, page, source}` rows), class [`MistralOCR`](langflow/mistral_ocr.py#L53) with [`_call`](langflow/mistral_ocr.py#L68) (POST with retries) and [`read_pages`](langflow/mistral_ocr.py#L86) (PDF to DataFrame) |
| [langflow/ingest.flowspec.json](langflow/ingest.flowspec.json) | FlowSpec: `mistral_ocr` → `split_text` → `qdrant` (with `embeddings`) → `chat_output` |
| [langflow/chat.flowspec.json](langflow/chat.flowspec.json) | FlowSpec: `chat_input` → `qdrant` (with `embeddings`) → `parser` → `prompt` → `gemini` → `chat_output` |
| [langflow/check_flows.py](langflow/check_flows.py) | Offline test using Flowgen's venv. Takes the Flowgen folder and `catalog.json` as arguments |

---

## 6. HTTP API

All endpoints are served only on `127.0.0.1` or `localhost`. POSTs from other origins get 403.

| Method and path | Request | Response | Errors |
|---|---|---|---|
| `GET /` | | The web page | |
| `POST /upload` | Multipart `file` | `{doc_id, status: "queued" or "already ingested"}` | 413 over 300 MB, 415 not a PDF |
| `GET /status/{doc_id}` | | `status.json` contents, plus `audit` when ready | 400 bad ID |
| `GET /docs` | | `[{doc_id, title, file, pages, complete}]` | |
| `POST /ask` | `{question, doc_ids}`; `doc_ids` may be null | Answer | 400 empty question, 409 busy ingesting, 503 API failed with no fallback |
| `GET /mode` | | `{provider, chosen, env_override, keys: {mistral, google}}`, where `keys` are booleans | |
| `POST /mode` | `{provider, mistral_api_key, google_api_key}` | Same as `GET /mode` | 400 bad provider, 409 busy |
| `GET /doctor` | | The `doctor.run` report | |

---

## 7. Settings and thresholds

Set these in `.env` unless marked as a code constant.

| Name | Default | Where it applies | Effect |
|---|---|---|---|
| `PROVIDER` | `local` | Everywhere | `local` or `cloud`. Overrides `settings.json` at startup |
| `MISTRAL_API_KEY`, `GOOGLE_API_KEY` | none | `cloud.py` | Override `settings.json` |
| `MISTRAL_OCR_MODEL`, `GEMINI_MODEL` | `mistral-ocr-latest`, `gemini-2.5-flash` | `cloud.py` | API models |
| `DEVICE` | `cuda` | `models.py`, `parse.py` | `cuda`, `mps` or `cpu` |
| `HF_HOME` | `models` | `config.py` | Model folder |
| `OCR_CONFIDENCE_MIN` | 0.95 | `handwriting._lines`, `audit.run` | Lower sends fewer lines to TrOCR |
| `HANDWRITING_CONFIDENCE_MIN` | 0.6 | `handwriting.py` | Higher rejects more TrOCR readings |
| `RELEVANCE_MIN` | 0.005 | `query._ask` | Higher gives more "Not found" answers |
| `FORMULA_LATEX` | `0` | `parse._converter` | `1` converts formulas to LaTeX in local mode |
| `LLM_REPO`, `LLM_FILE` | Qwen2.5-3B GGUF | `config.py` | Local LLM; re-run the download script |
| `PAGE_BATCH` (code) | 10 | `parse.docling_pages` | Lower uses less memory |
| `CHUNK_TOKENS` (code) | 350 | `chunk.py` | Chunk size; requires `ingest --rebuild` |
| `TOP_K` (code) | 6 | `query.retrieve` | Passages kept after reranking |
| `CONTEXT_TOKENS` (code) | 2500 local, 12000 cloud | `config._derived` | Passage budget per answer |
| `LLM_CTX` (code) | 8192 | `models.llm` | Qwen context window |
| `MIN_WORDS`, `MAX_WORDS` (code) | 60, 1800 | `summarize.py` | Which sections get summaries, and how much text is read |
| `MAX_PAGES`, `MAX_BYTES` (code) | 1000, 50 MB | `parse_mistral.py` | Mistral request limits |
| `PRICES` (code) | See file | `cloud.py` | Used for cost reports |

---

## 8. Where to change things

| To change | Edit | Then |
|---|---|---|
| The answer's wording or style | `query.SYSTEM` | `rag eval` |
| Which route a question takes | Regexes at the top of `query.py` | `rag eval` |
| Retrieval quality | `index.search`, `query.retrieve`, `TOP_K`, `RELEVANCE_MIN` | `rag eval` |
| Citation format | `query._cite` | `rag eval` |
| Chunk size or rules | `chunk.build`, `CHUNK_TOKENS` | `rag.chunk`, then `ingest --rebuild` on each PDF |
| Margin note, caption or merge rules | `stitch.py` | `rag.stitch`, then `ingest --rebuild` |
| A new reference or URL pattern | `enrich.py` | `rag.enrich`, then `ingest --rebuild` |
| A new item type | `parse.LABELS` or `parse_mistral.blocks`, plus `chunk.ATOMIC` and `chunk._render` if needed | `ingest --force` |
| Figure description prompt | `cloud.DESCRIBE` | `ingest --force` in API mode |
| Summary prompt | `summarize.SYS` | Delete `summaries.json`, then `ingest --rebuild` |
| The local LLM | `LLM_REPO`, `LLM_FILE`; lower the budget if `LLM_CTX` is smaller | Run the download script, then `rag eval` |
| The API LLM | `GEMINI_MODEL`. Another provider: add a function in `cloud.py` and call it from `generate` and `describe_image` | `rag.cloud`, then `rag eval` |
| The API OCR provider | New adapter producing item records, called from `ingest._cloud_parse` | `ingest --force` |
| The embedding model | `config.REPOS`, `index._ensure` (vector size), `index.encode` | Delete `qdrant/`, re-ingest everything |
| A new pipeline stage | `ingest.ingest`, wrapped in `stage(...)` if it loads a model | `ingest --rebuild` |
| A new audit check | `audit.run`: append to `gaps` or `notes` | `ingest --rebuild` |
| A new endpoint | `api.py`, plus the `rag.api` self-check | `rag.api` |
| Requirements, prices or links | `doctor.py`, `cloud.PRICES`, and the card text in `index.html` | `rag.doctor` |
| Colours or theme | `:root` variables in `index.html` (dark in two places) | Check all three theme options |
| A CLI command | `__main__.main` and `__main__._run` | Run it |

`ingest --rebuild` keeps page checkpoints and summaries, so it costs no new OCR. `ingest --force` deletes the document folder and starts over, which **repeats API charges in API mode**.

**Before adding a cloud call:** keep it in `cloud.py`, behind `PROVIDER == "cloud"`, and route it through `_post` so it gets retries and usage logging. Never pass text or keys to `_log`.

---

## 9. Testing

```powershell
# Fast: no models or keys needed, except where noted
.venv\Scripts\python -m rag.chunk
.venv\Scripts\python -m rag.stitch
.venv\Scripts\python -m rag.enrich
.venv\Scripts\python -m rag.handwriting
.venv\Scripts\python -m rag.evaluate
.venv\Scripts\python -m rag.cloud
.venv\Scripts\python -m rag.parse_mistral
.venv\Scripts\python -m rag.doctor
.venv\Scripts\python -m rag.models
.venv\Scripts\python -m rag.ingest
.venv\Scripts\python -m rag.api

# Slow: uses the models, or the APIs in API mode
.venv\Scripts\python -m rag doctor
.venv\Scripts\python -m rag eval
.venv\Scripts\python scripts\compare_modes.py
```

There is no test framework: each module's `if __name__ == "__main__"` block holds its checks. Add new assertions there.

Latest local results on the 3 sample PDFs: the 33 baseline questions scored 33/33 with recall@5 of 1.0 and handwriting character error of 0.031. The 20 harder questions scored 17/20. The `rag eval` default now runs all 53 together, which hasn't been done as one run yet.

---

## 10. Known limits

**API mode**

- No real Mistral or Gemini call has been made yet; only mocked tests. The request formats, times and costs are unverified.
- There are no text positions or OCR confidences, so no margin notes or low-confidence list, and handwriting error can't be measured.
- If a key is missing or an API call fails and the local models are installed, parsing falls back to local mode. A failed Gemini figure description also triggers this fallback for the remaining pages.
- It still needs PyTorch, Docling and the other imports from `requirements.txt`, plus 3.4 GB of local models.

**Local mode**

- Qwen 3B can answer "Not found" on fragmentary handwriting even when the right page was retrieved (3 of 4 handwriting questions in the harder set).
- SmolVLM's figure descriptions can be poor, so figure answers depend heavily on captions.
- Badly OCR'd printed lines can get tagged as handwritten, though the text is still right.
- On the survey sample, Table 13 is parsed as text rather than a table, so its audit reports one gap.

**Both modes**

- Only one process can open the vector store, so run either the server or the CLI, not both.
- The server has no login: anyone with an account on the computer can use it, and `settings.json` holds keys in plain text in your user folder.
- Answers are judged by phrase matching in `evaluate.run`, which can't assess reasoning quality.
- Deleting a document folder leaves its vectors in `qdrant/` until the same file is re-ingested.

Deliberate shortcuts are marked in the code with `shortcut:` comments. List them with `/ponytail-debt`, or `git grep "shortcut:"`.






## Myself
# Complete PDF RAG workflow — from upload to answer

Your project has two modes:

- Local mode: Free, runs on your GTX 1650 using local models.
- API mode: Paid, uses Mistral OCR and Gemini for faster cloud processing, while keeping much of the pipeline local.

The overall workflow is the same, but the PDF extraction and LLM models differ.

## Part 1: PDF ingestion

Imagine uploading a 180-page PDF containing text, scanned pages, tables, formulas, images and handwritten notes.

## Stage 1 — Upload and identification

File: `ingest.py`, `api.py`, `store.py`

- Accepts the PDF.
- Calculates SHA-256 and creates a document ID using the first 16 hexadecimal characters.
- Saves the PDF in `data/` or `data-cloud/`.
- Skips an already-ingested document.

## Stage 2 — PDF triage

File: `parse.py`

Tool: PyMuPDF

- Checks all 180 pages.
- Identifies scanned pages.
- Collects bookmarks, links and page dimensions.
- Does not need an AI model.

Example: Page 40 has digital text; page 41 is identified as scanned.

## Stage 3 — Document extraction

Files: `parse.py`, `parse_mistral.py`, `ingest.py`

Local: Docling, Heron, TableFormer, RapidOCR and SmolVLM.

API: Mistral OCR.

- Extracts paragraphs, headings, tables, formulas, images and scanned text.
- Processes pages and saves individual page checkpoints.
- Converts extracted content into structured items.
- In API mode, Gemini describes extracted figures.

If Mistral fails, `ingest._cloud_parse()` switches the remaining pages to local parsing, provided the local models are installed.

## Stage 4 — Handwriting recognition

File: `handwriting.py`

Model: TrOCR-base-handwritten

- Identifies potentially handwritten or poorly recognised text.
- Re-reads selected lines.
- Keeps readings with confidence of at least 0.6.

Normally runs in local mode, but can also run during API fallback.

## Stage 5 — Stitching and document structure

File: `stitch.py`

Tool: Python

- Removes headers and footers.
- Merges qualifying paragraphs and tables across pages.
- Attaches captions to tables and figures.
- Identifies margin notes when positions are available.
- Builds the document's section hierarchy.

Output: Structured document items.

## Stage 6 — Metadata enrichment

File: `enrich.py`

Tool: Regex

- Extracts URLs.
- Identifies research references and DOIs.
- Connects references to tables, figures and equations.

Example: "See Table 3" becomes a link to Table 3.

## Stage 7 — Document summarisation

Files: `summarize.py`, `models.py`, `cloud.py`

Local model: Qwen2.5-3B Q4\_K\_M.

API model: Gemini 2.5 Flash.

- Creates summaries for eligible sections.
- Creates a whole-document summary.
- Saves summaries so they can be reused.
- Uses Qwen as a fallback if Gemini summarisation fails and local models are available.

These summaries are also indexed later for searching.

## Stage 8 — Chunking

File: `chunk.py`

Tool: Python

- Splits the document into meaningful chunks of approximately 350 tokens.
- Preserves section boundaries.
- Keeps tables, formulas and figures atomic where possible.
- Splits very large tables by rows and repeats headers.
- Adds document title, section, page and content type.

Example: `[Title][4 Methods > 4.2][p87-88][table]`

## Stage 9 — Embedding and indexing

File: `index.py`

Model: BGE-M3

Database: Qdrant

- Converts chunks into dense and sparse vectors.
- Stores vectors and chunk information in Qdrant.
- Makes document content searchable.

This remains local in both modes.

## Stage 10 — Audit and validation

File: `audit.py`

Tool: Python validation rules

- Checks whether pages have extracted content.
- Checks missing tables and figures.
- Checks whether bookmarks appear in the section tree.
- Reports low-confidence regions and extraction gaps.

Output: `audit.json`, showing whether ingestion is complete and what may have been missed.

## Part 2: Question answering

Now the PDF is ingested. You ask:

"What was the nitrate level at station C?"

## Stage 11 — Question routing

File: `query.py`

Tool: Regex

Identifies the question type and selects one of five routes:

- `lookup`: Count tables, figures, references, pages, etc.
- `whole_doc`: Summarise or ask about the entire document.
- `section`: Explain a particular section.
- `multi_hop`: Compare multiple concepts or facts.
- `fact`: Search for a specific answer.

Not every question needs an LLM or vector search.

## Stage 12 — Retrieval and reranking

Files: `query.py`, `index.py`, `models.py`

Models: BGE-M3 and `bge-reranker-base`.

Database: Qdrant.

For fact and multi-hop questions:

1. Dense search finds semantically related chunks.
2. Sparse search finds matching keywords.
3. RRF combines both result lists.
4. Retrieves 20 candidates.
5. Reranker scores the candidates against the question.
6. Keeps the best 6 chunks.

This is the main information-retrieval stage.

## Stage 13 — Relevance check and context expansion

File: `query.py`

- Checks the best reranker score against `RELEVANCE_MIN = 0.005`.
- If below the threshold, returns: "Not found in the documents."
- Otherwise, adds neighbouring passages and up to two linked tables, figures or formulas.
- Prepares up to 2,500 tokens locally or 12,000 tokens in API mode.

This helps prevent answers from losing surrounding context.

## Stage 14 — Final answer generation

Files: `query.py`, `models.py`, `cloud.py`

Local: Qwen2.5-3B.

API: Gemini 2.5 Flash.

- Receives the question and retrieved evidence.
- Generates a natural-language answer.
- Adds page citations.
- Returns the supporting sources.

Example: "The nitrate level at station C was 11.3 mg/L (p. 45)."

## Stage 15 — Logging and evaluation

Files: `evaluate.py`, `query.py`, `cloud.py`

- Logs questions and retrieval results in `queries.jsonl`.
- Logs API usage, time and estimated cost in `api_usage.jsonl`.
- Evaluates answer accuracy against known questions in `samples/golden.json`.
- Measures handwriting recognition accuracy in local mode.

Helps identify where the system needs improvement.

## The complete workflow at a glance
```
PDF Upload

Triage and Extraction

PyMuPDF + Docling / Mistral OCR

Handwriting + Stitching + Enrichment

Summarisation and Chunking

BGE-M3 Embeddings + Qdrant Index

PDF ready for questionsUser asks a question

Question Router

Hybrid Search → RRF → Reranking

Relevance Gate + Context Expansion

Qwen / Gemini generates answer

Answer + Page Citations
```
The most important idea: Extraction makes the PDF readable, chunking organises it, BGE-M3 and Qdrant make it searchable, reranking finds the most relevant evidence, and Qwen or Gemini turns that evidence into an answer. That's the complete system.