# Final plan: dual-mode PDF RAG (Local and free, or API and fast)

Status: Phases 6-7 coded on 2026-10-31: golden set extended to 53 questions (`category` field: formula, chart, handwriting, margin_note, multi_hop), `rag eval` reports accuracy by category, `scripts/compare_modes.py` runs both modes and writes `data/eval_compare.json` with API cost from the usage log; README rewritten for two modes. Local baseline on the 20 new questions: 17/20, recall@5 1.0 (formula 4/4, chart 4/4, margin_note 4/4, multi_hop 4/4, handwriting 1/4: the right page was retrieved but the local model answered "Not found"). The cloud half of the comparison waits for keys. Phases 0-5 coded on 2026-10-31 (`rag/cloud.py`, `rag/parse_mistral.py`, `rag/doctor.py`, fallbacks in `rag/ingest.py` and `rag/models.py`, `GET/POST /mode`, mode cards in `rag/static/index.html`; cloud mode stores documents in `data-cloud/`). Offline gates passed: doctor 11/11 local and both keys red; local eval still 33/33, CER 0.031; every module self-check; TestClient pass of `/`, `/mode`, `/doctor`, foreign-origin refusal; mocked API-down fallbacks for parse and answer. Still waiting on API keys: the live gates of Phases 1-4 (API-mode ingest of the samples, counts within ±1, API-mode eval 33/33, chart description, survey audit `complete: true`, network-off runs) and the browser pass of Phase 5. Supersedes the single-mode sections of [PLAN.md](PLAN.md) and the draft in [PLAN_API.md](PLAN_API.md). Read this file before starting any phase.

## What the user gets

On first open of http://127.0.0.1:8000 (and whenever they click "Mode" later) the page shows two cards:

| | 1. Local, free | 2. API, fast |
|---|---|---|
| Cost | $0 after the one-time 12 GB download | about $0.50 per 100-page PDF, about $0.002 per question |
| Speed, 100 pages | about 30 min ingest, 10–140 s per answer (GTX 1650) | about 3 min ingest, 2–5 s per answer |
| Privacy | nothing leaves the computer | page images go to Mistral; section text, passages, questions and images go to Google |
| Hardware | NVIDIA GPU with 4 GB VRAM recommended, 16 GB RAM, 20 GB free disk | any laptop, 8 GB RAM, 5 GB free disk, internet |

Selecting a card opens its **Requirements** panel. Each line shows a live check mark or cross from the `/doctor` endpoint, so the user sees what is missing before they try.

### Requirements panel, Local mode

| Requirement | Detail | Check |
|---|---|---|
| Python 3.11 + venv | `.venv` present | import works |
| PyTorch CUDA | torch sees the GPU, or `DEVICE=cpu` chosen knowingly (5–10× slower) | `torch.cuda.is_available()` |
| Disk | models ≈ 12 GB, venv ≈ 6 GB, data grows ≈ 20 MB per 100 pages → 20 GB free recommended | free space measured |
| RAM | 16 GB recommended, 8 GB minimum with `PAGE_BATCH=5` | measured |
| VRAM | 4 GB minimum; models load one stage at a time | measured |
| Models present in `models/` | Docling bundle (Heron layout, TableFormer, RapidOCR, DocumentFigureClassifier, SmolVLM-256M) ≈ 1.5 GB; TrOCR-base-handwritten ≈ 1.3 GB; BGE-M3 ≈ 2.3 GB; bge-reranker-base ≈ 1.1 GB; Qwen2.5-3B-Instruct Q4_K_M GGUF ≈ 2.0 GB | each folder present |
| One-time internet | `python scripts/download_models.py` | n/a |

Methods listed on the panel: PyMuPDF triage → Docling layout/tables/OCR/pictures → TrOCR handwriting (OCR confidence < 0.95, kept if TrOCR ≥ 0.6) → stitch (headers, cross-page merge, captions, margin notes, section tree) → regex references and cross-refs → Qwen summaries → structure-first chunks → BGE-M3 dense + sparse → Qdrant embedded → audit → regex router → RRF → bge-reranker gate 0.005 → Qwen answer with (p. N).

### Requirements panel, API mode

| Requirement | Detail | Where to get it | Price (mid-2026 list, verify) | Check |
|---|---|---|---|---|
| Mistral API key | OCR, tables, formulas (LaTeX), handwriting, image boxes. Model `mistral-ocr-latest` | https://console.mistral.ai/api-keys | $4 per 1,000 pages, $2 batch | 1-page test call |
| Google AI Studio key, paid tier | Gemini 2.5 Flash for summaries, answers, multi-hop decomposition, picture descriptions. Paid tier so prompts are not used for training | https://aistudio.google.com/apikey and https://ai.google.dev/gemini-api/docs/pricing | $0.30 in / $2.50 out per 1M tokens | 1-token test call |
| Internet | both calls, retries on 429/5xx | | | ping |
| Local models still needed | BGE-M3 ≈ 2.3 GB and bge-reranker-base ≈ 1.1 GB run on CPU or GPU so chunk text and the index stay on the machine | `python scripts/download_models.py --api-only` | free | folders present |
| RAM / disk | 8 GB RAM, 5 GB free disk, no GPU required | | | measured |

Methods listed: PyMuPDF triage → Mistral OCR (one call per PDF, per-page markdown) → markdown-to-items adapter → stitch (no margin notes) → regex → Gemini summaries → chunks → BGE-M3 → Qdrant → audit → router → RRF → reranker gate → Gemini answer with (p. N). Fallback: if an API call fails after 3 retries the stage runs locally when the local models are installed, otherwise the user sees a clear error.

Keys typed into the panel are saved to `data/settings.json` (file permissions user-only, never logged, never returned by any endpoint). `PROVIDER` can also be set in `.env`.

## Phases, model, time

| Phase | Scope | Model | Time | Gate (I run it before moving on) |
|---|---|---|---|---|
| 0 Switch and doctor | `PROVIDER` in config; `data/settings.json`; `/doctor` endpoint and `python -m rag doctor` reporting every requirement above; `api_usage.jsonl`; offline flags only in local mode; SDKs added | Opus 5.5 | 0.5 day | `rag doctor` prints all local checks green on this PC and both API checks red with no keys; local mode ingest + eval unchanged (33/33) |
| 1 Mistral parse | `rag/parse_mistral.py`: upload, one OCR call, split PDFs over 1,000 pages or 50 MB, markdown → item schema (headings, tables + captions, `$$` formulas, images with bbox saved to `images/`, lists, text), same `pages/NNNN.json` checkpoints | Fable | 1.5 days | three sample PDFs ingest in API mode; `python -m rag.stitch` and `rag.chunk` pass; table and section counts within ±1 of local mode on the survey; adapter unit test with a fixed markdown fixture |
| 2 Gemini LLM and vision | `generate` and `count_tokens` route to Gemini; new `describe_image`; prompt budget 12,000 in API mode; `free()` no-op | Opus 5.5 | 0.5 day | `rag eval` in API mode: 33/33 gold phrases, every answer has a `(p. N)` citation; one chart figure gets a description naming an axis label |
| 3 Stitch and audit adaptation | heading level from markdown; margin notes skipped and flagged; caption attachment verified; audit image check; low-confidence list empty without false gaps | Fable | 1 day | survey audit in API mode: 0 TOC misses, 0 false caption gaps, `complete: true`; `rag.stitch` self-check extended for markdown input |
| 4 Fallback and errors | 3 retries with backoff, 60 s timeout, usage log; document-level fallback to Docling; question-level fallback to Qwen; status and UI messages | Opus 5.5 | 0.5 day | network off mid-ingest → finishes via local, status shows it; network off mid-question → answer from Qwen marked `provider: local`; no key + no local models → readable error, no traceback |
| 5 Mode UI | two cards, requirements panels with live checks, key entry, mode persisted; `GET/POST /mode`; `GET /doctor`; existing upload/ask untouched; still no external URLs in the page except the two provider links which open in a new tab | Opus 5.5 | 1 day | TestClient self-check for `/mode` and `/doctor`; manual browser pass: choose each mode, see checks, switch back; same-origin middleware still blocks foreign POSTs |
| 6 Evaluation | golden set + 20 questions on formulas, charts, handwriting, margin notes, multi-hop; both modes on the same PDFs; `data/eval_compare.json`; costs from usage log | Opus 5.5 runs, Fable judges | 1 day | comparison table written; API mode ≥ local on every category except margin notes, or the gap is documented on the API card |
| 7 Docs and cleanup | README mode section, PLAN.md pointer, `download_models.py --api-only`, memory note | Opus 5.5 | 0.5 day | README commands run as written on a clean checkout |

Total: 6.5 working days. Model switch at each phase boundary with `/model`; each phase starts with "read docs/PLAN_FINAL.md, phase N".

## Evaluator protocol (applies to every phase)

1. Run the gate command(s) listed above and paste the real output.
2. Run the module self-checks that the phase touched (`python -m rag.<module>`).
3. Confirm local mode is unchanged: `python -m rag eval` with `PROVIDER=local` still reports recall 1.0, 33/33, CER 0.031 on the samples.
4. Confirm no new network call in local mode: grep the phase's diff for `requests`, `httpx`, `mistralai`, `google` outside the `PROVIDER == "cloud"` branches.
5. State pass or fail in one line with the numbers. A fail blocks the next phase.

## Langflow export (API mode only)

A Langflow flow can mirror the API mode once `catalog.json` is available: PDF file → Mistral OCR (custom component or HTTP node, since Langflow has no native Mistral OCR node) → markdown splitter → embeddings → Qdrant → Gemini chat with a citation prompt. What Langflow cannot express with stock nodes is stitching, margin notes, the audit, the regex router, the reranker gate and the citation normaliser; those become custom Python components wrapping the `rag` modules, or are dropped for a simpler flow. Which of the two is decided after reading the catalog.

## Risks

- Privacy in API mode is opt-in and shown on the card; local mode stays the default.
- Mistral returns no text bboxes or OCR confidence: margin-note attachment and the low-confidence list do not exist in API mode.
- Prices and model names change; both are config values and the panel says "verify on the provider page".
- API mode still downloads 3.4 GB of local models for embeddings and reranking; a fully model-free mode would require a cloud embedding API and is out of scope.
