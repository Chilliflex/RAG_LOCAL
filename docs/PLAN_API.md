# Hybrid cloud plan: Mistral OCR + Gemini Flash with local fallbacks

Status: proposed, not started. The fully-local build in [PLAN.md](PLAN.md) stays the default until this plan is approved.

Goal: cut ingest of a 100-page mixed PDF from ~30 min to ~3 min and answers from 10–140 s to 2–5 s by moving the two expensive AI stages (document parsing, language model) to APIs, while keeping triage, stitching, enrichment, chunking, embeddings, the vector store, retrieval, reranking, audit and the web app local and unchanged.

Mode switch: `PROVIDER=cloud|local` in `.env`. `local` is today's behaviour byte for byte.

## APIs and what each stage uses

| Stage | cloud mode | local mode (fallback) | Reason for the choice |
|---|---|---|---|
| Fingerprint, upload, status | local | local | no model involved |
| Triage | PyMuPDF | PyMuPDF | <1 s, decides which pages are scanned |
| Layout, tables, OCR, formulas | Mistral OCR 4 (`mistral-ocr-latest`) | Docling | one request per PDF; per-page markdown, tables, image bboxes, LaTeX formulas |
| Handwriting | Mistral OCR, same call | TrOCR | read inline; separate crop stage disappears |
| Picture descriptions | Gemini 2.5 Flash vision | SmolVLM-256M | reads charts and photos properly |
| Stitch | Python (adapted to markdown input) | Python | geometric rules, free |
| Enrich | regex | regex | free |
| Section + doc summaries | Gemini 2.5 Flash | Qwen2.5-3B GGUF | 40 summaries in ~40 s vs ~10 min |
| Chunking | Python | Python | unchanged |
| Embeddings | BGE-M3 local | BGE-M3 local | fast enough, keeps chunk text on the machine |
| Vector store | Qdrant embedded | Qdrant embedded | unchanged |
| Audit | Python | Python | unchanged |
| Router | regex | regex | unchanged |
| Retrieval + rerank | BGE-M3 + bge-reranker-base | same | recall@5 already 1.0 |
| Multi-hop decomposition, answers | Gemini 2.5 Flash | Qwen2.5-3B GGUF | 1–3 s per call, better citation discipline |
| Web app, eval | FastAPI | FastAPI | unchanged |

What leaves the machine in cloud mode: page images (Mistral), section text, retrieved passages, questions and flagged images (Google). Embeddings, the index, the audit and the structured document never leave.

## Phases

| Phase | Scope | Done when |
|---|---|---|
| 0 Prerequisites | API keys, `PROVIDER` switch, SDKs, offline flags scoped to local mode, cost/latency log | `python -m rag doctor` prints both providers reachable; local mode unchanged |
| 1 Mistral parse | `rag/parse_mistral.py`: PDF → Mistral OCR → per-page item JSON in the existing schema; checkpoints and resume kept | `samples/*.pdf` ingest in cloud mode; `document.json` passes the stitch and chunk self-checks |
| 2 Gemini LLM | `models.generate` and a new `models.describe_image` route to Gemini in cloud mode; summaries, decomposition, answers unchanged in call shape | same 33 golden questions answered with `(p. N)` citations |
| 3 Stitch and audit adaptation | section tree from markdown heading levels, captions from markdown, margin notes marked "unavailable" in cloud mode, audit `low_confidence` replaced by Mistral image count check | audit shows no false gaps on the 144-page survey |
| 4 Fallback and failure handling | retries with backoff, rate-limit handling, per-document fallback to local parse, per-question fallback to Qwen, clear status messages in the UI | killing the network mid-ingest and mid-question both finish with a correct result |
| 5 Evaluation | golden set extended with 20 questions on formulas, charts, margin notes and handwriting; both modes measured on the same PDFs; cost per PDF logged | comparison table in `data/eval_compare.json`; go/no-go decision |
| 6 Cleanup | README, PLAN.md pointer, memory note, optional removal of local models if fallbacks are dropped | docs match behaviour |

Estimated effort: 0 half a day, 1 one to two days, 2 half a day, 3 one day, 4 half a day, 5 one day, 6 half a day. Total about one working week.

## Phase details

### Phase 0: Prerequisites
- Create a Mistral API key (console.mistral.ai) and a Google AI Studio key on the paid tier (the free tier may use prompts for training).
- `.env`: `PROVIDER=cloud`, `MISTRAL_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_MODEL=gemini-2.5-flash`, `MISTRAL_OCR_MODEL=mistral-ocr-latest`. `HF_HUB_OFFLINE` and the other offline flags are set only when `PROVIDER=local`.
- `requirements.txt`: `mistralai`, `google-genai`.
- `rag/config.py`: read the new variables; `PROVIDER` defaults to `local` so existing installs do not change behaviour.
- `rag/__main__.py doctor`: prints provider, key presence, one tiny test call per API with latency, GPU visibility.
- `data/api_usage.jsonl`: one line per API call with provider, model, pages or tokens, seconds, estimated cost. Needed for Phase 5.

### Phase 1: Mistral OCR parse
- Upload the PDF with `client.files.upload`, get a signed URL, call `client.ocr.process(model, document={"type": "document_url", ...}, include_image_base64=True)`. One call per PDF up to Mistral's limits (1,000 pages, 50 MB). Larger PDFs are split with PyMuPDF into parts and page numbers offset.
- Response: `pages[]`, each with `index`, `markdown`, `images[]` (id, bbox corners, base64), `dimensions`. Convert each page's markdown into items in the existing schema:

| Markdown | Item type | Notes |
|---|---|---|
| `#`, `##`, `###` lines | section_header | heading level kept as `level` |
| pipe tables | table | caption = the nearest "Table N" line above or below |
| `$$ … $$` and inline `$ … $` | formula | text = LaTeX |
| `![id](id)` | figure | bbox from `images[]`, image saved to `data/<doc_id>/images/` |
| `- ` / `1. ` lines | list_item | |
| other blocks | text | |

- Item fields `bbox` (figures only), `confidence` (None), `handwritten` (False) are kept so stitch, chunk and audit run untouched.
- Per-page checkpoints `pages/NNNN.json` are written exactly as today, so resume and the status endpoint keep working. Since Mistral returns the whole document at once, "resume" means skipping the API call when all pages exist.
- Scanned-page handwriting comes back inline. No confidence is available, so the handwriting stage is skipped in cloud mode.
- Test: ingest `mixed_handwriting.pdf`, `digital_attention.pdf`, `long_llm_survey.pdf`; run `python -m rag.stitch` and `python -m rag.chunk` self-checks; compare table count and section count with local mode.

### Phase 2: Gemini for language and vision
- `models.generate(system, user, max_new_tokens)` keeps its signature. In cloud mode it calls `client.models.generate_content(model=GEMINI_MODEL, contents=user, config={"system_instruction": system, "max_output_tokens": ...})`.
- `models.count_tokens` uses `client.models.count_tokens` in cloud mode so the prompt budget logic in `query._context` is unchanged. Budget can rise from 2,500 to 12,000 tokens in cloud mode since the 8,192 context limit no longer applies; keep top-k at 6 so citations stay precise.
- New `models.describe_image(png_bytes, caption)` → one sentence plus any readable values (axis labels, legend, numbers). Called from the parse stage for every figure Mistral returns, except logos and icons by size heuristic. Result stored in the figure item's `text` as today.
- `summarize.py`, `query._decompose`, `query._answer`, `query._whole_doc` are untouched: they call `generate`.
- `models.free()` becomes a no-op in cloud mode.
- Test: `python -m rag eval` on the golden set in cloud mode; all 33 must still contain the gold phrase and a `(p. N)` citation.

### Phase 3: Stitch and audit adaptation
- `stitch._level` prefers markdown heading level, then bookmarks, then numbering.
- `stitch._margin_notes` needs text bboxes, which Mistral does not provide. In cloud mode the function is skipped and `document.json` records `"margin_notes": "unavailable"`. Notes still appear as inline text in reading order, so they are searchable, only their attachment to a paragraph is lost.
- `stitch._attach_captions` works on text already; verify on the survey that "Table N" lines land on the right table.
- `stitch._merge_cross_page` works on markdown tables and paragraphs without change.
- `audit.run`: the low-confidence list is empty in cloud mode; add a check that every `![img]` reference in the markdown has a matching image, and keep the TOC and mention-vs-caption checks.
- Test: survey audit has 0 TOC misses and no false "figure mentioned but not captioned" gaps.

### Phase 4: Fallback and failure handling
- Every API call: 3 retries with exponential backoff on 429 and 5xx, 60 s timeout, errors logged to `api_usage.jsonl`.
- Ingest: if Mistral fails after retries for a document, status shows "cloud parse failed, using local" and `docling_pages` runs. The per-page checkpoints make the two paths interchangeable.
- Ask: if Gemini fails, the same prompt goes to local Qwen, and the response carries `"provider": "local"` so the UI can show it.
- Rate limits: Mistral and Gemini free tiers are low; the GPU lock already serialises ingests, which is enough for one user.
- Test: disable the network adapter during an ingest and during a question; both complete with the local fallback and the logs show it.

### Phase 5: Evaluation and decision
- Extend `samples/golden.json` with 20 questions targeting formulas (LaTeX output), chart values, handwriting lines, margin notes and multi-hop comparisons on the user's own PDFs.
- Run `python -m rag eval` in both modes on the same documents. Record recall@5, accuracy, handwriting CER (local only, cloud has no per-line confidence so CER is measured against the same `.truth.json`), ingest time, mean answer time, cost per PDF from `api_usage.jsonl`.
- Write `data/eval_compare.json` and a short table in README.
- Go if cloud accuracy ≥ local on every category except margin notes, and time drop matters. No-go keeps `PROVIDER=local`.

### Phase 6: Cleanup
- README: provider switch, keys, what data leaves the machine, costs observed.
- PLAN.md gets a one-line pointer to this file.
- Memory note `fully-local-no-api.md` updated to describe the switch and the user's decision.
- If the user drops local fallbacks for a GPU-free machine, remove TrOCR, Docling and Qwen from `download_models.py` and `requirements.txt`; keep BGE-M3 and the reranker.

## Expected accuracy change

| Content | Expected | Why |
|---|---|---|
| Formulas | better | LaTeX from Mistral; local formula conversion is off |
| Complex or scanned tables | better | stronger table reconstruction |
| Charts and photos | better | Flash reads values; SmolVLM-256M gives a vague sentence |
| Whole-document and multi-hop answers | better | 3B model vs frontier model |
| Printed OCR, page citations | same | both near perfect |
| Handwriting | unknown until Phase 5 | TrOCR is tuned to CER 0.031 on the sample; Mistral unmeasured |
| Margin notes | worse | no text bboxes; notes become inline text |

The current golden set is saturated (recall 1.0, 33/33), so improvement is only measurable on the new questions from Phase 5.

## Costs (mid-2026 list prices, verify before use)

| Item | Price | Per 100-page PDF |
|---|---|---|
| Mistral OCR 4 | $4 / 1,000 pages ($2 batch) | $0.40 |
| Gemini 2.5 Flash | $0.30 in / $2.50 out per 1M tokens | ~$0.10 (summaries + 40 images) |
| Gemini per question | ~6k in / 100 out tokens | ~$0.002 |
| Total ingest | | ~$0.50 |

## Risks
- Privacy: page images, section text, passages and questions leave the machine. Use paid tiers and set data retention in both consoles.
- Schema loss: no per-paragraph bboxes and no OCR confidence from Mistral; margin-note attachment and the low-confidence audit are lost in cloud mode.
- API drift: model names and prices change; `GEMINI_MODEL` and `MISTRAL_OCR_MODEL` are configurable.
- Offline use: cloud mode needs a network; local mode remains available as long as the models stay installed.
- Rate limits and outages: handled by Phase 4, but a long outage makes cloud ingest fall back to the slow local path.
