# PDF RAG

Ask questions about PDFs: digital, scanned, handwritten, tables, formulas, figures, long reports. Two modes, chosen in the web page or with `PROVIDER` in `.env`:

| | Local, free (default) | API, fast |
|---|---|---|
| What runs where | everything on this computer, no network at runtime | Mistral OCR reads the pages, Gemini writes summaries, figure descriptions and answers; embeddings, reranking and the index stay local |
| Cost | $0 after a one-time 12 GB model download | about $0.50 per 100-page PDF and $0.002 per question (list prices, verify on the provider pages) |
| Speed, 100 pages | about 30 min to ingest, 10–140 s per answer on a GTX 1650 | about 3 min to ingest, 2–5 s per answer |
| Privacy | nothing leaves the computer | page images go to Mistral; passages, questions and figures go to Google |
| Hardware | NVIDIA GPU with 4 GB VRAM recommended, 16 GB RAM, 20 GB free disk | any laptop with 8 GB RAM, 5 GB free disk and internet |

Each mode keeps its own documents and index (`data/` and `data-cloud/`), because the two parsers produce different content. Plan and design: [docs/PLAN_FINAL.md](docs/PLAN_FINAL.md) (modes), [docs/PLAN.md](docs/PLAN.md) (pipeline).

## Setup (once, needs internet)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install --only-binary=:all: llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py             # local mode: ~12 GB into .\models
.venv\Scripts\python scripts\download_models.py --api-only  # API mode only: BGE-M3 + reranker, ~3.4 GB
.venv\Scripts\python -m rag doctor                          # checks the requirements of both modes
```

Local mode then works offline: `.env` sets `HF_HUB_OFFLINE=1` and turns telemetry off. Without an NVIDIA GPU, set `DEVICE=cpu` in `.env` (slow).

API mode needs two keys: Mistral (https://console.mistral.ai/api-keys) and Google AI Studio (https://aistudio.google.com/apikey). Enter them in the web page's "API, fast" card (saved to `data/settings.json`, never shown again) or put `MISTRAL_API_KEY=` and `GOOGLE_API_KEY=` in `.env`. If an API call fails after 3 retries, that step runs locally when the local models are installed, and the status and answer say so.

## Use

```powershell
.venv\Scripts\python -m rag serve                    # web UI at http://127.0.0.1:8000: choose a mode, upload, ask
.venv\Scripts\python -m rag ingest my.pdf            # parse + index (resumable; re-run continues where it stopped)
.venv\Scripts\python -m rag ask "What was the nitrate level at station C?"
.venv\Scripts\python -m rag docs                     # list ingested documents (of the active mode)
.venv\Scripts\python -m rag eval                     # recall@5, accuracy by category, handwriting CER on samples/golden.json
.venv\Scripts\python scripts\compare_modes.py        # both modes on the same PDFs -> data/eval_compare.json, with API cost
```

The CLI uses the mode from `PROVIDER` in `.env`, else the one last chosen in the web page. Run either the server or the CLI commands, not both at once: the on-disk vector store is locked by whichever process opened it first.

The server answers only requests addressed to `127.0.0.1`/`localhost` and refuses POSTs from other origins, so a web page
you visit cannot read, upload or switch modes through it. It has no login: anyone with an account on this computer can use it.

Each document lives in `data/<doc_id>/` (or `data-cloud/`): `document.json` (structured content), `audit.json` (completeness check),
`timings.json`, `pages/` (per-page checkpoints), `images/` (figure crops, API mode). Delete that folder and the document is no longer searched. Its vectors stay in the mode's `qdrant/` until that file is ingested again; to remove everything, delete the data folder. Re-run `ingest --force` to rebuild a document.

Logs: `<data>/<doc_id>/timings.json` (seconds and peak VRAM per ingest stage), `<data>/queries.jsonl` (one line per question:
route, provider, seconds, top score), `<data>/eval_*.json` (per-question eval detail), `data-cloud/api_usage.jsonl` (one line per API call: pages or tokens and seconds, never keys or text).

Self-checks (no models or keys needed): `python -m rag.chunk`, `rag.stitch`, `rag.enrich`, `rag.handwriting`, `rag.evaluate`, `rag.cloud`, `rag.parse_mistral`, `rag.doctor`, `rag.models`, `rag.ingest`, `rag.api`.

## Models

| Job | Local mode | API mode |
|---|---|---|
| Layout, tables, OCR, formulas | Docling (Heron layout, TableFormer, RapidOCR) | Mistral OCR (`mistral-ocr-latest`) |
| Handwriting | TrOCR-base-handwritten on low-confidence OCR lines | Mistral OCR |
| Figure descriptions | SmolVLM-256M | Gemini 2.5 Flash |
| Embeddings (dense + sparse) | BGE-M3 | BGE-M3 (local) |
| Reranker | bge-reranker-base | bge-reranker-base (local) |
| Answers, summaries | Qwen2.5-3B-Instruct, Q4_K_M GGUF via llama.cpp | Gemini 2.5 Flash |

API mode has no word positions or OCR confidences, so margin notes are not detected and the low-confidence list is empty; the audit says so.

## Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `PROVIDER` | `local` | `local` or `cloud`; set here it overrides the mode chosen in the web page at the next start |
| `MISTRAL_API_KEY`, `GOOGLE_API_KEY` | | API mode keys (or enter them in the web page) |
| `MISTRAL_OCR_MODEL`, `GEMINI_MODEL` | `mistral-ocr-latest`, `gemini-2.5-flash` | API models; a Gemini model without a thinking budget setting may reject requests |
| `DEVICE` | `cuda` | `cuda`, `mps` or `cpu` |
| `OCR_CONFIDENCE_MIN` | see `rag/config.py` | OCR lines below this go to TrOCR (local mode) |
| `HANDWRITING_CONFIDENCE_MIN` | see `rag/config.py` | TrOCR's reading is kept only above this |
| `RELEVANCE_MIN` | see `rag/config.py` | below this reranker score: "Not found in the documents." |
| `FORMULA_LATEX` | `0` | `1` = convert formulas to LaTeX in local mode (about 30 s per formula on a GTX 1650) |
| `LLM_REPO` / `LLM_FILE` | Qwen2.5-3B GGUF | any chat GGUF; re-run the download script |


## Local mode (free, offline)
```
Imagine you upload a 180-page environmental research PDF containing printed text, scanned pages, tables, formulas, diagrams, images and handwritten notes. First, the system creates a unique ID using SHA-256 and saves the PDF locally, skipping it if it has already been completely processed. Next, PyMuPDF examines every page and identifies scanned pages, bookmarks, links and images. Docling then reads the PDF in batches of 10 pages: Heron identifies the layout and reading order, TableFormer reconstructs tables, RapidOCR extracts printed and scanned text, a picture classifier identifies charts and diagrams, and SmolVLM describes images. If RapidOCR struggles with handwriting, TrOCR tries to read it and replaces the original text only when its confidence is at least 0.6. After extraction, Python code removes repeated headers and footers, joins paragraphs across pages, attaches captions and margin notes, and builds the document's section structure. Regex then finds references, URLs, DOIs and connections such as "see Table 3". Qwen2.5-3B creates summaries of individual sections and the entire report. Python divides the extracted information into searchable chunks of approximately 350 tokens, keeping tables, formulas, figures and code together wherever possible. BGE-M3 converts these chunks into dense and sparse numerical representations, and Qdrant stores them locally so they can be searched later. An audit then checks missing pages, extraction quality, bookmarks, tables, figures and low-confidence regions. Once ingestion is complete, you can ask questions through the web page: for example, if you ask "What was the nitrate level at station C?", the system first identifies the question as a factual question, searches Qdrant using BGE-M3, combines results using Reciprocal Rank Fusion, reranks the best 20 results using bge-reranker-base and selects the top six. If the best reranking score is below 0.005, it responds that the answer was not found; otherwise, it adds neighbouring paragraphs and linked tables or figures, sends up to 2,500 tokens of relevant information to Qwen, and generates an answer such as "The nitrate level at station C was 11.3 mg/L (p. 1)." Simpler questions, such as how many tables exist, are answered directly from stored JSON without calling an AI model. Broad questions use the cached summaries, while complicated comparison questions can be divided into smaller questions. Everything runs on your computer using Python, your GTX 1650 and locally stored models, without API charges or internet after setup. In your reported tests, the original 33 questions scored 33/33, while 20 harder questions scored 17/20, with handwriting-related misses.
```

## API mode (paid, fast)
```
In API mode (paid, fast), when you upload a 180-page PDF containing scanned pages, printed text, tables, formulas, images and handwriting, the app first creates a unique document ID using SHA-256 and stores the PDF in `data-cloud/`, while PyMuPDF quickly checks all pages for page counts, bookmarks, links and scanned-page information. Next, the PDF is sent to Mistral OCR, which reads printed text, scanned text, tables, formulas and handwriting using its cloud service instead of your GTX 1650, and saves the results page by page so completed pages don't need to be processed or paid for again. The app then converts Mistral's Markdown output into structured items such as headings, paragraphs, tables, formulas and figures. For images and diagrams, it sends the extracted pictures to Gemini 2.5 Flash, which describes charts, labels, flowcharts and handwritten content inside images. After that, the same local Python code stitches content across pages, attaches captions, builds the section hierarchy and extracts references, URLs and cross-references. Gemini then creates summaries of individual sections and the entire document. The app divides everything into searchable chunks, generates embeddings using BGE-M3, stores them in Qdrant on your computer and runs an audit to check whether content was missed. When you ask something like "What does the handwritten note on page 14 say?", the router selects the appropriate question-answering route, searches Qdrant using dense and sparse search, combines results using RRF, reranks the best 20 results using `bge-reranker-base`, and keeps the top 6. Finally, the relevant passages and their neighbouring content are sent to Gemini, which generates an answer with page citations. The estimated processing time is around 5 minutes for 180 pages, with answers taking around 2–5 seconds, although these figures need real testing. Mistral and Gemini cost money, but the advantage is that the slowest AI processing happens in the cloud rather than on your GTX 1650. Your local machine still handles stitching, chunking, embeddings, indexing and retrieval. The trade-off is privacy: the PDF content and relevant questions/passages are sent to external providers.
```