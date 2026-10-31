# Local PDF RAG

Ask questions about PDFs: digital, scanned, handwritten, tables, long reports. **Everything runs on this computer.**
There are no API keys and no network calls at runtime. Plan and design: [docs/PLAN.md](docs/PLAN.md).

## Setup (once, needs internet)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install --only-binary=:all: llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py      # ~12 GB into .\models
```

After that the app works offline. `.env` sets `HF_HUB_OFFLINE=1` and turns telemetry off. Without an NVIDIA GPU, set `DEVICE=cpu` in `.env`. It will be slow.

## Use

```powershell
.venv\Scripts\python -m rag ingest my.pdf            # parse + index (resumable; re-run continues where it stopped)
.venv\Scripts\python -m rag ask "What was the nitrate level at station C?"
.venv\Scripts\python -m rag docs                     # list ingested documents
.venv\Scripts\python -m rag eval                     # recall@5, accuracy, handwriting CER on samples/golden.json
.venv\Scripts\python -m rag serve                    # web UI at http://127.0.0.1:8000 (this computer only)
```

Run either the server or the CLI commands, not both at once: the on-disk vector store is locked by whichever process opened it first.

The server answers only requests addressed to `127.0.0.1`/`localhost` and refuses POSTs from other origins, so a web page
you visit cannot read or upload documents through it. It has no login: anyone with an account on this computer can use it.

Each document lives in `data/<doc_id>/`: `document.json` (structured content), `audit.json` (completeness check),
`timings.json`, `pages/` (per-page checkpoints). Delete that folder and the document is no longer searched. Its vectors stay in `data/qdrant` until that file is ingested again; to remove everything, delete `data/`. Re-run `ingest --force` to rebuild a document.

Logs: `data/<doc_id>/timings.json` (seconds and peak VRAM per ingest stage), `data/queries.jsonl` (one line per question:
route, seconds, top score), `data/eval_*.json` (per-question eval detail).

Self-checks (no models needed): `python -m rag.chunk`, `rag.stitch`, `rag.enrich`, `rag.handwriting`, `rag.evaluate`.

## Models (all local)

| Job | Model |
|---|---|
| Layout, tables, OCR, picture descriptions | Docling (Heron layout, TableFormer, RapidOCR, SmolVLM-256M) |
| Handwriting | TrOCR-base-handwritten |
| Embeddings (dense + sparse) | BGE-M3 |
| Reranker | bge-reranker-base |
| Answers, summaries | Qwen2.5-3B-Instruct, Q4_K_M GGUF via llama.cpp |

## Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `DEVICE` | `cuda` | `cuda`, `mps` or `cpu` |
| `OCR_CONFIDENCE_MIN` | see `rag/config.py` | OCR lines below this go to TrOCR |
| `HANDWRITING_CONFIDENCE_MIN` | see `rag/config.py` | TrOCR's reading is kept only above this |
| `RELEVANCE_MIN` | see `rag/config.py` | below this reranker score: "Not found in the documents." |
| `FORMULA_LATEX` | `0` | `1` = convert formulas to LaTeX (about 30 s per formula on a GTX 1650) |
| `LLM_REPO` / `LLM_FILE` | Qwen2.5-3B GGUF | any chat GGUF; re-run the download script |
