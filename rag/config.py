import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Relative HF_HOME in .env is relative to the project, not the cwd.
os.environ["HF_HOME"] = str((ROOT / os.environ.get("HF_HOME", "models")).resolve())

DEVICE = os.environ.get("DEVICE", "cuda")
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)  # the query log is written before any document exists
MODELS = Path(os.environ["HF_HOME"])
DOCLING_ARTIFACTS = MODELS / "docling"

# Hub repo -> local folder. Models load from these folders only (plain copies; Windows blocks HF cache symlinks).
REPOS = {"BAAI/bge-m3": "bge-m3", "BAAI/bge-reranker-base": "bge-reranker-base",
         "microsoft/trocr-base-handwritten": "trocr-base-handwritten"}
EMBED_MODEL, RERANK_MODEL, HANDWRITING_MODEL = (str(MODELS / d) for d in REPOS.values())
# llama.cpp GGUF: ~10x faster prefill than bitsandbytes 4-bit on a GTX 1650
LLM_GGUF = (os.environ.get("LLM_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF"), os.environ.get("LLM_FILE", "qwen2.5-3b-instruct-q4_k_m.gguf"))
LLM_MODEL = str(MODELS / "llm-gguf" / LLM_GGUF[1])
LLM_CTX = 8192

# tuned on samples: printed scanned lines median 0.995 / p5 0.967; RapidOCR misreads of handwriting 0.75-0.92
OCR_CONFIDENCE_MIN = float(os.environ.get("OCR_CONFIDENCE_MIN", "0.95"))
HANDWRITING_CONFIDENCE_MIN = float(os.environ.get("HANDWRITING_CONFIDENCE_MIN", "0.6"))  # TrOCR mean token prob
# shortcut: formula->LaTeX costs ~30 s per formula on a GTX 1650, so off by default (formulas keep their text-layer text); set FORMULA_LATEX=1 on a faster GPU
FORMULA_LATEX = os.environ.get("FORMULA_LATEX", "0") == "1"
PAGE_BATCH = 10
CHUNK_TOKENS = 350  # child chunk size, approx tokens (words * 1.3)
TOP_K = 6
# tuned on golden set: answerable top scores >= 0.0114 (tables score low), unanswerable 0.0002-0.0099; the LLM is the second gate
RELEVANCE_MIN = float(os.environ.get("RELEVANCE_MIN", "0.005"))
