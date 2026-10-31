import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Relative HF_HOME in .env is relative to the project, not the cwd.
os.environ["HF_HOME"] = str((ROOT / os.environ.get("HF_HOME", "models")).resolve())

# Mode and API keys chosen in the UI live here, whichever mode is active. .env values win over the file.
SETTINGS = ROOT / "data" / "settings.json"


def load_settings() -> dict:
    return json.loads(SETTINGS.read_text(encoding="utf-8")) if SETTINGS.exists() else {}


def save_settings(**kw):
    SETTINGS.parent.mkdir(exist_ok=True)
    SETTINGS.write_text(json.dumps({**load_settings(), **kw}, indent=1), encoding="utf-8")
    # shortcut: chmod only clears the read-only bit on Windows; the file sits in the user's own folder, so upgrade to icacls if data/ moves to a shared disk
    SETTINGS.chmod(0o600)


def _derived(provider: str):
    if provider not in ("local", "cloud"):
        raise ValueError(f"PROVIDER must be 'local' or 'cloud', not {provider!r}")
    # The two modes parse the same PDF differently, so each keeps its own documents and index.
    # Passages per answer: the local prompt must fit LLM_CTX; Gemini gets 12000.
    return ROOT / ("data" if provider == "local" else "data-cloud"), 12000 if provider == "cloud" else 2500


def set_mode(provider: str, **keys):
    """Switch mode for this process and persist it; an empty key keeps the stored one."""
    global PROVIDER, DATA, CONTEXT_TOKENS, MISTRAL_API_KEY, GOOGLE_API_KEY
    DATA, CONTEXT_TOKENS = _derived(provider)
    keys = {k: v for k, v in keys.items() if v}
    save_settings(provider=provider, **keys)
    PROVIDER = provider
    DATA.mkdir(exist_ok=True)
    MISTRAL_API_KEY = keys.get("mistral_api_key", MISTRAL_API_KEY)
    GOOGLE_API_KEY = keys.get("google_api_key", GOOGLE_API_KEY)


_settings = load_settings()
PROVIDER = os.environ.get("PROVIDER") or _settings.get("provider") or "local"  # "local": offline models; "cloud": Mistral OCR + Gemini
DATA, CONTEXT_TOKENS = _derived(PROVIDER)
DATA.mkdir(exist_ok=True)  # the query log is written before any document exists
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY") or _settings.get("mistral_api_key") or ""
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or _settings.get("google_api_key") or ""
MISTRAL_OCR_MODEL = os.environ.get("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DEVICE = os.environ.get("DEVICE", "cuda")
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
