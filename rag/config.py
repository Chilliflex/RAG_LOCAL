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
    return ROOT / ("data" if provider == "local" else "data-cloud")


# Depth = how much document text one answer may use: (passage budget in tokens, chunks kept after reranking).
# Local: the window is set by memory (models.fit_context); None = fill the window. Cloud: cost per question grows with the budget.
DEPTHS = {"local": {"fast": (2500, 6), "balanced": (6000, 10), "maximum": (None, 20)},
          "cloud": {"standard": (12000, 6), "thorough": (30000, 15), "maximum": (100000, 40)}}


def depth_settings(depth: str | None = None) -> tuple[str, int | None, int]:
    """-> (name, budget, k) for the requested depth, else the saved one, else the first for the mode."""
    table = DEPTHS[PROVIDER]
    name = depth if depth in table else DEPTH if DEPTH in table else next(iter(table))
    return (name, *table[name])


def set_mode(provider: str, depth: str | None = None, **keys):
    """Switch mode for this process and persist it; an empty key keeps the stored one."""
    global PROVIDER, DATA, DEPTH, MISTRAL_API_KEY, GOOGLE_API_KEY
    DATA = _derived(provider)
    keys = {k: v for k, v in keys.items() if v}
    save_settings(provider=provider, **({"depth": depth} if depth else {}), **keys)
    PROVIDER = provider
    DEPTH = depth or DEPTH
    DATA.mkdir(exist_ok=True)
    MISTRAL_API_KEY = keys.get("mistral_api_key", MISTRAL_API_KEY)
    GOOGLE_API_KEY = keys.get("google_api_key", GOOGLE_API_KEY)


_settings = load_settings()
PROVIDER = os.environ.get("PROVIDER") or _settings.get("provider") or "local"  # "local": offline models; "cloud": Mistral OCR + Gemini
DEPTH = os.environ.get("DEPTH") or _settings.get("depth") or ""
DATA = _derived(PROVIDER)
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
LLM_CTX = int(os.environ.get("LLM_CTX", "0"))  # 0 = choose from free memory (models.fit_context)
LLM_CTX_MAX = 32768  # Qwen2.5-3B's trained window
# Measured with torch.cuda.mem_get_info on a GTX 1650, 3B Q4_K_M GGUF: 2.36 GB at 8k and 2.90 GB at 16k, +0.12/0.24 GB peak while
# answering -> 1.8 GB + 0.08 GB per 1k tokens. Re-measure for another model (python -m rag doctor shows the resulting window).
LLM_VRAM_BASE_GB, LLM_VRAM_PER_1K_GB = 1.8, 0.08
RERANKER_VRAM_GB = 0.6  # the reranker stays resident next to the LLM, so both must fit; else it runs on the CPU (~9 s per question)
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "auto")  # auto | cuda | cpu
VRAM_RESERVE_GB = float(os.environ.get("VRAM_RESERVE_GB", "0.75"))  # kept free for the display and other apps; 0.5 trades safety for depth
ANSWER_TOKENS = 2048 if PROVIDER == "cloud" else 1024
# chunks scoring below this share of the best one are not sent; 0 = send all k. 0.1 was tried on 2026-10-31 and cost 5 answers of 53
# (the right chunk often scores far below the top one), so it stays off by default
SCORE_RATIO = float(os.environ.get("SCORE_RATIO", "0"))

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
