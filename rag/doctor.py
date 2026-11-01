"""Requirements check for both modes: python -m rag doctor, GET /doctor. Reports pass/fail per line; never reports key values."""
import shutil
import sys

import psutil

from rag import config

GB = 2**30
# What each mode needs, for the requirements panel. Sizes measured on this install; prices are list prices, verify on the provider page.
LOCAL_MODELS = [("docling", "Docling bundle (Heron layout, TableFormer, RapidOCR, figure classifier, SmolVLM-256M)", 1.5),
                ("trocr-base-handwritten", "TrOCR-base-handwritten", 1.3), ("bge-m3", "BGE-M3 embeddings", 2.3),
                ("bge-reranker-base", "bge-reranker-base", 1.1), ("llm-gguf", "Qwen2.5-3B-Instruct Q4_K_M GGUF", 2.0)]
API_MODELS = LOCAL_MODELS[2:4]
APIS = [{"name": "Mistral API key", "use": "OCR: text, tables, formulas, handwriting, figure crops", "model": config.MISTRAL_OCR_MODEL,
         "url": "https://console.mistral.ai/api-keys", "price": "$4 per 1,000 pages ($2 batch)"},
        {"name": "Google AI Studio key", "use": "Gemini: answers, summaries, figure descriptions", "model": config.GEMINI_MODEL,
         "url": "https://aistudio.google.com/apikey", "price": "$0.30 in / $2.50 out per 1M tokens; paid tier so prompts are not used for training"}]


def _check(name, ok, detail=""):
    return {"name": name, "ok": bool(ok), "detail": detail}


def _models(models):
    out = []
    for folder, label, gb in models:
        p = config.MODELS / folder
        present = p.is_dir() and any(p.iterdir())
        out.append(_check(label, present, f"{p} ({gb} GB)" if present else f"missing: run python scripts/download_models.py ({gb} GB)"))
    return out


def run() -> dict:
    free = shutil.disk_usage(config.ROOT).free / GB
    ram = psutil.virtual_memory().total / GB
    import torch
    cuda = torch.cuda.is_available()
    vram = torch.cuda.get_device_properties(0).total_memory / GB if cuda else 0
    local = [_check("Python 3.11+ in a virtual environment", sys.version_info >= (3, 11) and sys.prefix != sys.base_prefix, sys.version.split()[0]),
             _check("PyTorch sees the GPU", cuda or config.DEVICE == "cpu",
                    f"{torch.cuda.get_device_name(0)}" if cuda else "DEVICE=cpu: 5-10x slower" if config.DEVICE == "cpu" else "no CUDA GPU; set DEVICE=cpu to run slowly on the CPU"),
             _check("VRAM 4 GB or more", vram >= 3.5 or config.DEVICE == "cpu", f"{vram:.1f} GB" if cuda else "no GPU"),
             _check("RAM 8 GB or more (16 recommended)", ram >= 7.5, f"{ram:.0f} GB"),
             _check("Free disk 20 GB (2 GB once the models are downloaded)", free >= 20 or (free >= 2 and all(c["ok"] for c in _models(LOCAL_MODELS))), f"{free:.0f} GB free"),
             *_models(LOCAL_MODELS)]
    try:
        import llama_cpp  # noqa: F401
        local.append(_check("llama-cpp-python installed", True))
    except ImportError as e:
        local.append(_check("llama-cpp-python installed", False, str(e)))
    from rag import models
    free_gb, kind = models.free_memory_gb()
    ctx = models.fit_context()
    local.append(_check("LLM window from free memory", True,
                        f"{ctx:,} tokens ({free_gb:.1f} GB {kind} free now, {config.VRAM_RESERVE_GB} GB kept free); reranker on the {models.rerank_device().upper()}"
                        + ("; fixed by LLM_CTX in .env" if config.LLM_CTX else "; for more, close other GPU apps or set VRAM_RESERVE_GB=0.5 or LLM_CTX in .env" if ctx <= 4096 else "")))
    from rag import cloud
    api = [_check("Internet connection", cloud.online(), "https reaches api.mistral.ai"),
           _check(f"{APIS[0]['name']} ({APIS[0]['url']})", *cloud.check_mistral()),
           _check(f"{APIS[1]['name']} ({APIS[1]['url']})", *cloud.check_google()),
           _check("RAM 8 GB or more", ram >= 7.5, f"{ram:.0f} GB"),
           _check("Free disk 5 GB", free >= 5, f"{free:.0f} GB free"),
           *_models(API_MODELS)]
    return {"provider": config.PROVIDER, "local": local, "api": api, "apis": APIS,
            "local_models": [{"name": l, "gb": gb} for _, l, gb in LOCAL_MODELS], "api_models": [{"name": l, "gb": gb} for _, l, gb in API_MODELS]}


def render(report: dict) -> str:
    lines = [f"Active mode: {report['provider']}"]
    for group, title in (("local", "Local mode (free, offline)"), ("api", "API mode (Mistral OCR + Gemini)")):
        n_ok = sum(c["ok"] for c in report[group])
        lines.append(f"\n{title}: {n_ok}/{len(report[group])} checks pass")
        lines += [f"  [{'ok' if c['ok'] else ' X'}] {c['name']}{': ' + c['detail'] if c['detail'] else ''}" for c in report[group]]
    return "\n".join(lines)


if __name__ == "__main__":  # self-check: fake keys and network; the key value never appears in the report
    import json
    from unittest import mock

    config.MISTRAL_API_KEY = config.GOOGLE_API_KEY = "sk-test-secret"
    with mock.patch("httpx.get", side_effect=ConnectionError), mock.patch("httpx.post", side_effect=ConnectionError):
        import httpx
        with mock.patch("httpx.get", return_value=httpx.Response(200)), mock.patch("httpx.post", return_value=httpx.Response(403)):
            r = run()
    text = json.dumps(r) + render(r)
    assert "sk-test" not in text and len(r["local"]) == 12 and len(r["api"]) == 7
    by = {c["name"].split(" (")[0]: c for c in r["api"]}
    assert by["Mistral API key"]["ok"] and not by["Google AI Studio key"]["ok"] and by["Internet connection"]["ok"]
    print("ok")
