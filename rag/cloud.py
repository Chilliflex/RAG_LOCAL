"""Mistral OCR and Gemini over plain HTTPS. Imported only when PROVIDER == "cloud".
Every call is logged to data-cloud/api_usage.jsonl (counts and seconds; never keys or text)."""
import base64
import json
import time

import httpx

from rag import config

MISTRAL = "https://api.mistral.ai/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
RETRY = {429, 500, 502, 503, 504}
PRICES = {"mistral_page": 4 / 1000, "gemini_in": 0.30 / 1e6, "gemini_out": 2.50 / 1e6}  # USD, list prices mid-2026; verify on the provider pages
USAGE = None  # test override; normally data-cloud/api_usage.jsonl, resolved per call because the mode can change at runtime
DESCRIBE = ("Describe this figure for a search index in 2-4 sentences: what kind of figure it is, what it shows, "
            "axis labels and units, the main trend or the steps of the diagram in order, and any text or handwriting in it, transcribed.")


def require_keys():
    missing = [n for n, k in (("MISTRAL_API_KEY", config.MISTRAL_API_KEY), ("GOOGLE_API_KEY", config.GOOGLE_API_KEY)) if not k]
    if missing:
        raise ValueError(f"API mode needs {' and '.join(missing)}: set it in .env or data/settings.json (python -m rag doctor shows what is missing)")


def _log(op, seconds, **kw):
    path = USAGE or config.DATA / "api_usage.jsonl"
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "op": op, "seconds": round(seconds, 1), **kw}) + "\n")


def _post(op, url, body, headers, timeout) -> tuple[dict, float]:
    t = time.time()
    for attempt in range(3):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=timeout)
            if r.status_code not in RETRY:
                break
            problem = f"HTTP {r.status_code}"
        except httpx.TransportError as e:
            problem = type(e).__name__
        if attempt < 2:
            time.sleep(2 ** (attempt + 1))
    else:
        _log(op, time.time() - t, error=problem)
        raise ConnectionError(f"{op} failed after 3 attempts: {problem}")
    if r.status_code != 200:
        _log(op, time.time() - t, error=f"HTTP {r.status_code}")
        raise ValueError(f"{op}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json(), time.time() - t


def _mistral_headers():
    return {"Authorization": f"Bearer {config.MISTRAL_API_KEY}"}


def ocr(pdf_bytes: bytes) -> dict:
    """One OCR call: per-page markdown, images (base64), page dimensions. Headers/footers are split out so they never reach the text."""
    body = {"model": config.MISTRAL_OCR_MODEL,
            "document": {"type": "document_url", "document_url": "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()},
            "include_image_base64": True, "extract_header": True, "extract_footer": True}
    out, s = _post("mistral_ocr", f"{MISTRAL}/ocr", body, _mistral_headers(), 600)
    _log("mistral_ocr", s, pages=(out.get("usage_info") or {}).get("pages_processed"), bytes=len(pdf_bytes))
    return out


def _gemini(op, parts, system, max_tokens) -> str:
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingBudget": 0}}}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    out, s = _post(op, f"{GEMINI}/{config.GEMINI_MODEL}:generateContent", body, {"x-goog-api-key": config.GOOGLE_API_KEY}, 60)
    u = out.get("usageMetadata") or {}
    _log(op, s, in_tokens=u.get("promptTokenCount"), out_tokens=u.get("candidatesTokenCount"))
    cands = out.get("candidates") or []
    if not cands or "content" not in cands[0]:
        raise ValueError(f"Gemini returned no answer: {out.get('promptFeedback') or cands}")
    return "".join(p.get("text", "") for p in cands[0]["content"].get("parts", [])).strip()


def generate(system: str, user: str, max_new_tokens: int = 512) -> str:
    return _gemini("gemini_generate", [{"text": user}], system, max_new_tokens)


def describe_image(data: bytes, mime: str) -> str:
    return _gemini("gemini_describe", [{"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}, {"text": DESCRIBE}], None, 200)


def count_tokens(text: str) -> int:
    return len(text) // 4 + 1  # estimate: Gemini's 1M context makes exactness pointless, and this saves one HTTP call per passage


def check_mistral() -> tuple[bool, str]:
    if not config.MISTRAL_API_KEY:
        return False, "no key"
    try:
        r = httpx.get(f"{MISTRAL}/models", headers=_mistral_headers(), timeout=15)
    except httpx.TransportError as e:
        return False, f"offline ({type(e).__name__})"
    return r.status_code == 200, "key accepted" if r.status_code == 200 else f"key rejected (HTTP {r.status_code})"


def check_google() -> tuple[bool, str]:
    if not config.GOOGLE_API_KEY:
        return False, "no key"
    try:  # countTokens is free
        r = httpx.post(f"{GEMINI}/{config.GEMINI_MODEL}:countTokens", json={"contents": [{"parts": [{"text": "ping"}]}]},
                       headers={"x-goog-api-key": config.GOOGLE_API_KEY}, timeout=15)
    except httpx.TransportError as e:
        return False, f"offline ({type(e).__name__})"
    return r.status_code == 200, f"key accepted, {config.GEMINI_MODEL}" if r.status_code == 200 else f"key rejected (HTTP {r.status_code})"


def cost(entries: list[dict]) -> dict:
    """USD and counts for usage-log entries (from api_usage.jsonl)."""
    pages = sum(e.get("pages") or 0 for e in entries)
    tin = sum(e.get("in_tokens") or 0 for e in entries)
    tout = sum(e.get("out_tokens") or 0 for e in entries)
    return {"calls": len(entries), "ocr_pages": pages, "gemini_in_tokens": tin, "gemini_out_tokens": tout,
            "usd": round(pages * PRICES["mistral_page"] + tin * PRICES["gemini_in"] + tout * PRICES["gemini_out"], 4)}


def online() -> bool:
    try:
        httpx.get(f"{MISTRAL}/models", timeout=10)  # 401 without a key still proves the connection
        return True
    except httpx.TransportError:
        return False


if __name__ == "__main__":  # self-check with a fake network: retries, errors, usage log, key never logged
    import tempfile
    from pathlib import Path
    from unittest import mock

    config.MISTRAL_API_KEY = config.GOOGLE_API_KEY = "sk-test-secret"
    with tempfile.TemporaryDirectory() as tmp:
        USAGE = Path(tmp) / "usage.jsonl"
        ok = httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Six layers (p. 3)."}]}}],
                                       "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 7}})
        with mock.patch("httpx.post", side_effect=[httpx.Response(429), httpx.ConnectError("x"), ok]) as post, mock.patch("time.sleep"):
            assert generate("sys", "q") == "Six layers (p. 3)."
        assert post.call_count == 3 and post.call_args.kwargs["json"]["system_instruction"]["parts"][0]["text"] == "sys"
        with mock.patch("httpx.post", return_value=httpx.Response(200, json={"pages": [], "usage_info": {"pages_processed": 2}})) as post:
            ocr(b"%PDF-1.4")
        assert post.call_args.kwargs["json"]["document"]["document_url"].startswith("data:application/pdf;base64,JVBERi0xLjQ=")
        for responses, expected in (([httpx.Response(401, text="bad")], "HTTP 401"), ([httpx.Response(503)] * 3, "after 3 attempts"),
                                    ([httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})], "no answer")):
            with mock.patch("httpx.post", side_effect=responses), mock.patch("time.sleep"):
                try:
                    generate("s", "u"); raise AssertionError("expected a failure")
                except (ValueError, ConnectionError) as e:
                    assert expected in str(e) and "sk-test" not in str(e), e
        log = USAGE.read_text()
        assert log.count("\n") == 5 and "sk-test" not in log and '"pages": 2' in log and '"in_tokens": 40' in log, log
        with mock.patch("httpx.get", return_value=httpx.Response(401)):
            assert check_mistral() == (False, "key rejected (HTTP 401)") and online()
        with mock.patch("httpx.post", return_value=httpx.Response(200, json={"totalTokens": 1})):
            assert check_google()[0]
    assert count_tokens("a" * 400) == 101
    assert cost([{"pages": 100}, {"in_tokens": 1_000_000, "out_tokens": 100_000}, {"error": "x"}]) == \
        {"calls": 3, "ocr_pages": 100, "gemini_in_tokens": 1_000_000, "gemini_out_tokens": 100_000, "usd": 0.95}
    print("ok")
