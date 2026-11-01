"""Local model loaders. One GPU with 4 GB: load a model for a stage, free() it before the next."""
import atexit
import gc
import os
from functools import cache

import torch

from rag import config


def free():
    if llm.cache_info().currsize:
        llm().close()  # llama.cpp holds VRAM until closed
    for f in (embedder, reranker, llm, trocr):
        f.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@cache
def embedder(device: str = config.DEVICE):
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(config.EMBED_MODEL, use_fp16=device == "cuda", devices=device)


@cache
def reranker(device: str = config.DEVICE):
    from FlagEmbedding import FlagReranker
    return FlagReranker(config.RERANK_MODEL, use_fp16=device == "cuda", devices=device)


def free_memory_gb() -> tuple[float, str]:
    """Free VRAM when the LLM runs on the GPU, else available RAM; measured now, so other apps' use counts."""
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.cuda.mem_get_info()[0] / 2**30, "VRAM"
    import psutil
    return psutil.virtual_memory().available / 2**30, "RAM"


def _window(free: float, with_reranker: bool) -> int:
    spare = free - config.VRAM_RESERVE_GB - config.LLM_VRAM_BASE_GB - (config.RERANKER_VRAM_GB if with_reranker else 0)
    return max(4096, min(config.LLM_CTX_MAX, int(spare / config.LLM_VRAM_PER_1K_GB) * 1024))


@cache
def rerank_device() -> str:
    """GPU when the reranker and an LLM window of at least 8k fit beside each other within the reserve; otherwise the CPU."""
    if config.RERANK_DEVICE != "auto":
        return config.RERANK_DEVICE
    if config.DEVICE != "cuda" or not torch.cuda.is_available():
        return config.DEVICE
    return "cuda" if _window(free_memory_gb()[0], True) >= 8192 else "cpu"


def fit_context() -> int:
    """Largest LLM window that keeps VRAM_RESERVE_GB free, from the measured memory cost of the model; LLM_CTX in .env overrides.
    Room is left for the reranker only when it shares the GPU and is not loaded yet (once loaded, its memory is already gone from free)."""
    if config.LLM_CTX:
        return config.LLM_CTX
    return _window(free_memory_gb()[0], rerank_device() == "cuda" and not reranker.cache_info().currsize)


def budgets() -> dict:
    """Effective passage budget per depth on this machine: the configured one, capped by the window (cloud: as configured)."""
    cap = (1_000_000 if config.PROVIDER == "cloud" else fit_context()) - config.ANSWER_TOKENS - 600
    return {name: min(budget or cap, cap) for name, (budget, _) in config.DEPTHS[config.PROVIDER].items()}


@cache
def llm():
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))  # llama.cpp uses torch's CUDA DLLs
    from llama_cpp import Llama
    m = Llama(config.LLM_MODEL, n_ctx=fit_context(), n_gpu_layers=-1 if config.DEVICE == "cuda" else 0, verbose=False)
    atexit.register(m.close)  # avoid noisy teardown errors
    return m


def context_window() -> int:
    return 1_000_000 if config.PROVIDER == "cloud" else llm().n_ctx()


last_provider = config.PROVIDER  # set to "local" when a cloud call failed and the local model answered instead (query.ask resets it)


def generate(system: str, user: str, max_new_tokens: int = 512) -> str:
    global last_provider
    if config.PROVIDER == "cloud":
        from rag import cloud
        try:
            return cloud.generate(system, user, max_new_tokens)
        except (ConnectionError, ValueError) as e:
            if not os.path.exists(config.LLM_MODEL):
                raise ValueError(f"{e}. The local model is not installed either (python scripts/download_models.py).") from None
            print(f"  Gemini unavailable ({e}); answering with the local model", flush=True)
            last_provider = "local"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = llm().create_chat_completion(msgs, max_tokens=max_new_tokens, temperature=0.0)
    return out["choices"][0]["message"]["content"].strip()


def count_tokens(text: str) -> int:
    if config.PROVIDER == "cloud":
        from rag import cloud
        return cloud.count_tokens(text)
    return len(llm().tokenize(text.encode("utf-8"), add_bos=False))


@cache
def trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained(config.HANDWRITING_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(config.HANDWRITING_MODEL).to(config.DEVICE).eval()
    return proc, model


def read_handwriting(images) -> list[tuple[str, float]]:
    """images: PIL RGB images, one text line each -> (text, confidence = mean token probability)."""
    proc, model = trocr()
    out = []
    for i in range(0, len(images), 8):
        px = proc(images=images[i:i + 8], return_tensors="pt").pixel_values.to(config.DEVICE)
        with torch.inference_mode():
            g = model.generate(px, max_new_tokens=96, output_scores=True, return_dict_in_generate=True)
            probs = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).exp()
        texts = proc.batch_decode(g.sequences, skip_special_tokens=True)
        for t, p, seq in zip(texts, probs, g.sequences[:, 1:]):
            keep = seq != proc.tokenizer.pad_token_id
            out.append((t, float(p[keep].mean()) if keep.any() else 0.0))
    return out


if __name__ == "__main__":  # self-check: Gemini down -> local answer, marked; Gemini down and no local model -> readable error
    from unittest import mock

    config.PROVIDER = "cloud"
    fake_llm = mock.Mock(create_chat_completion=lambda *a, **k: {"choices": [{"message": {"content": " local answer "}}]})
    # run as __main__, so patch this module's own globals, not rag.models (a separate copy)
    with mock.patch("rag.cloud.generate", side_effect=ConnectionError("gemini_generate failed after 3 attempts: ConnectError")), \
            mock.patch.dict(globals(), {"llm": lambda: fake_llm}), mock.patch("os.path.exists", return_value=True):
        assert generate("s", "u") == "local answer" and last_provider == "local"
    with mock.patch("rag.cloud.generate", side_effect=ValueError("gemini_generate: HTTP 401")), mock.patch("os.path.exists", return_value=False):
        try:
            generate("s", "u"); raise AssertionError
        except ValueError as e:
            assert "HTTP 401" in str(e) and "download_models" in str(e), e
    # 3.2 GB free (a 4 GB GTX 1650 driving the display): both models would leave only a 4k window, so the reranker goes to the CPU
    # and the LLM gets 8k; 4.5 GB free: both on the GPU with 16k; plenty: the trained maximum
    config.LLM_CTX, config.RERANK_DEVICE, config.DEVICE = 0, "auto", "cuda"
    for free, device, expected in ((3.2, "cpu", 8192), (4.5, "cuda", 16384), (1.0, "cpu", 4096), (40.0, "cuda", 32768)):
        rerank_device.cache_clear()
        with mock.patch.dict(globals(), {"free_memory_gb": lambda: (free, "VRAM")}), mock.patch("torch.cuda.is_available", return_value=True):
            assert (rerank_device(), fit_context()) == (device, expected), (free, rerank_device(), fit_context())
    rerank_device.cache_clear()
    with mock.patch.dict(globals(), {"fit_context": lambda: 4096}):
        config.PROVIDER = "local"; config.ANSWER_TOKENS = 1024
        assert budgets() == {"fast": 2472, "balanced": 2472, "maximum": 2472}, budgets()  # every depth capped by a 4k window
    print("ok")
