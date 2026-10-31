"""One-time model download. The only step that uses the network; the app runs offline afterwards."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config  # noqa: E402  (sets HF_HOME)

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download  # noqa: E402

api_only = "--api-only" in sys.argv  # API mode only needs the embedder and reranker
for repo, folder in config.REPOS.items():
    if api_only and folder not in ("bge-m3", "bge-reranker-base"):
        continue
    print("downloading", repo, flush=True)
    ignore = ["onnx/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "imgs/*"]
    if any(f.endswith(".safetensors") for f in list_repo_files(repo)):
        ignore.append("*.bin")  # skip duplicate weights
    snapshot_download(repo, local_dir=config.MODELS / folder, ignore_patterns=ignore)

if api_only:
    print("done (API mode models only)")
    sys.exit()
print("downloading", *config.LLM_GGUF, flush=True)
hf_hub_download(*config.LLM_GGUF, local_dir=config.MODELS / "llm-gguf")

print("downloading docling models", flush=True)
subprocess.run([str(Path(sys.executable).with_name("docling-tools")), "models", "download",
                "-o", str(config.DOCLING_ARTIFACTS),
                "layout", "tableformer", "code_formula", "picture_classifier", "rapidocr", "smolvlm"], check=True)
print("done")
