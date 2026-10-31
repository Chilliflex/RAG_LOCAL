"""Document JSON on disk: data/<doc_id>/document.json is the structured store (sections, refs, URLs, registry, xrefs)."""
import json
from pathlib import Path

from rag import config


def write_json(p: Path, obj):
    """Atomic: write a temp file, then rename. A crash never leaves a half-written file that blocks a resume."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def path(doc_id: str):
    return config.DATA / doc_id / "document.json"


def save(doc: dict):
    write_json(path(doc["doc_id"]), doc)


def load(doc_id: str) -> dict:
    return json.loads(path(doc_id).read_text(encoding="utf-8"))


def list_ids() -> list[str]:
    return sorted(p.parent.name for p in config.DATA.glob("*/document.json")) if config.DATA.exists() else []
