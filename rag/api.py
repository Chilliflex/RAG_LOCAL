"""FastAPI service. Binds to 127.0.0.1 (see __main__). One GPU job at a time: ingest and ask share a lock."""
import json
import re
import threading
import traceback
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rag import config, store
from rag.ingest import _set_status, doc_id_of, ingest, status

# built-in Swagger/ReDoc pages load JS from a CDN (network) and would shadow our /docs route
app = FastAPI(title="Local PDF RAG", docs_url=None, redoc_url=None, openapi_url=None)
LOCAL = {"127.0.0.1", "localhost"}
# A web page you visit can point its own domain at 127.0.0.1 (DNS rebinding) or POST a form here.
# Only requests addressed to localhost, from localhost, are served.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL))
GPU = threading.Lock()
UPLOADS = config.DATA / "uploads"
MAX_BYTES = 300 * 2**20


@app.middleware("http")
async def same_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and urlparse(origin).hostname not in LOCAL:
        return JSONResponse({"detail": "cross-site request refused"}, 403)
    return await call_next(request)


@app.get("/")
def page():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/upload")
async def upload(file: UploadFile):
    name = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "upload.pdf").name)
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File larger than 300 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Not a PDF")
    doc_id = doc_id_of(data)
    if store.path(doc_id).exists():
        return {"doc_id": doc_id, "status": "already ingested"}
    path = UPLOADS / doc_id / name  # per-document folder: two uploads named report.pdf never overwrite each other
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _set_status(doc_id, stage="queued", file=name)
    threading.Thread(target=_ingest, args=(path, doc_id), daemon=True).start()
    return {"doc_id": doc_id, "status": "queued"}


def _ingest(path, doc_id):
    with GPU:
        try:
            ingest(path)
        except Exception as e:  # surface the failure in the UI instead of a silent dead thread
            traceback.print_exc()
            _set_status(doc_id, stage="error", error=str(e))


@app.get("/status/{doc_id}")
def get_status(doc_id: str):
    if not re.fullmatch(r"[0-9a-f]{16}", doc_id):
        raise HTTPException(400, "bad id")
    s = status(doc_id)
    audit = config.DATA / doc_id / "audit.json"
    if audit.exists():
        s["audit"] = json.loads(audit.read_text(encoding="utf-8"))
    return s


@app.get("/docs")
def docs():
    out = []
    for d in store.list_ids():
        doc = store.load(d)
        out.append({"doc_id": d, "title": doc["title"], "file": doc["file"], "pages": doc["pages"],
                    "complete": status(d).get("complete")})
    return out


class Ask(BaseModel):
    question: str
    doc_ids: list[str] | None = None


@app.post("/ask")
def ask(req: Ask):
    from rag.query import ask as run
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    if not GPU.acquire(timeout=2):
        raise HTTPException(409, "Busy ingesting a document; ask again when it finishes.")
    try:
        return run(req.question.strip()[:2000], req.doc_ids or None)
    finally:
        GPU.release()


if __name__ == "__main__":  # self-check: foreign host / origin refused, local accepted, non-PDF refused
    from fastapi.testclient import TestClient
    c = TestClient(app, base_url="http://127.0.0.1")
    assert c.get("/docs", headers={"host": "evil.com"}).status_code == 400
    assert c.post("/upload", headers={"origin": "https://evil.com"}, files={"file": ("x.pdf", b"%PDF-1.4")}).status_code == 403
    assert c.post("/upload", headers={"origin": "http://127.0.0.1:8000"}, files={"file": ("x.txt", b"hello")}).status_code == 415
    assert c.get("/status/../etc").status_code in (400, 404) and c.get("/docs").status_code == 200
    print("ok")
