"""Qdrant (embedded, on disk): dense cosine (HNSW) + sparse lexical vectors from BGE-M3."""
import atexit
from functools import cache

from qdrant_client import QdrantClient, models as qm

from rag import config
from rag.models import embedder

COLLECTION = "chunks"


@cache
def client() -> QdrantClient:
    c = QdrantClient(path=str(config.DATA / "qdrant"))
    atexit.register(c.close)  # flush before interpreter teardown
    return c


def reset():
    """After a mode switch: close the store so the next call opens the other mode's folder."""
    if client.cache_info().currsize:
        c = client()
        atexit.unregister(c.close)
        c.close()
        client.cache_clear()


def _ensure():
    if not client().collection_exists(COLLECTION):
        client().create_collection(
            COLLECTION,
            vectors_config={"dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )


def _sparse(weights: dict) -> qm.SparseVector:
    return qm.SparseVector(indices=[int(k) for k in weights], values=[float(v) for v in weights.values()])


def encode(texts: list[str], device: str = config.DEVICE) -> dict:
    return embedder(device).encode(texts, batch_size=8, max_length=512, return_dense=True, return_sparse=True)


def add(chunks: list[dict]):
    _ensure()
    delete_doc(chunks[0]["doc_id"])
    for i in range(0, len(chunks), 64):
        batch = chunks[i:i + 64]
        enc = encode([c["embed_text"] for c in batch])
        client().upsert(COLLECTION, [
            qm.PointStruct(id=c["uid"], vector={"dense": d.tolist(), "sparse": _sparse(s)},
                           payload={k: v for k, v in c.items() if k != "embed_text"})
            for c, d, s in zip(batch, enc["dense_vecs"], enc["lexical_weights"])
        ])


def delete_doc(doc_id: str):
    if client().collection_exists(COLLECTION):
        client().delete(COLLECTION, qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]))


def search(query: str, doc_ids: list[str] | None = None, limit: int = 30, device: str = "cpu") -> list[dict]:
    """Dense + sparse prefetch fused with reciprocal rank fusion."""
    if not client().collection_exists(COLLECTION):
        return []
    enc = encode([query], device)
    flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))]) if doc_ids else None
    res = client().query_points(
        COLLECTION,
        prefetch=[qm.Prefetch(query=enc["dense_vecs"][0].tolist(), using="dense", limit=limit, filter=flt),
                  qm.Prefetch(query=_sparse(enc["lexical_weights"][0]), using="sparse", limit=limit, filter=flt)],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF), limit=limit, with_payload=True,
    )
    return [p.payload for p in res.points]
