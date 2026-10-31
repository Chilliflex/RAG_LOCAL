"""Offline check of the API flows: no API key, no network, nothing saved to Langflow.

Run with Flowgen's venv, from any folder:
  <Automate_Langflow>/langflow_venv/Scripts/python.exe langflow/check_flows.py <Automate_Langflow> <catalog.json>
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).parent
sys.path[:0] = [sys.argv[1], str(HERE)]

import httpx
from langchain_core.embeddings import DeterministicFakeEmbedding
from lfx.base.vectorstores.model import LCVectorStoreComponent
from lfx.components.processing.parser import ParserComponent
from lfx.components.processing.split_text import SplitTextComponent
from lfx.custom.utils import build_custom_component_template
from lfx.custom.custom_component.component import Component
from lfx_bundles.qdrant.qdrant import QdrantVectorStoreComponent

from mistral_ocr import MistralOCR, pages_to_rows

# --- page numbers and figures ---
rows = pages_to_rows(
    [
        {"index": 1, "markdown": "second ![img-0.jpeg](img-0.jpeg)",
         "images": [{"id": "img-0.jpeg", "image_annotation": '{"description": "A flows to B"}'}]},
        {"index": 0, "markdown": "first"},
        {"index": 2, "markdown": "  "},
    ],
    "a.pdf",
)
assert [r["page"] for r in rows] == [1, 2], rows
assert rows[1]["text"] == "second [Figure: A flows to B]", rows

# --- retries, errors, and the key staying out of the message ---
with tempfile.TemporaryDirectory() as tmp:
    pdf = Path(tmp) / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    ocr = MistralOCR(pdf=str(pdf), api_key="secret-key")
    good = httpx.Response(200, json={"pages": [
        {"index": 0, "markdown": "alpha fact\n\nbeta fact"},
        {"index": 1, "markdown": "gamma fact"},
    ]})
    with mock.patch("httpx.post", side_effect=[httpx.Response(429), httpx.ConnectError("x"), good]) as post, \
            mock.patch("time.sleep"):
        pages = ocr.read_pages()
    assert post.call_count == 3
    body = post.call_args.kwargs["json"]
    assert body["document"]["document_url"].startswith("data:application/pdf;base64,")
    assert ocr.status == "2 pages read, 0 empty", ocr.status
    for side_effect, expected in ((httpx.Response(401, text="bad key"), "HTTP 401"),
                                  ([httpx.Response(500)] * 3, "after 3 attempts")):
        with mock.patch("httpx.post", side_effect=[side_effect] if isinstance(side_effect, httpx.Response) else side_effect), \
                mock.patch("time.sleep"):
            try:
                ocr.read_pages()
                raise AssertionError("expected a failure")
            except ValueError as exc:
                assert expected in str(exc) and "secret-key" not in str(exc), exc

    # --- page number survives split -> Qdrant -> Parser ---
    chunks = SplitTextComponent(data_inputs=pages, chunk_size=20, chunk_overlap=0, separator="\n\n",
                                text_key="text", keep_separator="False", clean_output=False).split_text()
    common = dict(collection_name="t", host="", port=6333, grpc_port=6334, api_key=None, prefix="", timeout=0,
                  path=str(Path(tmp) / "q"), url="", distance_func="Cosine", content_payload_key="page_content",
                  metadata_payload_key="metadata", should_cache_vector_store=False, number_of_results=3,
                  embedding=DeterministicFakeEmbedding(size=8))
    found = QdrantVectorStoreComponent(ingest_data=chunks, search_query="fact", **common).search_documents()
    assert len(found) == 3, found
    spec = json.loads((HERE / "chat.flowspec.json").read_text(encoding="utf-8"))
    params = next(n for n in spec["nodes"] if n["name"] == "parser")["params"]
    text = ParserComponent(input_data=found, mode="Parser", pattern=params["pattern"], sep=params["sep"]) \
        .parse_combined_text().text
    for expected in ("[a.pdf, p. 1]\nalpha fact", "[a.pdf, p. 1]\nbeta fact", "[a.pdf, p. 2]\ngamma fact"):
        assert expected in text, text
print("component and page-number path: ok")

# --- Flowgen's own normalize, lint and compile, on the real catalog ---
from flowgen.catalog import Catalog
from flowgen.compiler import compile_flow
from flowgen.linter import lint
from flowgen.models import FlowSpec
from flowgen.normalize import normalize_spec

raw = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
node, _ = build_custom_component_template(Component(_code=(HERE / "mistral_ocr.py").read_text(encoding="utf-8")))
# Flowgen asks the server for a prompt's {variables} when it builds; stand in for that answer offline.
prompt = raw["components"]["Prompt Template"]["template"]
for variable in ("context", "question"):
    prompt[variable] = {"name": variable, "type": "str", "input_types": ["Message"], "show": True, "value": ""}
catalog = Catalog({"all": {**raw["components"], "MistralOCR": node}})
catalog.metadata = raw["metadata"]
failed = False
for name in ("ingest", "chat"):
    spec = FlowSpec.model_validate_json((HERE / f"{name}.flowspec.json").read_text(encoding="utf-8"))
    spec, notes = normalize_spec(spec, catalog)
    issues = lint(spec, catalog)
    for issue in [*notes, *issues]:
        print(f"  {name}: {issue}")
    errors = [i for i in issues if str(getattr(i, "code", i)).startswith("E")]
    if errors:
        failed = True
        continue
    flow = compile_flow(spec, catalog)
    print(f"{name}: lint ok, compiled {len(flow['data']['nodes'])} nodes, {len(flow['data']['edges'])} edges")
sys.exit(1 if failed else 0)
