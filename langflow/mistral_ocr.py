import base64
import json
import time
from pathlib import Path

import httpx
from lfx.custom.custom_component.component import Component
from lfx.io import FileInput, Output, SecretStrInput
from lfx.schema.dataframe import DataFrame

OCR_URL = "https://api.mistral.ai/v1/ocr"
MAX_BYTES = 50 * 1024 * 1024
RETRY_STATUS = {429, 500, 502, 503, 504}
IMAGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "image_description",
        "schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What the figure shows. For a diagram or flow, list its nodes and arrows in order. "
                    "For a chart, give the axes and trend. Transcribe any handwriting.",
                }
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    },
}


def pages_to_rows(pages: list[dict], source: str) -> list[dict]:
    """Turn Mistral pages into one row per non-empty page, with figure descriptions in place of image links."""
    rows = []
    # The API's index base is not documented consistently, so number pages by document order.
    for number, page in enumerate(sorted(pages, key=lambda p: p["index"]), start=1):
        text = page.get("markdown") or ""
        for image in page.get("images") or []:
            note = image.get("image_annotation")
            if note:
                try:
                    note = json.loads(note)["description"]
                except (ValueError, KeyError, TypeError):
                    pass  # keep the raw annotation: a messy description beats losing the figure
                text = text.replace(f"![{image['id']}]({image['id']})", f"[Figure: {note}]")
        if text.strip():
            rows.append({"text": text, "page": number, "source": source})
    return rows


class MistralOCR(Component):
    display_name = "Mistral OCR"
    description = "Read a PDF with Mistral OCR. Returns one row per page with text, page and source columns."
    icon = "scan-text"
    name = "MistralOCR"

    inputs = [
        FileInput(name="pdf", display_name="PDF", file_types=["pdf"], required=True),
        SecretStrInput(name="api_key", display_name="Mistral API Key", required=True),
    ]

    outputs = [
        Output(display_name="Pages", name="pages", method="read_pages"),
    ]

    def _call(self, payload: dict) -> dict:
        for attempt in range(3):
            try:
                response = httpx.post(OCR_URL, json=payload, timeout=600,
                                      headers={"Authorization": f"Bearer {self.api_key}"})
                if response.status_code not in RETRY_STATUS:
                    break
                problem = f"HTTP {response.status_code}"
            except httpx.TransportError as exc:
                problem = type(exc).__name__
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
        else:
            raise ValueError(f"Mistral OCR failed after 3 attempts: {problem}")
        if response.status_code != 200:
            raise ValueError(f"Mistral OCR returned HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def read_pages(self) -> DataFrame:
        pdf = self.pdf[0] if isinstance(self.pdf, list) else self.pdf
        if not pdf:
            raise ValueError("Upload a PDF first")
        path = Path(self.resolve_path(pdf))
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError(f"{path.name} is over Mistral's 50 MB limit; split it first")
        result = self._call({
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64," + base64.b64encode(raw).decode(),
            },
            "bbox_annotation_format": IMAGE_SCHEMA,
        })
        pages = result.get("pages") or []
        rows = pages_to_rows(pages, path.name)
        if not rows:
            raise ValueError(f"Mistral OCR found no text in {path.name}")
        self.status = f"{len(pages)} pages read, {len(pages) - len(rows)} empty"
        return DataFrame(rows)
