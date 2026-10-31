"""Triage (PyMuPDF) and per-page parsing (Docling), checkpointed per page: data/<doc_id>/pages/NNNN.json."""
import json
from pathlib import Path

import pymupdf

from rag import config
from rag.store import write_json

LABELS = {"text": "text", "paragraph": "text", "list_item": "list", "section_header": "heading", "title": "heading",
          "table": "table", "picture": "figure", "chart": "figure", "formula": "formula", "code": "code",
          "footnote": "footnote", "page_header": "page_header", "page_footer": "page_footer",
          "handwritten_text": "text", "reference": "text", "caption": "caption"}


def triage(pdf: Path) -> tuple[list[dict], list, list[dict], dict]:
    """Per page: text-layer chars, image count, scanned flag. Also bookmarks (TOC), link URLs, page widths."""
    pages, links, widths = [], [], {}
    with pymupdf.open(pdf) as d:
        toc = d.get_toc()
        for n, page in enumerate(d, 1):
            chars = len(page.get_text().strip())
            images = len(page.get_images())
            pages.append({"page": n, "chars": chars, "images": images, "scanned": chars < 50 and images > 0})
            links += [{"page": n, "url": l["uri"]} for l in page.get_links() if l.get("uri")]
            widths[n] = page.rect.width
    return pages, toc, links, widths


def _converter():
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, smolvlm_picture_description
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(
        artifacts_path=str(config.DOCLING_ARTIFACTS), enable_remote_services=False,
        do_ocr=True, ocr_options=RapidOcrOptions(), do_table_structure=True, do_formula_enrichment=config.FORMULA_LATEX,
        do_picture_classification=True, do_picture_description=True, picture_description_options=smolvlm_picture_description,
        generate_parsed_pages=True, accelerator_options=AcceleratorOptions(device=config.DEVICE),
        layout_batch_size=2, table_batch_size=2, ocr_batch_size=2,  # keep peak VRAM inside 4 GB
    )
    opts.picture_description_options.batch_size = 2
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _ocr_cells(cells, bbox, height):
    """OCR'd text lines whose centre lies in bbox (top-left coords). Text-layer text has none."""
    out = []
    for c in cells:
        if not c.from_ocr:
            continue
        b = c.rect.to_bounding_box()
        if b.coord_origin.name == "BOTTOMLEFT":
            b = b.to_top_left_origin(height)
        cx, cy = (b.l + b.r) / 2, (b.t + b.b) / 2
        if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
            out.append({"bbox": [round(b.l, 1), round(b.t, 1), round(b.r, 1), round(b.b, 1)], "text": c.text,
                        "conf": round(c.confidence, 3)})
    return out


def _items(conv, pdf: Path) -> dict[int, list[dict]]:
    from docling_core.types.doc import PictureItem, TableItem
    doc = conv.document
    heights = {p.page_no: p.size.height for p in conv.pages}
    cells = {p.page_no: (p.parsed_page.textline_cells if p.parsed_page else []) for p in conv.pages}
    out: dict[int, list[dict]] = {}
    pdfdoc = pymupdf.open(pdf)
    captions = set()
    for item, _ in doc.iterate_items():
        if isinstance(item, (TableItem, PictureItem)):
            captions |= {c.cref for c in item.captions}
    for item, _ in doc.iterate_items():
        if not getattr(item, "prov", None) or item.self_ref in captions:
            continue
        prov = item.prov[0]
        page = prov.page_no
        b = prov.bbox.to_top_left_origin(heights[page])
        bbox = [round(b.l, 1), round(b.t, 1), round(b.r, 1), round(b.b, 1)]
        kind = LABELS.get(item.label.value, "text")
        rec = {"page": page, "type": kind, "bbox": bbox, "source": "docling", "label": item.label.value}
        if item.label.value == "handwritten_text":
            rec["handwritten"] = True
        if isinstance(item, TableItem):
            rec["text"] = item.export_to_markdown(doc)
            rec["caption"] = item.caption_text(doc)
        elif isinstance(item, PictureItem):
            meta = item.meta
            desc = meta.description.text if meta and meta.description else ""
            cls = meta.classification.predictions[0].class_name if meta and meta.classification and meta.classification.predictions else None
            rec["caption"] = item.caption_text(doc)
            rec["picture_class"] = cls
            rec["text"] = " ".join(t for t in (rec["caption"], desc) if t)
        else:
            rec["text"] = item.text or getattr(item, "orig", "")
            if not rec["text"].strip() and kind == "formula":  # LaTeX off: use the PDF's own text in the box
                rec["text"] = " ".join(pdfdoc[page - 1].get_textbox(pymupdf.Rect(bbox)).split())
            if kind == "heading":
                rec["level"] = getattr(item, "level", 1)
        ocr = _ocr_cells(cells.get(page, []), bbox, heights[page])
        rec["confidence"] = round(sum(c["conf"] for c in ocr) / len(ocr), 3) if ocr else None
        if ocr and kind not in ("figure", "table"):
            rec["ocr_cells"] = ocr
        out.setdefault(page, []).append(rec)
    pdfdoc.close()
    return out


def docling_pages(pdf: Path, doc_dir: Path, n_pages: int, on_progress=None):
    """Parse in page batches; pages already checkpointed are skipped (resume after a crash)."""
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    todo = {p for p in range(1, n_pages + 1) if not (pages_dir / f"{p:04d}.json").exists()}
    conv = None
    while todo:
        start = end = min(todo)
        while end + 1 in todo and end - start + 1 < config.PAGE_BATCH:
            end += 1
        conv = conv or _converter()
        res = conv.convert(pdf, page_range=(start, end), raises_on_error=False)
        found = _items(res, pdf) if res.document else {}
        for p in {pg.page_no for pg in res.pages} & todo:  # unprocessed pages get no checkpoint, so the next run retries them
            recs = found.get(p, [])
            for n, r in enumerate(recs):
                r["id"] = f"p{p}_{n}"
            write_json(pages_dir / f"{p:04d}.json", recs)
        todo -= set(range(start, end + 1))
        if on_progress:
            on_progress(n_pages - len(todo), n_pages)


def load_pages(doc_dir: Path) -> list[dict]:
    items = []
    for f in sorted((doc_dir / "pages").glob("*.json")):
        items += json.loads(f.read_text(encoding="utf-8"))
    return items
