"""API-mode parsing: Mistral OCR per PDF (pages still to do only), markdown -> the same item records and
data/<doc_id>/pages/NNNN.json checkpoints that parse.docling_pages writes. Figures are described by Gemini."""
import base64
import json
import re

import pymupdf

from rag import config
from rag.store import write_json

MAX_PAGES, MAX_BYTES = 1000, 50 * 2**20  # Mistral's per-request limits
LABEL = {"heading": "section_header", "text": "text", "list": "list_item", "table": "table", "figure": "picture", "formula": "formula", "code": "code"}
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


def _describe(data: bytes, mime: str) -> str:
    from rag import cloud
    return cloud.describe_image(data, mime)


def blocks(md: str) -> list[dict]:
    """Markdown -> ordered blocks: heading (level), text, list, table, formula ($$), code (```), figure (image id)."""
    out, para, table, fence, math = [], [], [], None, None
    for ln in md.splitlines() + [""]:
        s = ln.strip()
        if fence is not None:
            if s.startswith("```"):
                out.append({"type": "code", "text": "\n".join(fence)}); fence = None
            else:
                fence.append(ln)
            continue
        if math is not None:
            end = s.find("$$")
            math.append(s if end < 0 else s[:end])
            if end >= 0:
                out.append({"type": "formula", "text": " ".join(" ".join(math).split())}); math = None
            continue
        if s.startswith("|"):
            table.append(s); continue
        if table:
            out.append({"type": "table", "text": "\n".join(table)}); table = []
        if not s or HEADING.match(s) or s.startswith(("```", "$$")) or IMAGE.search(s) or (LIST.match(s) and not para):
            if para:
                out.append({"type": "list" if LIST.match(para[0]) else "text", "text": " ".join(para)}); para = []
        if not s:
            continue
        if s.startswith("```"):
            fence = []
        elif s.startswith("$$"):
            inner = s[2:]
            if inner.rstrip().endswith("$$") and inner.strip() != "":
                out.append({"type": "formula", "text": inner.rstrip()[:-2].strip()})
            else:
                math = [inner]
        elif (m := HEADING.match(s)):
            out.append({"type": "heading", "text": m.group(2), "level": len(m.group(1))})
        elif IMAGE.search(s):
            for img in IMAGE.findall(s):
                out.append({"type": "figure", "text": "", "image_id": img})
            if (rest := IMAGE.sub("", s).strip()):
                out.append({"type": "text", "text": rest})
        else:
            para.append(s)
    return out


def items(page: dict, pno: int, doc_dir) -> list[dict]:
    """One OCR page -> item records. Mistral gives no text boxes, so boxes are reading-order slots (full width: no margin notes)."""
    dims = page.get("dimensions") or {}
    scale = 72 / (dims.get("dpi") or 72)
    w, h = (dims.get("width") or 612) * scale, (dims.get("height") or 792) * scale
    images = {im["id"]: im for im in page.get("images") or []}
    recs = blocks(page.get("markdown") or "")
    step = h / max(len(recs), 1)
    for n, r in enumerate(recs):
        r.update(id=f"p{pno}_{n}", page=pno, label=LABEL[r["type"]], source="mistral", confidence=None,
                 bbox=[round(0.1 * w, 1), round(n * step, 1), round(0.9 * w, 1), round((n + 0.9) * step, 1)])
        if r["type"] == "figure":
            im = images.get(r.pop("image_id"))
            r.update(caption="", picture_class=None)
            if im and im.get("top_left_x") is not None:
                r["image_bbox"] = [round(im[k] * scale, 1) for k in ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")]
            if im and (b64 := im.get("image_base64")):
                head, _, data = b64.partition(",")
                mime = head[5:].split(";")[0] if head.startswith("data:") else "image/jpeg"
                raw = base64.b64decode(data)
                path = doc_dir / "images" / f"p{pno}_{im['id']}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                r["image"] = str(path.relative_to(doc_dir))
                r["text"] = r["description"] = _describe(raw, mime)  # the stitcher prepends the caption to text; the audit checks description
    return recs


def _chunks(pdf, todo: list[int]):
    """(page numbers, pdf bytes) per request: runs of <= MAX_PAGES pages, halved until under MAX_BYTES."""
    with pymupdf.open(pdf) as src:
        def part(pages):
            sub = pymupdf.open()
            for p in pages:
                sub.insert_pdf(src, from_page=p - 1, to_page=p - 1)
            data = sub.tobytes(garbage=3, deflate=True)
            sub.close()
            if len(data) > MAX_BYTES and len(pages) > 1:
                yield from part(pages[:len(pages) // 2]); yield from part(pages[len(pages) // 2:])
            else:
                yield pages, data
        for i in range(0, len(todo), MAX_PAGES):
            yield from part(todo[i:i + MAX_PAGES])


def pages(pdf, doc_dir, n_pages: int, on_progress=None):
    """Pages without a checkpoint are sent; a page missing from the response gets no checkpoint and is retried next run."""
    from rag import cloud
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in range(1, n_pages + 1) if not (pages_dir / f"{p:04d}.json").exists()]
    done = n_pages - len(todo)
    for nums, data in _chunks(pdf, todo):
        res = cloud.ocr(data)
        for pg in res.get("pages") or []:
            if not 0 <= pg["index"] < len(nums):
                raise ValueError(f"Mistral returned page index {pg['index']} for a {len(nums)}-page request")
            pno = nums[pg["index"]]
            write_json(pages_dir / f"{pno:04d}.json", items(pg, pno, doc_dir))
            done += 1
        if on_progress:
            on_progress(done, n_pages)


if __name__ == "__main__":  # self-check: fixed markdown fixture -> items -> stitch -> chunks, with a fake Gemini and no network
    import tempfile
    from pathlib import Path
    from unittest import mock

    from rag import chunk, stitch

    MD = ("# 2 Methods\n\nThe model uses\nsix layers.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nTable 1: Sizes\n\n$$\nE = mc^2\n$$\n\n"
          "![img-0.jpeg](img-0.jpeg)\n\nFigure 1: Loss curve\n\n- first\n- second\n\n```\nx = 1\n```\n\n$$a+b$$ \n\n## 2.1 Setup\n\nText.")
    _describe = lambda data, mime: f"A line chart ({mime}, {len(data)} bytes) with Epochs on the x-axis."  # noqa: F811
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
    page = {"index": 0, "markdown": MD, "dimensions": {"dpi": 200, "width": 1700, "height": 2200},
            "images": [{"id": "img-0.jpeg", "top_left_x": 200, "top_left_y": 1000, "bottom_right_x": 1500, "bottom_right_y": 1600,
                        "image_base64": "data:image/png;base64," + png}]}
    with tempfile.TemporaryDirectory() as tmp, mock.patch("httpx.post", side_effect=AssertionError("network")):
        recs = items(page, 1, Path(tmp))
        kinds = [r["type"] for r in recs]
        assert kinds == ["heading", "text", "table", "text", "formula", "figure", "text", "list", "code", "formula", "heading", "text"], kinds
        assert recs[0]["level"] == 1 and recs[-2]["level"] == 2 and recs[1]["text"] == "The model uses six layers."
        assert recs[4]["text"] == "E = mc^2" and recs[9]["text"] == "a+b" and recs[8]["text"] == "x = 1" and recs[7]["text"] == "- first - second"
        fig = recs[5]
        assert fig["text"].startswith("A line chart (image/png, 12 bytes)") and fig["image_bbox"] == [72.0, 360.0, 540.0, 576.0], fig
        assert (Path(tmp) / fig["image"]).read_bytes().startswith(b"\x89PNG") and fig["bbox"][2] - fig["bbox"][0] > 0.3 * 612
        assert all(r["bbox"][1] < recs[n + 1]["bbox"][1] for n, r in enumerate(recs[:-1]))  # reading order kept in the boxes
        doc = stitch.build("x", "T", 1, recs, {1: 612}, [])
        assert [s["title"] for s in doc["sections"]] == ["2 Methods", "2.1 Setup"]
        tab = next(i for i in doc["items"] if i["type"] == "table")
        assert tab["caption"] == "Table 1: Sizes" and next(i for i in doc["items"] if i["type"] == "figure")["caption"] == "Figure 1: Loss curve"
        assert not any(i["type"] == "margin_note" for i in doc["items"])
        assert {c["type"] for c in chunk.build(doc)} == {"text", "table", "formula", "figure", "code"}
        # request splitting: 3-page PDF, pages 1 and 3 still to do, 1-page limit -> two requests mapped back to the right pages
        pdf = Path(tmp) / "t.pdf"
        with pymupdf.open() as d:
            for n in range(3):
                d.new_page().insert_text((72, 72), f"page {n + 1}")
            d.save(pdf)
        MAX_PAGES = 1
        parts = list(_chunks(pdf, [1, 3]))
        assert [p[0] for p in parts] == [[1], [3]] and all(pymupdf.open(stream=p[1]).page_count == 1 for p in parts)
        MAX_PAGES = 1000
        assert [p[0] for p in _chunks(pdf, [1, 3])] == [[1, 3]]
        fake = lambda data: {"pages": [{"index": i, "markdown": f"page {i}"} for i in range(pymupdf.open(stream=data).page_count)]}
        (Path(tmp) / "pages").mkdir()
        write_json(Path(tmp) / "pages" / "0002.json", [])  # page 2 already checkpointed
        with mock.patch("rag.cloud.ocr", side_effect=fake) as ocr_:
            pages(pdf, Path(tmp), 3, on_progress=lambda d, t: None)
            pages(pdf, Path(tmp), 3)  # all checkpointed: no second call
        assert ocr_.call_count == 1 and sorted(p.name for p in (Path(tmp) / "pages").glob("*.json")) == ["0001.json", "0002.json", "0003.json"]
        assert json.loads((Path(tmp) / "pages" / "0003.json").read_text(encoding="utf-8"))[0]["text"] == "page 1"  # 2nd page of the request = page 3
    print("ok")
