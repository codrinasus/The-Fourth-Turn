"""Docling-backed PDF parsing.

Docling runs as a self-hosted HTTP service. `/ingest` sends the committed PDF to
Docling, receives structured JSON/Markdown, caches the artifacts under
`data/extracted/docling/`, and returns page-indexed text to the normal chunking
and retrieval pipeline.

This keeps parsing inside the app workflow while avoiding a commercial
"chat-with-PDF" shortcut: Docling extracts structure; our code still chunks,
embeds, retrieves, reasons, cites, and writes the graded answer files.
"""

from __future__ import annotations

import io
import base64
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import httpx

from ..config import get_settings


_SKIP_LABELS = {
    "page_header",
    "page_footer",
    "footnote",
}


def extract_pages(pdf_path: Path) -> list[str]:
    settings = get_settings()
    base_dir, json_path, markdown_path = _artifact_paths(pdf_path)

    if settings.docling_use_cache and json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        result = _convert_with_docling(pdf_path, base_dir)
        data = _document_json(result)
        _write_artifacts(base_dir, json_path, markdown_path, data, _document_markdown(result))

    pages = _pages_from_docling_json(data)
    if not any(page.strip() for page in pages):
        raise ValueError("Docling produced no page-grounded text")
    return pages


def _artifact_paths(pdf_path: Path) -> tuple[Path, Path, Path]:
    base = Path(get_settings().docling_output_dir) / pdf_path.stem
    return base, base / f"{pdf_path.stem}.json", base / f"{pdf_path.stem}.md"


def _convert_with_docling(pdf_path: Path, output_dir: Path) -> dict[str, Any]:
    settings = get_settings()
    url = settings.docling_base_url.rstrip("/") + "/v1/convert/file"

    form_fields = [
        ("from_formats", (None, "pdf")),
        ("to_formats", (None, "json")),
        ("to_formats", (None, "md")),
        ("do_ocr", (None, str(settings.docling_do_ocr).lower())),
        ("table_mode", (None, settings.docling_table_mode)),
        ("image_export_mode", (None, _docling_request_image_mode(settings.docling_image_export_mode))),
        ("include_images", (None, "true")),
    ]

    with httpx.Client(timeout=settings.docling_timeout) as client:
        with pdf_path.open("rb") as fh:
            response = client.post(
                url,
                files=[("files", (pdf_path.name, fh, "application/pdf")), *form_fields],
            )

        # Some form parsers prefer one value rather than repeated `to_formats`.
        # Keep this fallback local to Docling so `/ingest` still has one parser call.
        if response.status_code in {400, 422}:
            fallback_fields = [(key, value) for key, value in form_fields if key != "to_formats"]
            fallback_fields.append(("to_formats", (None, "json")))
            with pdf_path.open("rb") as fh:
                response = client.post(
                    url,
                    files=[("files", (pdf_path.name, fh, "application/pdf")), *fallback_fields],
                )

    response.raise_for_status()
    if _is_zip_response(response):
        return _read_zip_response(response.content, output_dir, pdf_path.stem)

    payload = response.json()
    if str(payload.get("status", "")).lower() in {"failure", "failed"}:
        raise ValueError(f"Docling conversion failed: {payload.get('errors')}")
    return payload


def _is_zip_response(response: httpx.Response) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    return "application/zip" in content_type or response.content.startswith(b"PK\x03\x04")


def _docling_request_image_mode(configured_mode: str) -> str:
    # Docling Serve's JSON response can expose referenced filenames without
    # returning the files. For our referenced mode, request embedded bytes and
    # write them to disk ourselves.
    return "embedded" if configured_mode == "referenced" else configured_mode


def _read_zip_response(content: bytes, output_dir: Path, stem: str) -> dict[str, Any]:
    """Save Docling's referenced-image zip and return its JSON/Markdown content."""

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (output_dir / member.filename).resolve()
            if not target.is_relative_to(output_dir.resolve()):
                raise ValueError(f"refusing unsafe Docling zip member: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))

    json_path = _find_zip_artifact(output_dir, stem, ".json")
    markdown_path = _find_zip_artifact(output_dir, stem, ".md")
    if json_path is None:
        raise ValueError(f"Docling zip did not contain a JSON artifact under {output_dir}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path else ""
    return {"document": {"json_content": data, "md_content": markdown}}


def _find_zip_artifact(output_dir: Path, stem: str, suffix: str) -> Path | None:
    exact = list(output_dir.rglob(f"{stem}{suffix}"))
    if exact:
        return exact[0]
    candidates = list(output_dir.rglob(f"*{suffix}"))
    return candidates[0] if candidates else None


def _document_json(result: dict[str, Any]) -> dict[str, Any]:
    content = (result.get("document") or {}).get("json_content")
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content
    raise ValueError("Docling response did not include document.json_content")


def _document_markdown(result: dict[str, Any]) -> str:
    content = (result.get("document") or {}).get("md_content")
    return content if isinstance(content, str) else ""


def _write_artifacts(
    output_dir: Path,
    json_path: Path,
    markdown_path: Path,
    data: dict[str, Any],
    markdown: str,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if get_settings().docling_image_export_mode == "referenced":
        markdown = _materialize_embedded_images(output_dir, data, markdown)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown:
        markdown_path.write_text(markdown, encoding="utf-8")


def _materialize_embedded_images(output_dir: Path, data: dict[str, Any], markdown: str) -> str:
    images_dir = output_dir / "images"
    replacements: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            image = value.get("image")
            if isinstance(image, dict):
                uri = image.get("uri")
                if isinstance(uri, str) and uri.startswith("data:image/"):
                    relative = replacements.get(uri)
                    if relative is None:
                        relative = _save_data_uri(images_dir, uri)
                        replacements[uri] = relative
                    image["uri"] = relative
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    for old_uri, relative in replacements.items():
        markdown = markdown.replace(old_uri, relative)
    return markdown


def _save_data_uri(images_dir: Path, uri: str) -> str:
    header, encoded = uri.split(",", 1)
    mime = header.removeprefix("data:").split(";", 1)[0]
    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime, ".img")
    payload = base64.b64decode(encoded)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    filename = f"image_{digest}{suffix}"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / filename).write_bytes(payload)
    return f"images/{filename}"


def _pages_from_docling_json(data: dict[str, Any]) -> list[str]:
    pages: dict[int, list[str]] = {}
    for item in _iter_content_items(data):
        page_no = _page_number(item)
        text = _item_text(item)
        if page_no is None or not text:
            continue
        pages.setdefault(page_no, []).append(text)

    if not pages:
        return []
    return ["\n\n".join(pages.get(page, [])).strip() for page in range(1, max(pages) + 1)]


def _iter_content_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("texts", "tables", "pictures"):
        value = data.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def _page_number(item: dict[str, Any]) -> int | None:
    prov = item.get("prov")
    if not isinstance(prov, list) or not prov:
        return None
    raw = prov[0].get("page_no")
    try:
        page_no = int(raw)
    except (TypeError, ValueError):
        return None
    return page_no if page_no > 0 else page_no + 1


def _item_text(item: dict[str, Any]) -> str:
    label = str(item.get("label", "")).lower()
    if label in _SKIP_LABELS:
        return ""

    text = str(item.get("text") or "").strip()
    if text:
        return _normalize(text)

    table = _table_to_markdown(item.get("data"))
    if table:
        return table

    captions = item.get("captions")
    if isinstance(captions, list):
        caption_text = " ".join(
            str(caption.get("text", "")) for caption in captions if isinstance(caption, dict)
        )
        return _normalize(caption_text)

    return ""


def _table_to_markdown(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    cells = data.get("table_cells")
    if not isinstance(cells, list):
        return ""

    max_row = 0
    max_col = 0
    parsed_cells: list[tuple[int, int, str]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        text = _normalize(str(cell.get("text") or ""))
        if not text:
            continue
        row = int(cell.get("start_row_offset_idx") or 0)
        col = int(cell.get("start_col_offset_idx") or 0)
        max_row = max(max_row, row)
        max_col = max(max_col, col)
        parsed_cells.append((row, col, text))

    if not parsed_cells:
        return ""

    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for row, col, text in parsed_cells:
        grid[row][col] = text

    rows = [" | ".join(row).strip() for row in grid if any(cell.strip() for cell in row)]
    return "\n".join(rows)


def _normalize(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
