"""
Preprocess unstructured JSONL elements into clean, embedding-ready chunks.

Example:
  env/bin/python -m rag.preprocess \
    --input data/processed/attention-is-all-you-need/fast/elements_fast.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BANNED_TYPES = {"Footer", "Header", "PageBreak"}
ALLOWED_TYPES = {"Title", "NarrativeText", "ListItem", "Text", "Table", "Image"}


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise_text(text: str, element_type: str) -> bool:
    if not text:
        return True

    compact = re.sub(r"\s+", " ", text).strip()
    alnum_tokens = re.findall(r"[A-Za-z0-9]+", compact)

    # Pure page numbers or isolated numbers.
    if re.fullmatch(r"[\[(]?\d{1,4}[\])]?", compact):
        return True

    # Pattern like "g u A 2" / "3 2 0 2".
    if re.fullmatch(r"(?:[A-Za-z0-9]\s+){2,}[A-Za-z0-9]", compact):
        return True

    if alnum_tokens:
        one_char_ratio = sum(1 for t in alnum_tokens if len(t) == 1) / len(alnum_tokens)
        if len(alnum_tokens) >= 4 and one_char_ratio >= 0.7 and element_type in {"Text", "Title"}:
            return True

    # Minimal OCR/layout noise filtering for generic Text/Title blocks.
    if element_type == "Text" and len(compact) < 12:
        return True
    if element_type == "Title" and len(compact) < 4:
        return True

    return False


def clean_elements(raw_elements: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    cleaned: List[Dict[str, Any]] = []
    removed = 0

    for idx, el in enumerate(raw_elements):
        el_type = str(el.get("type", "")).strip()
        if el_type in BANNED_TYPES or el_type not in ALLOWED_TYPES:
            removed += 1
            continue

        md = el.get("metadata") or {}
        base: Dict[str, Any] = {
            "id": idx,
            "type": el_type,
            "metadata": {
                "filename": md.get("filename"),
                "page_number": md.get("page_number"),
            },
        }

        if el_type == "Image":
            base["image_base64"] = el.get("image_base64")
            base["image_mime_type"] = el.get("image_mime_type")
            cleaned.append(base)
            continue

        if el_type == "Table":
            base["table_html"] = (
                el.get("table_html")
                or md.get("text_as_html")
                or md.get("table_as_html")
                or md.get("html")
            )
            cleaned.append(base)
            continue

        text = normalize_text(str(el.get("text") or ""))
        if is_noise_text(text, el_type):
            removed += 1
            continue

        base["text"] = text
        cleaned.append(base)

    return cleaned, removed


def build_sections(cleaned_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current_title = "Document"
    current_text_parts: List[str] = []
    current_pages: List[int] = []
    current_has_table = False
    current_has_image = False
    current_tables_html: List[str] = []
    current_images_base64: List[str] = []
    section_id = 0

    def flush_section() -> None:
        nonlocal section_id, current_text_parts, current_pages, current_has_table, current_has_image
        nonlocal current_tables_html, current_images_base64
        if not current_text_parts and not current_has_table and not current_has_image:
            return
        section_text = normalize_text(" ".join(current_text_parts))
        if len(section_text) < 40 and not (current_has_table or current_has_image):
            current_text_parts = []
            current_pages = []
            current_tables_html = []
            current_images_base64 = []
            return
        section_id += 1
        sections.append(
            {
                "section_id": section_id,
                "section_title": current_title,
                "text": section_text,
                "page_start": min(current_pages) if current_pages else None,
                "page_end": max(current_pages) if current_pages else None,
                "has_table": current_has_table,
                "has_image": current_has_image,
                "tables_html": current_tables_html,
                "images_base64": current_images_base64,
            }
        )
        current_text_parts = []
        current_pages = []
        current_has_table = False
        current_has_image = False
        current_tables_html = []
        current_images_base64 = []

    for el in cleaned_elements:
        el_type = el["type"]
        text = el.get("text", "")
        page_number = el["metadata"].get("page_number")
        if isinstance(page_number, int):
            current_pages.append(page_number)

        if el_type == "Title":
            # Keep numbered sub-headings inside the current section.
            if re.match(r"^\d+\.\s+", text):
                current_text_parts.append(text)
                continue
            flush_section()
            current_title = text
            continue

        if el_type == "Table":
            current_has_table = True
            table_html = el.get("table_html")
            if table_html:
                current_tables_html.append(table_html)
            continue

        if el_type == "Image":
            current_has_image = True
            image_b64 = el.get("image_base64")
            if image_b64:
                current_images_base64.append(image_b64)
            continue

        # Grouping rule: each Title with following paragraph/list content.
        if el_type in {"NarrativeText", "ListItem"}:
            current_text_parts.append(text)

    flush_section()
    return sections


def chunk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(words):
        part = words[i : i + chunk_size]
        if not part:
            break
        chunks.append(" ".join(part))
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def build_chunks(
    sections: List[Dict[str, Any]],
    filename: str,
    chunk_size: int,
    overlap: int,
    min_chars: int,
    split_sections: bool = False,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for sec in sections:
        if split_sections:
            section_chunks = chunk_words(sec["text"], chunk_size=chunk_size, overlap=overlap)
        else:
            section_chunks = [sec["text"]]
        for idx, chunk in enumerate(section_chunks):
            chunk = normalize_text(chunk)
            if len(chunk) < min_chars:
                continue
            chunks.append(
                {
                    "chunk_id": f"{filename}:s{sec['section_id']}:c{idx}",
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "section_id": sec["section_id"],
                        "section_title": sec["section_title"],
                        "page_start": sec["page_start"],
                        "page_end": sec["page_end"],
                        "has_table": sec.get("has_table", False),
                        "has_image": sec.get("has_image", False),
                        "tables_html": sec.get("tables_html", []),
                        "images_base64": sec.get("images_base64", []),
                        "chunk_index": idx,
                    },
                }
            )
    return chunks


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def derive_output_paths(input_path: Path) -> Tuple[Path, Path, Path]:
    base = input_path.stem.replace("elements_", "")
    parent = input_path.parent
    return (
        parent / f"clean_elements_{base}.jsonl",
        parent / f"clean_sections_{base}.jsonl",
        parent / f"clean_chunks_{base}.jsonl",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean unstructured JSONL and build section/chunk outputs for RAG.")
    parser.add_argument("--input", required=True, help="Path to elements_*.jsonl produced by unstructured.")
    parser.add_argument("--chunk-size", type=int, default=220, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=40, help="Chunk overlap in words.")
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum chars per chunk to keep.")
    parser.add_argument(
        "--split-sections",
        action="store_true",
        help="Split long sections into multiple chunks (disabled by default).",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    raw = read_jsonl(input_path)
    cleaned, removed = clean_elements(raw)
    sections = build_sections(cleaned)
    filename = cleaned[0]["metadata"].get("filename") if cleaned else input_path.name
    chunks = build_chunks(
        sections=sections,
        filename=str(filename),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chars=args.min_chars,
        split_sections=args.split_sections,
    )

    clean_el_path, clean_sec_path, clean_chunk_path = derive_output_paths(input_path)
    write_jsonl(clean_el_path, cleaned)
    write_jsonl(clean_sec_path, sections)
    write_jsonl(clean_chunk_path, chunks)

    print(f"Input elements: {len(raw)}")
    print(f"Removed noisy elements: {removed}")
    print(f"Clean elements kept: {len(cleaned)}")
    print(f"Sections: {len(sections)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Clean elements file: {clean_el_path}")
    print(f"Clean sections file: {clean_sec_path}")
    print(f"Clean chunks file: {clean_chunk_path}")


if __name__ == "__main__":
    main()
