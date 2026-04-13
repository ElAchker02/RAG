"""
Process PDFs with Unstructured for this Django project.

Default project folders:
- Input PDFs:   <BASE_DIR>/data/pdfs
- Output files: <BASE_DIR>/data/processed

Usage examples:
  python -m rag.unstructured
  python -m rag.unstructured --strategy smart
  python -m rag.unstructured --strategy all --infer-tables
  python -m rag.unstructured --pdf "rapport.pdf" --strategy hi_res
  python -m rag.unstructured --input-dir "./my_pdfs" --out-dir "./outputs" --recursive

Notes:
- Some options (tables/images/OCR) may require extra deps:
    pip install "unstructured[all-docs]"
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from django.conf import settings as django_settings
except Exception:  # pragma: no cover
    django_settings = None

try:
    from rag.preprocess import build_chunks, build_sections, clean_elements, derive_output_paths, write_jsonl
except Exception:  # pragma: no cover
    try:
        from preprocess import build_chunks, build_sections, clean_elements, derive_output_paths, write_jsonl
    except Exception:  # pragma: no cover
        build_chunks = None
        build_sections = None
        clean_elements = None
        derive_output_paths = None
        write_jsonl = None


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_partition_pdf():
    import sys

    script_dir = Path(__file__).resolve().parent
    original_sys_path = list(sys.path)

    try:
        # Avoid shadowing the third-party `unstructured` package when this file
        # is executed directly as `python unstructured.py` from rag/.
        sys.path = [
            p for p in sys.path if Path(p or ".").resolve() != script_dir
        ]

        maybe_shadow = sys.modules.get("unstructured")
        if maybe_shadow is not None and Path(getattr(maybe_shadow, "__file__", "")).resolve() == Path(__file__).resolve():
            del sys.modules["unstructured"]

        from unstructured.partition.pdf import partition_pdf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Package 'unstructured' is not installed in this virtualenv. "
            'Install it with: pip install "unstructured[all-docs]"'
        ) from exc
    finally:
        sys.path = original_sys_path

    return partition_pdf


def element_to_record(el) -> Dict[str, Any]:
    t = type(el).__name__
    text = getattr(el, "text", None)
    md = {}
    meta = getattr(el, "metadata", None)
    if meta is not None and hasattr(meta, "to_dict"):
        md = meta.to_dict()
    # image path key varies by version/options
    image_path = md.get("image_path") or md.get("image_file_path") or md.get("image") or md.get("filename")
    record: Dict[str, Any] = {"type": t, "metadata": md, "image_path_guess": image_path}

    if t == "Table":
        table_html = md.get("text_as_html") or md.get("table_as_html") or md.get("html")
        record["table_html"] = table_html
        record["table_structure_available"] = bool(table_html)
        return record

    # Inline base64 for images if available.
    if t == "Image" and image_path:
        try:
            import base64
            from pathlib import Path

            p = Path(image_path)
            if p.exists():
                record["image_base64"] = base64.b64encode(p.read_bytes()).decode("ascii")
                record["image_mime_type"] = "image/jpeg"
        except Exception:
            pass
        return record

    record["text"] = text
    return record


def run_strategy(
    pdf_path: Path,
    strategy: str,
    out_dir: Path,
    infer_tables: bool,
    extract_images: bool,
    clean_outputs: bool,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    split_sections: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    """
    Returns:
      - records: list of element dicts
      - counts: dict(type -> count)
      - extracted_images: list of image paths found in metadata (best-effort)
    """
    safe_mkdir(out_dir)

    img_dir = out_dir / "images"
    if extract_images:
        safe_mkdir(img_dir)

    partition_pdf = load_partition_pdf()

    # Many Unstructured versions support these kwargs, but some may not.
    # We’ll pass them conditionally and handle TypeError gracefully.
    kwargs = {
        "filename": str(pdf_path),
        "strategy": strategy,
    }

    if infer_tables or strategy in {"hi_res", "auto"}:
        kwargs["infer_table_structure"] = True

    if extract_images:
        # For modern unstructured (>=0.15) image extraction uses these args
        kwargs["extract_images_in_pdf"] = True
        kwargs["extract_image_block_output_dir"] = str(img_dir)
        # Be explicit about which block types to save to cover most versions
        kwargs["extract_image_block_types"] = ["Image"]

    try:
        elements = partition_pdf(**kwargs)
    except TypeError as e:
        # Fallback: call with only filename + strategy
        print(f"[WARN] Some options not supported in your Unstructured version for strategy={strategy}: {e}")
        elements = partition_pdf(filename=str(pdf_path), strategy=strategy)
    except Exception as e:
        msg = str(e).lower()
        missing_table_model = any(
            token in msg
            for token in [
                "table-transformer-structure-recognition",
                "can't load image processor",
                "preprocessor_config.json",
            ]
        )
        missing_ocr_deps = any(
            token in msg
            for token in [
                "tesseract",
                "pdfinfo",
                "poppler",
                "pdfinfonotinstallederror",
                "unable to get page count",
            ]
        )
        if missing_table_model and kwargs.get("infer_table_structure"):
            print(
                "[WARN] Table structure model unavailable; "
                f"retrying '{pdf_path.name}' with infer_table_structure=False."
            )
            kwargs_no_tables = dict(kwargs)
            kwargs_no_tables.pop("infer_table_structure", None)
            elements = partition_pdf(**kwargs_no_tables)
        elif missing_ocr_deps and strategy != "fast":
            print(
                "[WARN] OCR dependency missing (tesseract/poppler); "
                f"retrying '{pdf_path.name}' with strategy='fast' instead of '{strategy}'."
            )
            elements = partition_pdf(filename=str(pdf_path), strategy="fast")
        else:
            raise

    records = [element_to_record(el) for el in elements]

    counts: Dict[str, int] = {}
    extracted_images: List[str] = []
    for r in records:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
        # best-effort image path detection
        md = r.get("metadata") or {}
        ip = md.get("image_path") or md.get("image_file_path")
        if ip:
            extracted_images.append(ip)

    # Fallback: if unstructured did not extract images, try raw PDF image extraction.
    fallback_images_count = 0
    if extract_images:
        saved_files = list(img_dir.glob("*")) if img_dir.exists() else []
        if not saved_files:
            fallback_images_count = extract_images_with_pdfimages(pdf_path, img_dir)
            if fallback_images_count:
                print(
                    f"[INFO] Unstructured returned no image files; "
                    f"extracted {fallback_images_count} embedded image(s) via pdfimages."
                )

    # Save outputs
    json_path = out_dir / f"elements_{strategy}.jsonl"
    with open(json_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    clean_summary: Dict[str, Any] = {"enabled": clean_outputs, "generated": False}
    if clean_outputs:
        clean_summary = generate_clean_outputs(
            elements_path=json_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
            split_sections=split_sections,
        )

    summary_path = out_dir / f"summary_{strategy}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf": str(pdf_path),
                "strategy": strategy,
                "infer_tables": infer_tables,
                "extract_images": extract_images,
                "out_dir": str(out_dir),
                "counts": counts,
                "extracted_images_count": len(extracted_images),
                "images_folder": str(img_dir) if extract_images else None,
                "pdfimages_fallback_images_count": fallback_images_count,
                "clean_outputs": clean_summary,
                "notes": [
                    "elements_*.jsonl contains element text + metadata",
                    "images are saved only if extract_images_in_pdf is supported and images exist",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return records, counts, extracted_images


def generate_clean_outputs(
    elements_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    split_sections: bool,
) -> Dict[str, Any]:
    if not all([build_chunks, build_sections, clean_elements, derive_output_paths, write_jsonl]):
        print("[WARN] preprocess module not available: clean_* files were not generated.")
        return {"enabled": True, "generated": False, "reason": "preprocess module unavailable"}

    raw_rows: List[Dict[str, Any]] = []
    with open(elements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))

    cleaned, removed = clean_elements(raw_rows)
    sections = build_sections(cleaned)
    filename = cleaned[0]["metadata"].get("filename") if cleaned else elements_path.name
    chunks = build_chunks(
        sections=sections,
        filename=str(filename),
        chunk_size=chunk_size,
        overlap=chunk_overlap,
        min_chars=min_chunk_chars,
        split_sections=split_sections,
    )

    clean_el_path, clean_sec_path, clean_chunk_path = derive_output_paths(elements_path)
    write_jsonl(clean_el_path, cleaned)
    write_jsonl(clean_sec_path, sections)
    write_jsonl(clean_chunk_path, chunks)

    return {
        "enabled": True,
        "generated": True,
        "removed_noisy_elements": removed,
        "clean_elements_count": len(cleaned),
        "sections_count": len(sections),
        "chunks_count": len(chunks),
        "files": {
            "clean_elements": str(clean_el_path),
            "clean_sections": str(clean_sec_path),
            "clean_chunks": str(clean_chunk_path),
        },
    }


def extract_images_with_pdfimages(pdf_path: Path, img_dir: Path) -> int:
    if shutil.which("pdfimages") is None:
        return 0

    output_prefix = img_dir / "raw"
    cmd = ["pdfimages", "-all", str(pdf_path), str(output_prefix)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return 0

    return len(list(img_dir.glob("raw-*")))


def print_run_summary(
    records: List[Dict[str, Any]],
    counts: Dict[str, int],
    imgs: List[str],
    out_dir: Path,
    print_first: int,
    extract_images: bool,
) -> None:
    print(f"Output dir: {out_dir}")
    print("Element counts:")
    for k in sorted(counts.keys()):
        print(f"  - {k}: {counts[k]}")

    if extract_images:
        img_folder = out_dir / "images"
        saved_files = list(img_folder.glob("*")) if img_folder.exists() else []
        print(f"Images (metadata referenced): {len(imgs)}")
        print(f"Images (files in folder): {len(saved_files)}")
        if saved_files:
            print(f"  Sample saved image: {saved_files[0].name}")

    n = min(print_first, len(records))
    print(f"\nFirst {n} elements preview:")
    for i in range(n):
        r = records[i]
        t = r["type"]
        txt = (r["text"] or "").replace("\n", " ").strip()
        if len(txt) > 120:
            txt = txt[:120] + "..."
        page = (r.get("metadata") or {}).get("page_number")
        print(f"[{i + 1:02d}] {t} (page={page}) -> {txt}")


def score_fast_extraction(records: List[Dict[str, Any]], counts: Dict[str, int]) -> Dict[str, float]:
    text_parts = [(r.get("text") or "").strip() for r in records]
    non_empty = [t for t in text_parts if t]
    full_text = " ".join(non_empty)
    total_chars = len(full_text)
    alnum_chars = sum(1 for ch in full_text if ch.isalnum())
    alnum_ratio = alnum_chars / total_chars if total_chars else 0.0
    long_lines = sum(1 for t in non_empty if len(t) >= 40)
    long_line_ratio = long_lines / len(non_empty) if non_empty else 0.0
    text_blocks = sum(counts.get(k, 0) for k in ["NarrativeText", "Text", "Title", "ListItem"])

    return {
        "total_chars": float(total_chars),
        "alnum_ratio": alnum_ratio,
        "long_line_ratio": long_line_ratio,
        "text_blocks": float(text_blocks),
    }


def should_use_fast(scores: Dict[str, float]) -> bool:
    enough_text = scores["total_chars"] >= 1200 or (
        scores["total_chars"] >= 500 and scores["text_blocks"] >= 20
    )
    readable_text = scores["alnum_ratio"] >= 0.45 and scores["long_line_ratio"] >= 0.15
    return enough_text and readable_text


def get_project_base_dir() -> Path:
    if django_settings is not None and getattr(django_settings, "configured", False):
        return Path(django_settings.BASE_DIR)
    return Path(__file__).resolve().parent.parent


def process_pdf(
    pdf_path: Path,
    out_root: Path,
    strategy: str,
    ocr_strategy: str,
    infer_tables: bool,
    extract_images: bool,
    clean_outputs: bool,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    split_sections: bool,
    print_first: int,
) -> None:
    primary_strategy = "hi_res"

    if strategy == "all":
        strategies = ["fast", "hi_res", "ocr_only", "auto"]
    elif strategy == "smart":
        strategies = ["smart"]
    else:
        strategies = [strategy]

    for strat in strategies:
        print("\n" + "=" * 80)
        print(f"PDF: {pdf_path.name} | strategy: {strat}")
        if strat == "smart":
            primary_out_dir = out_root / pdf_path.stem / primary_strategy
            primary_records, primary_counts, primary_imgs = run_strategy(
                pdf_path=pdf_path,
                strategy=primary_strategy,
                out_dir=primary_out_dir,
                infer_tables=infer_tables,
                extract_images=extract_images,
                clean_outputs=clean_outputs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_chars=min_chunk_chars,
                split_sections=split_sections,
            )
            scores = score_fast_extraction(primary_records, primary_counts)
            route_primary = should_use_fast(scores)
            print(
                "[ROUTE] fast_score: "
                f"chars={int(scores['total_chars'])}, "
                f"alnum_ratio={scores['alnum_ratio']:.2f}, "
                f"long_line_ratio={scores['long_line_ratio']:.2f}, "
                f"text_blocks={int(scores['text_blocks'])}"
            )
            if route_primary:
                print(f"[ROUTE] Selectable text detected -> keep strategy '{primary_strategy}'.")
                print_run_summary(
                    records=primary_records,
                    counts=primary_counts,
                    imgs=primary_imgs,
                    out_dir=primary_out_dir,
                    print_first=print_first,
                    extract_images=extract_images,
                )
            else:
                print(f"[ROUTE] Likely scanned PDF -> run strategy '{ocr_strategy}'.")
                ocr_out_dir = out_root / pdf_path.stem / ocr_strategy
                records, counts, imgs = run_strategy(
                    pdf_path=pdf_path,
                    strategy=ocr_strategy,
                    out_dir=ocr_out_dir,
                    infer_tables=infer_tables,
                    extract_images=extract_images,
                    clean_outputs=clean_outputs,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_chars=min_chunk_chars,
                    split_sections=split_sections,
                )
                print_run_summary(
                    records=records,
                    counts=counts,
                    imgs=imgs,
                    out_dir=ocr_out_dir,
                    print_first=print_first,
                    extract_images=extract_images,
                )
        else:
            out_dir = out_root / pdf_path.stem / strat
            records, counts, imgs = run_strategy(
                pdf_path=pdf_path,
                strategy=strat,
                out_dir=out_dir,
                infer_tables=infer_tables,
                extract_images=extract_images,
                clean_outputs=clean_outputs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_chars=min_chunk_chars,
                split_sections=split_sections,
            )
            print_run_summary(
                records=records,
                counts=counts,
                imgs=imgs,
                out_dir=out_dir,
                print_first=print_first,
                extract_images=extract_images,
            )


def process_folder(
    input_dir: Path,
    out_root: Path,
    strategy: str,
    ocr_strategy: str,
    infer_tables: bool,
    extract_images: bool,
    clean_outputs: bool,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    split_sections: bool,
    print_first: int,
    recursive: bool,
) -> None:
    glob_pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(input_dir.glob(glob_pattern))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    print(f"Found {len(pdf_files)} PDF file(s) in {input_dir}")
    for pdf_file in pdf_files:
        process_pdf(
            pdf_path=pdf_file.resolve(),
            out_root=out_root,
            strategy=strategy,
            ocr_strategy=ocr_strategy,
            infer_tables=infer_tables,
            extract_images=extract_images,
            clean_outputs=clean_outputs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
            split_sections=split_sections,
            print_first=print_first,
        )


def resolve_pdf_path(pdf_arg: str, input_dir: Path) -> Path:
    candidate = Path(pdf_arg).expanduser()
    search_paths = []

    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        # 1) Relative to current working directory
        search_paths.append(Path.cwd() / candidate)
        # 2) Relative to configured input directory
        search_paths.append(input_dir / candidate)

    for path in search_paths:
        resolved = path.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved

    attempted = ", ".join(str(p.resolve()) for p in search_paths)
    raise FileNotFoundError(f"PDF not found. Checked: {attempted}")


def main():
    base_dir = get_project_base_dir()
    default_input_dir = base_dir / "data" / "pdfs"
    default_out_dir = base_dir / "data" / "processed"

    parser = argparse.ArgumentParser(
        description="Process project PDFs with Unstructured (single file or whole folder)."
    )
    parser.add_argument("--pdf", help="Single PDF path or file name inside --input-dir.")
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help=f"Folder that contains PDFs. Default: {default_input_dir}",
    )
    parser.add_argument(
        "--out-dir",
        default=str(default_out_dir),
        help=f"Output folder for parsed files. Default: {default_out_dir}",
    )
    parser.add_argument(
        "--strategy",
        default="hi_res",
        choices=["fast", "hi_res", "ocr_only", "auto", "all", "smart"],
        help="Strategy to run. 'smart' routes hi_res vs OCR automatically.",
    )
    parser.add_argument(
        "--ocr-strategy",
        default="hi_res",
        choices=["hi_res", "ocr_only"],
        help="OCR strategy used when --strategy smart detects a scanned PDF.",
    )
    parser.add_argument(
        "--infer-tables",
        action="store_true",
        help="Try to infer table structure (mostly useful for hi_res).",
    )
    parser.add_argument("--no-images", action="store_true", help="Disable image extraction.")
    parser.add_argument("--no-clean", action="store_true", help="Disable clean/section/chunk generation.")
    parser.add_argument("--chunk-size", type=int, default=220, help="Chunk size in words for clean chunks.")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap in words for clean chunks.")
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum chars for a clean chunk.")
    parser.add_argument(
        "--split-sections",
        action="store_true",
        help="Split long sections into multiple chunks (disabled by default).",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    parser.add_argument("--print-first", type=int, default=15, help="Print first N elements (type + short text).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    safe_mkdir(input_dir)
    safe_mkdir(out_dir)

    extract_images = not args.no_images
    clean_outputs = not args.no_clean

    if args.pdf:
        pdf_path = resolve_pdf_path(args.pdf, input_dir)
        process_pdf(
            pdf_path=pdf_path,
            out_root=out_dir,
            strategy=args.strategy,
            ocr_strategy=args.ocr_strategy,
            infer_tables=args.infer_tables,
            extract_images=extract_images,
            clean_outputs=clean_outputs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_chars=args.min_chars,
            split_sections=args.split_sections,
            print_first=args.print_first,
        )
    else:
        process_folder(
            input_dir=input_dir,
            out_root=out_dir,
            strategy=args.strategy,
            ocr_strategy=args.ocr_strategy,
            infer_tables=args.infer_tables,
            extract_images=extract_images,
            clean_outputs=clean_outputs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_chars=args.min_chars,
            split_sections=args.split_sections,
            print_first=args.print_first,
            recursive=args.recursive,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
