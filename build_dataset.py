"""
Build `data/train.txt` from local raw corpus files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import config
from clean_corpus import FileStats, iter_clean_lines, write_report

# Files larger than this skip the per-file contamination check (the check
# buffers the file in RAM). Huge dumps (TinyStories, Wikipedia shards) have
# effectively zero chance of sharing 8-13 word n-grams with the 165 probes.
CONTAMINATION_MAX_FILE_BYTES = 64_000_000


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".log",
    ".xml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaned training corpus")
    parser.add_argument("--source", type=str, default=config.RAW_DATA_DIR)
    parser.add_argument("--output", type=str, default=config.DATA_PATH)
    parser.add_argument("--min-file-bytes", type=int, default=8_192)
    parser.add_argument("--min-line-chars", type=int, default=8)
    parser.add_argument("--max-line-chars", type=int, default=1_200)
    parser.add_argument("--max-mb-per-file", type=float, default=0.0,
                        help="Cap dung lượng mỗi file raw đóng góp vào corpus "
                             "(0 = không giới hạn). VD 300 = mỗi file tối đa 300MB, "
                             "hữu ích khi corpus gồm các dump nhiều GB mà GPU yếu.")
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append the existing output dataset instead of replacing it.",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Create a timestamped backup before overwriting the output dataset.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Also save the built corpus to data/curated/snapshots/ with a timestamped name.",
    )
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--no-contamination-check",
        action="store_true",
        help="Skip the benchmark-leakage guard (removes any document sharing "
             "an n-gram with data/eval/probes.jsonl).",
    )
    return parser.parse_args()


def iter_source_files(source_root: Path, min_file_bytes: int):
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < min_file_bytes:
            continue
        yield path


def build_corpus(
    source_root: str | Path,
    min_file_bytes: int,
    min_line_chars: int,
    max_line_chars: int,
    output_path: Path | None = None,
    *,
    max_chars_per_file: int = 0,
    check_contamination: bool = True,
) -> list[FileStats]:
    """Stream-clean every source file and append to ``output_path``.

    Constant memory: cleaned lines are written out incrementally instead
    of collecting the whole corpus in RAM (multi-GB raw files are common
    now). Set ``max_chars_per_file`` > 0 to cap each file's contribution.
    Files under ``CONTAMINATION_MAX_FILE_BYTES`` are additionally screened
    against the benchmark probes (leakage guard).
    """
    source_root = Path(source_root)
    stats: list[FileStats] = []
    out_handle = open(output_path, "w", encoding="utf-8") if output_path is not None else None
    contaminate_fn = None
    if check_contamination:
        from data_quality import check_contamination as contaminate_fn

    try:
        for path in sorted(iter_source_files(source_root, min_file_bytes=min_file_bytes)):
            rel_path = path.relative_to(source_root).as_posix()
            file_stats: dict = {"input_chars": 0, "kept_lines": 0, "skipped_lines": 0}
            file_out_chars = 0
            doc_open = False

            line_iter = iter_clean_lines(
                path,
                min_line_chars=min_line_chars,
                max_line_chars=max_line_chars,
                strip_gutenberg="gutenberg" in path.as_posix().lower(),
                strip_xml=path.suffix.lower() == ".xml",
                max_chars=max_chars_per_file or None,
                stats=file_stats,
            )

            buffered_lines = None
            if contaminate_fn is not None and path.stat().st_size <= CONTAMINATION_MAX_FILE_BYTES:
                buffered_lines = list(line_iter)
                doc = "\n".join(buffered_lines)
                if doc.strip():
                    _, removed = contaminate_fn([doc])
                    if removed:
                        print(f"[Corpus] Contamination: dropped {rel_path} "
                              f"(leaked data/eval/probes.jsonl n-grams)")
                        buffered_lines = []
                        file_stats = {"input_chars": file_stats["input_chars"], "kept_lines": 0, "skipped_lines": file_stats["skipped_lines"]}

            for line in (buffered_lines if buffered_lines is not None else line_iter):
                if out_handle is not None:
                    if not doc_open:
                        out_handle.write(f"<doc path=\"{rel_path}\">\n")
                        doc_open = True
                    else:
                        out_handle.write("\n")
                    out_handle.write(line)
                file_out_chars += len(line)

            if file_out_chars and out_handle is not None and doc_open:
                out_handle.write("\n</doc>\n\n")
            if file_out_chars:
                stats.append(
                    FileStats(
                        path=rel_path,
                        input_chars=file_stats["input_chars"],
                        output_chars=file_out_chars,
                        kept_lines=file_stats["kept_lines"],
                        skipped_lines=file_stats["skipped_lines"],
                    )
                )
    finally:
        if out_handle is not None:
            out_handle.close()
    return stats


def _stream_copy_text(src: Path, out_handle) -> int:
    """Copy a text file into ``out_handle`` in chunks; returns char count."""
    copied = 0
    with open(src, "r", encoding="utf-8", errors="replace") as src_handle:
        while True:
            chunk = src_handle.read(1 << 20)
            if not chunk:
                break
            out_handle.write(chunk)
            copied += len(chunk)
    return copied


def write_snapshot(output_path: Path) -> Path:
    os.makedirs(config.DATA_SNAPSHOT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    snapshot_path = Path(config.DATA_SNAPSHOT_DIR) / f"{output_path.stem}_{stamp}{output_path.suffix}"
    shutil.copyfile(output_path, snapshot_path)
    return snapshot_path


def backup_existing_file(output_path: Path) -> Path | None:
    if not output_path.exists():
        return None
    os.makedirs(config.DATA_SNAPSHOT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = Path(config.DATA_SNAPSHOT_DIR) / f"{output_path.stem}_backup_{stamp}{output_path.suffix}"
    shutil.copy2(output_path, backup_path)
    return backup_path


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    output_path = Path(config.output_path(args.output))

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source directory not found: {source_root}\n"
            f"Put raw text files under this folder first."
        )

    os.makedirs(output_path.parent, exist_ok=True)
    os.makedirs(config.CURATED_DATA_DIR, exist_ok=True)
    os.makedirs(config.DATA_SNAPSHOT_DIR, exist_ok=True)

    if args.backup_existing:
        backup_path = backup_existing_file(output_path)
        if backup_path is not None:
            print(f"[Corpus] Backup saved:   {backup_path}")

    building_path = output_path.with_suffix(output_path.suffix + ".building")
    existing_chars = 0

    with open(building_path, "w", encoding="utf-8", newline="\n") as out:
        if args.append_existing and output_path.exists():
            if output_path.stat().st_size > 0:
                print(f"[Corpus] Appending after existing corpus ({output_path.stat().st_size/1e6:.1f} MB)")
                existing_chars = _stream_copy_text(output_path, out)
                out.write("\n\n<doc path=\"existing/train.txt\">\n")

        stats = build_corpus(
            source_root=source_root,
            min_file_bytes=args.min_file_bytes,
            min_line_chars=args.min_line_chars,
            max_line_chars=args.max_line_chars,
            output_path=building_path,
            max_chars_per_file=int(args.max_mb_per_file * 1024 * 1024),
        )

    if not stats and existing_chars == 0:
        building_path.unlink(missing_ok=True)
        raise RuntimeError(
            "No usable text files were found. Add larger local text files under data/raw/."
        )

    os.replace(building_path, output_path)

    output_bytes = output_path.stat().st_size
    output_mb = output_bytes / (1024 * 1024)
    if existing_chars:
        print(f"[Corpus] Existing chars: {existing_chars:,}")
    print(f"[Corpus] Files kept:     {len(stats):,}")
    print(f"[Corpus] Output size:    {output_mb:.2f} MB")
    print(f"[Corpus] Saved -> {output_path}")

    if args.snapshot:
        snapshot_path = write_snapshot(output_path)
        print(f"[Snapshot] Saved -> {snapshot_path}")

    if args.report:
        report_path = output_path.with_suffix(output_path.suffix + ".report.json")
        write_report(report_path, stats)
        print(f"[Report] Saved -> {report_path}")

    if output_mb < 500:
        print(
            "[Corpus] Note: current dataset is still below 500 MB. "
            "Add more raw files and rerun."
        )


if __name__ == "__main__":
    main()
