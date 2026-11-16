"""Budgeted acquisition only: no conversion, preparation or training.
python scripts/python_d.py scripts/download_training_data.py --output-dir data/downloads/general_v1
Add --execute to download the planned complete files; default is metadata only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import os
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from storage import configure_runtime, output_path
configure_runtime(ROOT)
from fetch_datasets import resolve_hf_files, download_with_progress, USER_AGENT
from lineage import atomic_json, sha256_file

ORIGINS = ("human_documented", "synthetic_documented")


def choose_files(entries: list, budgets: dict, cache: Path) -> tuple[list, list]:
    selected, skipped = [], []
    remaining = dict(budgets)
    for entry in entries:
        origin = entry["origin"]
        if origin not in budgets: raise ValueError("Unbudgeted origin")
        for item in entry["resolved_files"]:
            size = item.get("bytes")
            if not isinstance(size, int) or size <= 0:
                skipped.append({"dataset_id": entry["id"], "path": item["path"], "reason": "unknown_size"})
                continue
            existing = cache / ("hf-" + entry["id"]) / Path(item["path"]).name
            if existing.is_file():
                skipped.append({"dataset_id": entry["id"], "path": item["path"],
                                "reason": "existing_cache_preserved_not_counted_as_new", "local_bytes": existing.stat().st_size})
                continue
            if size > remaining[origin]:
                skipped.append({"dataset_id": entry["id"], "path": item["path"], "reason": "origin_budget"})
                continue
            remaining[origin] -= size
            selected.append({"dataset_id": entry["id"], "origin": origin, "repository": entry["hf_repo"],
                             "revision": entry["resolved_revision"], "remote_path": item["path"],
                             "bytes": size, "expected_sha256": item.get("sha256"),
                             "license": entry["license"], "dataset_card": entry["dataset_card"],
                             "quality_evidence": entry["quality_evidence"], "domain": entry["domain"],
                             "text_columns": entry["text_columns"],
                             "url": f"https://huggingface.co/datasets/{entry['hf_repo']}/resolve/{entry['resolved_revision']}/{item['path']}"})
    return selected, skipped


def make_plan(catalog: dict, total_bytes: int, cache: Path) -> dict:
    if total_bytes <= 0: raise ValueError("Positive budget required")
    budgets = {ORIGINS[0]: total_bytes * 7 // 10}
    budgets[ORIGINS[1]] = total_bytes - budgets[ORIGINS[0]]
    entries = copy.deepcopy(catalog["datasets"])
    for entry in entries: resolve_hf_files(entry)
    selected, skipped = choose_files(entries, budgets, cache)
    return {"schema_version": 1, "status": "planned_not_downloaded", "total_download_budget_bytes": total_bytes,
            "origin_budget_bytes": budgets, "files": selected, "skipped": skipped,
            "reuse_existing": catalog.get("reuse_existing", []),
            "new_planned_bytes": sum(item["bytes"] for item in selected),
            "note": "70/30 byte acquisition budgets are NOT measured training-token ratios. Origin/accuracy not guaranteed per record."}


class DownloadPaused(Exception):
    """Keep the partial payload when the caller's time budget is reached."""


def download_partial(item: dict, dest: Path, *, deadline: float | None = None) -> None:
    """Resume only a pinned complete file; reject ignored/malformed Range responses.

    A final SHA256 is mandatory; partial files never become usable artifacts.
    This function does not execute data or remove failed partials.
    """
    dest = Path(output_path(dest))
    expected = item.get("expected_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Partial resume requires a pinned SHA256")
    if dest.exists():
        raise FileExistsError(f"Destination exists; inspect it first: {dest}")
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if part.is_symlink(): raise ValueError("Refusing partial symlink")
    offset = part.stat().st_size if part.exists() else 0
    total = item["bytes"]
    if offset > total: raise ValueError("Partial exceeds pinned size")
    if offset < total:
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset: headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(item["url"], headers=headers)
        print(f"[GET] {Path(item['remote_path']).name} from {offset:,}/{total:,} bytes", flush=True)
        with urllib.request.urlopen(request, timeout=45) as response:
            if offset:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match or tuple(map(int, match.groups())) != (offset, total-1, total):
                    raise ValueError("Server did not honor exact Range; partial preserved")
            elif response.status != 200:
                raise ValueError("Expected HTTP 200 for new complete-file request")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != total-offset:
                raise ValueError("Response length differs from pinned file")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Encoded response cannot safely resume")
            written = offset
            with part.open("ab" if offset else "xb") as stream:
                while written < total:
                    if deadline is not None and time.monotonic() >= deadline:
                        stream.flush(); os.fsync(stream.fileno())
                        raise DownloadPaused()
                    chunk = response.read(min(4 << 20, total-written))
                    if not chunk: raise ValueError("Truncated response; partial preserved")
                    stream.write(chunk); written += len(chunk)
                    if written % (64 << 20) < len(chunk):
                        print(f"[Download] {written:,}/{total:,} bytes", flush=True)
                if response.read(1): raise ValueError("Response exceeds pinned size")
                stream.flush(); os.fsync(stream.fileno())
    if part.stat().st_size != total or sha256_file(str(part)) != expected:
        raise ValueError("Pinned size/SHA256 mismatch; partial preserved, not published")
    # Atomic no-clobber publication; same-volume hard link retains original bytes.
    os.link(part, dest)
    part.unlink()
    atomic_json(str(dest) + ".manifest.json", {"url":item["url"], "bytes":total,
                "sha256":expected, "revision":item["revision"], "status":"complete"})


def execute(plan: dict, output: Path, *, max_files: int | None = None, resume: bool = False,
            resume_partial: bool = False, max_seconds: float | None = None) -> dict:
    output = Path(output_path(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / (output.name + ".download.lock")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        return _execute(plan, output, max_files=max_files, resume=resume,
                        resume_partial=resume_partial, max_seconds=max_seconds)
    finally:
        lock.unlink()


def _execute(plan: dict, output: Path, *, max_files: int | None = None, resume: bool = False,
             resume_partial: bool = False, max_seconds: float | None = None) -> dict:
    if output.exists() and not resume: raise ValueError("Output exists; choose a new batch, no overwrite/retry hidden")
    if max_files is not None and max_files < 1: raise ValueError("max-files must be positive")
    if max_seconds is not None and max_seconds <= 0: raise ValueError("max-seconds must be positive")
    if max_seconds is not None and not resume_partial:
        raise ValueError("Time-bounded downloads require --resume-partial to preserve progress")
    deadline = time.monotonic() + max_seconds if max_seconds is not None else None
    if sum(i["bytes"] for i in plan["files"]) > plan.get("total_download_budget_bytes", plan["new_planned_bytes"]):
        raise ValueError("Plan exceeds total download budget")
    required = plan["new_planned_bytes"] + (1 << 30)
    if shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free < required:
        raise ValueError("Insufficient free disk plus 1 GiB reserve")
    output.mkdir(parents=True, exist_ok=resume)
    report_path = output / "download_report.json"
    if resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("failures"):
            raise ValueError("Previous download failed; inspect report/files before retrying (no automatic budget reset)")
        if report.get("files") != plan["files"]:
            raise ValueError("Resume must use the saved plan")
    else:
        report = dict(plan, status="downloading", completed=[], failures=[], started_at=time.time())
    completed = set()
    consumed = Counter()
    for prior in report["completed"]:
        path = Path(prior["local_path"])
        if path.stat().st_size != prior["bytes"] or sha256_file(str(path)) != prior["sha256"]:
            raise ValueError("Previously completed file changed; do not silently reuse")
        completed.add((prior["dataset_id"], prior["remote_path"]))
        consumed[prior["origin"]] += prior["bytes"]
    report["status"] = "downloading"
    atomic_json(str(report_path), report)
    count = 0
    for item in plan["files"]:
        if (item["dataset_id"], item["remote_path"]) in completed:
            continue
        if (max_files is not None and count >= max_files) or (deadline is not None and time.monotonic() >= deadline):
            report["status"] = "paused_more_files_pending"
            atomic_json(str(report_path), report)
            return report
        count += 1
        dest = output / item["origin"] / item["dataset_id"] / Path(item["remote_path"]).name
        try:
            # Bound all transferred payload to the pinned complete-file size.
            if resume_partial:
                download_partial(item, dest, deadline=deadline)
            else:
                download_with_progress(item["url"], dest, limit_bytes=item["bytes"])
            consumed[item["origin"]] += dest.stat().st_size
            actual = sha256_file(str(dest))
            if dest.stat().st_size != item["bytes"] or (item["expected_sha256"] and actual != item["expected_sha256"]):
                raise ValueError("Pinned size/SHA256 mismatch; do not use this file")
            report["completed"].append(dict(item, local_path=str(dest.resolve()), sha256=actual,
                                             status="downloaded_hash_verified_not_fact_checked"))
            report.pop("active_partial", None)
        except DownloadPaused:
            report["status"] = "paused_partial_file"
            report["active_partial"] = str(dest.with_suffix(dest.suffix + ".part").relative_to(output))
            atomic_json(str(report_path), report)
            return report
        except Exception as exc:
            report["failures"].append({"dataset_id": item["dataset_id"], "remote_path": item["remote_path"], "error": str(exc)})
            # Stop, rather than automatic retries spending the budget on failed payloads.
            report["status"] = "stopped_on_error"
            atomic_json(str(report_path), report)
            print(f"STOP: {exc}", flush=True)
            return report
        report.update(downloaded_bytes_by_origin=dict(consumed), elapsed_s=round(time.time()-report["started_at"], 2))
        atomic_json(str(report_path), report)
    report["status"] = "downloaded_not_prepared_not_trained"
    atomic_json(str(report_path), report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="docs/training/download_catalog.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--total-gib", type=float, default=20)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-files", type=int, help="Download only this many new files, then pause")
    p.add_argument("--resume", action="store_true", help="Continue the saved successful/paused plan; reverify completed files")
    p.add_argument("--status", action="store_true", help="Read local progress only; no network or downloads")
    p.add_argument("--resume-partial", action="store_true", help="Use validated HTTP Range, pinned SHA256 and preserve partial progress")
    p.add_argument("--max-seconds", type=float, help="Pause after this time budget; requires --resume-partial")
    args = p.parse_args()
    output = Path(output_path(args.output_dir))
    if args.status:
        report = json.loads((output / "download_report.json").read_text(encoding="utf-8"))
        print(json.dumps({"status":report["status"], "completed_files":len(report["completed"]),
                          "planned_files":len(report["files"]), "downloaded_bytes_by_origin":report.get("downloaded_bytes_by_origin"),
                          "failures":report["failures"],
                          "lock_exists":(output.parent/(output.name+".download.lock")).exists(),
                          "partial_files":[{"path":str(p.relative_to(output)),"bytes":p.stat().st_size} for p in output.rglob("*.part")]}, indent=2), flush=True)
        return
    if args.resume:
        if not args.execute: p.error("--resume requires --execute")
        plan = json.loads((output / "download_report.json").read_text(encoding="utf-8"))
    else:
        catalog = json.loads((ROOT / args.catalog).read_text(encoding="utf-8"))
        plan = make_plan(catalog, int(args.total_gib * (1 << 30)), ROOT / "cache/hf")
    if args.execute:
        print(f"Planned: {len(plan['files'])} files, {plan['new_planned_bytes']:,} bytes", flush=True)
        result = execute(plan, output, max_files=args.max_files, resume=args.resume,
                         resume_partial=args.resume_partial, max_seconds=args.max_seconds)
        print(json.dumps({"status":result["status"],"completed_files":len(result["completed"]),
                          "bytes_by_origin":result.get("downloaded_bytes_by_origin"),
                          "failures":result["failures"]}, ensure_ascii=True, indent=2), flush=True)
        if result["failures"]: raise SystemExit(1)
    else:
        print(json.dumps(plan, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
