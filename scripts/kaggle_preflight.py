"""Phase 17 §3 Stage A — environment preflight, for Kaggle or anywhere else.

Nothing here assumes a GPU model, a VRAM figure, a runtime limit or a dataset
slug. It reports what is actually present so a run can refuse to start on the
wrong machine instead of discovering it 40 minutes in.

Usage
-----
    python scripts/kaggle_preflight.py --out bench_results/phase17/kaggle_preflight.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def nvidia_smi() -> dict:
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    try:
        query = ("name,memory.total,memory.used,driver_version,compute_cap")
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        rows = [line.split(", ") for line in out.stdout.strip().splitlines() if line]
        return {"available": True,
                "gpus": [{"name": r[0], "memory_total_mb": int(r[1]),
                          "memory_used_mb": int(r[2]), "driver": r[3],
                          "compute_capability": r[4]} for r in rows if len(r) >= 5]}
    except Exception as exc:                     # noqa: BLE001 - report, never crash
        return {"available": False, "error": str(exc)}


def total_ram_gb() -> float | None:
    try:
        import psutil                            # optional dependency
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except Exception:
        return None
    return None


def host_kind() -> str:
    """"kaggle" only on a real Kaggle kernel.

    ``os.path.isdir("/kaggle")`` is not that test: on Windows a leading slash
    resolves against the current drive, so a local ``D:\\kaggle`` folder made a
    Windows desktop report ``is_kaggle: true``. Kaggle always sets a kernel run
    type and mounts ``/kaggle/working``.
    """
    if sys.platform.startswith("win"):
        return "local"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.path.isdir("/kaggle/working") and os.path.isdir("/kaggle/input"):
        return "kaggle"
    return "local"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 17 environment preflight")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench_results", "phase17",
                                                  "kaggle_preflight.json"))
    ap.add_argument("--checkpoint", default=os.path.join(
        ROOT, "checkpoints", "gpt2_small", "phase14_wiki", "best.pt"))
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()

    import torch

    report: dict = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "ram_gb": total_ram_gb(),
            "is_kaggle": host_kind() == "kaggle",
            "host_kind": host_kind(),
        },
        "torch": {"version": str(torch.__version__),
                  "cuda_available": torch.cuda.is_available(),
                  "cuda_version": torch.version.cuda,
                  "device_count": torch.cuda.device_count()},
        "nvidia_smi": nvidia_smi(),
        "storage": {
            "repo_disk_free_gb": round(shutil.disk_usage(ROOT).free / 1024**3, 1),
            "kaggle_input": os.path.isdir("/kaggle/input"),
            "kaggle_working": os.path.isdir("/kaggle/working"),
            "kaggle_temp": os.path.isdir("/kaggle/temp"),
        },
        "repository": {
            "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip() or None,
            "dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip()),
        },
        "checkpoint": {
            "path": os.path.relpath(args.checkpoint, ROOT) if os.path.isabs(
                args.checkpoint) else args.checkpoint,
            "exists": os.path.isfile(args.checkpoint),
            "bytes": os.path.getsize(args.checkpoint) if os.path.isfile(
                args.checkpoint) else None,
            "sha256": sha256_file(args.checkpoint),
        },
        "data": {},
    }

    if torch.cuda.is_available():
        report["torch"]["devices"] = [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "total_vram_gb": round(
                 torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)}
            for i in range(torch.cuda.device_count())]
    for name in ("gpt2_corpus_v5", "gpt2_corpus_v6a", "curated_v6", "curated_v7"):
        path = os.path.join(ROOT, "data", name)
        man = os.path.join(path, "manifest.json")
        report["data"][name] = {
            "exists": os.path.isdir(path),
            "manifest_sha256": sha256_file(man),
            "train_tokens": (json.load(open(man, encoding="utf-8"))
                             .get("train", {}).get("tokens") if os.path.isfile(man) else None),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"host      {report['host']['platform']} | cpu {report['host']['cpu_count']} | "
          f"ram {report['host']['ram_gb']} GB")
    print(f"torch     {report['torch']['version']} | cuda {report['torch']['cuda_version']} | "
          f"available {report['torch']['cuda_available']}")
    for gpu in report["torch"].get("devices", []):
        print(f"gpu[{gpu['index']}]   {gpu['name']} | {gpu['total_vram_gb']} GB")
    print(f"host kind {report['host']['host_kind']}")
    print(f"storage   repo free {report['storage']['repo_disk_free_gb']} GB | "
          f"kaggle input {report['storage']['kaggle_input']} | "
          f"working {report['storage']['kaggle_working']}")
    print(f"repo      {report['repository']['head']} (dirty={report['repository']['dirty']})")
    print(f"ckpt      {report['checkpoint']['path']} exists={report['checkpoint']['exists']} "
          f"sha={str(report['checkpoint']['sha256'])[:12]}")
    print(f"Wrote {os.path.relpath(args.out, ROOT)}")

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("[Fatal] --require-cuda was requested and CUDA is unavailable.")
    if not report["checkpoint"]["exists"]:
        raise SystemExit("[Fatal] Phase 14 checkpoint is missing: continued pretraining "
                         "cannot start from random weights.")


if __name__ == "__main__":
    main()
