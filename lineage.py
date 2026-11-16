"""Content-addressed lineage for future runs. Legacy ancestry stays unknown."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import uuid


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: str, value) -> None:
    from storage import output_path
    path = output_path(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".manifest-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def file_manifest(paths: list[str]) -> list[dict]:
    return [{"path": str(Path(p).resolve()), "sha256": sha256_file(p),
             "bytes": os.path.getsize(p)} for p in sorted(set(paths)) if os.path.isfile(p)]


def start_run(model, tokenizer, training_config: dict, datasets: list[str],
              parent_path: str | None, output_path: str) -> dict:
    import torch
    root = Path(__file__).resolve().parent
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, timeout=5).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, timeout=5))
    except (OSError, subprocess.SubprocessError):
        git_sha, dirty = None, None
    code = [{"path": p.name, "sha256": sha256_file(str(p))} for p in sorted(root.glob("*.py"))]
    hardware = {"platform": platform.platform(), "python": platform.python_version(),
                "torch": str(torch.__version__), "device": str(next(model.parameters()).device),
                "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        hardware["gpu"] = torch.cuda.get_device_name(0)
    run = {"schema_version": 1, "run_id": uuid.uuid4().hex,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_sha": git_sha, "git_dirty": dirty, "code": code,
           "code_hash": object_hash(code), "model_config_hash": object_hash(model.export_config()),
           "tokenizer_hash": object_hash(tokenizer.to_state_dict()),
           "datasets": file_manifest(datasets), "training_config": training_config,
           "hardware": hardware, "seed": training_config.get("seed"),
           "parent_checkpoint": file_manifest([parent_path])[0] if parent_path and os.path.isfile(parent_path) else None,
           "tokens_seen_before_run": None, "tokens_seen_this_run": 0,
           "supervised_tokens_this_run": 0, "step": 0,
           "validation_metrics": {}, "benchmark_results": None,
           "output_checkpoint": output_path, "status": "started"}
    run["manifest_path"] = os.path.join(str(root), "runs", run["run_id"] + ".json")
    atomic_json(run["manifest_path"], run)
    return run


def attach_run(model, run: dict):
    model._run_manifest = run


def record_batch(model, tokens: int, supervised_tokens: int | None = None):
    run = getattr(model, "_run_manifest", None)
    if run is not None:
        run["tokens_seen_this_run"] += int(tokens)
        run["supervised_tokens_this_run"] += int(supervised_tokens if supervised_tokens is not None else tokens)


def export_inference(source: str, destination: str) -> dict:
    """Explicit separate artifact, never overwrite a training checkpoint."""
    import torch
    from utils import load_checkpoint_file
    from storage import output_path
    destination = output_path(destination)
    if os.path.abspath(source) == os.path.abspath(destination) or os.path.exists(destination):
        raise ValueError("Inference destination must be new and distinct from source")
    checkpoint = load_checkpoint_file(source, map_location="cpu")
    keys = ("model_state_dict", "model_config", "tokenizer_state", "vocab_size", "iteration", "loss", "lineage")
    payload = {k: checkpoint[k] for k in keys if k in checkpoint}
    payload.update(artifact_kind="inference_weights", parent_sha256=sha256_file(source), format_version=3)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    tmp = destination + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, destination)
    report = {"source_sha256": payload["parent_sha256"], "output_sha256": sha256_file(destination),
              "artifact_kind": "inference_weights", "bytes": os.path.getsize(destination)}
    atomic_json(destination + ".manifest.json", report)
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--export-inference", nargs=2, metavar=("SOURCE", "DESTINATION"), required=True)
    a = p.parse_args()
    print(json.dumps(export_inference(*a.export_inference), indent=2))
