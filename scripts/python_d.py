"""Run Python/pip with project-local caches and temp, without global OS changes.
python scripts/python_d.py --storage-report
python scripts/python_d.py -m pip install -r requirements.txt
python scripts/python_d.py fetch_datasets.py --only fineweb2-vi --sample-rows 1000
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from storage import configure_runtime, output_path


def main():
    locations = configure_runtime(ROOT)
    if sys.argv[1:] == ["--storage-report"]:
        import shutil
        report = {"project_root": str(ROOT), "python": sys.executable,
                  "locations": locations,
                  "outputs": {name: output_path(ROOT / name) for name in
                              ("data", "cache", "checkpoints", "runs", "bench_results")},
                  "disk_free_bytes": shutil.disk_usage(ROOT).free,
                  "policy": "Windows output drive D; environment applies only to this process and children"}
        from lineage import atomic_json
        atomic_json(str(ROOT / "bench_results/storage_report.json"), report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/python_d.py --storage-report | SCRIPT [ARGS] | -m MODULE [ARGS]")
    # Inherit the configured environment before imports in the new interpreter.
    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]], cwd=ROOT))


if __name__ == "__main__":
    main()
