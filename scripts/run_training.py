"""Reproducible launcher, works in Kaggle after copying this code workspace.
python scripts/run_training.py --config training_run.json        # preflight only
python scripts/run_training.py --config training_run.json --execute
No uploads, guessed glob paths, downloads, notebook source patches or secrets.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from storage import configure_runtime, output_path
configure_runtime(ROOT)
from lineage import atomic_json, sha256_file, object_hash

ALLOWED={"mode","device","optimizer","batch-size","grad-accum","block-size","lr",
         "seed","run-iters","steps","eval-interval","eval-iters","checkpoint-interval",
         "warmup","pack","low-vram","resume","no-pretrained-init","max-convs",
         "hf-seed-dir","sft-val-dir","grounded-dir","duplication"}


def preflight(spec: dict, *, root: Path=ROOT):
    if spec.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(spec.get("inputs"),dict):
        raise ValueError("inputs object required")
    resolved={}
    for role,record in spec["inputs"].items():
        if role not in {"data","tokenizer","parent","mixture"}:
            raise ValueError(f"unknown input role {role}")
        path=Path(record["path"])
        if not path.is_absolute(): path=root/path
        if not path.is_file(): raise ValueError(f"missing {role}: {path}")
        digest=sha256_file(str(path))
        if digest != record.get("sha256"):
            raise ValueError(f"hash mismatch for {role}")
        resolved[role]=str(path)
    if "tokenizer" not in resolved:
        raise ValueError("pinned tokenizer required")
    # Existing trainer reads bpe.json unless restoring it from its checkpoint.
    if "parent" not in resolved and sha256_file(str(root/"bpe.json")) != spec["inputs"]["tokenizer"]["sha256"]:
        raise ValueError("workspace bpe.json differs from pinned tokenizer")
    args=spec.get("training",{})
    if set(args)-ALLOWED: raise ValueError("unsupported training options")
    mode=args.get("mode","pretrain")
    if mode not in {"pretrain","instruct"}: raise ValueError("invalid mode")
    if mode=="pretrain" and "data" not in resolved:
        raise ValueError("pretrain requires pinned data")
    output=Path(output_path(spec["output_checkpoint"],root=root))
    if any(output.resolve()==Path(p).resolve() for p in resolved.values()):
        raise ValueError("output must not overwrite input")
    if output.exists() and not args.get("resume"):
        raise ValueError("output exists; use a new run path")
    command=[sys.executable,str(root/"train.py"),"--checkpoint",str(output)]
    for key,value in args.items():
        if isinstance(value,bool):
            if value: command.append("--"+key)
        else: command.extend(["--"+key,str(value)])
    if "data" in resolved: command += ["--data",resolved["data"]]
    if "mixture" in resolved: command += ["--mixture",resolved["mixture"]]
    if "parent" in resolved:
        if mode=="pretrain":
            # Pretrain has only resume initialization: copy parent to a NEW
            # output path is explicitly performed on --execute, never preflight.
            if not args.get("resume"): command.append("--resume")
        else: command += ["--init-from",resolved["parent"]]
    return command,resolved,output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    spec=json.loads(Path(a.config).read_text(encoding="utf-8"))
    command,inputs,output=preflight(spec)
    print(json.dumps({"command":command,"inputs":inputs,"output":str(output)},indent=2))
    if not a.execute: return
    if spec.get("training",{}).get("device")=="cuda":
        import torch
        if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable; refusing silent CPU training")
    if spec.get("training",{}).get("mode","pretrain")=="pretrain" and "parent" in inputs and not output.exists():
        import shutil
        output.parent.mkdir(parents=True,exist_ok=True)
        with open(output,"xb") as target,open(inputs["parent"],"rb") as source:
            shutil.copyfileobj(source,target)
    atomic_json(str(output)+".launch.json",{"config":spec,"config_hash":object_hash(spec),"command":command,"status":"launching"})
    result=subprocess.run(command,cwd=ROOT)
    raise SystemExit(result.returncode)

if __name__=="__main__":main()
