"""Prepare a small Kaggle upload on D, without training/uploading.
python scripts/python_d.py scripts/pack_kaggle.py --output-dir kaggle_upload/v4_chat
Use --checkpoint checkpoints/jingpt_latest.pt for continued pretrain instead.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from storage import configure_runtime, output_path
configure_runtime(ROOT)
from lineage import atomic_json, object_hash, sha256_file


def pack(output:Path, prepared:Path, checkpoint:Path, *, root:Path=ROOT)->dict:
    output=Path(output_path(output,root=root))
    if output.exists():raise ValueError("Upload directory already exists; choose a new name")
    if not checkpoint.is_file():raise ValueError(f"Parent checkpoint missing: {checkpoint}")
    report=json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    mixture=json.loads((prepared/"mixture.json").read_text(encoding="utf-8"))
    if report.get("status")!="ready" or object_hash(mixture)!=report.get("mixture_hash"):
        raise ValueError("Prepared dataset not ready or mixture changed")
    expected={prepared/"pretrain/validation.txt":report["validation_sha256"]}
    for domain,info in report["pretrain"].items():
        expected[prepared/"pretrain"/(domain+".txt")]=info["train_sha256"]
        if info.get("val_sha256"):expected[prepared/"pretrain"/(domain+"_val.txt")]=info["val_sha256"]
    for did,info in report["sft"].items():
        for split in ("train","val"):
            if info.get(split+"_sha256"):expected[prepared/("sft_"+split)/(did+".jsonl")]=info[split+"_sha256"]
    for path,digest in expected.items():
        if sha256_file(str(path))!=digest:raise ValueError(f"Prepared file hash mismatch: {path}")
    code=list(root.glob("*.py"))+list((root/"scripts").glob("*.py"))
    code += [root/name for name in ("bpe.json","requirements.txt","hf_datasets.json","sources.json")]
    data=list(expected)+[prepared/"manifest.json",prepared/"mixture.json",prepared/"pretrain_sources.jsonl"]
    if (prepared/"verification.json").is_file():data.append(prepared/"verification.json")
    data += [root/"data/eval/probes.jsonl",root/"data/eval/heldout.txt",root/"data/seed/conversations.jsonl"]
    for path in code+data:
        if not path.is_file():raise ValueError(f"Required file missing: {path}")
        if path.is_symlink():raise ValueError(f"Refusing symlink in upload: {path}")
    # Avoid accidentally archiving a file above the workspace.
    for path in code+data:path.relative_to(root)
    output.mkdir(parents=True)
    archives=[]
    for name,paths in (("jin-code.zip",code),("jin-data.zip",data)):
        target=output/name;files=[]
        with zipfile.ZipFile(target,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for path in sorted(set(paths)):
                arcname=path.relative_to(root).as_posix()
                archive.write(path,arcname)
                files.append({"path":arcname,"bytes":path.stat().st_size,"sha256":sha256_file(str(path))})
        archives.append({"filename":name,"bytes":target.stat().st_size,"sha256":sha256_file(str(target)),"files":files})
    result={"schema_version":1,"status":"packaged_not_uploaded_not_trained","prepared_relative":prepared.relative_to(root).as_posix(),
            "archives":archives,"parent":{"filename":checkpoint.name,"local_path":str(checkpoint.resolve()),
                "bytes":checkpoint.stat().st_size,"sha256":sha256_file(str(checkpoint))},
            "note":"Upload the two ZIPs, upload_manifest.json and the EXACT parent checkpoint separately; parent not copied into this folder."}
    atomic_json(str(output/"upload_manifest.json"),result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True)
    p.add_argument("--prepared",default="data/prepared_v4")
    p.add_argument("--checkpoint",default="checkpoints/jingpt_chat.pt")
    args=p.parse_args()
    result=pack(Path(output_path(args.output_dir)),Path(output_path(args.prepared)),Path(output_path(args.checkpoint)))
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=="__main__":main()
