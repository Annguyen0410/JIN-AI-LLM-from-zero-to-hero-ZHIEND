"""JIN data preparation: reuse downloaded data before buying more disk/network.
No training, no tokenizer rebuild, no replacement of existing corpora.
python data_pipeline.py --inventory
python data_pipeline.py --prepare --output-dir data/prepared_v1 --mb-per-domain 32 --sft-per-source 2000
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
import unicodedata

import config
from lineage import atomic_json, object_hash, sha256_file

# Starting recipe, NOT experimentally optimized weights. VI intentionally
# prominent because current model fails all 22 Vietnamese probes.
WEIGHTS = {"vietnamese": .35, "educational": .30, "knowledge": .15,
           "code": .10, "math": .08, "stories": .02}


def inventory(root: Path) -> dict:
    records=[]
    for directory in (root/"data/raw",root/"data/seed/hf",root/"cache/hf"):
        groups={}
        if not directory.exists():continue
        for p in directory.rglob("*"):
            if not p.is_file() or p.name.endswith((".part", ".manifest.json")):continue
            relative=p.relative_to(directory)
            group=relative.parts[0] if len(relative.parts)>1 else p.stem
            item=groups.setdefault(group,{"group":group,"directory":str(directory.relative_to(root)),"files":0,"bytes":0})
            item["files"]+=1;item["bytes"]+=p.stat().st_size
        records.extend(groups.values())
    disk=shutil.disk_usage(root)
    return {"generated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "groups":records,"total_listed_bytes":sum(r["bytes"] for r in records),
            "disk_free_bytes":disk.free,"train_bytes":(root/"data/train.txt").stat().st_size if (root/"data/train.txt").exists() else 0}


class ContaminationGuard:
    """Build probe ngrams ONCE; stream candidate docs instead of loading corpus."""
    def __init__(self, path: str):
        from data_quality import load_probe_texts, _contamination_ngrams_adaptive
        self.ngrams=set()
        for text in load_probe_texts(path):self.ngrams.update(_contamination_ngrams_adaptive(text,13))
        self.lengths={len(g) for g in self.ngrams}
        self.first_tokens={g[0] for g in self.ngrams}
    def contaminated(self,text: str)->bool:
        words=re.findall(r"[A-Za-zÀ-ỹ0-9]+",text.lower())
        return any(tuple(words[i:i+n]) in self.ngrams for i,word in enumerate(words)
                   if word in self.first_tokens for n in self.lengths if i+n <= len(words))


def iter_documents(path: Path, max_chars: int=12000):
    """Legacy blank-separated or new EOT-delimited text, with bounded chunks."""
    lines=[];size=0
    with path.open(encoding="utf-8",errors="replace") as probe:
        has_eot = "<|endoftext|>" in probe.read(1 << 20)
    if has_eot:
        # Modern fetched samples carry explicit record boundaries. NEVER split
        # one article into independently hash-split train/val paragraphs.
        oversized = False
        with path.open(encoding="utf-8",errors="replace") as handle:
            for line in handle:
                if line.strip()=="<|endoftext|>":
                    if lines and not oversized:yield "".join(lines).strip()
                    lines=[];size=0;oversized=False;continue
                size += len(line)
                if size > 100000:
                    oversized=True;lines=[]
                if not oversized:lines.append(line)
        if lines and not oversized:yield "".join(lines).strip()
        return
    with path.open(encoding="utf-8",errors="replace") as handle:
        for line in handle:
            if (not line.strip() and not has_eot) or line.strip()=="<|endoftext|>":
                if lines:yield "".join(lines).strip()
                lines=[];size=0;continue
            # Split oversized paragraphs rather than discard Wikipedia lines.
            while len(line)>max_chars:
                cut=line.rfind(" ",0,max_chars)
                cut=cut if cut>0 else max_chars
                if lines:yield "".join(lines).strip();lines=[];size=0
                yield line[:cut].strip();line=line[cut:]
            if size+len(line)>max_chars and lines:
                yield "".join(lines).strip();lines=[];size=0
            lines.append(line);size+=len(line)
    if lines:yield "".join(lines).strip()


def text_quality(text:str,domain:str)->str|None:
    if len(text)<80:return "short"
    if "\ufffd" in text:return "encoding"
    if sum(c.isalpha() for c in text)/max(1,len(text)) < (.2 if domain in {"code","math"} else .5):return "low_information"
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    if len(lines)>8 and len(set(lines))/len(lines)<.5:return "repeated_lines"
    from grounding import INJECTION
    if INJECTION.search(text):return "instruction_injection"
    return None


def source_lines(paths):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:yield path,line


def source_metadata(path:Path)->dict:
    manifest=Path(str(path)+".manifest.json")
    if not manifest.exists():return {"revision_status":"unknown_legacy_source"}
    metadata=json.loads(manifest.read_text(encoding="utf-8"))
    return {"manifest_sha256":sha256_file(str(manifest)),"details":metadata}


def source_domain(dataset_id:str,catalog:dict)->str|None:
    if dataset_id in catalog and catalog[dataset_id].get("domain"):
        return catalog[dataset_id]["domain"]
    if "wikipedia-vi" in dataset_id or "fineweb2-vi" in dataset_id:return "vietnamese"
    if "cosmopedia" in dataset_id or "fineweb-edu" in dataset_id:return "educational"
    if "wikipedia" in dataset_id:return "knowledge"
    if "python-edu" in dataset_id:return "code"
    if "tinystories" in dataset_id:return "stories"
    return None  # do not silently train on arbitrary logs/JSON/private files


def downloaded_sources(path: Path, root: Path) -> tuple[list[Path], dict, dict]:
    """Use only checksum-verified completed parquet entries, never .part files."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "downloaded_not_prepared_not_trained" or report.get("failures"):
        raise ValueError("Download batch is not complete; finish/review it before preparation")
    records = report.get("completed", [])
    planned = {(r["dataset_id"], r["remote_path"]): r for r in report.get("files", [])}
    identities = {(r["dataset_id"], r["remote_path"]) for r in records}
    if len(records) != len(planned) or identities != set(planned):
        raise ValueError("Incomplete download file list")
    paths, sources, catalog = [], {}, {}
    for record in records:
        original = Path(record["local_path"])
        p = original.resolve()
        p.relative_to(root.resolve())  # Never ingest arbitrary files outside project.
        pinned = planned[(record["dataset_id"], record["remote_path"])]
        if any(record.get(key) != pinned.get(key) for key in ("bytes", "expected_sha256", "origin", "domain", "revision")):
            raise ValueError("Completed metadata differs from pinned plan")
        if p.suffix != ".parquet" or not p.is_file() or original.is_symlink():
            raise ValueError("Expected completed project-local parquet")
        if p.stat().st_size != record["bytes"] or sha256_file(str(p)) != record["sha256"]:
            raise ValueError("Downloaded source hash/size mismatch")
        if record.get("expected_sha256") and record["sha256"] != record["expected_sha256"]:
            raise ValueError("Downloaded source differs from pinned upstream hash")
        if record["domain"] not in WEIGHTS:
            raise ValueError("Unsupported downloaded domain")
        if record["origin"] not in {"human_documented", "synthetic_documented"}:
            raise ValueError("Downloaded origin missing or unknown")
        if p in sources: raise ValueError("Duplicate downloaded source path")
        paths.append(p);sources[p]=record
        catalog[record["dataset_id"]] = record
    return paths, sources, catalog


def prepare(root:Path, output:Path, *, mb_per_domain:float=32, sft_per_source:int=2000,seed:int=1337,
            allow_noncommercial:bool=False, download_report:Path|None=None)->dict:
    output=Path(config.output_path(output,root=root))
    if output.exists():raise ValueError("Output already exists; select a new version directory (no overwrite)")
    if mb_per_domain<=0 or sft_per_source<1:raise ValueError("Budgets must be positive")
    free=shutil.disk_usage(output.parent if output.parent.exists() else root).free
    if free < int(mb_per_domain*6*(1<<20))+ (1<<30):raise ValueError("Not enough disk plus 1 GiB safety reserve")
    added_paths, downloaded, added_catalog = downloaded_sources(download_report, root) if download_report else ([], {}, {})
    output.mkdir(parents=True)
    (output/"pretrain").mkdir();(output/"sft_train").mkdir();(output/"sft_val").mkdir()
    catalog_list=json.loads((root/"hf_datasets.json").read_text(encoding="utf-8"))["datasets"]
    catalog={e["id"]:e for e in catalog_list}
    catalog.update(added_catalog)
    guard=ContaminationGuard(str(root/"data/eval/probes.jsonl"))
    report={"schema_version":1,"seed":seed,"status":"building","recipe_note":"Starting weights, not capability-optimized; exact dedupe only",
            "budgets":{"mb_per_domain":mb_per_domain,"sft_per_source":sft_per_source},
            "license_policy":"includes NC research data" if allow_noncommercial else "exclude NC and uncertain per-source licenses","pretrain":{},"sft":{},"sources":[],
            "benchmark_sha256":sha256_file(str(root/"data/eval/probes.jsonl")),"heldout_guard":"probe ngrams + fixed heldout 13-grams"}
    if download_report:
        report["download_report"] = {"path":str(download_report.resolve()),"sha256":sha256_file(str(download_report))}
        report["origin_mixture_status"] = "Not enforced: current domain weights do not guarantee human/synthetic 70/30"
    atomic_json(str(output/"manifest.json"),report)
    # Existing heldout phrases are protected independently of probe labels.
    heldout=root/"data/eval/heldout.txt"
    if heldout.exists():
        words=re.findall(r"[A-Za-zÀ-ỹ0-9]+",heldout.read_text(encoding="utf-8").lower())
        guard.ngrams.update(tuple(words[i:i+13]) for i in range(len(words)-12));guard.lengths.add(13)
        guard.first_tokens={g[0] for g in guard.ngrams}
        report["heldout_sha256"]=sha256_file(str(heldout))
    seen=set();cap=int(mb_per_domain*(1<<20));domains={}
    provenance = (output/"pretrain_sources.jsonl").open("w", encoding="utf-8")
    paths=sorted((root/"data/raw").glob("hf-*/*.txt")) + added_paths
    groups={d:[] for d in WEIGHTS}
    for path in paths:
        did=downloaded[path]["dataset_id"] if path in downloaded else path.parent.name.removeprefix("hf-")
        domain=source_domain(did,catalog)
        if re.search(r"(?:^|[-_])(valid|validation|test)(?:$|[-_])",did):continue
        if domain in groups:groups[domain].append((did,path))
    # Domain cap is divided among source files to avoid one dump dominating.
    for domain,files in groups.items():
        if not files:continue
        print(f"[Prepare] pretrain {domain}: {len(files)} source file(s)", flush=True)
        train_path=output/"pretrain"/(domain+".txt")
        val_path=output/"pretrain"/(domain+"_val.txt")
        counts=Counter();per_file=max(1,cap//len(files))
        with train_path.open("w",encoding="utf-8",newline="\n") as train_out,val_path.open("w",encoding="utf-8",newline="\n") as val_out:
            for did,path in files:
                local_bytes=0;source_counts=Counter()
                # Prefer original parquet row boundaries. Legacy converted text
                # lost article boundaries by treating blank paragraphs as docs.
                parquets=[path] if path in downloaded else sorted((root/"cache/hf"/f"hf-{did}").glob("*.parquet"))
                if parquets:
                    from fetch_datasets import iter_parquet_text
                    documents=(text for pq in parquets for text in iter_parquet_text(pq,["text"]))
                    source_counts["original_parquet_rows_used"] = 1
                else:
                    cache_texts=sorted((root/"cache/hf"/f"hf-{did}").glob("*.txt"))
                    if cache_texts:
                        documents=(text for cached in cache_texts for text in iter_documents(cached))
                        source_counts["original_text_records_used"] = 1
                    else:
                        documents=iter_documents(path)
                        source_counts["converted_text_boundaries"] = 1
                for text in documents:
                    counts["examined"]+=1
                    if counts["examined"]>500000:break
                    # Preserve document boundaries/split identity; exceptionally
                    # large articles are excluded, not fragmented across splits.
                    if len(text)>100000:
                        counts["oversized_document"]+=1;continue
                    text=unicodedata.normalize("NFC",text).strip()
                    reason=text_quality(text,domain)
                    if reason:counts[reason]+=1;continue
                    digest=hashlib.sha256(re.sub(r"\s+"," ",text).casefold().encode()).hexdigest()
                    if digest in seen:counts["duplicate"]+=1;continue
                    if guard.contaminated(text):counts["contaminated"]+=1;continue
                    encoded=(text+"\n<|endoftext|>\n").encode("utf-8")
                    if local_bytes+len(encoded)>per_file:break
                    seen.add(digest)
                    is_val=int(digest[:8],16)%50==0
                    (val_out if is_val else train_out).write(encoded.decode("utf-8"))
                    counts["val_docs" if is_val else "train_docs"]+=1
                    counts["val_bytes" if is_val else "train_bytes"]+=len(encoded)
                    source_counts["kept_docs"]+=1;source_counts["kept_bytes"]+=len(encoded);local_bytes+=len(encoded)
                    provenance.write(json.dumps({"content_sha256":digest,"dataset_id":did,"domain":domain,
                        "split":"val" if is_val else "train","license":catalog.get(did,{}).get("license"),
                        "origin":downloaded[path]["origin"] if path in downloaded else "unknown_legacy_origin",
                        "source_file":str(path.relative_to(root))},ensure_ascii=False)+"\n")
                report["sources"].append({"dataset_id":did,"path":str(path.relative_to(root)),"bytes":path.stat().st_size,
                    "license":catalog.get(did,{}).get("license","unknown"),"domain":domain,"prepared":dict(source_counts),
                    "acquisition":downloaded.get(path),
                    "provenance":source_metadata(path),"original_parquets":[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,
                        "provenance":source_metadata(p)} for p in parquets],
                    "note":"Source revisions only where available; viewer samples are not revision-pinned"})
        if not counts["train_docs"]:
            train_path.unlink();val_path.unlink();continue
        domains[domain]=str(train_path.resolve())
        report["pretrain"][domain]=dict(counts,train_sha256=sha256_file(str(train_path)),val_sha256=sha256_file(str(val_path)))
    provenance.close()
    validation=output/"pretrain"/"validation.txt"
    with validation.open("wb") as target:
        for path in sorted((output/"pretrain").glob("*_val.txt")):
            with path.open("rb") as source:shutil.copyfileobj(source,target)
    # SFT should preserve complete conversations and should not include NC or
    # uncertain-per-source licenses by default. Those stay in raw inventory.
    sft_seen=set();sft_files={}
    for path in sorted((root/"data/seed/hf").glob("*.jsonl")):
        did=next((e["id"] for e in sorted(catalog_list,key=lambda e:-len(e["id"])) if path.name.startswith(e["id"]+"-")),None)
        if did:sft_files.setdefault(did,[]).append(path)
    for did,paths in sft_files.items():
        print(f"[Prepare] SFT {did}: {len(paths)} shard(s)", flush=True)
        entry=catalog[did];license_=entry.get("license","")
        if ("NC" in license_.upper() and not allow_noncommercial) or license_.lower().startswith("various"):
            report["sft"][did]={"status":"excluded_by_default_license_policy","license":license_};continue
        counts=Counter();train_path=output/"sft_train"/(did+".jsonl");val_path=output/"sft_val"/(did+".jsonl")
        with train_path.open("w",encoding="utf-8",newline="\n") as train_out,val_path.open("w",encoding="utf-8",newline="\n") as val_out:
            # Deterministic reservoir sampling avoids always using only first N rows.
            reservoir=[];rng=random.Random(seed)
            for path,line in source_lines(paths):
                # Bounded scan for first preparation batch; reservoir is uniform
                # within this scanned prefix, not claimed globally uniform.
                if counts["examined"] >= 30000:
                    counts["scan_budget_reached"] = 1
                    break
                counts["examined"]+=1
                try:record=json.loads(line)
                except json.JSONDecodeError:counts["malformed"]+=1;continue
                msgs=record.get("messages")
                if not isinstance(msgs,list) or not all(isinstance(m,dict) and isinstance(m.get("content"),str) and m.get("role") in {"user","assistant","system"} for m in msgs):
                    counts["malformed"]+=1;continue
                if not any(m["role"]=="user" for m in msgs) or not msgs or msgs[-1]["role"]!="assistant":counts["malformed"]+=1;continue
                text="\n".join(m["content"] for m in msgs)
                if len(text)>16000 or len(text)<40:counts["length"]+=1;continue
                from grounding import INJECTION
                if INJECTION.search(text):counts["instruction_injection"]+=1;continue
                # This is a conservative heuristic, not a PII classifier.
                if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?:AKIA[0-9A-Z]{16}|-----BEGIN .*PRIVATE KEY)",text):
                    counts["possible_pii_or_secret"]+=1;continue
                if guard.contaminated(text):counts["contaminated"]+=1;continue
                digest=object_hash(msgs)
                if digest in sft_seen:counts["duplicate"]+=1;continue
                sft_seen.add(digest);counts["eligible"]+=1
                row={"messages":msgs,"provenance":{"dataset_id":did,"license":license_,"source_file":str(path.relative_to(root)),"conversation_sha256":digest}}
                if len(reservoir)<sft_per_source:reservoir.append(row)
                else:
                    index=rng.randrange(counts["eligible"])
                    if index<sft_per_source:reservoir[index]=row
            for row in reservoir:
                is_val=int(row["provenance"]["conversation_sha256"][:8],16)%20==0
                (val_out if is_val else train_out).write(json.dumps(row,ensure_ascii=False)+"\n")
                counts["val_conversations" if is_val else "train_conversations"]+=1
        report["sft"][did]=dict(counts,license=license_,sampling="seeded reservoir within up to 30000 scanned rows",
            train_sha256=sha256_file(str(train_path)),val_sha256=sha256_file(str(val_path)),
            sources=[{"path":str(p.relative_to(root)),"provenance":source_metadata(p)} for p in paths])
    # Reuse vetted TRAIN-only problem/solution pairs as continued-pretrain
    # math/code text; no new invented answers and no execution of external code.
    for domain,source_ids in (("math",("gsm8k-train","metamathqa")),("code",("opencodeinstruct",))):
        target=output/"pretrain"/(domain+".txt")
        if target.exists():continue
        counts=Counter()
        with target.open("w",encoding="utf-8",newline="\n") as handle:
            for did in source_ids:
                source=output/"sft_train"/(did+".jsonl")
                if not source.exists():continue
                with source.open(encoding="utf-8") as rows:
                    for line in rows:
                        row=json.loads(line)
                        text="\n".join(m["content"] for m in row["messages"])
                        record=text+"\n<|endoftext|>\n";size=len(record.encode("utf-8"))
                        if counts["train_bytes"]+size>cap:break
                        handle.write(record);counts["train_bytes"]+=size;counts["train_docs"]+=1
        if counts["train_docs"]:
            domains[domain]=str(target.resolve())
            report["pretrain"][domain]=dict(counts,train_sha256=sha256_file(str(target)),
                origin="prepared SFT TRAIN pairs only; external answers not locally proven",validation="separate SFT val pairs")
            with validation.open("a",encoding="utf-8",newline="\n") as val_out:
                for did in source_ids:
                    source=output/"sft_val"/(did+".jsonl")
                    if source.exists():
                        with source.open(encoding="utf-8") as rows:
                            for line in rows:
                                row=json.loads(line)
                                val_out.write("\n".join(m["content"] for m in row["messages"])+"\n<|endoftext|>\n")
        else:target.unlink()
    if not domains:raise ValueError("No usable existing pretrain sources; see inventory")
    active={d:WEIGHTS[d] for d in domains};total=sum(active.values())
    stages=[{"name":"balanced_vi_en","start":0,"end":1,"weights":{d:w/total for d,w in active.items()}}]
    mixture={"status":"materialized","domains":domains,"validation_path":str(validation.resolve()),"stages":stages,
             "provenance_report":str((output/"manifest.json").resolve()),"seed":seed,"split":"content_hash_document_disjoint","requires_manifest_status":"ready"}
    atomic_json(str(output/"mixture.json"),mixture)
    report.update(status="ready",mixture_hash=object_hash(mixture),validation_sha256=sha256_file(str(validation)),
                  total_pretrain_bytes=sum(Path(p).stat().st_size for p in domains.values())+validation.stat().st_size,
                  total_sft_train=sum(r.get("train_conversations",0) for r in report["sft"].values()))
    atomic_json(str(output/"manifest.json"),report)
    return report


def verify_prepared(output:Path)->dict:
    """Content hashes and exact split-disjointness, no generation or training."""
    report=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    if report.get("status")!="ready":raise ValueError("Prepared dataset is not ready")
    mixture=json.loads((output/"mixture.json").read_text(encoding="utf-8"))
    if object_hash(mixture)!=report["mixture_hash"]:raise ValueError("Mixture hash mismatch")
    hashes={"train":set(),"val":set()}
    with (output/"pretrain_sources.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            rec=json.loads(line);hashes[rec["split"]].add(rec["content_sha256"])
    if hashes["train"]&hashes["val"]:raise ValueError("Pretrain train/val leakage")
    for domain,r in report["pretrain"].items():
        if sha256_file(str(output/"pretrain"/(domain+".txt")))!=r["train_sha256"]:
            raise ValueError(f"Hash mismatch: {domain}")
        if r.get("val_sha256") and sha256_file(str(output/"pretrain"/(domain+"_val.txt")))!=r["val_sha256"]:
            raise ValueError(f"Validation domain hash mismatch: {domain}")
    actual={"train":set(),"val":set()}
    for split,paths in (("train",[output/"pretrain"/(d+".txt") for d in report["pretrain"]]),
                        ("val",[output/"pretrain/validation.txt"])):
        for path in paths:
            for text in iter_documents(path):
                actual[split].add(hashlib.sha256(re.sub(r"\s+"," ",text).casefold().encode()).hexdigest())
    if actual["train"]&actual["val"]:raise ValueError("Pretrain content train/val leakage")
    if any(not hashes[s].issubset(actual[s]) for s in hashes):raise ValueError("Provenance does not match pretrain content")
    sft_hashes={"train":set(),"val":set()};counts=Counter()
    for split in ("train","val"):
        for path in (output/("sft_"+split)).glob("*.jsonl"):
            did=path.stem;r=report["sft"][did]
            if sha256_file(str(path))!=r[split+"_sha256"]:raise ValueError("SFT output hash mismatch")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record=json.loads(line);digest=object_hash(record["messages"])
                    if digest!=record["provenance"]["conversation_sha256"]:raise ValueError("Conversation hash mismatch")
                    if digest in sft_hashes[split]:raise ValueError("Duplicate conversation")
                    sft_hashes[split].add(digest);counts[split]+=1
    if sft_hashes["train"]&sft_hashes["val"]:raise ValueError("SFT train/val leakage")
    if sha256_file(str(output/"pretrain/validation.txt"))!=report["validation_sha256"]:raise ValueError("Validation hash mismatch")
    result={"status":"verified","pretrain_source_hashes":{k:len(v) for k,v in hashes.items()},
            "pretrain_content_hashes":{k:len(v) for k,v in actual.items()},
            "sft_conversations":dict(counts),"exact_split_overlap":0,
            "limitations":"Exact document/conversation hash disjointness; paraphrase/near-duplicate families not proven disjoint"}
    atomic_json(str(output/"verification.json"),result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--inventory",action="store_true");p.add_argument("--prepare",action="store_true");p.add_argument("--verify",action="store_true")
    p.add_argument("--output-dir",default="data/prepared_v1");p.add_argument("--mb-per-domain",type=float,default=32)
    p.add_argument("--sft-per-source",type=int,default=2000);p.add_argument("--seed",type=int,default=1337)
    p.add_argument("--allow-noncommercial",action="store_true",help="Explicit personal/research profile incl. Bactrian-X VI; resulting data has NC restrictions")
    p.add_argument("--download-report",help="Import completed checksum-verified parquet batch; no copying or generated labels")
    a=p.parse_args();root=Path(config.ROOT_DIR)
    if a.inventory:
        result=inventory(root);atomic_json(str(root/"bench_results/data_inventory.json"),result)
    elif a.prepare:result=prepare(root,root/a.output_dir,mb_per_domain=a.mb_per_domain,sft_per_source=a.sft_per_source,seed=a.seed,allow_noncommercial=a.allow_noncommercial,download_report=Path(a.download_report).resolve() if a.download_report else None)
    elif a.verify:result=verify_prepared(root/a.output_dir)
    else:p.error("Choose --inventory, --prepare or --verify")
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=="__main__":main()
