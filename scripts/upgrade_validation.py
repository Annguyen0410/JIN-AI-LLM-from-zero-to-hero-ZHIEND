"""One-shot validation artifacts; never trains or replaces the served model."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    torch.set_num_threads(2)
    from lineage import sha256_file,object_hash,atomic_json
    from bpe_tokenizer import BPETokenizer
    from chat_dataset import build_dataset
    import config,train
    from data_quality import load_probe_texts,check_contamination
    report={"served_weights_modified":False,"scope":"CPU training integration; NOT capability training"}
    with tempfile.TemporaryDirectory(prefix="jin-trainer-test-") as td:
        td=Path(td);seed=td/"seed";seed.mkdir()
        corpus="\n".join(f"Training example {i}: oranges cost {i+2} credits and the basket is blue." for i in range(40))
        tok=BPETokenizer();tok.train(corpus,vocab_size=300);tok.save(str(td/"bpe.json"))
        (td/"data.txt").write_text(corpus*20,encoding="utf-8")
        convs=[{"messages":[{"role":"user","content":f"Describe training example number {i} for this fictional inventory."},
                            {"role":"assistant","content":f"Example {i} has {i+2} oranges. The basket is blue and this is a fictional test."}]} for i in range(24)]
        (seed/"conversations.jsonl").write_text("\n".join(json.dumps(c) for c in convs),encoding="utf-8")
        tiny={"embed_dim":32,"n_layers":1,"n_heads":2,"n_kv_heads":2,"block_size":64,"dropout":0,"modern":True,"qk_norm":True}
        with mock.patch.multiple(config, BPE_PATH=str(td/"bpe.json"),CACHE_DIR=str(td/"cache"),
            CHECKPOINT_DIR=str(td/"checkpoints"),BEST_CHECKPOINT_PATH=str(td/"none.pt"),
            SEED_CONVERSATIONS_PATH=str(seed/"conversations.jsonl"),DEVICE="cpu"), \
            mock.patch.object(config,"get_model_config",side_effect=lambda vocab:dict(tiny,vocab_size=vocab)), \
            mock.patch.object(train,"sample_from_model"),mock.patch.object(train,"_quick_chat_sample"):
            for mode in ("pretrain","instruct"):
                output=str(td/(mode+".pt"))
                argv=["train.py","--mode",mode,"--device","cpu","--checkpoint",output,
                      "--data",str(td/"data.txt"),"--block-size","32","--batch-size","1","--grad-accum","1",
                      "--run-iters","2","--steps","2","--eval-interval","1","--eval-iters","1",
                      "--checkpoint-interval","1","--no-pretrained-init","--duplication","2",
                      "--hf-seed-dir","","--grounded-dir","","--val-fraction","0.25"]
                with mock.patch.object(sys,"argv",argv):
                    args=train.parse_args()
                (train.train_pretrain if mode=="pretrain" else train.train_instruct)(args)
                from utils import load_checkpoint_file
                ck=load_checkpoint_file(output,map_location="cpu")
                run=ck["lineage"]
                assert ck["iteration"]==2 and run["tokens_seen_this_run"]==64
                assert ck["val_loss"] > 0 and run["supervised_tokens_this_run"] > 0
                if mode=="instruct":
                    split=run["dataset_split"]
                    assert not set(split["train_conversation_hashes"]) & set(split["val_conversation_hashes"])
                report[mode]={"step":ck["iteration"],"tokens":run["tokens_seen_this_run"],
                              "supervised_tokens":run["supervised_tokens_this_run"],"val_loss":ck["val_loss"],
                              "manifest_path":run["manifest_path"],"verified":True}
    # Exercise both HTTP answer modes using the actual served checkpoint,
    # persistent memory, and deterministic local evidence (no live web claim).
    import app
    from utils import build_inference_bundle
    from retriever import HybridRetriever,VectorIndex,BM25Index
    mdl,tok,meta=build_inference_bundle(str(ROOT/"checkpoints/jingpt_chat.pt"),device="cpu")
    with tempfile.TemporaryDirectory(prefix="jin-api-test-") as td:
        store=__import__("memory").MemoryStore(str(Path(td)/"memory.jsonl"))
        store.add(tier="semantic",text="Hà Nội là thủ đô của Việt Nam.",source_url="https://example.org/vietnam")
        with mock.patch.multiple(app,model=mdl,tokenizer=tok,checkpoint_meta=meta,MEMORY_STORE=store,
             MEMORY_RETRIEVER=HybridRetriever(VectorIndex(mdl.embed_dim),BM25Index()),MEMORY_ENABLED=True), \
             mock.patch.object(app,"_rate_limit_ok",return_value=True),mock.patch.object(app,"_auth_ok",return_value=True):
            client=app.app.test_client()
            memory=client.post("/api/memory/search",json={"query":"thủ đô Việt Nam"})
            qa=client.post("/api/chat",json={"message":"Thủ đô Việt Nam là gì?","memory":True,"web_search":False})
            stream=client.post("/api/chat/stream",json={"message":"Thủ đô Việt Nam là gì?","memory":True,"web_search":False})
            calc=client.post("/api/agent",json={"message":"Calculate (3+4)*9","web_search":False})
            assert "Hà Nội" in qa.json["response"] and qa.status_code==200
            stream_records=[json.loads(line[6:]) for line in stream.data.decode().splitlines() if line.startswith("data: ")]
            assert memory.json["count"]==1 and any("Hà Nội" in record.get("reply",record.get("delta","")) for record in stream_records)
            assert calc.json["answer"]=="63"
            raw=client.post("/api/chat",json={"message":"Hello","memory":False,"web_search":False,"grounded":False,"max_tokens":8,"temperature":0})
            assert raw.status_code==200 and raw.json["tokens_generated"]>0
            report["actual_checkpoint_api"]={"memory_search":True,"grounded_sync":True,"grounded_sse":True,
                "verified_calculator":True,"model_generation":True,"request_id_header":bool(raw.headers.get("X-Request-ID"))}
    # Audit existing grounded shards without rewriting user's dataset.
    hashes=set();rows=dupes=0
    for path in sorted((ROOT/"data/seed/grounded").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            try:record=json.loads(line)
            except json.JSONDecodeError:continue
            rows+=1;digest=object_hash(record.get("messages",[]))
            if digest in hashes:dupes+=1
            hashes.add(digest)
    report["grounded_data_audit"]={"rows":rows,"exact_duplicates":dupes,"unique":len(hashes),
        "quality_warning":"Legacy answers were paired with unrelated search results; citations are not entailment verification. Rebuild into a separate directory."}
    report["checkpoint_inventory"]=[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size}
                                     for p in (ROOT/"checkpoints").glob("*.pt")]
    report["recovery"]={"source_zip_exists":(ROOT/".freebuff/upgrade_recovery/source.zip").exists(),
                        "index_snapshot_exists":(ROOT/".freebuff/upgrade_recovery/git-index.snapshot").exists()}
    atomic_json(str(ROOT/"bench_results/upgrade_validation.json"),report)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
