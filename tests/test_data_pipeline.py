from __future__ import annotations
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import fetch_datasets as fetch
from data_pipeline import prepare,iter_documents,ContaminationGuard,verify_prepared


class Response(io.BytesIO):
    def __init__(self,body,total=None):
        super().__init__(body);self.headers={}
        if total is not None:self.headers["Content-Length"]=str(total)
    def geturl(self):return "https://example.org/train"


class DownloadTests(unittest.TestCase):
    def test_complete_download_hash_and_manifest(self):
        with tempfile.TemporaryDirectory() as td,mock.patch.object(fetch,"open_url",return_value=Response(b"abcdef",6)):
            p=Path(td)/"file.json";fetch.download_with_progress("https://example.org/data",p,limit_bytes=6)
            self.assertEqual(p.read_bytes(),b"abcdef")
            m=json.loads(Path(str(p)+".manifest.json").read_text());self.assertEqual(m["status"],"complete")
            self.assertFalse(p.with_suffix(".json.part").exists())
    def test_budget_does_not_publish_truncated_json_or_parquet(self):
        for total in (None,100):
            with tempfile.TemporaryDirectory() as td,mock.patch.object(fetch,"open_url",return_value=Response(b"x"*100,total)):
                p=Path(td)/"file.parquet"
                with self.assertRaises(ValueError):fetch.download_with_progress("https://example.org/data",p,limit_bytes=8)
                self.assertFalse(p.exists());self.assertFalse(p.with_suffix(".parquet.part").exists())
    def test_short_response_fails_and_existing_files_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"file.txt"
            with mock.patch.object(fetch,"open_url",return_value=Response(b"x",100)):
                with self.assertRaises(ValueError):fetch.download_with_progress("https://example.org/data",p)
            p.write_text("owned")
            with self.assertRaises(FileExistsError):fetch.download_with_progress("https://example.org/data",p)
            self.assertEqual(p.read_text(),"owned")
    def test_existing_partial_not_deleted(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"file.txt";partial=p.with_suffix(".txt.part");partial.write_text("recover me")
            with self.assertRaises(FileExistsError):fetch.download_with_progress("https://example.org/data",p)
            self.assertEqual(partial.read_text(),"recover me")
    def test_remote_shards_pinned_and_test_split_excluded(self):
        info={"sha":"a"*40,"siblings":[{"rfilename":"data/train-00000.parquet","size":123},
                {"rfilename":"data/test-00000.parquet","size":456}]}
        e={"hf_repo":"owner/repo","file_pattern":r"data/train-.*\.parquet","max_shards":1}
        with mock.patch.object(fetch,"open_url",return_value=Response(json.dumps(info).encode())):
            urls=fetch.resolve_hf_files(e)
        self.assertEqual(len(urls),1);self.assertIn("a"*40,urls[0]);self.assertNotIn("test",urls[0])
    def test_raw_writer_utf8_budget_and_eot(self):
        with tempfile.TemporaryDirectory() as td:
            n,size=fetch.write_raw_text_shard(Path(td),"vi","fixture",["Tiếng Việt "*20],max_bytes=1000)
            p=Path(td)/"hf-vi/fixture.txt"
            self.assertEqual(size,p.stat().st_size);self.assertEqual(n,1);self.assertIn("<|endoftext|>",p.read_text(encoding="utf-8"))
    def test_parquet_qa_filter(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"code.parquet"
            pq.write_table(pa.Table.from_pylist([{"input":"Question","output":"Answer","average_test_score":"1.0"},
                {"input":"Bad","output":"Unverified","average_test_score":"0.5"}]),p)
            rows=list(fetch.iter_parquet_qa(p,{"question_column":"input","answer_column":"output","quality_column":"average_test_score","min_quality":1}))
            self.assertEqual(len(rows),1)
    def test_gzip_json_array_stream(self):
        import gzip
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"rows.json.gz"
            with gzip.open(p,"wt",encoding="utf-8") as h:json.dump([{"instruction":"Q","output":"A"}],h)
            self.assertEqual(len(list(fetch.iter_alpaca_json(p))),1)


class PreparationTests(unittest.TestCase):
    def test_eot_preserves_blank_paragraphs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"rows.txt";p.write_text("First paragraph\n\nSecond paragraph\n<|endoftext|>\nNext document\n<|endoftext|>\n")
            docs=list(iter_documents(p));self.assertEqual(len(docs),2);self.assertIn("Second",docs[0])
    def test_eot_long_record_not_fragmented_and_oversized_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"rows.txt"
            long_record="A whole article "*2000
            p.write_text(long_record+"\n<|endoftext|>\n"+"x"*100001+"\n<|endoftext|>\nLast document\n<|endoftext|>\n")
            docs=list(iter_documents(p))
            self.assertEqual(docs,[long_record.strip(),"Last document"])
    def test_mixture_refuses_building_and_hash_changes(self):
        from utils import MixtureSampler
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=root/"manifest.json";mixture=root/"mix.json"
            spec={"requires_manifest_status":"ready","provenance_report":str(manifest),"domains":{}}
            mixture.write_text(json.dumps(spec));manifest.write_text(json.dumps({"status":"building"}))
            with self.assertRaisesRegex(ValueError,"not ready"):MixtureSampler(None,str(mixture))
            manifest.write_text(json.dumps({"status":"ready","mixture_hash":"incorrect"}))
            with self.assertRaisesRegex(ValueError,"differs"):MixtureSampler(None,str(mixture))
    def test_prepared_mixture_split_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"data/raw/hf-wikipedia-vi-sample").mkdir(parents=True)
            (root/"data/seed/hf").mkdir(parents=True);(root/"data/eval").mkdir()
            (root/"data/eval/probes.jsonl").write_text(json.dumps({"prompt":"Forbidden exact question in benchmark","reference":"known reference answer with several words"})+"\n")
            catalog={"datasets":[{"id":"wikipedia-vi-sample","license":"CC-BY-SA-3.0"},{"id":"dolly-15k","license":"CC-BY-SA-3.0"}]}
            (root/"hf_datasets.json").write_text(json.dumps(catalog))
            texts=[f"Bài viết tiếng Việt số {i} nói về khoa học tự nhiên và giáo dục cho học sinh. Đây là nội dung kiểm thử độc lập." for i in range(120)]
            (root/"data/raw/hf-wikipedia-vi-sample/train.txt").write_text("\n<|endoftext|>\n".join(texts),encoding="utf-8")
            records=[{"messages":[{"role":"user","content":f"Explain example number {i} in this fictional collection"},{"role":"assistant","content":f"This fictional example {i} has a valid response for checking data preparation."}]} for i in range(50)]
            source=root/"data/seed/hf/dolly-15k-fixture.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in records+records[:3]),encoding="utf-8")
            second_shard=root/"data/seed/hf/dolly-15k-fixture2.jsonl"
            second_shard.write_text(json.dumps({"messages":[{"role":"user","content":"Explain a unique question from the second shard"},
                {"role":"assistant","content":"This second shard must be combined, never overwrite the first shard."}]}))
            output=root/"prepared";report=prepare(root,output,mb_per_domain=.1,sft_per_source=40)
            mixture=json.loads((output/"mixture.json").read_text());self.assertEqual(mixture["status"],"materialized")
            provenance=[json.loads(l) for l in (output/"pretrain_sources.jsonl").read_text().splitlines()]
            self.assertFalse({p["content_sha256"] for p in provenance if p["split"]=="train"}&{p["content_sha256"] for p in provenance if p["split"]=="val"})
            train_rows=[json.loads(l) for l in (output/"sft_train/dolly-15k.jsonl").read_text().splitlines()]
            val_rows=[json.loads(l) for l in (output/"sft_val/dolly-15k.jsonl").read_text().splitlines()]
            self.assertFalse({r["provenance"]["conversation_sha256"] for r in train_rows}&{r["provenance"]["conversation_sha256"] for r in val_rows})
            self.assertEqual(report["sft"]["dolly-15k"]["duplicate"],3)
            self.assertEqual(len(report["sft"]["dolly-15k"]["sources"]),2)
            self.assertEqual(report["sft"]["dolly-15k"]["eligible"],51)
            self.assertEqual(verify_prepared(output)["exact_split_overlap"],0)
            from utils import MixtureSampler
            import torch
            with mock.patch("utils.load_or_create_token_data_from_file",return_value=(torch.arange(100),0,True)):
                sampler=MixtureSampler(None,str(output/"mixture.json"),block_size=8,batch_size=2,device="cpu")
                self.assertEqual(sampler.batch(.5)[0].shape,(2,8))
            train_file=output/"pretrain/vietnamese.txt"
            with train_file.open("a",encoding="utf-8") as handle:handle.write("tampered")
            with self.assertRaisesRegex(ValueError,"Hash mismatch"):verify_prepared(output)
            with self.assertRaisesRegex(ValueError,"domain hash mismatch"):MixtureSampler(None,str(output/"mixture.json"))
            with self.assertRaises(ValueError):prepare(root,output)
    def test_explicit_sft_validation_used_and_overlap_rejected(self):
        import config
        from bpe_tokenizer import BPETokenizer
        from chat_dataset import build_dataset
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"seed").mkdir();(root/"train").mkdir();(root/"val").mkdir()
            seed=root/"seed/conversations.jsonl";seed.write_text("")
            tokenizer_path=root/"bpe.json";BPETokenizer().save(str(tokenizer_path))
            def row(i):
                return {"messages":[{"role":"user","content":f"Explain fictional sample {i} for a unit test."},
                    {"role":"assistant","content":f"Sample {i} is a fictional example for testing disjoint conversations. "*4}]}
            train_file=root/"train/a.jsonl";val_file=root/"val/a.jsonl"
            train_file.write_text("\n".join(json.dumps(row(i)) for i in range(4)))
            val_file.write_text("\n".join(json.dumps(row(i)) for i in range(4,6)))
            with mock.patch.object(config,"BPE_PATH",str(tokenizer_path)),mock.patch("data_quality.check_contamination",side_effect=lambda docs:(docs,[])):
                train,val,_=build_dataset(str(seed),(str(root/"train"),),block_size=32,duplication=5,
                    val_fraction=.2,validation_dirs=(str(root/"val"),))
                self.assertEqual(train.split_manifest["method"],"explicit_conversation_disjoint_validation")
                self.assertEqual(len(train.split_manifest["train_conversation_hashes"]),4)
                self.assertEqual(len(train.split_manifest["val_conversation_hashes"]),2)
                self.assertGreater(len(val),0)
                val_file.write_text(json.dumps(row(0)))
                with self.assertRaisesRegex(ValueError,"overlap"):
                    build_dataset(str(seed),(str(root/"train"),),block_size=32,validation_dirs=(str(root/"val"),))
    def test_explicit_sft_validation_empty_fails(self):
        import config
        from bpe_tokenizer import BPETokenizer
        from chat_dataset import build_dataset
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);seed=root/"conversations.jsonl";seed.write_text("")
            tokenizer_path=root/"bpe.json";BPETokenizer().save(str(tokenizer_path))
            with mock.patch.object(config,"BPE_PATH",str(tokenizer_path)),mock.patch("data_quality.check_contamination",return_value=([],[])):
                with self.assertRaisesRegex(ValueError,"no JSONL"):
                    build_dataset(str(seed),validation_dirs=(str(root/"missing"),))
    def test_download_batch_ingestion_and_integrity(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from lineage import sha256_file
        from data_pipeline import downloaded_sources
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/"data/eval").mkdir(parents=True)
            (root/"data/eval/probes.jsonl").write_text(json.dumps({"prompt":"Protected benchmark sentinel"})+"\n")
            (root/"hf_datasets.json").write_text(json.dumps({"datasets":[]}))
            source=root/"train.parquet"
            pq.write_table(pa.Table.from_pylist([{"text":f"Article {i} about fictional natural science examples and education. This original document is used only for testing ingestion and disjoint splits."} for i in range(250)]),source)
            rec={"local_path":str(source),"bytes":source.stat().st_size,"sha256":sha256_file(str(source)),
                 "expected_sha256":sha256_file(str(source)),"dataset_id":"fixture-knowledge","remote_path":"train.parquet","domain":"knowledge",
                 "origin":"human_documented","license":"fixture","revision":"pinned"}
            path=root/"downloads.json"
            value={"status":"downloaded_not_prepared_not_trained","files":[rec],"completed":[rec],"failures":[]}
            path.write_text(json.dumps(value))
            out=root/"prepared"
            report=prepare(root,out,mb_per_domain=.1,sft_per_source=1,download_report=path)
            self.assertIn("knowledge",report["pretrain"])
            self.assertIn("Not enforced",report["origin_mixture_status"])
            self.assertEqual(verify_prepared(out)["exact_split_overlap"],0)
            provenance=[json.loads(l) for l in (out/"pretrain_sources.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["origin"]=="human_documented" for r in provenance))
            value["status"]="paused_partial_file";path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"not complete"):downloaded_sources(path,root)
            self.assertFalse((root/"bad").exists())
            value["status"]="downloaded_not_prepared_not_trained";path.write_text(json.dumps(value))
            source.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"hash/size"):downloaded_sources(path,root)

    def test_contamination_guard(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"probes.jsonl";p.write_text(json.dumps({"prompt":"These exact six words are forbidden"})+"\n")
            g=ContaminationGuard(str(p));self.assertTrue(g.contaminated("prefix These exact six words are forbidden suffix"))
            self.assertFalse(g.contaminated("Something else entirely"))

if __name__=="__main__":unittest.main()
