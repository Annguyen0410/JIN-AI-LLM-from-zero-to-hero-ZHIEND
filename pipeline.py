"""
Unified pipeline orchestrator for JIN AI.

Merges the functionality of the old ``train_jin_v2.py`` and
``auto_pipeline.py`` into a single command.

Modes:
    ``chat``   build instruct dataset -> train BPE -> train (instruct)
    ``raw``    fetch raw corpora -> build dataset -> optionally train (pretrain)
    ``evolve`` bench baseline -> run curiosity cycles -> gate report
              (Phase 5; needs a real train/eval provider — see
              ``learn_daemon.run_daemon``)
    ``bootstrap`` the full v3 path: data -> BPE -> pretrain -> SFT ->
              DPO -> bench (Phase 7)

Examples:
    python pipeline.py chat
    python pipeline.py chat --resume --skip-bpe
    python pipeline.py raw --train --resume
    python pipeline.py raw --only gutenberg --max-sources 3 --skip-build
    python pipeline.py evolve --cycles 5 --topics data/curiosity/topics.txt
    python pipeline.py bootstrap --steps 500 --pref-steps 200 --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import config
from build_dataset import build_corpus
from fetch_datasets import fetch_source, load_sources


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.check_call(cmd, cwd=config.ROOT_DIR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JIN AI pipeline orchestrator")
    p.add_argument("mode", choices=("chat", "raw", "evolve", "bootstrap"),
                   help="``chat`` builds instruct data + BPE + instruct-train. "
                        "``raw``  fetches + builds pretrain data + optionally trains. "
                        "``evolve`` runs curiosity cycles (Phase 5). "
                        "``bootstrap`` full v3 path: data -> BPE -> pretrain -> SFT -> DPO -> bench.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", type=str, default=None, metavar="NAME",
                   choices=config.DEVICE_CHOICES)
    # Chat mode options
    p.add_argument("--skip-build-data", action="store_true")
    p.add_argument("--skip-bpe", action="store_true")
    p.add_argument("--vocab-size", type=int, default=config.BPE_VOCAB_SIZE)
    p.add_argument("--synth-samples", type=int, default=4_000)
    p.add_argument("--literary-target-mb", type=float, default=15.0)
    p.add_argument("--run-iters", type=int, default=config.RUN_ITERS)
    p.add_argument("--max-iters", type=int, default=None)
    # Raw mode options
    p.add_argument("--sources", type=str, default=os.path.join(config.ROOT_DIR, "sources.json"))
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--max-sources", type=int, default=None)
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--checkpoint", type=str, default=config.LATEST_CHECKPOINT_PATH)
    # Evolve mode (Phase 5)
    p.add_argument("--cycles", type=int, default=config.CURIOSITY_MAX_CYCLES)
    p.add_argument("--topics", type=str, default=None,
                   help="File with one curiosity topic per line.")
    p.add_argument("--dry-run", action="store_true",
                   help="Evolve: plan cycles without training.")
    # Bootstrap mode (Phase 7)
    p.add_argument("--steps", type=int, default=500,
                   help="Bootstrap: SFT steps after pretrain.")
    p.add_argument("--pref-steps", type=int, default=200,
                   help="Bootstrap: DPO/ORPO steps after SFT."
                        " (0 = skip preference tuning)")
    p.add_argument("--pref-method", choices=("orpo", "dpo"), default="orpo")
    p.add_argument("--pref-data", type=str, default=None,
                   help="Bootstrap: preference pairs JSONL (default: generate "
                        "from self_distill on the instruct data).")
    p.add_argument("--keep-pretrain", action="store_true",
                   help="Bootstrap: keep the pretrain checkpoint as the SFT base.")
    return p.parse_args()


def pipeline_chat(args: argparse.Namespace) -> None:
    if not args.skip_build_data or not os.path.exists(config.CHAT_DATA_PATH):
        run([sys.executable, os.path.join(config.ROOT_DIR, "build_instruct_dataset.py"),
             "--synth-samples", str(args.synth_samples),
             "--literary-target-mb", str(args.literary_target_mb),
             "--output", config.CHAT_DATA_PATH])
    else:
        print(f"[Pipeline] Using existing {config.CHAT_DATA_PATH}")

    if not args.skip_bpe and not os.path.exists(config.BPE_PATH):
        run([sys.executable, os.path.join(config.ROOT_DIR, "bpe_tokenizer.py"),
             "--data", config.CHAT_DATA_PATH, "--out", config.BPE_PATH,
             "--vocab-size", str(args.vocab_size)])
    else:
        if os.path.exists(config.BPE_PATH):
            print(f"[Pipeline] Using existing BPE at {config.BPE_PATH}")

    train_cmd = [sys.executable, os.path.join(config.ROOT_DIR, "train.py"),
                 "--mode", "instruct", "--data", config.CHAT_DATA_PATH,
                 "--run-iters", str(args.run_iters)]
    if args.resume:
        train_cmd.append("--resume")
    if args.max_iters is not None:
        train_cmd.extend(["--max-iters", str(args.max_iters)])
    if args.device is not None:
        train_cmd.extend(["--device", args.device])
    run(train_cmd)


def pipeline_raw(args: argparse.Namespace) -> None:
    if not args.skip_fetch:
        sources = load_sources(args.sources)
        selected = [s for s in sources if s.get("enabled", True)]
        only_set = set(args.only or [])
        if only_set:
            selected = [s for s in selected if s["id"] in only_set]
        if args.max_sources:
            selected = selected[:args.max_sources]
        if not selected:
            raise RuntimeError("No enabled sources matched.")
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        from fetch_datasets import fetch_source
        for src in selected:
            try:
                fetch_source(src, output_dir=config.RAW_DATA_DIR)
            except Exception as exc:
                print(f"[Error] {src.get('id')}: {exc}")

    if not args.skip_build:
        import shutil as _shutil

        building = config.DATA_PATH + ".building"
        had_existing = os.path.exists(config.DATA_PATH) and os.path.getsize(config.DATA_PATH) > 0
        existing_chars = 0
        with open(building, "w", encoding="utf-8", newline="\n") as out:
            if had_existing:
                # Stream-copy the existing corpus instead of read() (multi-GB safe).
                with open(config.DATA_PATH, "r", encoding="utf-8", errors="replace") as src:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        existing_chars += len(chunk)
                out.write("\n\n<doc path=\"existing/train.txt\">\n")
            stats = build_corpus(source_root=config.RAW_DATA_DIR, min_file_bytes=8_192,
                                  min_line_chars=8, max_line_chars=1_200,
                                  output_path=building)
        if not stats and existing_chars == 0:
            building and os.path.exists(building) and os.remove(building)
            raise RuntimeError("No usable corpus files found.")
        os.replace(building, config.DATA_PATH)
        os.makedirs(config.DATA_SNAPSHOT_DIR, exist_ok=True)
        snap = os.path.join(config.DATA_SNAPSHOT_DIR,
                            f"train_auto_{time.strftime('%Y%m%d_%H%M%S')}_{os.path.basename(config.DATA_PATH)}")
        _shutil.copyfile(config.DATA_PATH, snap)
        size_mb = os.path.getsize(config.DATA_PATH) / 1e6
        print(f"[Pipeline] Built -> {config.DATA_PATH} ({size_mb:.2f} MB, {len(stats)} new files)")
        print(f"[Pipeline] Snapshot -> {snap}")

    if args.train:
        cmd = [sys.executable, os.path.join(config.ROOT_DIR, "train.py"),
               "--run-iters", str(args.run_iters)]
        if args.resume:
            cmd.append("--resume")
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        if args.device:
            cmd.extend(["--device", args.device])
        raise SystemExit(subprocess.call(cmd, cwd=config.ROOT_DIR))


def pipeline_evolve(args: argparse.Namespace) -> None:
    """Phase 5: bench baseline -> curiosity cycles -> gate report.

    The train/eval providers are wired by ``learn_daemon.make_engine``;
    without a real train provider the cycles run in ``--dry-run`` mode
    (plan everything, touch nothing). The report is the ledger summary.
    """
    from learn_daemon import run_daemon

    topics = []
    if args.topics and os.path.exists(args.topics):
        with open(args.topics, "r", encoding="utf-8") as f:
            topics = [ln.strip() for ln in f if ln.strip()]
    if not topics:
        topics = ["quantum computing", "Vietnam history", "Fibonacci sequence"]

    # Optional baseline bench on the current checkpoint.
    if not args.dry_run and os.path.exists(args.checkpoint):
        run([sys.executable, os.path.join(config.ROOT_DIR, "bench.py"),
             "--checkpoint", args.checkpoint, "--no-train-bench"])

    results = run_daemon(topics, cycles=args.cycles, dry_run=args.dry_run,
                         seed=1337, device=args.device or "cpu")
    merged = sum(1 for r in results if r.get("decision") == "merged")
    rolled = sum(1 for r in results if r.get("decision") == "rolled_back")
    print("=" * 72)
    print(f"[Evolve] {len(results)} cycles | merged={merged} rolled_back={rolled} "
          f"| see data/curiosity/ledger.jsonl")
    print("=" * 72)


def pipeline_bootstrap(args: argparse.Namespace) -> None:
    """Phase 7: full v3 path — data -> BPE -> pretrain -> SFT -> DPO -> bench.

    Each stage is the existing CLI (same flags, same checkpoints), so any
    stage can be resumed/skipped by re-running with the corresponding
    ``--skip-*``/existing-file checks. The chain:

      1. corpus   : fetch raw sources -> build data/train.txt (if missing)
      2. BPE      : train the byte-level tokenizer on that corpus
      3. pretrain : train.py --mode pretrain on data/train.txt
      4. SFT      : build instruct data -> train.py --mode instruct
                    (base = the pretrain checkpoint)
      5. DPO      : train_pref.py --method {orpo,dpo} on preference pairs
                    (pairs come from self_distill on the instruct data)
      6. bench    : bench.py on the final instruct checkpoint
    """
    root = config.ROOT_DIR

    # 1. Corpus (pretrain data) — reuse the raw-mode fetch+build.
    if not os.path.exists(config.DATA_PATH):
        print("[bootstrap] Step 1/6 — fetching + building pretrain corpus")
        pipeline_raw(args)
    else:
        print(f"[bootstrap] Step 1/6 — corpus exists: {config.DATA_PATH}")

    # 2. BPE on the corpus.
    if not os.path.exists(config.BPE_PATH):
        print("[bootstrap] Step 2/6 — training BPE tokenizer")
        run([sys.executable, os.path.join(root, "bpe_tokenizer.py"),
             "--data", config.DATA_PATH, "--out", config.BPE_PATH,
             "--vocab-size", str(args.vocab_size)])
    else:
        print(f"[bootstrap] Step 2/6 — BPE exists: {config.BPE_PATH}")

    # 3. Pretrain (raw next-token on the corpus).
    pretrain_ckpt = args.checkpoint
    if args.keep_pretrain:
        print("[bootstrap] Step 3/6 — SKIPPED (--keep-pretrain)")
    else:
        print("[bootstrap] Step 3/6 — pretrain")
        cmd = [sys.executable, os.path.join(root, "train.py"),
               "--run-iters", str(args.run_iters)]
        if args.device:
            cmd.extend(["--device", args.device])
        run(cmd)
        pretrain_ckpt = config.LATEST_CHECKPOINT_PATH

    # 4. SFT on the instruct corpus, base = pretrain checkpoint.
    print("[bootstrap] Step 4/6 — instruct SFT")
    if not os.path.exists(config.CHAT_DATA_PATH):
        run([sys.executable, os.path.join(root, "build_instruct_dataset.py"),
             "--synth-samples", str(args.synth_samples),
             "--literary-target-mb", str(args.literary_target_mb),
             "--output", config.CHAT_DATA_PATH])
    sft_cmd = [sys.executable, os.path.join(root, "train.py"),
               "--mode", "instruct", "--data", config.CHAT_DATA_PATH,
               "--run-iters", str(args.steps),
               "--checkpoint", pretrain_ckpt]
    if args.device:
        sft_cmd.extend(["--device", args.device])
    run(sft_cmd)
    sft_ckpt = config.LATEST_CHECKPOINT_PATH

    # 5. Preference tuning (DPO/ORPO) — pairs from self_distill if none given.
    if args.pref_steps > 0:
        print("[bootstrap] Step 5/6 — preference tuning")
        pref_data = args.pref_data
        if not pref_data or not os.path.exists(pref_data):
            # Prompts: verified reasoning data (math/logic/code) so the
            # verifier can accept/reject candidates against ground truth.
            reasoning = os.path.join(config.ROOT_DIR, "data", "reasoning.jsonl")
            if not os.path.exists(reasoning):
                print("[bootstrap]   generating verified reasoning prompts "
                      "(build_reasoning_data.py)")
                run([sys.executable, os.path.join(root, "build_reasoning_data.py"),
                     "--n-math", "500", "--n-logic", "200", "--n-code", "300",
                     "--out", reasoning])
            dist = os.path.join(config.ROOT_DIR, "data", "distill.jsonl")
            pref_out = os.path.join(config.ROOT_DIR, "data", "pref.jsonl")
            print("[bootstrap]   rejection sampling via self_distill "
                  "(best vs worst of k=4) -> preference pairs")
            run([sys.executable, os.path.join(root, "self_distill.py"),
                 "--checkpoint", sft_ckpt, "--prompts", reasoning,
                 "--k", "4", "--out", dist, "--pref-out", pref_out]
                + (["--device", args.device] if args.device else []))
            pref_data = pref_out
        run([sys.executable, os.path.join(root, "train_pref.py"),
             "--data", pref_data, "--method", args.pref_method,
             "--checkpoint", sft_ckpt, "--steps", str(args.pref_steps),
             "--out", os.path.join(config.CHECKPOINT_DIR, "jingpt_pref.pt")]
            + (["--device", args.device] if args.device else []))

    # 6. Bench the final instruct checkpoint.
    print("[bootstrap] Step 6/6 — bench final checkpoint")
    run([sys.executable, os.path.join(root, "bench.py"),
         "--checkpoint", sft_ckpt, "--no-train-bench"])
    print("[bootstrap] done. Final SFT checkpoint: " + sft_ckpt)


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print(f"JIN AI Pipeline ({args.mode} mode)")
    print("=" * 72)
    if args.mode == "chat":
        pipeline_chat(args)
    elif args.mode == "evolve":
        pipeline_evolve(args)
    elif args.mode == "bootstrap":
        pipeline_bootstrap(args)
    else:
        pipeline_raw(args)


if __name__ == "__main__":
    main()