#!/usr/bin/env bash
# One-off: bench 4 checkpoints cùng protocol (--device cuda --no-train-bench)
# Thứ tự: pretrain_best -> v4 57400 -> v4 60020 -> serving chat
set -u
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../.." || exit 1
LOG="bench_results/run_logs/20261002_4ckpts.log"
exec >>"$LOG" 2>&1
echo "START $(date '+%F %T')"
for ck in \
  "checkpoints/pretrain/general_v1_01/jin/checkpoints/kaggle_v4_pretrain_best.pt" \
  "checkpoints/kaggle_v4_60020/kaggle_v4_instruct_best.pt" \
  "checkpoints/kaggle_v4_60020/kaggle_v4_instruct.pt" \
  "checkpoints/jingpt_chat.pt" ; do
  echo "===== BEGIN $ck $(date '+%F %T') ====="
  ./.venv/Scripts/python.exe -u bench.py --checkpoint "$ck" --device cuda --no-train-bench
  echo "===== END rc=$? $ck $(date '+%F %T') ====="
done
echo "ALL DONE $(date '+%F %T')"
