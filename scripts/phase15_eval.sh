#!/usr/bin/env bash
# Phase 15D/15E/15I — evaluation on the clean unseen-by-both slice + generation suite.
# One model at a time, so the GPU never runs two jobs at once.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"
CLEAN="data/curated_v6/benchmark/clean_unseen.jsonl"
OUT="bench_results/phase15"
mkdir -p "$OUT"

eval_ckpt() {   # name, path
  local name="$1" path="$2"
  [ -f "$path" ] || { echo "[skip] $name (no $path)"; return; }
  echo "[eval] $name"
  "$PY" -u scripts/eval_phase15.py --checkpoint "$path" --val-jsonl "$CLEAN" \
      --docs-per-stratum 24 --seq-len 1024 --out "$OUT/clean_$name.json" \
      > "runs/eval_clean_$name.log" 2>&1 || echo "[fail] $name"
  grep -E "^\[.*\] (overall|loss)" "runs/eval_clean_$name.log" | tail -2
}

echo "[eval] gpt2 reference"
"$PY" -u scripts/eval_phase15.py --hf-model gpt2 --val-jsonl "$CLEAN" \
    --docs-per-stratum 24 --seq-len 1024 --out "$OUT/clean_gpt2_reference.json" \
    > runs/eval_clean_gpt2.log 2>&1 || echo "[fail] gpt2"

eval_ckpt jin_pilot_best_160    checkpoints/gpt2_small/pilot/best.pt
eval_ckpt jin_phase14_wiki      checkpoints/gpt2_small/phase14_wiki/best.pt
eval_ckpt jin_phase15_A_seed2024 checkpoints/phase15/A_seed2024/best.pt
eval_ckpt jin_phase15_B_humanheavy checkpoints/phase15/B_humanheavy/best.pt
eval_ckpt jin_phase15_B_synthetic checkpoints/phase15/B_synthetic/best.pt

# Generation suite: the Phase 14 model and the best Phase 15 run, same prompts.
for spec in "phase14:checkpoints/gpt2_small/phase14_wiki/best.pt" \
            "phase15_A_seed2024:checkpoints/phase15/A_seed2024/best.pt"; do
  name="${spec%%:*}"; path="${spec#*:}"
  [ -f "$path" ] || continue
  echo "[gen] $name"
  "$PY" -u scripts/generation_suite.py --checkpoint "$path" --per-category 6 \
      --max-new-tokens 64 --out "$OUT/generation_$name.json" \
      > "runs/gen_$name.log" 2>&1 || echo "[fail] gen $name"
  grep -E "^\[" "runs/gen_$name.log" | tail -10
done
echo "[eval+gen done]"
