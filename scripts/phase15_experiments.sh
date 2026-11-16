#!/usr/bin/env bash
# Phase 15H — controlled pretraining experiments.
#
# Everything is bounded and reproducible:
#   * 250 optimizer steps x 16,384 tokens = 4.10M tokens per run
#   * seq 512, batch 4, grad-accum 8 (same architecture, same recipe)
#   * a new output directory per run; no checkpoint is ever overwritten
#   * each run writes a manifest with its corpus hash, config fingerprint and
#     hardware metadata (train_gpt2.py does this on every save)
#
# Usage: bash scripts/phase15_experiments.sh [--skip-build]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=".venv/Scripts/python.exe"
STEPS="${STEPS:-250}"
SEQ="${SEQ:-512}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-8}"

build_corpus() {   # name, target tokens, mix
  local name="$1" tokens="$2" mix="$3"
  if [ -f "data/$name/manifest.json" ]; then echo "[skip] $name exists"; return; fi
  echo "[build] $name ($tokens tokens)"
  "$PY" -u scripts/build_tokenized_corpus.py \
      --splits data/curated_v6/splits --out "data/$name" \
      --max-train-tokens "$tokens" --mix "$mix" > "runs/build_$name.log" 2>&1 || {
    echo "[fail] build $name"; return 1; }
}

run() {            # corpus, out-dir, seed, tag
  local corpus="$1" out="$2" seed="$3" tag="$4"
  if [ -f "$out/best.pt" ]; then echo "[skip] $out exists"; return; fi
  echo "[train] $tag seed=$seed corpus=$corpus -> $out"
  "$PY" -u train_gpt2.py --preset gpt2_small_reference \
      --data-dir "data/$corpus" --out-dir "$out" \
      --seq-len "$SEQ" --batch-size "$BATCH" --grad-accum "$ACCUM" \
      --max-steps "$STEPS" --warmup 30 --lr 6e-4 --min-lr-ratio 0.1 \
      --eval-interval 50 --eval-iters 4 --save-interval 125 --dropout 0.0 \
      --seed "$seed" > "runs/train_${tag}_s${seed}.log" 2>&1 || {
    echo "[fail] $tag seed $seed"; return 1; }
  echo "[done] $tag seed=$seed: $(grep -o 'best val loss [0-9.]*' "runs/train_${tag}_s${seed}.log" | tail -1)"
}

if [ "${1:-}" != "--skip-build" ]; then
  # A: the v6 default mixture.  B1: human-heavy.  B2: synthetic-heavier.
  build_corpus gpt2_corpus_v6a  12000000 "wikipedia_en_20231101:0.65,cosmopedia_v2_sample:0.25,prepared_v4:0.08,project_seed:0.02"
  build_corpus gpt2_corpus_v6b1  8000000 "wikipedia_en_20231101:0.85,prepared_v4:0.10,project_seed:0.02,cosmopedia_v2_sample:0.03"
  build_corpus gpt2_corpus_v6b2  8000000 "wikipedia_en_20231101:0.50,cosmopedia_v2_sample:0.45,prepared_v4:0.03,project_seed:0.02"
fi

# Experiment A — seed variance on an identical corpus and recipe.
for seed in 1337 2024 42; do
  run gpt2_corpus_v6a "checkpoints/phase15/A_seed${seed}" "$seed" "A"
done

# Experiment B — data mixture, one seed, identical budget.
run gpt2_corpus_v6b1 "checkpoints/phase15/B_humanheavy" 1337 "B1"
run gpt2_corpus_v6b2 "checkpoints/phase15/B_synthetic" 1337 "B2"

echo "[all done]"
