#!/usr/bin/env bash
# Re-evaluate each method's best checkpoint on the same held-out tasks, in the
# same order, logging per-task outcomes so the methods can be compared pairwise.
# Then rebuild the report. Safe to re-run.
set -u
cd "$(dirname "$0")"
PIXI="$HOME/.pixi/bin/pixi"

# Wait for the whole supervisor, not just the current method: run_all.sh has a
# short gap between methods and starting an eval inside it would put two models
# on the GPU at once.
while pgrep -f "train[.]p[y] --method" >/dev/null 2>&1 \
   || pgrep -f "run_a[l]l\.sh" >/dev/null 2>&1 \
   || pgrep -f "gaca_onl[y]\.sh" >/dev/null 2>&1; do
  sleep 30
done

COMMON=(
  --model Qwen/Qwen2.5-1.5B-Instruct
  --task_types pick_and_place_simple,look_at_obj_in_light
  --max_adm0 20 --n_train 256 --n_eval 64
  --max_steps 25 --eval_batch 16 --max_new_tokens 8 --history_len 6 --seed 0
)

for M in grpo gaca; do
  if [ -d "runs/${M}_best" ]; then
    echo "=== $(date -Is) paired eval: $M ==="
    "$PIXI" run python src/train.py --method "$M" --eval_only "runs/${M}_best" \
      --out "runs/${M}_paired" "${COMMON[@]}" 2>&1 | tail -3
  fi
done
"$PIXI" run report > runs/report.out 2>&1
echo "=== $(date -Is) finalize done ==="
