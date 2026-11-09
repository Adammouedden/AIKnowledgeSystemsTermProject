#!/usr/bin/env bash
# GRPO under mean-norm: the matched baseline for runs/gaca_mn.
#
# Remark 3 of the paper predicts that mean-and-std normalization shrinks the
# *gain from adapting*, i.e. the GACA-minus-GRPO margin, not GACA's absolute
# success. Testing that needs both arms under each normalization:
#
#            mean-and-std-norm      mean-norm
#   GRPO     runs/grpo              runs/grpo_mn   <- this run
#   GACA     runs/gaca              runs/gaca_mn
#
# Waits on the mean-norm GACA paired-eval file rather than on pgrep: a pgrep
# pattern written into a script also appears in the command line of the shell
# that created it, which self-matches and hangs forever.
set -u
cd "$(dirname "$0")"
PIXI="$HOME/.pixi/bin/pixi"

while [ ! -f runs/gaca_mn_paired.eval.json ]; do sleep 60; done
sleep 30

COMMON=(
  --model Qwen/Qwen2.5-1.5B-Instruct
  --task_types pick_and_place_simple,look_at_obj_in_light
  --max_adm0 20 --n_train 256 --n_eval 64
  --iters 75 --eval_every 15
  --tasks_per_iter 2 --G 8 --max_steps 25
  --gamma 0.95 --invalid_penalty 0.0 --beta_kl 0.01 --norm mean
  --lr 1e-5 --lora_r 32 --clip_eps 0.2
  --rows_per_update 96 --micro_bs 8
  --gen_batch 16 --eval_batch 16 --max_new_tokens 8 --history_len 6
  --seed 0 --time_budget 5.0 --ckpt_every 10 --resume
)

for attempt in 1 2 3; do
  echo "=== $(date -Is) starting grpo mean-norm (attempt $attempt) ==="
  if "$PIXI" run python src/train.py --method grpo --out runs/grpo_mn "${COMMON[@]}" >> runs/grpo_mn.out 2>&1; then
    echo "=== $(date -Is) finished ==="; break
  fi
  echo "=== $(date -Is) attempt $attempt FAILED, resuming ==="; sleep 30
done

if [ -d runs/grpo_mn_best ]; then
  "$PIXI" run python src/train.py --method grpo --eval_only runs/grpo_mn_best --out runs/grpo_mn_paired \
    --task_types pick_and_place_simple,look_at_obj_in_light --max_adm0 20 --n_train 256 --n_eval 64 \
    --max_steps 25 --eval_batch 16 --max_new_tokens 8 --history_len 6 --seed 0 >> runs/grpo_mn.out 2>&1
fi
"$PIXI" run report > runs/report.out 2>&1
echo "=== $(date -Is) all done ==="
