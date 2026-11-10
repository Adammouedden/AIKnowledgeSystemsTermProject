#!/bin/bash
# Gradient signal-to-noise of uniform vs action-span vs COMMIT token credit (src/analysis/grad_snr.py).
# Two policies: the base model and the trained COMMIT checkpoint. ~2-3 h on the A100; run from the
# training queue (an item named grad_snr), never alongside training.
set -euo pipefail
ROOT=/workspace/AIKnowledgeSystemsTermProject
cd $ROOT
. /workspace/verl-agent/.venv/bin/activate
export HF_HOME=$ROOT/models ALFWORLD_DATA=$ROOT/data/alfworld TOKENIZERS_PARALLELISM=false
DATA=$ROOT/runs/grad_snr; OUT=$ROOT/src/runs/verl/grad_snr; mkdir -p $DATA $OUT
for pair in "base:Qwen/Qwen2.5-1.5B-Instruct" "commit:$ROOT/models/ckpt/commit_s1"; do
  name=${pair%%:*}; model=${pair#*:}
  [ -f $DATA/$name.npz ] && { echo "$name done, skipping"; continue; }
  # separate processes so vLLM's GPU memory is fully released before the gradient pass
  python src/analysis/grad_snr.py rollouts --model $model --out $DATA/${name}_rollouts.json
  python src/analysis/grad_snr.py grads --model $model --inp $DATA/${name}_rollouts.json --out $DATA/$name.npz
done
python src/analysis/grad_snr.py report --npz $DATA/base.npz $DATA/commit.npz --names "base model" "trained COMMIT" --out $OUT/grad_snr
