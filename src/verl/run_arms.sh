#!/bin/bash
# The three arms, one after another, identical except for the estimator.
#   grpo        : episode-level advantage only (Eq. 2)
#   gaca        : the paper's GACA (Eq. 8, NLL-gated lambda_t)
#   gaca_msize  : GACA with the anchor-group-size reliability correction
# Usage: STEPS=<iters> TEST_FREQ=<k> run_arms.sh [arm ...]
set -u
ROOT=/workspace/AIKnowledgeSystemsTermProject
ARMS=${@:-grpo gaca gaca_msize}
# Every arm runs exactly 34 iterations (rollout-matched); no wall-clock cap.
export TRAIN_BS=${TRAIN_BS:-4} VAL_BS=${VAL_BS:-128} STEPS=${STEPS:-34} TEST_FREQ=${TEST_FREQ:-10}
for arm in $ARMS; do
  # one training process at a time
  while pgrep -f "verl.trainer.main_ppo" > /dev/null; do sleep 30; done
  extra=""
  case $arm in
    grpo)       m=grpo; r=none ;;
    gaca)       m=gaca; r=none ;;
    gaca_msize) m=gaca; r=msize ;;
    gaca_random) m=gaca; r=none; extra="algorithm.gaca.score_mode=random" ;;  # ablation (paper Table 2)
    gaca_loo)   m=gaca; r=none; extra="algorithm.gaca.score_mode=loo" ;;
    gaca_probe_s1) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe env.seed=1 +env.val_seed=1000" ;;  # treatment: probe-weighted token credit (IDEAS.md 1); val games pinned
    gaca_probe_c05) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe algorithm.gaca.probe_clip_lo=0.5 algorithm.gaca.probe_clip_hi=2.0" ;;  # probe, clip [0.5, 2]
    gaca_probe_pos) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe algorithm.gaca.probe_sign=pos" ;;  # probe weights on positive advantages only
    gaca_probe_ref) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe algorithm.gaca.probe_scorer=ref" ;;  # probe scored by the frozen reference model (IDEAS 1e.2)     # treatment: anchor-pooled score (App. B.4, never trained in the paper)
    *) echo "unknown arm $arm"; continue ;;
  esac
  if grep -q "Training Progress: 100%" $ROOT/src/runs/verl/$arm/train.log 2>/dev/null; then
    echo "$arm already finished, skipping"; continue
  fi
  echo "[$(date +%T)] start $arm ($m, rel=$r)" | tee -a $ROOT/src/runs/verl/arms.log
  $ROOT/src/verl/run_arm.sh $arm $m $r $extra > /dev/null 2>&1
  /workspace/verl-agent/.venv/bin/ray stop --force > /dev/null 2>&1
  echo "[$(date +%T)] end $arm" | tee -a $ROOT/src/runs/verl/arms.log
done
