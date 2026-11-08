#!/bin/bash
# The three arms, one after another, identical except for the estimator.
#   grpo        : episode-level advantage only (Eq. 2)
#   gaca        : the paper's GACA (Eq. 8, NLL-gated lambda_t)
#   gaca_msize  : GACA with the anchor-group-size reliability correction
# Usage: STEPS=<iters> TEST_FREQ=<k> run_arms.sh [arm ...]
set -u
ROOT=/workspace/AIKnowledgeSystemsTermProject
ARMS=${@:-grpo gaca gaca_msize}
export TRAIN_BS=${TRAIN_BS:-4} VAL_BS=${VAL_BS:-128} STEPS=${STEPS:-150} TEST_FREQ=${TEST_FREQ:-10}
export GACA_TIME_BUDGET_H=${GACA_TIME_BUDGET_H:-3.75} GACA_VAL_SECS=${GACA_VAL_SECS:-420}
for arm in $ARMS; do
  # one training process at a time
  while pgrep -f "verl.trainer.main_ppo" > /dev/null; do sleep 30; done
  extra=""
  case $arm in
    grpo)       m=grpo; r=none ;;
    gaca)       m=gaca; r=none ;;
    gaca_msize) m=gaca; r=msize ;;
    gaca_random) m=gaca; r=none; extra="algorithm.gaca.score_mode=random" ;;  # ablation (paper Table 2)
    gaca_loo)   m=gaca; r=none; extra="algorithm.gaca.score_mode=loo" ;;     # treatment: anchor-pooled score (App. B.4, never trained in the paper)
    *) echo "unknown arm $arm"; continue ;;
  esac
  if grep -q "Training Progress: 100%" $ROOT/runs/verl/$arm/train.log 2>/dev/null; then
    echo "$arm already finished, skipping"; continue
  fi
  echo "[$(date +%T)] start $arm ($m, rel=$r)" | tee -a $ROOT/runs/verl/arms.log
  $ROOT/verl/run_arm.sh $arm $m $r $extra > /dev/null 2>&1
  /workspace/verl-agent/.venv/bin/ray stop --force > /dev/null 2>&1
  echo "[$(date +%T)] end $arm" | tee -a $ROOT/runs/verl/arms.log
done
