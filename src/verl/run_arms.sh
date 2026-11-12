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
  if [ "$arm" = "grad_snr" ]; then
    echo "[$(date +%T)] start grad_snr (gradient signal-to-noise analysis)" | tee -a $ROOT/src/runs/verl/arms.log
    $ROOT/src/analysis/grad_snr.sh > $ROOT/src/runs/verl/grad_snr.log 2>&1; rc=$?
    echo "[$(date +%T)] end grad_snr (exit $rc)" | tee -a $ROOT/src/runs/verl/arms.log
    continue
  fi
  extra=""; lr=1e-6
  case $arm in
    grpo)       m=grpo; r=none ;;
    gaca)       m=gaca; r=none ;;
    gaca_msize) m=gaca; r=msize ;;
    gaca_random) m=gaca; r=none; extra="algorithm.gaca.score_mode=random" ;;  # ablation (paper Table 2)
    gaca_loo)   m=gaca; r=none; extra="algorithm.gaca.score_mode=loo" ;;
    commit_s1) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe env.seed=1 +env.val_seed=1000" ;;  # treatment: probe-weighted token credit (IDEAS.md 1); val games pinned
    commit_c05) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe algorithm.gaca.probe_clip_lo=0.5 algorithm.gaca.probe_clip_hi=2.0" ;;  # probe, clip [0.5, 2]
    commit_pos) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe algorithm.gaca.probe_sign=pos" ;;  # probe weights on positive advantages only
    commit_ref) m=gaca; r=none; extra="algorithm.gaca.token_mode=probe +algorithm.gaca.probe_scorer=ref" ;;  # probe scored by the frozen reference model (IDEAS 1e.2)     # treatment: anchor-pooled score (App. B.4, never trained in the paper)
    # hill-climbing queue (2026-11-10): performance, not ablation rigor
    commit_lr3) m=gaca; r=none; lr=3e-6; extra="algorithm.gaca.token_mode=probe env.seed=1 +env.val_seed=1000" ;;  # COMMIT (as commit_s1) at lr 3e-6
    gaca_lr3)   m=gaca; r=none; lr=3e-6 ;;                                                                         # GACA at lr 3e-6
    gaca_actw)  m=gaca; r=none; extra="algorithm.gaca.token_mode=action algorithm.gaca.action_weight=2.0" ;;      # probe-free: <action> span 2x, reasoning 1x
    commit_ds)  m=gaca; r=none; lr=3e-6; extra="algorithm.gaca.token_mode=probe env.seed=1 +env.val_seed=1000 algorithm.filter_groups.enable=True algorithm.filter_groups.max_num_gen_batches=3" ;;  # COMMIT at lr 3e-6 + DAPO dynamic sampling (drop all-same-reward groups, up to 3 rollout passes)
    # learning-rate sweep (2026-11-10, DECISIONS.md): does token weighting let larger steps help?
    gaca_actw_lr3) m=gaca; r=none; lr=3e-6; extra="algorithm.gaca.token_mode=action algorithm.gaca.action_weight=2.0" ;;
    gaca_actw_lr6) m=gaca; r=none; lr=6e-6; extra="algorithm.gaca.token_mode=action algorithm.gaca.action_weight=2.0" ;;
    gaca_lr6)      m=gaca; r=none; lr=6e-6 ;;
    commit_lr3_s0) m=gaca; r=none; lr=3e-6; extra="algorithm.gaca.token_mode=probe env.seed=0 +env.val_seed=1000" ;;  # seed control: COMMIT lr 3e-6 on the seed every non-COMMIT arm uses
    # COMMIT hill-climb (2026-11-11): lr 6e-6; then frozen-reference probe only if lr6 shows no gain; then probe every 8 tokens
    commit_lr6_s0) m=gaca; r=none; lr=6e-6; extra="algorithm.gaca.token_mode=probe env.seed=0 +env.val_seed=1000" ;;
    commit_ref_lr3_s0) m=gaca; r=none; lr=3e-6; extra="algorithm.gaca.token_mode=probe env.seed=0 +env.val_seed=1000 +algorithm.gaca.probe_scorer=ref" ;;
    commit_ref_lr6_s0) m=gaca; r=none; lr=6e-6; extra="algorithm.gaca.token_mode=probe env.seed=0 +env.val_seed=1000 +algorithm.gaca.probe_scorer=ref" ;;
    commit_e8_lr3_s0) m=gaca; r=none; lr=3e-6; extra="algorithm.gaca.token_mode=probe env.seed=0 +env.val_seed=1000 algorithm.gaca.probe_every=8" ;;
    commit_e8_lr6_s0) m=gaca; r=none; lr=6e-6; extra="algorithm.gaca.token_mode=probe env.seed=0 +env.val_seed=1000 algorithm.gaca.probe_every=8" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  if grep -q "Training Progress: 100%" $ROOT/src/runs/verl/$arm/train.log 2>/dev/null; then
    echo "$arm already finished, skipping"; continue
  fi
  echo "[$(date +%T)] start $arm ($m, rel=$r)" | tee -a $ROOT/src/runs/verl/arms.log
  echo "[$(date +%T)]   lr=$lr extra: $extra" >> $ROOT/src/runs/verl/arms.log
  LR=$lr $ROOT/src/verl/run_arm.sh $arm $m $r $extra > /dev/null 2>&1
  /workspace/verl-agent/.venv/bin/ray stop --force > /dev/null 2>&1
  echo "[$(date +%T)] end $arm" | tee -a $ROOT/src/runs/verl/arms.log
done
