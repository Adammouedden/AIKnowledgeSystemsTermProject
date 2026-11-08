#!/bin/bash
# After GACA-LOO: smoke-test the probe integration, then run the probe arm.
cd /workspace/AIKnowledgeSystemsTermProject
until grep -q "end gaca_loo" runs/verl/arms.log; do sleep 30; done
sleep 30
LOG=runs/verl/probe_smoke/train.log
rm -rf runs/verl/probe_smoke
TRAIN_BS=4 VAL_BS=16 STEPS=2 TEST_FREQ=100 VAL_BEFORE=False GACA_TIME_BUDGET_H=0 \
  verl/run_arm.sh probe_smoke gaca none algorithm.gaca.token_mode=probe > /dev/null 2>&1
/workspace/verl-agent/.venv/bin/ray stop --force > /dev/null 2>&1
if grep -qE "Traceback|Error executing job" $LOG || ! grep -q "probe/rows" $LOG; then
  echo "[$(date +%T)] probe smoke FAILED, arm not started" >> runs/verl/arms.log; exit 1
fi
echo "[$(date +%T)] probe smoke passed" >> runs/verl/arms.log
git add runs/verl/probe_smoke/train.log runs/verl/arms.log && git -c user.name="Adam Mouedden" -c user.email="adam.mouedden@gmail.com" commit -q -m "Probe integration smoke test passed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
# matched to the controls on iterations (34), not wall-clock; 6 h safety cap
STEPS=34 GACA_TIME_BUDGET_H=6 verl/run_arms.sh gaca_probe
git add runs/verl/gaca_probe runs/verl/arms.log && git -c user.name="Adam Mouedden" -c user.email="adam.mouedden@gmail.com" commit -q -m "Results: GACA + probe-weighted token credit arm complete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
