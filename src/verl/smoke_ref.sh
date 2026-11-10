#!/bin/bash
# One-shot: 1-iteration GPU smoke of probe_scorer=ref after the routing fix.
cd /workspace/AIKnowledgeSystemsTermProject
rm -rf src/runs/verl/ref_smoke
TRAIN_BS=4 VAL_BS=16 STEPS=1 TEST_FREQ=100 VAL_BEFORE=False \
  src/verl/run_arm.sh ref_smoke gaca none algorithm.gaca.token_mode=probe algorithm.gaca.probe_scorer=ref > /dev/null 2>&1
/workspace/verl-agent/.venv/bin/ray stop --force > /dev/null 2>&1
if grep -q "probe/rows" src/runs/verl/ref_smoke/train.log && ! grep -q "Error executing" src/runs/verl/ref_smoke/train.log; then
  echo "[$(date +%T)] ref-scorer smoke passed" >> src/runs/verl/arms.log
  echo commit_ref >> src/runs/verl/queue.txt
else
  echo "[$(date +%T)] ref-scorer smoke FAILED" >> src/runs/verl/arms.log
fi
