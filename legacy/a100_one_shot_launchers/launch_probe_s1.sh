#!/bin/bash
# After gaca_probe (seed 0): smoke-test the final-weights save+upload, then run
# the probe arm with training seed 1 on the same 128 validation games.
cd /workspace/AIKnowledgeSystemsTermProject
until grep -q "end gaca_probe$" runs/verl/arms.log; do sleep 60; done
sleep 240   # launch_probe.sh commits; backup_after_probe.sh uploads
REPO=Adammouedden/gaca-term-checkpoints
export GACA_HF_REPO=$REPO
rm -rf runs/verl/save_smoke
GACA_SAVE_FINAL=$PWD/runs/verl/save_smoke/final GACA_HF_PATH=smoke \
TRAIN_BS=4 VAL_BS=16 STEPS=1 TEST_FREQ=100 VAL_BEFORE=False GACA_TIME_BUDGET_H=0 \
  verl/run_arm.sh save_smoke gaca none algorithm.gaca.token_mode=probe > /dev/null 2>&1
/workspace/verl-agent/.venv/bin/ray stop --force > /dev/null 2>&1
OK=$(HF_HOME=$PWD/models /workspace/verl-agent/.venv/bin/python -c "
from huggingface_hub import HfApi
f=HfApi().list_repo_files('$REPO'); need={'smoke/config.json','smoke/TRAINING_INFO.json','smoke/tokenizer_config.json'}
w=[x for x in f if x.startswith('smoke/') and x.endswith('.safetensors')]
print('yes' if need<=set(f) and w else 'no')" 2>/dev/null)
if [ "$OK" != "yes" ]; then
  echo "[$(date +%T)] save smoke FAILED, gaca_probe_s1 not started" >> runs/verl/arms.log; exit 1
fi
HF_HOME=$PWD/models /workspace/verl-agent/.venv/bin/python -c "
from huggingface_hub import HfApi
HfApi().delete_folder('smoke', repo_id='$REPO', commit_message='remove save smoke test')" > /dev/null 2>&1
echo "[$(date +%T)] save smoke passed (uploaded + verified + removed)" >> runs/verl/arms.log
git add runs/verl/save_smoke/train.log runs/verl/arms.log && git -c user.name="Adam Mouedden" -c user.email="adam.mouedden@gmail.com" commit -q -m "Final-weights save+upload smoke test passed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
GACA_SAVE_FINAL=$PWD/runs/verl/gaca_probe_s1/final GACA_HF_PATH=gaca_probe_s1 \
STEPS=34 GACA_TIME_BUDGET_H=6 verl/run_arms.sh gaca_probe_s1
git add runs/verl/gaca_probe_s1 runs/verl/arms.log && git -c user.name="Adam Mouedden" -c user.email="adam.mouedden@gmail.com" commit -q -m "Results: GACA + probe, seed 1 (final weights on HF)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
verl/hf_backup.sh >> runs/verl/hf_backup.log 2>&1
