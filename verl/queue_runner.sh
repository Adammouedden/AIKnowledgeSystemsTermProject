#!/bin/bash
# Runs arms from runs/verl/queue.txt one at a time. Edit the queue file at any
# time: the runner takes its first line only when it is ready for the next arm.
# Lines: an arm name from verl/run_arms.sh, or "save_smoke". Blank/# ignored.
cd /workspace/AIKnowledgeSystemsTermProject
Q=runs/verl/queue.txt
REPO=Adammouedden/gaca-term-checkpoints
PY=/workspace/verl-agent/.venv/bin/python
gitc() { git -c user.name="Adam Mouedden" -c user.email="adam.mouedden@gmail.com" commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; }
publish() {  # report, commit, push (if credentials ever exist), HF backup
  $PY src/report.py verl > /dev/null 2>&1
  git add docs/RESULTS.md runs/verl/learning_curves.png runs/verl/arms.log runs/verl/queue.txt $1 2>/dev/null
  gitc "$2"
  GIT_TERMINAL_PROMPT=0 git push -q origin a100 > /dev/null 2>&1
  verl/hf_backup.sh >> runs/verl/hf_backup.log 2>&1
}
while true; do
  while pgrep -f "verl.trainer.main_ppo" > /dev/null; do sleep 30; done
  sleep 120   # let a finishing launcher make its commit first
  if [ -n "$(git status --porcelain runs/verl/gaca_probe 2>/dev/null)" ]; then publish runs/verl/gaca_probe "Results: GACA + probe (seed 0) complete"; fi
  next=$(grep -vE '^\s*(#|$)' $Q | head -1)
  if [ -z "$next" ]; then sleep 60; continue; fi
  $PY - "$Q" "$next" <<'PY'
import sys
p, n = sys.argv[1], sys.argv[2]
lines = open(p).read().splitlines()
for i, l in enumerate(lines):
    if l.strip() == n:
        del lines[i]; break
open(p, "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
  if [ "$next" = "save_smoke" ]; then
    rm -rf runs/verl/save_smoke
    GACA_SAVE_FINAL=$PWD/runs/verl/save_smoke/final GACA_HF_REPO=$REPO GACA_HF_PATH=smoke \
    TRAIN_BS=4 VAL_BS=16 STEPS=1 TEST_FREQ=100 VAL_BEFORE=False \
      verl/run_arm.sh save_smoke gaca none algorithm.gaca.token_mode=probe > /dev/null 2>&1
    /workspace/verl-agent/.venv/bin/ray stop --force > /dev/null 2>&1
    OK=$(HF_HOME=$PWD/models $PY -c "
from huggingface_hub import HfApi
f=set(HfApi().list_repo_files('$REPO'))
print('yes' if {'smoke/config.json','smoke/TRAINING_INFO.json','smoke/tokenizer_config.json'}<=f and any(x.startswith('smoke/') and x.endswith('.safetensors') for x in f) else 'no')" 2>/dev/null)
    if [ "$OK" = "yes" ]; then
      HF_HOME=$PWD/models $PY -c "from huggingface_hub import HfApi; HfApi().delete_folder('smoke', repo_id='$REPO', commit_message='remove save smoke test')" > /dev/null 2>&1
      echo "[$(date +%T)] save smoke passed (uploaded + verified + removed)" >> runs/verl/arms.log
    else
      echo "[$(date +%T)] save smoke FAILED: arms will run but final weights may not upload" >> runs/verl/arms.log
    fi
    git add runs/verl/save_smoke/train.log runs/verl/arms.log runs/verl/queue.txt; gitc "Final-weights save smoke test: $OK"
    continue
  fi
  echo "[$(date +%T)] queue -> $next" >> runs/verl/arms.log
  GACA_SAVE_FINAL=$PWD/runs/verl/$next/final GACA_HF_REPO=$REPO GACA_HF_PATH=$next verl/run_arms.sh $next
  publish runs/verl/$next "Results: $next complete (final weights -> $REPO/$next)"
done
