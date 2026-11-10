#!/bin/bash
# Regenerate docs/RESULTS.md after each probe arm ends; commit; back up to HF.
cd /workspace/AIKnowledgeSystemsTermProject
for arm in gaca_probe gaca_probe_s1; do
  until grep -qE "end $arm$" runs/verl/arms.log; do sleep 60; done
  sleep 300   # let the launchers make their own commits first
  /workspace/verl-agent/.venv/bin/python src/report.py verl > /dev/null 2>&1
  git add docs/RESULTS.md runs/verl/learning_curves.png && git -c user.name="Adam Mouedden" -c user.email="adam.mouedden@gmail.com" commit -q -m "RESULTS: regenerated after $arm

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  git push -q origin a100 2>/dev/null
  verl/hf_backup.sh >> runs/verl/hf_backup.log 2>&1
done
