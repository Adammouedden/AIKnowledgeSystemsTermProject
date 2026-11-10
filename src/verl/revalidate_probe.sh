#!/bin/bash
# One-shot (IDEAS 1e.1): token flip-test oracle with the trained commit_s1 policy, then requeue commit_pos.
cd /workspace/AIKnowledgeSystemsTermProject
until git log --oneline -20 | grep -q "Results: commit_c05 complete"; do sleep 30; done
while pgrep -f "verl.trainer.main_ppo" > /dev/null; do sleep 30; done
sleep 30
echo "[$(date +%T)] start probe re-validation (trained commit_s1)" >> src/runs/verl/arms.log
export HF_HOME=$PWD/models ALFWORLD_DATA=$PWD/data/alfworld TOKENIZERS_PARALLELISM=false
PY=/workspace/verl-agent/.venv/bin/python
$PY src/measure.py tokens --model models/trained/commit_s1 --out src/runs/tokens/trained_s1 \
  --task_offset 400 --n_games 48 --n_steps 320 --K 8 > src/runs/tokens/trained_s1.out 2>&1
$PY src/measure.py tokens_robust src/runs/tokens/trained_s1.jsonl --task_offset 400 > /dev/null 2>&1
echo "[$(date +%T)] end probe re-validation" >> src/runs/verl/arms.log
git add src/runs/tokens/trained_s1.jsonl src/runs/tokens/trained_s1.analysis.json src/runs/tokens/trained_s1.robust.json src/runs/tokens/trained_s1.out src/runs/verl/arms.log src/verl/revalidate_probe.sh
git -c user.name="Adam Mouedden" -c user.email="adam.mouedden@gmail.com" commit -q -m "Probe re-validation on the trained commit_s1 policy (token flip-test oracle)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git fetch -q . a100:main && GIT_TERMINAL_PROMPT=0 git push -q origin a100 main > /dev/null 2>&1
printf 'commit_pos\n' > src/runs/verl/queue.txt
