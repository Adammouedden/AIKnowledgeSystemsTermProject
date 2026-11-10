#!/bin/bash
# Waits for the running arm's results commit, then records the demo trace on the free GPU.
cd /workspace/AIKnowledgeSystemsTermProject
until git log --oneline -5 | grep -q "Results: commit_ref complete"; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
source /workspace/verl-agent/.venv/bin/activate
export HF_HOME=$PWD/models
python src/demo/make_trace.py --model models/ckpt/commit_s1 --out src/demo/trace.json > src/demo/make_trace.log 2>&1
python src/demo/build_demo.py src/demo/trace.json src/demo/credit_debugger.html >> src/demo/make_trace.log 2>&1
TZ=America/New_York date +%r >> src/demo/make_trace.log
