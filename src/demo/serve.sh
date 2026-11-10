#!/bin/bash
# Live Credit Debugger on 127.0.0.1:17100 (exposed by Caddy on port 10100).
cd /workspace/AIKnowledgeSystemsTermProject
source /workspace/verl-agent/.venv/bin/activate
export HF_HOME=$PWD/models TOKENIZERS_PARALLELISM=false
exec python src/demo/server.py --model models/ckpt/commit_s1 --port 17100 "$@"
