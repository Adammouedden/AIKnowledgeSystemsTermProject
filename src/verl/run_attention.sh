#!/bin/bash
cd /workspace/AIKnowledgeSystemsTermProject
export HF_HOME=$PWD/models TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
nice -n 15 /workspace/verl-agent/.venv/bin/python -c "
import sys; sys.argv=['x']; sys.path.insert(0,'src'); import paper_figures as P; P.attention(n=128)" > src/runs/tokens/attention.out 2>&1
