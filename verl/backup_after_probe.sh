#!/bin/bash
cd /workspace/AIKnowledgeSystemsTermProject
until grep -q "end gaca_probe" runs/verl/arms.log; do sleep 60; done
sleep 120   # let launch_probe.sh make its results commit first
verl/hf_backup.sh >> runs/verl/hf_backup.log 2>&1
