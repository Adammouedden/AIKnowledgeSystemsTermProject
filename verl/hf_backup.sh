#!/bin/bash
# Mirror the committed state of branch a100 to the private HF repo (no weights/data).
ROOT=/workspace/AIKnowledgeSystemsTermProject
S=$(mktemp -d); trap 'rm -rf $S' EXIT
cd $ROOT && git archive a100 | tar -x -C $S
HF_HOME=$ROOT/models /workspace/verl-agent/.venv/bin/python - "$S" <<'PY' 2>&1 | grep -E "^uploaded|Error"
import sys, subprocess
from huggingface_hub import HfApi
sha = subprocess.check_output(["git", "-C", "/workspace/AIKnowledgeSystemsTermProject", "rev-parse", "--short", "a100"]).decode().strip()
info = HfApi().upload_folder(folder_path=sys.argv[1], repo_id="Adammouedden/gaca-term-project",
                             commit_message=f"Backup of branch a100 @ {sha}")
print("uploaded:", sha)
PY
