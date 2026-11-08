# Equivalent of `pixi shell` on machines without pixi: source this, then use `python`.
# The .venv is built with uv from the exact versions pinned in pixi.lock (see DECISIONS.md).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$ROOT/models" ALFWORLD_DATA="$ROOT/data/alfworld"
export PYTORCH_ALLOC_CONF="expandable_segments:True" TOKENIZERS_PARALLELISM=false
source "$ROOT/.venv/bin/activate"
