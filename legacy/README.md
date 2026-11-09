# Legacy code (not used by the current experiments)

Kept for the record; nothing here is run any more. Current code is in `src/`.

- `4060_hf_pipeline/`: the original HF/LoRA trainer of attempts 1-3 (RTX 4060) and the A100 zero-shot
  format tests (`train.py`, `env.sh`, `test_train.py`, `status.py`), the Monte-Carlo step-advantage
  oracle that was piloted but never run to completion (`measure_mc_oracle.py`, the full `measure.py` of
  that time), and the generator of `docs/RESULTS_4060.md` (`report_4060.py`). Paths inside assume the old
  layout (`src/`, `runs/*.jsonl`).
- `4060_launchers/`: retired launchers of attempts 1-3. **Do not run them**: none waits for the GPU.
- `a100_one_shot_launchers/`: single-purpose launchers of the A100 arms, superseded by
  `src/verl/queue_runner.sh` + `src/runs/verl/queue.txt`.

pixi (`pixi.toml`, `pixi.lock`) was removed: the experiments run in verl-agent's own venv
(`/workspace/verl-agent/.venv`), see `docs/HANDOFF.md`. It remains in git history.
