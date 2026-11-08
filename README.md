# GACA vs GRPO on ALFWorld

A small-scale reproduction of **GACA** (Granularity-Adaptive Credit Assignment,
Liang et al., ICLR 2026) against vanilla **GRPO**, with **GiGPO** as the control
that separates GACA's state-adaptive mixing weight from a fixed one. Everything
runs on a single 8 GB consumer GPU.

```
references/      the GACA paper and the project proposal
src/env.py       ALFWorld (TextWorld) wrapper, prompt construction, anchor states
src/advantages.py  GRPO / GiGPO / fixed-lambda / GACA advantage estimators
src/train.py     rollout, clipped surrogate (Eq. 1), evaluation, CLI
src/report.py    builds RESULTS.md and the learning-curve figure
src/status.py    one-line progress report for a running job
src/test_*.py    test suites
DECISIONS.md     every judgement call about scale, model and data, with reasons
RESULTS.md       generated: the comparison tables and figure
runs/            logs (*.jsonl), stdout (*.out), LoRA checkpoints
```

## Running

```bash
pixi run test          # advantage-estimator unit tests (fast, no GPU)
pixi run test-train    # end-to-end rollout/update tests on the real model
./run_all.sh           # GRPO, then GACA, then GiGPO, then the report
pixi run report        # rebuild RESULTS.md from whatever logs exist
pixi run python src/status.py   # progress of a running job
```

A single method, with every other setting shared:

```bash
pixi run train --method gaca --out runs/gaca \
  --task_types pick_and_place_simple,look_at_obj_in_light --max_adm0 20 \
  --iters 90 --tasks_per_iter 2 --G 8 --max_steps 25
```

`--method` is the only thing that changes between the compared runs:

| `--method` | advantage |
|---|---|
| `grpo` | `A^E` — one group-relative scalar per trajectory (Eq. 2) |
| `gigpo` | `w A^S_t + A^E` — anchor-state step advantage at a fixed weight (Eq. 3) |
| `fixed` | `lam0 w A^S_t + (1-lam0) A^E` — the paper's fixed-weight control |
| `gaca` | `lam_t w A^S_t + (1-lam_t) A^E`, `lam_t` from the NLL criticality score (Eqs. 4-8) |

GACA's own hyperparameters follow the paper's Table 4 defaults: `--lam0 0.5
--alpha_lam 0.5 --beta_lam 0.5 --alpha_c 1.0 --w 1.0`. The ablation controls of
the paper's Table 2 are reachable with `--score_mode uniform` (forces
`c_t == 1`) and `--score_mode random` (permutes the scores within a trajectory).

## What differs from the paper

The short version: ALFWorld only, Qwen2.5-1.5B-Instruct with LoRA instead of
full fine-tuning, actions chosen by index from the admissible list, the two
shortest-horizon task types, 25 environment steps instead of 50, and 90
iterations on one seed instead of 150 epochs on three. Each of those is a
consequence of a 7 GB VRAM budget and is argued for in `DECISIONS.md`.

Absolute success rates here are therefore **not** comparable to the paper's
Table 1. Only the ordering between estimators, which is what the paper's claim
is about, is.
