# Where this stands (2026-11-07, 23:40)

## Finished — the 2x2 is complete, nothing is running

|                    | mean-and-std-norm | mean-norm | GACA - GRPO |
|--------------------|-------------------|-----------|-------------|
| GRPO               | **14.1%**         | 10.9%     |             |
| GACA               | **15.6%**         | 9.4%      |             |
| margin             | +1.6              | -1.6      | flips sign  |

All four runs: 75 iterations, same 207 training tasks in the same order, same
64 held-out tasks, seed 0, differing only in `--method` and `--norm`. All start
from the same 3.1% zero-shot baseline.

**The conclusion is a clean null.** Across all six pairwise comparisons of the
four best checkpoints, the smallest McNemar p is 0.344 and no two runs solve
more than 3 tasks in common (each solves 6-10 of 64). The GACA-minus-GRPO
margin is one task in favour under one normalization and one task against under
the other. A quantity that flips sign with an arbitrary normalization choice is
noise, not an estimator effect.

What *did* reproduce is the mechanism: anchor coverage 0.82-0.84 against the
paper's 0.86, criticality mean exactly 1.00, realized mixing weight averaging
lambda_0. See `RESULTS.md` for all tables and `runs/learning_curves.png`.

## Read the mean-norm column with this caveat
Mean-norm advantages are ~4x smaller (mean |A| 0.056 against 0.234-0.239), so
gradient norms fall to ~0.9 against ~2.6-3.1 and much of the update no longer
clears the 1.0 clip. Both mean-norm arms take that same smaller step, so the
*margin* comparison stays matched, but the lower absolute numbers in that
column are partly undertraining rather than a normalization effect. Testing
Remark 3 properly needs the mean-norm arms re-run at a learning rate that
matches median gradient norms. See `DECISIONS.md`.

## Highest-value next runs, in order
1. **`--max_steps 40` or `50`** (paper uses 50; we used 25). Measured: winning
   episodes run right up to the cap (21, 23, 25 steps) with no decay, and every
   failure burns the full budget. The cap is truncating episodes still making
   progress, which suppresses success rate, which is what starves the credit
   signal. ~4 h per run.
2. **`--method gigpo`** — the paper's own control, implemented and unit-tested
   but never run. Without it, the step-level advantage cannot be separated from
   the adaptive weight. ~3 h.
3. **`--score_mode loo`** — Appendix B.4's debiased criticality, implemented
   and tested today. Likely more relevant here than in the paper because our
   criticality score is 3x more dispersed than theirs.

## Housekeeping
- `runs/attempt1_penalty0.1/` — the failed first attempt (penalty 0.1), kept.
- `runs/partial_grpo_seed1/` — a partial second-seed GRPO run, stopped at ~20
  iterations, kept out of `runs/` so the report ignores it.
- `run_all.sh.superseded`, `chain.sh.superseded`, `seeds.sh.superseded`,
  `gaca_only.sh.superseded`, `run_meannorm.sh.superseded` — retired launchers.
  Do not run; none of them waits for the GPU and they would collide with a live
  run. Still runnable: `finalize.sh` (paired evals + report) and
  `run_grpo_meannorm.sh` (spent, but harmless -- it resumes a finished run).
- `runs/*_paired.eval.json` — per-task outcomes for each best checkpoint on the
  same 64 tasks in the same order. These are what the McNemar tests consume.
