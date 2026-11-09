# Project brief

Reproduce **GACA** (Granularity-Adaptive Credit Assignment, ICLR 2026, in
`references/`) and compare it against vanilla **GRPO** on the benchmark the
paper uses. Ignore everything about *learning value* in
`references/Credit_Assignment_Proposal (2).pdf`, and ignore the model sizes it
names.

Read `docs/DECISIONS.md` before changing anything. It records every judgement call
with the measurement behind it, including two that were expensive to learn.
`docs/RESULTS.md` has the numbers. `docs/HANDOFF.md` has the state of the last session.

---

## Hard budget constraint

**4-6 hours per experiment. This has not changed and will not be relaxed by
moving to a cloud GPU.** Do not propose or start anything longer. If a plan
needs 30 hours, it is the wrong plan; shrink the scope, not the honesty of the
conclusion.

A corollary that matters: **the paper's absolute numbers are out of reach and
chasing them is a mistake.** They train ~8x more trajectories than fits in this
budget. Aim instead for a comparison that is *resolvable* - see "Spend compute
on resolution" below.

## Environment

Written for a 7 GB RTX 4060; now moving to a cloud GPU. On more VRAM, change
things in this order and re-measure after each:

1. **Free-form text actions with a proper few-shot prompt.** The current
   index-based action selection (model emits the number of an admissible
   action, 1-3 tokens) was forced by an 88% invalid-action rate at 1.5B. That
   rate was very likely caused by a weak prompt - a single one-line example
   rather than the 1-2 complete worked trajectories that ALFWorld/ReAct and
   verl-agent use. **Test this first; it costs 15 minutes, not GPU hours**, and
   if invalid actions fall below ~20% then delete the index hack. It is the
   largest fidelity deviation we have and it does not transfer to real tasks.
2. **Full fine-tuning instead of LoRA** (needs ~24 GB with 8-bit Adam at 1.5B).
   Probably the biggest single lever after training budget.
3. **`--max_steps 50`** to match the paper. Measured: the current cap of 25
   binds - winning episodes run right up against it and every failure burns the
   full budget.
4. **More prompts per optimizer step** (`--tasks_per_iter`). We used 2; the
   public GiGPO recipes the paper aligns to appear to use ~8.

**Stay at 1.5B.** The paper's result is 1.5B; a 3B model adds a variable while
leaving every deficit above in place.

Consider **verl-agent** only if the GPU is 40 GB+. It gives GiGPO and the
paper's exact prompts for free, but it assumes sharded training plus a
colocated vLLM worker.

## Spend compute on resolution, not on length

The completed GRPO-vs-GACA comparison returned **McNemar p = 1.000**, and the
two policies solved only **1 task in common** out of ~9 each - about what
independence predicts. Separately, two GRPO seeds differed by 3.1 points at a
matched checkpoint, which is larger than GACA's entire best-checkpoint lead.

So **seed variance, not training length, is the binding constraint on what can
be concluded.** Within 4-6 hours, prefer:

- several short runs across seeds over one long run,
- more evaluation tasks (we used 64 of the paper's 128),
- paired per-task tests (McNemar, in `docs/RESULTS.md`) over comparing two rates,
- and **measurement experiments over training experiments** wherever possible.

The proposal's own first experiment is a measurement study - does a cheap
estimator predict the thing it claims to? That style is feasible here; head-to-
head training largely is not.

## Findings that must not be re-litigated

- **`--invalid_penalty 0.0`, not the paper's 0.1.** With index-based actions an
  inadmissible action is nearly impossible, so the penalty stopped being a
  corrective and became the dominant gradient: the policy optimised it to 0.2%
  while held-out success *fell* 4.7% -> 1.6%. Removing it moved the best result
  to 14.1%. If free-form actions are restored, **revisit this** - the penalty
  may become legitimate again, because inadmissible actions become real.
- **A penalty *schedule* cannot fix that.** Under mean-and-std normalization the
  weight cancels exactly: mean |A| is 0.8255 at lambda = 0.1, 0.03 and 0.01. It
  is a step function at zero, not a dial.
- **Reward shaping dominated credit assignment at low success rates.** Removing
  the penalty moved the result by 9.4 points; switching estimator moved it by
  1.5. Audit the reward before comparing estimators.
- **Qwen ships a sampling preset** (`repetition_penalty 1.1`, `temperature 0.7`,
  `top_p 0.8`, `top_k 20`). Leaving it on makes the recorded log-probs not the
  policy's. `build_model` pins all of it to identity - do not remove that.
- **TextWorld leaks ~1 MB per distinct game loaded** even with reuse, and ~5 MB
  if you construct a fresh env per task. Keep the pooled `AlfEnv` and keep
  `--n_train` bounded.
- **Mean-norm arms are confounded with undertraining.** Their advantages are
  3.5x smaller, so gradient norms fall below the 1.0 clip and the effective step
  shrinks. To test normalization properly, first match median gradient norms.

## The open research direction (not learning value)

GACA sets the mixing weight lambda_t from one scalar: a single-rollout NLL,
which Proposition 1 admits is only a *one-sided lower bound* on one of the five
quantities in the closed-form optimum
`lambda* = M_E / (M_E + M_S)`.

The unused axis is **estimator reliability, `M_S`** - not decision importance.
Anchor states currently build `A^S_t` but play no part in deciding how much to
trust it. Two free signals follow:

- **anchor-group size `m_z`**: the paper *proves* lambda* is nondecreasing in it
  and then never puts it in the gate. It is exactly known, being the bucket size
  already computed. It fixes a real pathology - a singleton step has no
  within-group contrast at all, yet a high NLL can still hand it lambda ~ 0.9.
- **observed disagreement / return spread within the anchor bucket**: measure
  `C_z` instead of bounding it.

Nearest prior art is Appendix B.4's leave-one-out score, already implemented as
`--score_mode loo`; it uses the anchor group for a *better estimate of the same
quantity*, not for estimator reliability. Beat that baseline.

## Working rules

- Keep the code small and simple: 5 source files in `src/`, no frameworks beyond
  TRL/PEFT/transformers. Test rigorously before long runs.
- **Append every new judgement call about scale, model, or data to
  `docs/DECISIONS.md`** (never to this file), with the measurement that motivated it.
- During training, report progress about every 10 minutes with an ETA.
- One training process at a time. Multiple agents/terminals have collided on
  this GPU before; the launchers in `legacy/4060/` are retired - do not run
  them.
- `pixi run test` (CPU, fast) and `pixi run test-train` (GPU) must pass before a
  long run.
