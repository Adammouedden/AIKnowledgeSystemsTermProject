# Judgement calls

Running log of every decision about model choice, data volume, or scale, with
the reason. Appended to as decisions are made. Nothing here changes between the
compared methods: GRPO, GiGPO and GACA share every setting except the advantage
estimator, which is the whole point of the comparison.

Hardware: single RTX 4060, 8188 MiB total with ~1.1 GiB already taken by the
display, so ~7 GiB usable. 16 CPU cores, 16 GiB RAM.

---

## 2026-11-07 — Benchmark: ALFWorld only, not WebShop

GACA evaluates on ALFWorld and WebShop. We reproduce **ALFWorld only**.

WebShop needs a local server over a 1.18M-product index (Java + pyserini +
~10 GiB of data) and is the heavier of the two to stand up. ALFWorld's text
(TextWorld) backend is pure CPU, installs from pip, and its data is ~2.3 GiB.
With one 7 GiB GPU, standing up a second benchmark would cost more than it
would add: the GACA-vs-GRPO contrast is already testable on ALFWorld, which is
also the benchmark the paper's own ablation grid (Table 2) runs on.

## 2026-11-07 — Model: Qwen2.5-1.5B-Instruct with LoRA

The paper uses Qwen2.5-Instruct at 1.5B and 7B with full fine-tuning. We use
**Qwen2.5-1.5B-Instruct**, the paper's smaller scale, but with **LoRA**
(r=32, alpha=64, on all attention and MLP projections, 36.9M trainable
parameters) rather than full fine-tuning.

Measured: bf16 weights are 3.09 GiB. Full fine-tuning would need weights +
bf16 grads + fp32 Adam moments = 3.1 + 3.1 + 12.4 = 18.6 GiB, and even 8-bit
Adam lands near 9.3 GiB. Neither fits in 7 GiB. LoRA keeps peak usage at
4.5-6.4 GiB measured end to end, including generation at batch 16.

Qwen2.5-0.5B-Instruct was measured as the cheaper alternative and rejected: on
the easiest ALFWorld task type it scored 0/24 episodes zero-shot and produced
off-task text (`'evaporate all fish...'`), so GRPO would have had an all-zero
reward signal and no gradient. The 1.5B scored 2/24 on the same tasks.

Consequence to state plainly: because the backbone is LoRA-adapted rather than
fully fine-tuned, our absolute success rates are not comparable to the paper's
table. Only the ordering between estimators, which is what the paper's claim is
about, is.

## 2026-11-07 — Learning rate 1e-5 instead of the paper's 1e-6

The paper's shared config uses lr 1e-6 for full fine-tuning. LoRA adapters need
a larger step to move the policy at all, so we use **1e-5**, constant, AdamW,
no weight decay. Same value for every method.

## 2026-11-07 — No KL penalty (beta_KL = 0 instead of 0.01)

The paper keeps a KL term to a reference policy at 0.01. We drop it. With LoRA
the reference policy is free in principle (disable the adapter), but evaluating
it costs a second forward pass over every step row, which would roughly double
the cost of the update, and we are compute-bound. Dropping the KL is standard
in recent critic-free recipes (e.g. DAPO) and, over the few hundred updates we
can afford, drift from the reference is small. Same for every method.

## 2026-11-07 — Action selection by index from the numbered admissible list

This is the largest deviation from the paper's setup and it was forced by
measurement, not preference.

The paper's agents emit free-form text actions and are charged a small penalty
for an inadmissible one. Measured on Qwen2.5-1.5B-Instruct zero-shot:

| prompt format | invalid-action rate | success |
|---|---|---|
| `<think>...</think><action>...</action>` | 69% | 2/24 |
| "reply with the action text, verbatim" | 88% | 0/32 |
| numbered list, "reply with the number" | 0.3% | see below |

At 1.5B the model cannot reliably copy an action out of a 10-40 item list, so
nearly every action was inadmissible and the reward signal was almost pure
penalty. Selecting by index fixes that and also shortens the sampled action to
1-3 tokens, which cuts generation time roughly fourfold.

This is a regime the paper itself names: Appendix B's `M_z` is "the number of
textual realizations of any canonical action", and it states `M_z = 1` is
exactly the case where "the policy is constrained to emit a canonical
admissible action". So index selection sits inside the paper's own analysis
rather than outside it, and it makes the action NLL `H_t` of Eq. 4 a clean
per-action quantity. The invalid-action penalty is retained (it fires on an
out-of-range or unparseable reply) but now rarely triggers.

We also drop the `examine X` and `close X` commands from the admissible set.
Neither is required by any ALFWorld task type, and removing them shrinks both
the action space and the prompt.

## 2026-11-07 — Task types: the two shortest-horizon ones

ALFWorld has six task types. We train and evaluate on **`pick_and_place_simple`
and `look_at_obj_in_light`** only (1098 of 3553 train games; 42 of 274
validation games before the room-size filter below).

The other four (`pick_clean_then_place_in_recep`, `pick_heat_...`,
`pick_cool_...`, `pick_two_obj_and_place`) add a sub-goal and need roughly twice
the horizon. At the success rate a LoRA-adapted 1.5B starts from, they would
have contributed almost no non-zero rewards, and a GRPO group in which every
rollout fails produces an exactly zero advantage and therefore no gradient at
all. Restricting to the two shortest-horizon types is what keeps the learning
signal non-degenerate inside our budget.

## 2026-11-07 — Max 25 environment steps instead of 50

The paper allows 50 steps on ALFWorld. We use **25**. Failed episodes always run
to the cap, so the cap sets the cost of the whole run almost single-handedly.
The two task types we keep need roughly 8-15 steps when solved, so 25 leaves
real slack for search while halving the rollout bill.

## 2026-11-07 — Prompt kept deliberately short

Every environment step re-encodes the prompt, so prompt length sets the cost of
both the rollout prefill and the policy update. Measured: the rollout is
prefill-bound (about 4 s per lockstep generation call at batch 16 with ~900
token prompts, for only 8 generated tokens). The prompt therefore carries a
compact system message, the last 6 steps with observations truncated to 110
characters, a one-line list of already-visited receptacles (loop avoidance at
~15 tokens instead of a 300-token history), what the agent is carrying, the
current observation, and the numbered action list. This brought prompts from
~900-1300 tokens down to ~570-820.

## 2026-11-07 — Env instances are pooled and reused

Not a scale decision but it constrains one. TextWorld interns game logic in
process-global caches, so each freshly constructed ALFWorld environment leaks
about 4.7 MiB that neither `close()` nor `gc.collect()` returns; a first attempt
to scan all 3553 games in a 12-process pool exhausted 16 GiB of RAM and swap.
Reusing one environment object via `load()` costs **1.0 MiB per distinct game**
instead. The trainer therefore keeps a fixed pool of 16 reusable runners and
caps the number of distinct games a run touches (`--n_train`, default 256), so
the leak stays bounded at a few hundred MiB over a full run.

## 2026-11-07 — Qwen's sampling preset had to be overridden

Qwen2.5's shipped `generation_config.json` sets `repetition_penalty: 1.1`,
`temperature: 0.7`, `top_p: 0.8`, `top_k: 20`. Those are logits processors, so
leaving them on meant the log-probs recorded during generation were *not* the
policy's: sampling was off-policy and the action NLL feeding GACA's criticality
score was biased. All four are pinned to the identity. The alignment test's
measured error between recorded and recomputed log-probs dropped from 0.089 to
0.035 nats once this was fixed, and now sits at the bf16 noise floor.

## 2026-11-07 — Rollout log-probs come from generation scores, not a second pass

`H_t` is read off `generate`'s own scores rather than recomputed with a
dedicated forward pass. This is what the paper prescribes ("It costs nothing to
compute and requires no second forward pass"), and a second pass over every
step row would have cost about a third of an update. The price is that bf16
prefill-versus-decode arithmetic makes the recorded NLL differ from a recomputed
one by ~0.03-0.05 nats, measured to be the same size as the bf16 noise floor
itself. Against a within-trajectory NLL spread of a few tenths of a nat this is
real but modest noise, and it is the same single-sample noise the paper
acknowledges in its limitations.

## 2026-11-07 — Room-size filter: rooms with <= 20 admissible actions at reset

Kitchens expose ~36 receptacles; bedrooms and bathrooms ~10-17. We keep only
tasks whose room offers **at most 20 admissible actions at reset**, which keeps
207 of 256 sampled training tasks, 37 of 48 `valid_seen` and 27 of 42
`valid_unseen` (64 held-out evaluation tasks in total).

Two reasons, both measured. The action list is the largest block of the prompt,
so smaller rooms cut prompt tokens and therefore both rollout and update cost.
And search is the binding difficulty: finding one object among 36 receptacles
inside a 25-step cap is close to hopeless for this policy, so the big-kitchen
tasks contributed almost exclusively all-fail groups, which carry no gradient.

## 2026-11-07 — Scale of the run: 90 iterations, 1 seed, 3 methods

Measured on this GPU: **76 s rollout + 44 s update = 120 s per iteration** at
2 tasks x 8 rollouts x <=25 steps, with peak VRAM 5.96 GiB. Evaluating all 64
held-out tasks costs about 5 minutes.

The paper trains 150 epochs and averages 3 seeds. We run **90 iterations, one
seed, per method** (about 3.0 h of training plus ~0.5 h of evaluation each),
evaluating every 18 iterations for a 6-point learning curve. Three methods
therefore cost ~10.5 h. The brief's 6-hour ceiling is per run; each run here is
about 3.5 h.

Order is GRPO, then GACA (the comparison the brief asks for), then GiGPO (the
control that separates GACA's adaptive weight from GiGPO's fixed one), so the
required pair completes first.

Consequence to state plainly: ~90 optimizer steps on 180 task-groups is one to
two orders of magnitude less training than the paper's, and at a ~6% starting
success rate roughly half of the GRPO groups have every rollout fail and so
produce an exactly zero advantage and no gradient. A null result between
methods is a likely and legitimate outcome at this budget.

## 2026-11-07 — Methods are matched on iterations, not wall-clock

An iteration whose advantages are all exactly zero carries no gradient, so the
update is skipped. This happens more often for GRPO (an all-fail group zeroes
the episode advantage outright) than for GACA or GiGPO, so GRPO finishes its
90 iterations sooner. The budget that is held fixed is the **rollout budget**
(iterations, tasks, group size, step cap), which is what the paper and the
proposal match on; wall-clock is allowed to differ.

## 2026-11-07 — One optimizer step per rollout batch

TRL's GRPO exposes `num_iterations` (inner epochs over one batch of rollouts).
We use one, so the update is exactly on-policy: the importance ratio is 1 and
the clip never binds. The surrogate is still written out in full as Eq. 1. More
inner epochs would buy more optimizer steps per rollout, but they need stored
old-policy log-probs for the ratio, and the only ones we have carry the bf16
prefill/decode discrepancy noted above. Given the instruction to keep the code
simple and avoid backtracking on bugs, the extra machinery was not worth it.

## 2026-11-07 — Advantage estimators are our own code, not TRL's GRPOTrainer

The brief asks for TRL's GRPO. We read `trl.GRPOTrainer` (TRL 1.17, 3528 lines)
and did not use it, because its contract does not fit multi-turn agent RL with
per-step credit:

- it computes one advantage per *sequence* and broadcasts it to that sequence's
  tokens, whereas GiGPO and GACA need one advantage per *environment step*, with
  a trajectory contributing a variable number of step rows;
- `_prepare_inputs` slices a generation batch into `steps_per_generation`
  equal microbatches, so the number of rows per iteration has to be fixed and
  divisible, while a rollout iteration yields a variable count (episodes end at
  different times).

Making it fit would have meant overriding both `_generate_and_score_completions`
and `_compute_loss` -- that is, both halves of the trainer -- leaving only the
optimizer plumbing, at a much higher bug risk than the ~150-line update loop in
`src/train.py`. That loop implements Eq. 1 directly (token-level clipped
surrogate, per-step advantage broadcast through the response mask, length
normalization inside each step) and is covered by `src/test_train.py`, which
pins the log-prob alignment against a deliberately shifted control and a
measured bf16 noise floor.

## 2026-11-07 — Time budget raised to 5 h for GACA and GiGPO

Measured in flight: GRPO runs at **125 s/iteration**, but that average already
includes skipping the update on the ~half of its iterations whose advantages
are all zero. GACA and GiGPO skip far less often, so they cost roughly
145 s/iteration and would have hit the original 3.8 h ceiling at about
iteration 85 of 90.

Truncating one method and not another would break the matched-iteration
comparison, which is the whole basis of the result, so the ceiling for the
remaining two runs is 5 h. It is a safety net against a hang, not a target:
all three are expected to complete the full 90 iterations.

## 2026-11-07 — Restart: sparse-only reward and the paper's KL penalty restored

The first GRPO attempt (archived in `runs/attempt1_penalty0.1/`) was stopped at
iteration 37 of 90 because it was degrading, not learning:

| iters | held-out success | train success | invalid actions | action NLL |
|---|---|---|---|---|
| 0  | 3.1% | --   | 3.8% | 0.531 |
| 18 | 4.7% | 5.7% | 0.9% | 0.533 |
| 36 | **1.6%** | 2.1% | **0.2%** | **0.385** |

The policy drove invalid actions to zero and its action NLL down, and ended up
*worse than zero-shot*. Diagnosis: with index-based action selection the
invalid-action rate starts near zero anyway, so a 0.1 penalty was the only
reliably learnable term in the return, while task success (~5%, sparse) was
not. GRPO optimized the dense term to extinction and lost the entropy that a
search task needs. Two changes:

**Invalid-action penalty 0.1 -> 0.** The paper charges this penalty because its
agents emit free-form text and get it wrong often. Ours select an index, so the
penalty was mostly a distractor with more learnable signal in it than the task
itself. With it removed the return is exactly the sparse success indicator the
paper's credit-assignment problem is posed over. A side effect is honest: when
every rollout in a group fails, all advantages are exactly zero and the
iteration is skipped, which is the correct response to no signal.

**KL penalty restored to the paper's 0.01.** Reverses the earlier decision to
drop it. It is the regularizer that resists exactly the entropy collapse
observed above, and it is what the paper's Table 3 specifies. With LoRA the
reference policy is the same weights with the adapter switched off, so it costs
one extra forward pass per microbatch and no extra parameters. Implemented with
the low-variance k3 estimator, as the paper's "KL loss (low-variance
estimator)" row indicates.

The implementation is pinned by a test that is exact rather than statistical:
with LoRA's B initialised to zero the adapter is the identity, so the measured
KL must be *exactly* 0.0, and it is; after perturbing the adapter it becomes
0.2768, and the loss at zero advantage equals beta * KL * rows to within 1e-3.

Iterations reduced 90 -> 75 (eval every 15) to absorb the extra reference
forward pass and keep each run near 3 h.

---

# Attempt 2

## 2026-11-07 — Attempt 1 failed: the invalid-action penalty was the dominant gradient

Attempt 1 (GRPO, 38 iterations, archived in `runs/attempt1_penalty0.1/`) did not
plateau, it **degraded**:

| iteration | 0 | 18 | 36 |
|---|---|---|---|
| held-out success | 3.1% | 4.7% | 1.6% |

| block of 9 iterations | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| invalid-action rate | 4.4% | 1.6% | 0.36% | 0.17% | 0.37% |
| train success | 5.6% | 6.9% | 3.5% | 2.1% | 0.0% |
| mean abs advantage | 0.81 | 0.61 | 0.31 | 0.25 | 0.38 |

Diagnosis. At a ~5% success rate almost no group of 8 rollouts contains a win,
so the group-relative advantage was set almost entirely by *which rollout made
fewer invalid actions*. The policy duly optimized that to ~0.2%, grew more
confident while doing so (action NLL 0.53 -> 0.39), and solved fewer tasks.

The root cause is a design error of ours, not of the paper. The paper charges a
small invalid-action penalty because its agents emit free-form text actions that
are frequently inadmissible. Under the index-based action selection we adopted,
inadmissible actions are nearly impossible (0.3% zero-shot), so the penalty
stopped being a small corrective and became the only thing the gradient could
see.

**Change: `--invalid_penalty 0.0`.** The reward is now exactly the paper's
`r_T in {0,1}` success indicator with no shaping.

This worked. Measured over the first 41 iterations of attempt 2, against the
same stretch of attempt 1:

| | attempt 1 (penalty 0.1) | attempt 2 (penalty 0.0) |
|---|---|---|
| best held-out success | 4.7% | **7.8%** |
| mean abs advantage | 0.81 -> 0.25 (collapsing) | 0.10 -> 0.28 (growing) |
| action NLL | 0.53 -> 0.39 (collapsing) | 0.55 -> 0.51 (stable) |
| gradient norm | ~6 (clipped 6x) | 1.4-3.0 |
| invalid-action rate | driven to 0.2% | steady ~5%, untouched |

## 2026-11-07 — Collapse guard: KL penalty 0.01, learning rate left at 1e-5

Attempt 1 drifted into a confident, narrow policy (action NLL 0.53 -> 0.39)
with gradient norms around 6x the 1.0 clip threshold. Two guards were on the
table: halve the learning rate (free) or restore the paper's KL penalty to a
reference policy (a second forward pass per microbatch, about +35 s/iteration).

**The configuration actually running uses the KL penalty at beta_KL = 0.01 with
the learning rate left at 1e-5**, which is what the paper's Table 3 specifies.
With LoRA the reference policy is just the adapter switched off, so it needs no
extra parameters, and it is implemented with Schulman's k3 estimator
(`exp(d) - d - 1`, non-negative and lower-variance than the plain log-ratio).

This supersedes an earlier note in this file that said the KL term was dropped
for compute reasons; it is now on. The measured cost is visible in the run
logs as `kl_ref`, which rose to ~0.24 and settled around 0.08-0.15.

## 2026-11-07 — Zero-advantage rows dropped before subsampling

With a pure sparse reward, every row of a group in which no rollout succeeded
has an advantage of exactly zero and contributes exactly zero gradient. Since
only `rows_per_update`=96 of ~400 collected rows enter an update, roughly half
the update budget would have been spent on rows that cannot move the policy
whenever one of the two groups had no win. Rows are now filtered to those with a
non-zero advantage before subsampling, which doubles the informative batch at no
cost. Advantages are still computed from all steps, so credit assignment is
unchanged.

## 2026-11-07 — Qwen2.5-0.5B-Instruct measured and rejected (again)

Re-tested under the full attempt-2 configuration, because the earlier rejection
used the free-form prompt that 0.5B could not follow and index selection might
have rescued it. Measured on the same 64 held-out tasks:

| backbone | zero-shot success | eval wall-clock |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 3.1% (2/64) | 333 s |
| Qwen2.5-0.5B-Instruct | 1.6% (1/64) | **528 s** |

0.5B is both worse *and slower*. Slower is the counter-intuitive part: at
`max_new_tokens=8` a batch finishes as soon as every sequence emits EOS, and the
1.5B reliably emits it after one or two tokens while the 0.5B keeps generating
to the cap. The saved parameter FLOPs are then more than cancelled by decoding
~4x more tokens per step, on top of the per-step environment and tokenization
work that is identical for both models. **Staying at 1.5B.**

## 2026-11-07 — Scope cut to fit the compute budget

Three runs at 90 iterations is ~9 h, which is more than is available. Attempt 2
therefore runs **75 iterations** per method, evaluating every 15 for a 6-point
curve. Measured cost is 123 s/iteration (90 s rollout + 33 s update) and 333 s
per 64-task evaluation, so each method takes about **3.0 h**.

Order is GRPO, GACA, GiGPO. The brief requires GACA versus vanilla GRPO, so that
pair finishes first and **GiGPO, which is our own addition, is the one to drop
if the budget runs out.** Cutting iterations further was rejected: at this
success rate the experiment is already signal-starved, and halving training
would make a null result near-certain for reasons having nothing to do with
credit assignment.

## 2026-11-07 — GiGPO dropped; GRPO vs GACA only

The third run (GiGPO, the control separating GACA's state-adaptive weight from
a fixed one) is dropped. GRPO took 2.92 h, so three runs plus evaluation would
have finished near 22:00, and the brief asks specifically for GACA against
vanilla GRPO. The two runs that remain are the ones the comparison needs.

Config parity was checked programmatically rather than by eye: the GRPO and
GACA run arguments differ in exactly two keys, `method` and `out`. Everything
else -- tasks, task order, seed, group size, step cap, learning rate, LoRA
rank, KL weight, row budget -- is identical.

GiGPO remains one flag away (`--method gigpo`) for anyone with more compute,
and `src/advantages.py` already implements it and the paper's `fixed`-weight
and `uniform`/`random` ablation controls.

## 2026-11-07 — Index actions widen GACA's criticality score 3x (a consequence worth naming)

First training-run diagnostics for GACA, against the paper's Table 7 (ALFWorld):

| diagnostic | ours | paper |
|---|---|---|
| step-advantage coverage | 0.84 | 0.86 |
| c_t mean / std | 1.00 / 0.98 | 1.00 / 0.31 |
| lambda_t mean / std | 0.49 / 0.21 | 0.50 / 0.08 |
| clamped steps | 0.049 | < 0.01 |

Coverage matching to within 0.02 says the anchor-state grouping GACA inherits
from GiGPO behaves exactly as published, which is the part of the pipeline most
likely to have been silently wrong.

The criticality score, though, is three times more dispersed than theirs, and
that follows directly from our index-based action selection. `H_t` is the *mean
per-token* NLL of the action. The paper's free-form actions run to ~20 tokens,
so that mean is an average over many tokens and concentrates tightly. Ours are
1-3 tokens, so there is almost no averaging and the per-step score swings far
more widely.

The knock-on effect is that `lambda_t` has std 0.21 rather than 0.08 and the
clamp binds on ~5% of steps rather than <1%. Our gate therefore sweeps most of
[0, 1] instead of hovering in a narrow band around `lambda_0`, making this a
more aggressive version of the mechanism than the one the paper tuned. The
paper's own ablation found over-modulation hurts: `alpha = beta = 1.0` scored
1.44 points below `0.5`. We keep `alpha = beta = 0.5` because it is the paper's
default and changing it would stop being a reproduction, but the honest reading
is that our GACA sits somewhere between the paper's `alpha = 0.5` and `1.0`
rows. Matching their realized `lambda` spread would need `alpha` near 0.15.

## 2026-11-07 — Seed-variance control added (seed 1, both methods)

The first comparison came back indistinguishable: GRPO best 14.1%, GACA best
15.6%, McNemar p = 1.000 on the paired per-task outcomes. The paired data also
showed the two methods' success sets overlapping on exactly 1 task where chance
alone predicts 1.4, i.e. *which* tasks a run solves looks like a property of the
run rather than of the estimator.

That makes the most valuable remaining use of the GPU a direct measurement of
the noise floor rather than a third estimator. Both methods are therefore re-run
at `--seed 1`. Task selection and ordering go through `random.Random(1234)`
independently of `--seed`, so these runs see the same tasks in the same order
and differ only in sampling randomness; the spread between two seeds of the
*same* method is then the correct yardstick for judging any difference between
methods.

This was preferred over running GiGPO. A third estimator measured on the same
unknown noise floor would not have resolved anything; knowing the noise floor
makes every comparison in the report interpretable, including the existing one.

---

# Attempt 3: mean-norm GACA

## 2026-11-07 — Re-running GACA under mean-norm, on the paper's own reasoning

We ran ALFWorld with `mean-and-std-norm`, which is the paper's stated ALFWorld
convention (Table 3). Remark 3 of the paper says that is the setting in which
GACA should look *worst*:

> "Mean-and-std normalization, used on ALFWorld, rescales each estimator by its
> own group standard deviation and therefore removes exactly that information,
> driving lambda*_z toward 1/2 uniformly in z. Read literally, the theory then
> predicts a smaller spread of per-state optima under mean-and-std norm than
> under mean-norm, and by Proposition 2 a smaller gain from adapting."

Three things follow:

1. Mean-norm is a sanctioned variant, not a departure: Table 4 lists the search
   range as `{mean-norm, mean-and-std-norm}`, and the paper says the per-
   benchmark convention was chosen "for comparability with Feng et al. (2026)
   rather than because the theory prefers either."
2. Their own numbers agree with the prediction: gains of +13.6/+8.4 on the
   mean-norm benchmark (WebShop) against +10.2/+2.4 on the mean-and-std one
   (ALFWorld).
3. They concede the controlled experiment was never run: "the two benchmarks
   differ in far more than their normalization, so this is a consistency check
   and **not a test**."

Holding ALFWorld fixed and varying only the normalization is therefore a test
the paper itself identifies as missing. The mechanism hypothesis is concrete:
under mean-and-std, GACA's per-step lambda_t is adapting on top of a
normalization that has already forced the optimal weight to 1/2 everywhere, so
there is nothing left for adaptation to exploit.

Configuration: identical to `runs/gaca` except `--norm mean`. Same seed, same
task order, same 64 evaluation tasks.

Checked before launching: an all-fail group still yields *exactly* zero
advantage under mean-norm, so the zero-gradient skip still fires. Advantage
magnitudes are about 2.3x smaller (0.356 vs 0.833 on a 2-of-8-wins group), but
observed gradient norms were 2.4-2.6 against a clip threshold of 1.0, so updates
stay in the clipped regime where direction dominates and the effective step size
is largely unchanged. The learning rate is left at 1e-5, as the paper also uses
one learning rate across both normalizations.

**Caveat on what this can conclude.** This is a GACA-vs-GACA ablation. A better
mean-norm GACA number cannot be compared against the mean-and-std GRPO run:
normalization affects GRPO too. Claiming anything about GACA versus GRPO under
mean-norm needs a matched GRPO mean-norm run, which is a further ~3 h.

---

# Fidelity audit (sweep of the paper against this repo)

## Settings: Table 3 / Table 4 versus ours

| Setting | Paper (ALFWorld) | Ours | |
|---|---|---|---|
| Group size G | 8 | 8 | match |
| Discount gamma | 0.95 | 0.95 | match |
| KL loss (k3 estimator) | 0.01 | 0.01 | match |
| Max prompt length | 2048 | 2048 | match |
| Advantage normalization | mean-and-std | mean-and-std (+ mean-norm run) | match |
| Reported checkpoint | best held-out | best held-out | match |
| criticality score / alpha_c / lambda_0 / alpha_lam / beta_lam / w | entropy / 1.0 / 0.5 / 0.5 / 0.5 / 1.0 | identical | match |
| Backbone | Qwen2.5-Instruct 1.5B & 7B, full fine-tune | 1.5B + LoRA r=32 | VRAM |
| Learning rate | 1e-6 | 1e-5 | LoRA needs a larger step |
| Training epochs | 150 | 75 iterations | compute |
| **Max environment steps** | **50** | **25** | **compute - and it binds, see below** |
| Invalid-action penalty | 0.1 | 0.0 | measured failure, see attempt 1 |
| Held-out eval tasks | 128 | 64 | compute |
| Seeds | 3 | 1 (a second GRPO seed is now running) | compute |
| Prompts per optimizer step | unspecified; "aligned with the public GiGPO recipes" | 2 | unknown gap, likely 4x small |

Appendix D is implemented in full and covered by tests: row-wise group
statistics with the unbiased `ddof=1` divisor and the length-weighted mean; the
singleton anchor-group fallback to prompt-group statistics; reward-gated
criticality with `Delta_1 = 0`; per-step advantage scalars broadcast through the
response mask; and both ablation controls (`--score_mode uniform|random`).

## The 25-step cap is costing us successes

Step counts of the successful episodes in the paired evaluation:

```
GRPO  winners:  3, 3, 3, 5, 10, 14, 18, 21, 23
GACA  winners:  5, 6, 9, 9, 13, 14, 16, 20, 21, 25
every failure: exactly 25 (the cap)
```

The distribution of winning episode lengths does not decay before the cap -
three of nine GRPO winners and three of ten GACA winners needed 18 or more
steps, and one GACA win landed exactly on 25. ALFWorld has no failure
termination, so every unsuccessful episode runs the full budget. Together these
say the cap is truncating episodes that were still making progress, and that
raising it toward the paper's 50 would convert some near-misses into successes.

That matters out of proportion to its size, because success rate is what feeds
the credit-assignment signal: at ~10% success a third of GRPO iterations have no
gradient at all. Raising the cap is the most direct lever on the regime problem
we have. The cost is close to linear in the cap (rollout is ~70% of iteration
time and every failure runs to the limit), so 50 steps would put an iteration
near 190 s and a 75-iteration run near 4 h - still inside the 6 h per-run
ceiling in the brief.

**Recommendation: the next run to spend compute on is max_steps 40-50, not
another estimator variant.**

## Components implemented and tested but never run

These are all one flag away and would each cost ~3 h:

- **GiGPO** (`--method gigpo`). The paper's own control, "the control that
  isolates GACA's contribution, since the two share the entire two-level
  decomposition". Without it we cannot separate the step-level advantage from
  the adaptive weight.
- **Uniform and random-permutation controls** (`--score_mode uniform|random`).
  The paper calls the random control "our cleanest evidence", because it keeps
  the whole distribution of mixing weights and changes only which step gets
  which. Running it is only worthwhile once there is a gain to attribute.
- **The lambda_0 sweep** {0, 0.3, 0.5, 0.7, 0.9, 1.0} via `--lam0`, which
  reproduces the paper's Figure 2c, and the modulation sweep {0, 0.25, 0.5,
  1.0} via `--alpha_lam`, and the fusion sweep {1.0, 0.7, 0.5} via `--alpha_c`.

## Implemented after the sweep

- **The debiased criticality variant of Appendix B.4** is now available as
  `--score_mode loo`. It scores a step by the leave-one-out mean of `ell_j`
  over the *other* visits to its anchor group, so the score is conditionally
  independent of the action this rollout actually took. That makes the
  reweighting an unbiased rescaling of the policy gradient, which the default
  single-sample score is not, and cuts the score variance from
  `Var(ell | z)` to `Var(ell | z) / (m_z - 1)`. It gives up within-group
  resolution, and singleton anchor groups keep their own `H_t` because no
  leave-one-out average exists. Covered by `test_loo_criticality`.

  The paper calls the choice between this and the single-sample score "an
  empirical question we have not settled". It is plausibly *more* relevant here
  than in the paper: our criticality score is three times more dispersed
  (c_t std 0.98-1.04 against their 0.31) precisely because a 1-3 token action
  gives `H_t` almost no within-action averaging, and the leave-one-out mean
  restores averaging across the anchor group instead. With step coverage at
  0.82-0.84, most steps would actually get one.

## Not implemented

- **7B scale and WebShop**, both out of reach on this hardware.

## 2026-11-07 — Normalization study replaces the seed-variance runs

The seed-1 runs were stopped after 18 GRPO iterations (archived in
`runs/partial_grpo_seed1/`) and the GPU redirected to a normalization study:
GACA under **mean-norm** (`runs/gaca_mn`) instead of the mean-and-std-norm used
for ALFWorld so far.

The motivation is the paper's own Remark 3: mean-and-std normalization rescales
each estimator by its group standard deviation, which "removes exactly that
information, driving lambda*_z toward 1/2 uniformly in z", and by Proposition 2
shrinks the gain from adapting. The paper notes its measured gains were indeed
larger on the mean-norm benchmark (WebShop, +13.6/+8.4) than on the
mean-and-std-norm one (ALFWorld, +10.2/+2.4), but says plainly that the two
benchmarks "differ in far more than their normalization, so this is a
consistency check and not a test". Holding the benchmark fixed and varying only
the normalization is the controlled test the paper did not run.

**A matched GRPO mean-norm arm was added** (`runs/grpo_mn`), queued behind it.
Remark 3 predicts a change in the GACA-minus-GRPO *margin*, not in GACA's
absolute success, so GACA under mean-norm is only interpretable against GRPO
under mean-norm. The design is a 2x2:

|        | mean-and-std-norm | mean-norm        |
|--------|-------------------|------------------|
| GRPO   | `runs/grpo`       | `runs/grpo_mn`   |
| GACA   | `runs/gaca`       | `runs/gaca_mn`   |

All four share every other setting. The caveat from the seed data still stands
and applies here too: a margin of one or two tasks is inside run-to-run noise,
so this 2x2 can only show a difference large enough to clear that.

## 2026-11-07 — Mean-norm carries a mild step-size handicap (measured mid-run)

Before launching I argued that mean-norm's smaller advantages would be absorbed
by gradient clipping, because observed gradient norms sat above the 1.0 clip
threshold. Measured on the live runs, that is mostly but not entirely true:

| | GACA mean-and-std | GACA mean-norm |
|---|---|---|
| mean abs advantage | 0.314 | 0.091 |
| gradient norm, mean / median | 4.11 / 4.07 | 1.47 / 1.37 |
| fraction of updates above the 1.0 clip | 0.98 | 0.88 |
| KL drift from the reference policy | 0.246 | 0.099 |

The advantage magnitude is 3.5x smaller, not the 2.3x a synthetic two-of-eight
group predicted, because real groups are mostly all-fail or single-win and the
group standard deviation they are divided by is correspondingly small.

Consequence: 88% of mean-norm updates are still clipped to unit norm and so take
the same size step as the mean-and-std run, but 12% now fall below the threshold
and take a genuinely smaller one, against 2% for mean-and-std. The lower KL
drift confirms the mean-norm policy moves less overall. So a null or negative
mean-norm result is **partly confounded with undertraining** and should not be
read as a clean refutation of Remark 3.

The clean version of this experiment would either raise the learning rate for
the mean-norm arm until median gradient norms match, or drop gradient clipping
so that advantage magnitude passes through to the step size. We keep lr at 1e-5
and the clip at 1.0 because the paper uses one learning rate across both
normalizations, and because changing two things at once would make the arm
uninterpretable against the existing runs.

## 2026-11-07 — Scale confound in the normalization study

Measured mid-run, mean-norm does not only remove the per-state rescaling the
paper's Remark 3 is about; it also changes the magnitude of the advantage:

| | mean-and-std-norm (`runs/gaca`) | mean-norm (`runs/gaca_mn`) |
|---|---|---|
| mean abs advantage | 0.234 | 0.061 |
| mean gradient norm | 3.07 | 0.88 |
| train success | 9.3% | 8.1% |

Standardizing divides by a group standard deviation well below one at these
success rates, so it inflates the advantage; dropping it leaves the raw
return contrast, roughly four times smaller. The mean-norm arms therefore train
with a much smaller effective step.

Both arms of the mean-norm column take the same smaller step, so the
*margin* comparison (GACA - GRPO within a normalization) stays matched and the
2x2 is still worth reading. But a smaller mean-norm margin would be consistent
with either less to gain from adapting (Remark 3) or simply less effective
training, and these runs cannot separate the two. Doing so would need the
mean-norm arms re-run at a compensating learning rate. This caveat is printed
in `RESULTS.md` alongside the 2x2 so the table cannot be read without it.

## 2026-11-07 — Spent launchers retired

`seeds.sh`, `gaca_only.sh` and `run_meannorm.sh` renamed to `*.superseded`,
matching the convention already used for `run_all.sh` and `chain.sh`. All three
had finished or been abandoned, and none of them waits for the GPU any more
(the `pgrep` wait in `seeds.sh` had to be removed because the pattern written
into a script also appears in the command line of the shell that created it,
so it self-matched and hung). Left runnable: `finalize.sh`, and
`run_grpo_meannorm.sh` while its run is in flight.
