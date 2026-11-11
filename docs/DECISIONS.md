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

---

# Attempt 4: A100 40 GB (2026-11-08)

## 2026-11-08 — Hardware and environment

Moved to a single **A100-PCIE 40 GB**, 24 CPU cores, 165 GiB RAM, 32 GB disk,
no persistent volume (anything not pushed off-box is lost on recycle). pixi's
installer could not be run here, so the environment is a uv venv (`.venv`,
`source env.sh`) built from the *exact* versions in `pixi.lock` (torch
2.10.0+cu128, transformers 5.22.2, trl 1.17.1, peft 0.21.2, alfworld 0.4.2,
textworld 1.7.0). `pixi.toml` paths were made project-relative. Both suites
pass unchanged, and the index-action zero-shot evaluation reproduces the
4060's number exactly (3.1%, 2/64, invalid 3.9%), so the port is faithful.

## 2026-11-08 — Free-form text actions restored (the index hack goes)

The brief's first item: was the 88% invalid-action rate a prompt problem? Yes.
Zero-shot, greedy, the same 64 held-out tasks, 25-step cap:

| action format | invalid | success |
|---|---|---|
| index into numbered list (attempts 1-3) | 3.9% | 3.1% (2/64) |
| text, 2 worked trajectories as one transcript in the system prompt | 67.7% | 3.1% (2/64) |
| text, the same 2 trajectories as prior chat turns *with the admissible list shown at every step* | **19.5%** | **7.8% (5/64)** |

The middle row is instructive. Its examples showed actions but never the list
they were chosen from, so the model learned to *compose* actions: 40% of its
invalid replies named objects copied out of the examples (`desk 2`,
`desklamp 1`, `sidetable 1`). Showing the worked trajectories in exactly the
live-step format, admissible list included, teaches copying instead. The
residual 20-27% (sampled, T=1) is spread over genuine mistakes -- `open` on a
non-openable receptacle, `take` of an absent object, paraphrases -- i.e. real
inadmissible actions of the kind the paper's penalty exists for.

Parsing is strict: a reply is admissible only if, up to case, whitespace and
trailing punctuation, it *is* an admissible command. Anything else goes to the
environment verbatim ("Nothing happens.") and is counted invalid. No fuzzy
rescue, so the rate above is the real one.

Under the brief's own criterion (below ~20%) this clears the bar, and the
zero-shot success rate more than doubles, so text actions become the default
and the index path is retired. Cost: the few-shot turns put prompts at
~2,600 tokens, so `--max_prompt_len` is 4096 for text mode (2048 would have
truncated the live turn), and actions are ~6-10 tokens (`--max_new_tokens`
24). The prompt-length cost is recovered by the prefix cache below.

## 2026-11-08 — The paper's response format: reasoning + action, ~512-token cap

Checked against the sources the paper says it follows (verl-agent, "the public
GiGPO recipes"). Its ALFWorld prompt asks the model to "reason step-by-step ...
within <think> </think> tags" and then give the action inside <action>
</action>, with `max_response_length=512`. So a step's "action" a_t in the
paper -- and hence GACA's H_t, the mean NLL over a_t's tokens -- is the whole
reasoning-plus-action response, not a bare 6-token command. Our text mode
(6.5 tokens/step) and index mode (1-3) both mismatch this, and this is also the
direct cause of our 3x wider c_t spread (DECISIONS: "Index actions widen
GACA's criticality score 3x"). The paper's format is now the default.

The same recipe also fixes, per arm: 16 tasks x 8 rollouts per iteration (the
brief guessed 8), 150 iterations, lr 1e-6 full fine-tuning, PPO mini-batch 256
step-rows, KL 0.01 low_var_kl, invalid-action penalty 0.1 (format-based: both
tags present, no Chinese characters), validation sampled at T=0.4 on 128
episodes.

## 2026-11-08 — Moved the experiment to verl-agent (vLLM rollouts)

With ~100-500 reasoning tokens per step, generation dominates and HF
`generate` in lockstep batches is the wrong tool: a 64-task zero-shot
evaluation with the reasoning prompt was still unfinished after ~55 minutes.
The paper's own per-iteration cost (Table 5: 371 s, 319 s of it rollout, on
>=62 GB-per-GPU hardware) puts 150 iterations at ~15.5 h per arm; the brief's
~4 h per arm (training + evaluation, confirmed by the user) therefore caps us
near 20% of the paper's iterations even at the paper's speed, and an HF
pipeline would reach ~6-10%. verl-agent closes most of the throughput part of
that gap and is the paper's exact pipeline (prompts, anchor grouping,
mini-batched PPO, KL estimator, penalty), so most remaining deviations vanish.

What was changed in verl-agent (all marked "term project" in the source):

- `gigpo/core_gaca.py` + `algorithm.adv_estimator=gaca`: every batch row
  becomes the step record of our `src/advantages.py` (symlinked in as
  `gigpo/gaca_advantages.py`, so the unit-tested code *is* the code that
  runs). All arms -- GRPO, GACA, GACA + reliability -- go through it and
  differ only in `algorithm.gaca.method` / `algorithm.gaca.rel`. H_t is read
  from `old_log_probs`, which verl computes before the advantage step anyway,
  exactly as the paper says ("no second forward pass").
- `AlfworldMultiWorker`: 8 environments per Ray actor. The container caps its
  whole process tree at 2,816 threads (`pids.max`); one actor per env (128
  train + 128 val) cannot start (`fork: Resource temporarily unavailable`).
  Each env keeps its own instance and seed, so behaviour is unchanged.
- `rollout_loop._dump_episodes`: per-episode validation outcomes (gamefile,
  won, length) to `val_episodes.jsonl`, for paired McNemar tests.

Single-GPU deviations from the recipe (`src/verl/run_arm.sh`): tensor parallel 1
(recipe 2), vLLM at 35% of the 40 GB card (recipe 60% of 80 GB), Adam state
offloaded to CPU, vLLM's own FlashAttention backend (the recipe's xformers
wheel does not load against torch 2.8 here), KV cache kept resident because
releasing it requires disabling CUDA graphs.

Environment: `/workspace/verl-agent/.venv`, Python 3.12, vllm 0.11.0 (torch
2.8.0+cu128), flash-attn 2.8.3 prebuilt wheel, transformers 4.57.3, ray 2.50,
alfworld 0.4.2. The earlier HF venv was deleted to free disk (32 GB total);
it can be rebuilt from `pixi.lock` in ~3 minutes.

## 2026-11-08 — Full ALFWorld benchmark, not the two-task-type subset

The two-task-type, small-room subset of attempts 1-3 existed to keep a
LoRA-on-7 GB policy above zero reward. Under verl-agent the paper's own
setting -- all six task types, its 128 validation episodes -- is the default
and costs nothing extra, and it is the only setting the paper's numbers refer
to. We use it.

## 2026-11-08 — Batch shape and per-arm budget (measured on verl-agent)

Measured, one A100 40 GB, paper recipe otherwise:

| tasks x G | rollout | old+ref log-probs | update | **iteration** | trajectories / h |
|---|---|---|---|---|---|
| 16 x 8 (paper) | 334 s | 109 + 147 s | 482 s | **1076 s** | 428 |
| 4 x 8 (used) | 228 s | 32 + 44 s | 145 s | **451 s** | 255 |
| paper, Table 5 (16 x 8, larger hardware) | 319 s | 23 s | 30 s | 371 s | 1242 |

Rollout now matches the paper; the training passes are 10-16x slower than
theirs (one GPU against what must be several). Rollout is latency-bound by
the 50-step lockstep, so it barely shrinks with the batch, which makes the
paper's 16 x 8 the more trajectory-efficient shape (~1.7x).

**Chosen: 4 x 8, wall-clock budget 3.75 h of training per arm** (+ ~10 min of
setup and the initial validation, = the user's ~4 h per arm), up to the paper's
150 iterations, validation on 128 episodes every 10 iterations and at the end.
That is ~29 iterations / ~930 trajectories per arm, against 150 / 19,200 in
the paper (~19% of its iterations, ~5% of its trajectories). 16 x 8 would give
~12 iterations / ~1,500 trajectories. With one PPO update per iteration at
this batch (the 256-prompt mini-batch exceeds the batch), 4 x 8 gives ~2.4x
more policy updates, and at a ~4% starting success rate the number of
improvement steps is the scarcer resource. Same for every arm.

The budget is enforced inside verl's loop (`GACA_TIME_BUDGET_H`): the
iteration that would cross it becomes the last one, so the final validation
still runs.

## 2026-11-08 — Invalid-action penalty back to the paper's 0.1

Attempt 1-3 set it to 0 because index actions made inadmissible output nearly
impossible, so the penalty became the only learnable signal. Under the paper's
free-form reasoning format that premise is gone: 16-17% of steps are
format-invalid zero-shot (valid_action_ratio 0.83-0.84), a genuine failure mode
the penalty exists to correct. The brief anticipated exactly this ("revisit
... the penalty may become legitimate again"). It is the paper's value and is
identical across arms. Watch item: if valid_action_ratio rises to ~1 while
success falls, the attempt-1 failure is recurring.

## 2026-11-08 — First measurements under the paper's format

Base model, paper protocol (all six task types, 128 in-distribution validation
episodes, T=0.4): **4.7% (6/128)**. GACA's diagnostics on the first 16 x 8
iteration now sit next to the paper's Table 7: c_t std 0.40 [0.31], lambda_t
std 0.10 [0.08], clamped 0.001 [<0.01], step coverage 0.96 [0.86]. Under index
actions they were 1.04 / 0.22 / 0.056 / 0.82; matching the response format is
what closed that gap.

## 2026-11-08 — Probe-weighted token credit: offline evidence and arm design

Full analysis in `IDEAS.md`; data in `src/runs/tokens/`. A flip-test oracle
(resample a response after j tokens, does the final action change?) on 640
real steps over two disjoint game sets found that the commitment probe
q(j) = P(original action | first j tokens + forced "</think>\n<action>") is the
only token-weighting signal that tracks where the decision is made. Using the
exact training function (`advantages.probe_token_weights`, clip [0.25, 4],
mean-preserving), it puts **1.65x [1.57, 1.74]** more credit on decisive
segments than uniform, on held-out games, consistently across all six task
types (1.52-1.96), both game halves (1.73 / 1.57) and early/late steps, and
most on open decisions (2.37x). Entropy (1.04x), NLL (1.08x) and an
entropy-plus-action-span hybrid (1.07x) do not.

Arm design: GACA unchanged; each step's advantage is redistributed over its
response tokens by the probe weights (cut every 16 reasoning tokens + the
<action> tag; invalid-format rows stay uniform). Integration
(`verl-agent/gigpo/core_probe.py`) scores the probe through verl's own
log-prob routine; CPU tests pin the batch layout against direct scoring
(<2e-4 nats) and against the offline oracle's values (bf16 noise).

One definitional change from the offline oracle: the action cut is where the
`<action>` tag *begins* (clean context), not the token that completes it (the
oracle's, whose context ended in a tag fragment). It affects only the last
reasoning cut.

**Matched on iterations (34), not wall-clock.** The probe adds a scoring pass
per iteration; on the controls' 3.75 h budget it would get fewer iterations,
i.e. less data, and lose for reasons unrelated to credit assignment. Rollout
budget (iterations x tasks x G) is what the comparison holds fixed, as in
attempts 1-3. The arm therefore runs past 4 h by the measured probe overhead.

## 2026-11-09 — Fixed 34 iterations per arm; wall-clock cap retired; repo layout

Every arm now runs exactly 34 iterations (`STEPS=34` in `src/verl/run_arms.sh` / `src/verl/run_arm.sh`) and the
wall-clock budget (`GACA_TIME_BUDGET_H`) is unset, at the user's direction. The arms are matched on
rollout budget, which the cap could violate: it stopped GACA + m_z and GACA-LOO at 33 iterations, one
short of the controls (accepted as is, noted in RESULTS).

Documents moved to `docs/` (DECISIONS, RESULTS, IDEAS, HANDOFF, RESULTS_4060); the 4060-era launchers to
`legacy/4060_launchers/` and their logs to `src/runs/4060/`. Paths used by running jobs (`verl/`, `src/`, `src/runs/verl/`)
were left unchanged; `src/report.py` writes `docs/RESULTS.md` and `docs/RESULTS_4060.md`.

## 2026-11-09 — Probe arm and its variants queued

All 34 iterations, all uploading final weights: `gaca_probe_s1` (the GACA + probe arm; training seed 1,
validation games pinned by `+env.val_seed=1000`, identical to every other arm's), `gaca_probe_c05`
(clip [0.5, 2]) and `gaca_probe_pos` (probe weights only on rows with a positive advantage). The two
variants use the controls' training seed. Motivation is in docs/IDEAS.md (1b). The queue is a plain
file, `src/runs/verl/queue.txt`, consumed by `src/verl/queue_runner.sh`, so it can change between arms.

## 2026-11-09 — Layout: all live code under src/

At the user's direction `verl/` and `runs/` now live in `src/` (`src/verl/`, `src/runs/`), so `src/`
holds all live code and its outputs; code that is no longer run is in `legacy/` (see its README), and
pixi was removed. The move was made between arms (no job running) and every launcher, the report and
the docs were updated; the verl-agent symlink to `src/advantages.py` is unaffected.

## 2026-11-09 — CIR (Yu et al., 2026) logged from the probe passes

The commitment probe is the measurement behind Causal Importance of Reasoning (arXiv 2604.22074); that
paper uses it as a reward bonus, not for token-level advantage redistribution. CIR (mean JS divergence
between the probe at each cut and at the full reasoning) and q_T are now logged per iteration by
`core_probe.apply_probe_weights` at no extra cost, from `gaca_probe_c05` on (the running `gaca_probe_s1`
loaded its code before the change). Motivation: the paper finds that outcome-driven RL makes reasoning
less causally important, and concentrating credit on the committing segment could do the same.

## 2026-11-09 — GACA + probe: 20.3%

`gaca_probe_s1` finished at **20.3% (26/128)**, the highest final of any arm (GACA 18.8%, GRPO 14.1%).
Paired: vs GACA 16 vs 14 games solved by one arm only (p = 0.86); vs GRPO 16 vs 8 (p = 0.15) -- not
significant on one seed. q(0) fell 0.19 -> 0.14 over training: the reasoning became more causally
important, not less (IDEAS 1c). Final weights: `Adammouedden/gaca-term-checkpoints/gaca_probe_s1`
(verified listed).

## 2026-11-09 — Clip [0.5, 2]: 14.1%; trained-policy probe re-validation inserted before positive-only

`gaca_probe_c05` ended at 14.1% (18/128): below GACA + probe (20.3%; 8 vs 16 games, p = 0.15) and level
with GRPO (13 vs 13, p = 1.0); vs GACA 10 vs 16 (p = 0.33). The tighter clip kept update size at GACA's
level (gradient norm 0.93 vs GACA 0.90).
At the user's direction the token oracle is re-run with the trained `gaca_probe_s1` policy
(`src/verl/revalidate_probe.sh`, IDEAS 1e.1) before `gaca_probe_pos`. A launch-shell command line
containing the text "verl.trainer.main_ppo" made its GPU-free check self-match (as in 2026-11-07); the
GPU idled ~23 min until the stale shell was killed. Scripts must be launched from files, never heredocs
that contain the patterns they pgrep for.

## 2026-11-09 — Probe re-validated on a trained policy; positive-only proceeds

The token flip-test oracle with `gaca_probe_s1`'s final weights, on the same 48 held-out games: probe
credit on decision importance 1.64x [1.55, 1.72] at the training configuration (base model 1.65x
[1.57, 1.74]); unclipped 3.09x vs 2.94x; entropy still ~1x. Training did not degrade the probe as a
measurement, so the frozen-reference scorer (implemented, `gaca_probe_ref`) is not needed for accuracy;
it stays defined but unqueued. `gaca_probe_pos` was re-queued automatically afterwards.

## 2026-11-09 — gaca_probe_ref queued after positive-only

At the user's direction the frozen-reference scorer arm (`probe_scorer=ref`: the probe is scored by the
reference model already loaded for the KL term; rollouts and the PPO update stay on-policy) runs after
`gaca_probe_pos`, with the controls' training seed, 34 iterations, final weights uploaded. It tests the feedback-loop hypothesis
(policy grading its own reasoning drives early commitment, IDEAS 1c/1e.2) rather than probe accuracy,
which the trained-policy re-validation showed holds.

## 2026-11-09 — gaca_probe_ref crashed at its first probe call; fixed, GPU smoke before re-running

`probe_scorer=ref` called `compute_ref_log_prob` on the actor/rollout worker group, but with full
fine-tuning verl keeps the reference model in a separate `ref` worker group (`assert self._is_ref`).
The unit test mocked the dispatch and could not catch it. Fixed to use verl's own routing
(`ref_policy_wg` unless `ref_in_actor`), with a test reproducing the crash condition; the arm is re-run
only after a 1-iteration GPU smoke (`src/verl/smoke_ref.sh`). The crashed attempt's logs are kept in
`src/runs/verl/gaca_probe_ref_crashed/` (no training happened). Lesson: anything that crosses the
worker boundary gets a GPU smoke before a full arm.

## 2026-11-10 — Run excluded

The first GACA + probe run (training seed 0) is excluded from the analysis at the user's direction,
because the user judged it flawed; its raw logs are kept in `src/runs/excluded/gaca_probe_seed0/`.


## Method renamed to COMMIT (2026-11-10)

Probe-weighted token credit is now called **COMMIT** (COMMItment-weighted Token credit) in the paper,
code and demo. "GACA + probe" undersold it: the token weights are estimator-agnostic and GACA is only
the step estimator underneath. Arm keys and run folders were renamed `gaca_probe_*` -> `commit_*`.
Older records keep the old names: `arms.log`, commit messages, earlier entries in this file, and the
Hugging Face checkpoint folders (`Adammouedden/gaca-term-checkpoints/gaca_probe_s1` etc.).
`report.py` matches both names in `arms.log`. The frozen-reference arm (`commit_ref`) was stopped at
iteration 33/34 for the project deadline; it saved no checkpoint and must be re-run from scratch.

## Hill-climbing queue replaces the frozen-reference re-run (2026-11-10)

The user's priority moved from ablation rigor to performance. The commit_ref re-run was killed a few
minutes in (logs in `src/runs/verl/commit_ref_restart_killed`); the frozen-reference question stays open.
Queue, in order, each 34 iterations like every other arm:
1. `commit_lr3`, `gaca_lr3`: learning rate 3e-6 instead of 1e-6. Our runs are 34 iterations where the
   paper's are 150, so the conservative paper rate is the first suspect for slow learning.
2. `gaca_actw`: probe-free token weighting, `<action>` span 2x and reasoning 1x, mean-preserving. Tests
   whether COMMIT's gain needs the probe, which costs ~40% wall-clock (5.19 h vs 3.67 h).
3. `commit_ds`: COMMIT plus DAPO dynamic sampling (verl-agent's `algorithm.filter_groups`: groups whose 8
   rollouts all get the same reward carry zero advantage and are dropped and refilled, up to 3 rollout
   passes). At ~5% base success most groups have no win, so most of each batch carries no gradient.
   Launched at lr 3e-6 (user decision after `commit_lr3`: 23.4% vs 20.3%, ahead at every evaluation; no
   time for extra seeds), so it stacks on the best configuration so far.
Caveat for reading these: one 128-game evaluation has a standard deviation of ~3.4 points near 18%, so
a gain smaller than ~5 points from one seed is not distinguishable from noise.

## Does token weighting let larger steps help? Learning-rate sweep + gradient SNR (2026-11-10)

Evidence that motivated it (training logs, mean over iterations 17-34): at lr 3e-6 COMMIT drifts from
the reference as far as GACA or farther (KL loss 0.113 vs 0.094) and clips more (1.1% vs 0.5%), yet
gains 18.7 points to GACA's 8.6. So weighting does not shrink the step; the hypothesis is that it points
a step of the same size in a less noisy direction. Continuing COMMIT past 34 iterations was rejected:
68 iterations is comparable to neither our baselines nor the paper's 150.
- Training (34 iterations each): `gaca_actw_lr3`, `gaca_lr6`, `gaca_actw_lr6`. With the existing arms
  this gives success vs learning rate for uniform credit (GACA) and action-span weighting at 1e-6,
  3e-6 and 6e-6. Prediction: the weighted-minus-uniform gap grows with the learning rate.
- Measurement (`src/analysis/grad_snr.py`, queue item `grad_snr`, ~2.5 h): at two fixed policies
  (base model, trained COMMIT), on the same rollout groups (16 groups with mixed wins, 8 rollouts,
  30 steps), the policy gradient under uniform, action-span and COMMIT probe weights. Reports gradient
  coherence across groups, the cosine between two 4-group batches (one training batch), and the
  gradient noise scale (McCandlish et al., 2018), with jackknife intervals and paired differences.
  Gradients are kept on a strided 1/128 coordinate subsample (12M of 1.54B parameters; inner products
  unbiased up to scale). 30 steps instead of 50 to fit the time; the fixed policy makes it a
  within-sample comparison either way. CPU smoke test passed (all coordinates nonzero; same-rollout
  cosine uniform/action 0.95, uniform/probe 0.57).

## Seed confound in the learning-rate sweep; seed-0 COMMIT control (2026-11-11)

Every COMMIT arm trains on env.seed=1 (training-game order); every other arm on seed 0. At lr 3e-6 both
seed-1 runs reached ~23% while both seed-0 runs stalled for 30 iterations (GACA 13.3% final, action-span
weighting 6.2%), and GACA at 6e-6 (seed 0) collapsed to 0%. So "COMMIT tolerates a larger step" and
"seed 1 suits a larger step" cannot be told apart. `gaca_actw_lr6` was dropped (action-span weighting
already failed at 3e-6) and replaced by `commit_lr3_s0`: COMMIT at lr 3e-6 on seed 0, validation games
pinned as before. Reaching ~20% would support the probe; stalling like the other seed-0 runs would point
to the seed. The probe-vs-action-span contrast at 3e-6 carries the same confound until this run is in.

## Gradient SNR result (2026-11-11)

16 mixed-outcome groups per checkpoint (base Qwen2.5-1.5B-Instruct and trained COMMIT seed 1); per-group GACA policy gradients under uniform, action-span (×2) and probe token weights on identical rollouts; 1/128 coordinate subsample; jackknife 95% CIs.

- Probe weighting gives the most coherent gradients on both checkpoints: 4-group batch cosine 0.077 vs 0.026 uniform (base), 0.049 vs -0.004 (COMMIT); noise scale B 46 vs 146 (base). Action-span sits in between.
- Paired probe − uniform batch-cosine difference is +0.052 [-0.010, +0.113] (base) and +0.053 [-0.020, +0.125] (COMMIT): the same direction and size on both checkpoints, but the CIs include zero at 16 groups.
- Reading: consistent with "probe weights point the step in a better direction" (and with probe at 3e-6 > uniform at 3e-6), not proof. Per-group gradients are nearly orthogonal under every scheme (coherence ≈ 0.01), so the absolute signal is small. More groups would tighten the CIs; not queued, given time.

## commit_lr3_s0 relaunch after disk-full crash (2026-11-11)

The first launch (16:10 UTC) died at env reset: "No space left on device" copying libdownward.so to /tmp. grad_snr had left 4.6 GB of gradient samples (runs/grad_snr/*.npz) on the 32 GB root overlay. The report is committed and the samples can be regenerated, so they were deleted (9.1 GB free); the crashed log is kept as commit_lr3_s0_diskfull. Requeued unchanged.
