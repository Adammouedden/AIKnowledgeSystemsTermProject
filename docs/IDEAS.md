# Ideas: adaptive credit assignment beyond GACA

Working notes for the term project's new direction. Controls are **GRPO** and
**GACA**; each idea below is a *treatment* compared against both on the same
128 held-out games (paired McNemar). Measurements behind each idea are in
`DECISIONS.md` and `src/runs/tokens/`, `src/runs/verl/`.

## What the offline tests established (2026-11-08)

- **GACA's step score is mostly noise.** At anchor states visited 2+ times,
  ~85% of the variance of the per-step criticality score (sampled NLL *or*
  exact entropy) is within-state, i.e. it depends on which reasoning path was
  drawn, not on the state. Swapping NLL for exact entropy changes lambda by
  >0.1 on only 0.9% of steps -> not worth an arm.
  (`src/runs/verl/entropy_check_iter1-5.json`)
- **Most of a decision is made in the reasoning, not in the action tokens.**
  Resampling a response after its first j tokens and checking whether the
  final `<action>` changes (the flip test): 82% of the decision is settled
  before `<action>`; fully resampling changes the action ~72% of the time.
- **Entropy is a poor guide to where the decision is made.** The final tenth
  of a response holds ~20% of decision importance but ~3% of entropy: the
  action tokens look certain yet are where the commitment lands. Entropy
  token weights put only 1.04x more credit on decisive segments than uniform;
  an entropy + protected-action-span hybrid 1.07x (did not hold out of sample).
- **The commitment probe works.** q(j) = P(original action | reasoning up to
  token j, then a forced `</think><action>`), one teacher-forced scoring pass
  per cut point, no generation. Weighting a segment by how much q rises across
  it puts **2.94x** more credit on decisive segments than uniform and is the
  only scheme that fits measured importance better than uniform (TV -0.065,
  95% CI [-0.093, -0.037]), on games not used to design it.
  (`src/runs/tokens/heldout.analysis.json`)
  Caveats: the probe and the flip-test ground truth measure related things,
  so part of the agreement is by construction; it shows *where* the decision
  is made, not that crediting those tokens trains better; base policy only.

## Treatments

### 1. Probe-weighted token credit (next arm)

Keep GACA's step advantage A_t unchanged; distribute it across the step's
response tokens by the probe:

    A_{t,j} = A_t * w_{t,j},   mean_j w_{t,j} = 1   (mean-preserving)

with w proportional to the rise in q across the segment containing token j,
clipped to [0.25, 4] and renormalized. Uniform q reproduces GACA exactly
(unit-tested). A step's total credit is unchanged; only where it lands inside
the response moves. Cost: ~6-8 extra teacher-forced passes per response
(cut points every ~16 tokens), est. +3-5 min/iteration -> ~25-28 iterations
in the 4 h budget instead of 34. Measure before launching.

Why it should help: a good step's advantage reinforces mostly the reasoning
that produced the decision, a bad step's advantage penalizes mostly that
reasoning, instead of spreading both over boilerplate.

Risk: per-token weights up to 4x let some tokens' gradients grow, so updates can get larger and the
policy can drift faster; [0.5, 2] is the tighter-clip variant (offline: 1.26x credit on decisive
segments, vs 1.65x at [0.25, 4]).

### 1b. Probe variants (2026-11-09)

Two hypotheses about how token concentration could hurt, one arm each:

- **Clip [0.5, 2]** (`gaca_probe_c05`): if concentration makes updates too large, a tighter clip should
  keep KL down (offline: 1.26x credit on decisive segments instead of 1.65x). **Result: 14.1%**, level
  with GRPO; the gradient norm fell to GACA's level (0.93 vs 0.90).
- **Positive advantages only** (`gaca_probe_pos`): the probe moves *where* credit lands, not whether it
  is right. At 5-15% success many step advantages rest on one or two wins, and a wrong-signed advantage
  concentrated on the decisive reasoning does the most damage. Probe weights only where A_t > 0;
  zero/negative rows keep uniform credit. **Result: 5.5%**, significantly below GRPO (p = 0.027) and
  GACA (p < 0.001), with high KL (0.059 mean). Keeping blame uniform did not help; if anything,
  asymmetric credit destabilized training.

### 1c. Relation to CIR, and a new failure hypothesis (2026-11-09)

Yu, Tartaglini, Hase, Guestrin & Potts, *Outcome Rewards Do Not Guarantee Verifiable or Causally
Important Reasoning* (arXiv 2604.22074, ICML 2026) define **Causal Importance of Reasoning**:
truncate the reasoning after k tokens, force the answer, p_k = P(answer | q, t_{1:k});
CIR = mean_k JS(Bern(p_k) || Bern(p_T)). That is the same measurement as our commitment probe. They
use it as a **scalar reward bonus** (r_out + beta CIR), and find that outcome-only RL lowers CIR on
19/40 ReasoningGym tasks (reasoning becomes decorative); the bonus or a little SFT restores it. They do
not redistribute the advantage across tokens -- every token of a response still gets the same
advantage. **Our contribution is that redistribution** (the per-segment *rise* of p_k sets each token's
share of the step advantage, in multi-turn agent RL), not the probe itself, which must be credited to
them (and to earlier truncation-based "early answering" faithfulness tests).

Their warning is a risk for our method specifically: concentrating credit on whichever segment commits
to the action could reward early commitment, so the decision drifts forward and the rest of the
reasoning becomes decorative. q(0) -- the probability of the eventual action *before* any reasoning --
and CIR are logged per iteration to watch for it. In the GACA + probe arm it did not happen: q(0) fell
0.19 -> 0.14 over training (reasoning became *more* causally important) while it reached 20.3%. The
positive-only variant failed with q(0) flat (0.20 -> 0.19), so early commitment does not explain that
failure.

Candidate arm: **GACA + probe + CIR bonus** -- token redistribution plus the paper's reward term
(computable from the probe passes already run), so concentration cannot make the reasoning decorative.

### 1d. Concentrate only credit we trust (generalizes positive-only)

The positive-only arm is a special case of a more principled rule: **scale the concentration of token
credit by how much the step advantage can be trusted.** Interpolate per row between uniform and probe
weights, w = 1 + kappa_t (w_probe - 1), with kappa_t in [0, 1] from signals already in the batch:

- the number of successful rollouts in the step's group (one win out of 8 makes every advantage in the
  group rest on a single trajectory -> low kappa; 3+ wins -> high kappa);
- the step's anchor-group size m_z and whether its bucket shows return spread (A^S built from a real
  within-state contrast vs the prompt-group fallback);
- the sign of A_t (positive-only is kappa = 1[A_t > 0]).

Rationale: the probe moves *where* credit lands, so its benefit is bounded by whether the credit is
right; at 5-15% success most step advantages rest on one or two wins. This keeps the probe's precision
where the signal is reliable and falls back to GACA where it is not. Testable offline first: on the
step dumps, check whether |A_t|-sign agreement with the MC oracle (or with a second seed's advantage at
the same anchor) rises with group win count, before spending an arm.

### 1e. Further probe improvements (2026-11-09)

Ranked by cost and by how directly they address a measured problem:

1. **Re-validate the probe on a trained policy (measurement, ~15 min GPU).** The flip-test oracle that
   justified the probe ran on the *base* model. The GACA + probe arm's final weights are on HF; rerun
   `src/measure.py tokens` with them to see whether the probe still tracks decision importance after
   training. Decides whether (2) matters.
   **Done 2026-11-09:** on the trained GACA + probe policy the probe tracks decision importance as well
   as on the base model (training config 1.64x [1.55, 1.72] vs 1.65x; unclipped 3.09x vs 2.94x). The
   probe does not degrade as a measurement.
2. **Score the probe with the frozen reference model.** (Implemented: `probe_scorer=ref`, arm
   `gaca_probe_ref`, running 2026-11-09.) Today the policy scores its own reasoning: credit goes to the
   committing segment, which can push commitment earlier, which moves the credit again -- a possible
   feedback loop. The reference model is already loaded for the KL term, so this costs nothing.
3. **Pool the probe across visits to the same anchor state** (average the q-rise profiles of the group's
   visits, as LOO does for the NLL score); GACA's per-visit score was ~85% within-state noise.
4. **Guard against decorative reasoning:** CIR bonus in the reward (1c), or a penalty on q(0) rising
   above its starting level.
5. **Trust-scaled concentration** (1d); positive-only is its sign-only special case.
6. **Log-space or JS gain instead of raw q differences.** Raw dq over-weights the final jump to
   near-certainty; log-gain credits an early 0.05 -> 0.2 shift as much as a late 0.5 -> 0.95, and JS
   would make the token weights the per-segment terms of CIR itself.
7. **Cut at sentence/clause boundaries** rather than every 16 tokens (as Thought Anchors does), so a
   decision is not split across two segments.

### 2. Probe as the step gate

q(0) -- the probability of the eventual action before any reasoning -- says
how predetermined the action was. Near-certain -> not really a decision ->
lean on episode credit (lambda down); open -> lean on step credit (lambda up).
Replaces GACA's noisy single-sample NLL with a direct measure of how much of a
decision the step was.

### 3. Both

The probe sets lambda_t per step (idea 2) and distributes credit within the
step (idea 1).

### Step-level noise reduction (running)

GACA-LOO: score each visit by the leave-one-out mean NLL of the *other* visits
to its anchor state (paper Appendix B.4). Directly targets the 85%
within-state noise. The paper flags it as unsettled and never trains it.

## Not pursued, and why

- **Exact-entropy step gate**: agrees with GACA's NLL gate on >99% of steps.
- **Entropy token weighting / hybrid**: barely tracks decision importance.
- **m_z reliability gate**: only ~4% of steps are singletons, so it barely
  changes anything (12.5% vs GACA 18.8%, p=0.17).
- **Learning-value credit** (proposal): per-decision effect on held-out
  performance is far below measurement noise at this scale.
- **Random-score ablation**: already in the paper (Table 2); stopped at iter 10.

## Prior art to check before claiming novelty

- Truncate-and-force-answer measurement of reasoning importance: CIR (Yu et al., 2026, arXiv
  2604.22074) -- the commitment probe is this measurement; see 1c.
- Resampling-based importance of reasoning steps ("Thought Anchors", Bogdan
  et al., 2025) -- the flip test is close to this.
- Monte-Carlo credit from token prefixes (VinePPO).
- High-entropy minority tokens in single-turn reasoning RL (Wang et al.).
Using that measurement's per-segment rise to distribute a step's advantage
over tokens in multi-turn agent RL appears new; the measurement itself is not.
