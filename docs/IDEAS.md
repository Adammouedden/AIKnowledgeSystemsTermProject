# Ideas: adaptive credit assignment beyond GACA

Working notes for the term project's new direction. Controls are **GRPO** and
**GACA**; each idea below is a *treatment* compared against both on the same
128 held-out games (paired McNemar). Measurements behind each idea are in
`DECISIONS.md` and `runs/tokens/`, `runs/verl/`.

## What the offline tests established (2026-11-08)

- **GACA's step score is mostly noise.** At anchor states visited 2+ times,
  ~85% of the variance of the per-step criticality score (sampled NLL *or*
  exact entropy) is within-state, i.e. it depends on which reasoning path was
  drawn, not on the state. Swapping NLL for exact entropy changes lambda by
  >0.1 on only 0.9% of steps -> not worth an arm.
  (`runs/verl/entropy_check_iter1-5.json`)
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
  (`runs/tokens/heldout.analysis.json`)
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

Watch item (from the running arm, iter 6): KL to the reference is ~0.045,
~3x the controls at the same point (0.014-0.015). Per-token weights up to 4x
let some tokens' gradients grow, so the policy drifts faster. If KL keeps
climbing while held-out success stalls, the upper clip is the suspect;
[0.5, 2] is the follow-up variant (offline: 1.26x credit on decisive
segments, vs 1.65x at [0.25, 4]).

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

- Resampling-based importance of reasoning steps ("Thought Anchors", Bogdan
  et al., 2025) -- the flip test is close to this.
- Monte-Carlo credit from token prefixes (VinePPO).
- High-entropy minority tokens in single-turn reasoning RL (Wang et al.).
The cheap teacher-forced probe used to distribute a step's advantage in
multi-turn agent RL appears new, but has not been checked against the
literature.
