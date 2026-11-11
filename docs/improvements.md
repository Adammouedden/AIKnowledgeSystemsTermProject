# Improving COMMIT: evidence log

A living record of what makes COMMIT better, and why. Every claim names the runs behind it and how strong it is right now. Update it as runs finish. It feeds the updated paper and presentation.

**COMMIT, in one line.** The GACA step advantage, spread over a response's tokens by a commitment probe. The probe is the model's probability `q(c)` of the action it finally took if it stopped reasoning at token `c`, scored every 16 tokens. Tokens where `q` rises get more credit. Weights are clipped to [0.25, 4] and rescaled to mean 1, so the total advantage per response is unchanged.

**Setup for every number below.** ALFWorld, Qwen2.5-1.5B-Instruct, verl-agent, 34 iterations of 4 games × 8 rollouts, final checkpoint evaluated on the same 128 held-out games. One game is 0.78 points. The evaluation SD is about ±3.4 points near 18% success. Every row is one training run. Paired p-values are exact McNemar tests on the 128 games (`docs/RESULTS.md`).

---

## 1. Scoreboard

| Change to COMMIT | Final success | Reference | Paired p | Verdict |
|---|---|---|---|---|
| COMMIT, lr 1e-6 (original) | 20.3 | GACA 18.8 | 0.856 | on par with GACA |
| **lr 3e-6** | **23.4** | COMMIT lr 1e-6, 20.3 | 0.523 | best so far; seed-confounded (§2) |
| lr 3e-6 + dynamic sampling (DAPO) | 22.7 | COMMIT lr 3e-6, 23.4 | 1.0 (13/12 discordant) | no gain, 1.9× wall-clock |
| narrower weight clip [0.5, 2] | 14.1 | COMMIT, 20.3 | 0.152 | worse: flatter weights hurt |
| weights on positive advantages only | 5.5 | COMMIT, 20.3 | < 0.001 | collapses |
| lr 6e-6 | queued | GACA lr 6e-6, 0.0 | | §5 |
| frozen-reference probe | conditional | | | §5 |
| probe every 8 tokens | queued | | | §5 |

Probe-free comparison point, action-span weighting (the `<action>` span ×2, reasoning ×1, mean 1):

| Arm | Final success |
|---|---|
| action-span weighting, lr 1e-6 | 20.3 |
| action-span weighting, lr 3e-6 | 6.2 |

---

## 2. Does token weighting let larger steps help?

Final held-out success by learning rate:

| Arm (training seed) | lr 1e-6 | lr 3e-6 | lr 6e-6 |
|---|---|---|---|
| GACA (seed 0) | 18.8 | 13.3 | 0.0 |
| GACA + action-span weighting (seed 0) | 20.3 | 6.2 | not run (dropped) |
| COMMIT (seed 1) | 20.3 | **23.4** | |
| COMMIT (seed 0) | | running (`commit_lr3_s0`) | queued (`commit_lr6_s0`) |

Paired tests:

- **At lr 3e-6:**
  - COMMIT vs GACA: 30 vs 17 wins, p = 0.007.
  - GACA vs action-span weighting: 17 vs 8, p = 0.035.
- **Raising lr from 1e-6 to 3e-6:**
  - GACA: 24 → 17, p = 0.143.
  - action-span weighting: 26 → 8, p < 0.001.
  - COMMIT: 26 → 30, p = 0.523.

**Reading.** Uniform credit (GACA) and the fixed action-span heuristic both get worse as the step size grows; action-span weighting falls fastest. COMMIT is the only arm that improves at 3e-6. So the probe, not reweighting as such, is what tolerates the larger step: a wrong token weighting amplifies the damage that a large learning rate does.

**Caveat: seed confound.** Every COMMIT run so far used training seed 1; every GACA and action-span run used seed 0. The validation games are the same. `commit_lr3_s0` removes the confound: COMMIT at lr 3e-6 on seed 0, finishing about 4:40 PM ET on 2026-11-11. Until it lands, the COMMIT row in this section is suggestive only.

### Step size and stability (iterations 17–34, lr 3e-6)

| Metric | COMMIT | GACA |
|---|---|---|
| KL to reference (`kl_loss`) | 0.113 | 0.094 |
| PPO clip fraction | 1.1% | 0.5% |

COMMIT moves *further* from the reference and clips *more*, yet learns more. Token weighting does not make the steps smaller or safer. If it helps, it does so by pointing each step in a better direction. Section 3 tests that directly.

---

## 3. Gradient analysis: do probe weights give a cleaner gradient?

**Question.** On identical rollouts, does weighting tokens by the probe make per-game policy gradients agree more with each other (higher signal-to-noise) than uniform or action-span weighting?

**Method** (`src/analysis/grad_snr.py`, `grad_snr.sh`; report `src/runs/verl/grad_snr/grad_snr.{md,json}`):

- **Checkpoints.** The base model, and trained COMMIT (seed 1, lr 1e-6, `models/ckpt/commit_s1`).
- **Rollouts.** 8 rollouts per game at T = 1.0, max 30 steps. Games with mixed outcomes (0 < wins < 8) are kept, 16 per checkpoint, drawn from 48 games each. Advantages are GACA, as in training.
- **Per-game gradients.** For each game, ∇ of −Σ A_gaca · w · log π over its rows, under three token weightings `w`: uniform, action-span (×2) and probe (COMMIT).
- **Subsample.** A strided 1/128 subsample of the parameters (12M of 1.54B).
- **Statistics.**
  - coherence: the mean pairwise cosine between games;
  - batch cosine: the cosine between two disjoint 4-game batch gradients, over 400 random splits;
  - gradient noise scale `B = tr Σ / |μ|²` (McCandlish et al., 2018; lower means less noise relative to signal);
  - jackknife 95% CIs over games, and paired differences on the same games.

**Results** (point estimate [95% CI]):

| Checkpoint | Weighting | Coherence | 4-game batch cosine | Noise scale B |
|---|---|---|---|---|
| base | uniform | 0.007 [−0.013, 0.027] | 0.026 [−0.048, 0.100] | 146 [−410, 702] |
| base | action-span | 0.011 [−0.009, 0.031] | 0.038 [−0.034, 0.111] | 97 [−100, 295] |
| base | **probe** | **0.021** [−0.000, 0.043] | **0.077** [0.002, 0.153] | **46** [−3, 95] |
| trained COMMIT | uniform | −0.001 [−0.009, 0.007] | −0.004 [−0.038, 0.030] | ∞ (mean ≈ 0) |
| trained COMMIT | action-span | 0.004 [−0.007, 0.015] | 0.014 [−0.028, 0.056] | 314 |
| trained COMMIT | **probe** | **0.013** [−0.006, 0.032] | **0.049** [−0.021, 0.119] | **85** [−61, 231] |

Paired differences on the same games:

| Checkpoint | Comparison | Coherence diff | Batch cosine diff |
|---|---|---|---|
| base | action − uniform | +0.003 [−0.002, +0.009] | +0.013 [−0.006, +0.031] |
| base | probe − uniform | +0.014 [−0.004, +0.031] | +0.052 [−0.010, +0.113] |
| trained COMMIT | action − uniform | +0.005 [−0.002, +0.012] | +0.018 [−0.008, +0.044] |
| trained COMMIT | probe − uniform | +0.014 [−0.005, +0.033] | +0.053 [−0.020, +0.125] |

**Reading.**

- The ordering is the same on both checkpoints and every statistic: probe > action-span > uniform.
- Probe weighting roughly triples the batch cosine, and cuts the noise scale about 3× on the base model.
- The probe − uniform gain is the same size on both checkpoints, about +0.05 batch cosine.
- On the trained model the uniform gradient's mean is indistinguishable from zero (B = ∞): late in training, uniform credit adds mostly noise.
- This matches §2. Probe weights give a more consistent update direction, which is what a larger learning rate needs.

**Strength: suggestive, not significant.** Every paired CI includes zero at 16 games. The one CI that clears zero is the base-model probe batch cosine, and that is an unpaired level, not a difference. Absolute agreement is small under every scheme (coherence about 0.01): per-game gradients are nearly orthogonal at this batch size.

**To strengthen.**
- More games (about 48 per checkpoint would roughly halve the CIs; about 6 h).
- The seed-0 COMMIT checkpoint, to add a second trained model.
- Measure the same statistics on the lr 3e-6 checkpoints, where §2's divergence appears.

---

## 4. Claims and their status

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | COMMIT at lr 3e-6 beats GACA at lr 3e-6 | 23.4 vs 13.3, p = 0.007 | supported, but seed-confounded until `commit_lr3_s0` |
| C2 | The probe, not reweighting as such, is what tolerates larger steps | action-span weighting drops 20.3 → 6.2 at 3e-6 while COMMIT rises | suggestive (seed confound) |
| C3 | Probe weights give a more coherent gradient | §3: consistent +0.05 batch cosine on two checkpoints | suggestive (CIs include 0) |
| C4 | Token weighting does not shrink steps | §2 stability table | supported (descriptive) |
| C5 | Flatter or one-sided probe weights hurt | clip [0.5, 2]: 14.1; positive-only: 5.5 | supported for positive-only (p < 0.001); clip suggestive |
| C6 | Dynamic sampling adds nothing on top | 22.7 vs 23.4, p = 1.0 | supported (no detectable gain) |

---

## 5. Running and planned

All on training seed 0 (validation seed 1000), so each pairs with the GACA arm at the same learning rate.

| Order | Arm | What it tests | Rule |
|---|---|---|---|
| now | `commit_lr3_s0` | removes the seed confound in C1 and C2 | |
| 1 | `commit_lr6_s0` | does COMMIT survive lr 6e-6, where GACA collapses to 0.0? | gain = final strictly above `commit_lr3_s0`, and no collapse |
| 2 | `commit_ref_lr3_s0` | probe scored by the frozen reference model, so the weights don't drift with the policy | runs **only if** lr 6e-6 shows no gain |
| 3 | `commit_e8_lr{3,6}_s0` | probe every 8 tokens instead of 16 (finer placement of the commitment point) | at the better learning rate from step 1 |

Ideas not yet scheduled (from the 2026-11-11 discussion):

- **Sharper weights:** clip range [0.1, 10], or the squared rise in `q`. C5 suggests flatter weights hurt.
- **Probe weights on the step advantage only:** leave GACA's episode component uniform.
- **Weights on negative advantages only:** the untested mirror of the collapsed positive-only run.

---

## 6. Caveats to carry into the paper

- **One seed per configuration.** A 4-point difference (5 games) is within evaluation noise. Lean on the paired tests, not on gaps in the table.
- **No extra seeds by choice:** time budget (user decision, 2026-11-10). State it as a limitation.
- **Final checkpoint only.** The "best" column in `RESULTS.md` is optimistic: it picks the best of five noisy evaluations.
- **The gradient analysis subsamples parameters.** A 1/128 strided subsample gives unbiased cosines in expectation, but is not the full gradient.
