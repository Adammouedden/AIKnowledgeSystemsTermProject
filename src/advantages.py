"""Critic-free advantage estimators: GRPO, GiGPO and GACA.

Everything operates on a flat list of per-step records produced by one rollout
iteration. A record is a dict with:
    group  : prompt-group (task) id -- group-relative normalization unit
    traj   : rollout id inside the group
    t      : 0-based step index inside the trajectory
    ret    : R(tau), the trajectory return, shared by all its steps
    g_hat  : discounted step return \\hat g_t = sum_{t'>=t} gamma^{t'-t} r_t'
    anchor : hash of the environment state at this step (anchor state)
    nll    : H_t, mean per-token NLL of the sampled action (Eq. 4)

Equation numbers refer to Liang et al., "Granularity-Adaptive Credit Assignment
for Long-Horizon LLM Agent Reinforcement Learning" (ICLR 2026).
"""

import numpy as np

EPS = 1e-6


def discounted_step_returns(rewards, gamma: float):
    """\\hat g_t for one trajectory (Section 2.1)."""
    out = np.zeros(len(rewards), dtype=np.float64)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


def _norm(x, mode: str):
    """Group-relative normalization. `x` is the vector of all rows in a group."""
    x = np.asarray(x, dtype=np.float64)
    c = x - x.mean()
    if mode == "mean":
        return c
    # Unbiased (ddof=1) std over step-rows, per Appendix D.
    s = x.std(ddof=1) if x.size > 1 else 0.0
    return c / (s + EPS)


def episode_advantage(steps, norm: str):
    """Eq. 2, with the verl-agent row-wise convention of Appendix D.

    Statistics are taken over every step-row of a prompt group, each row
    carrying its own trajectory's return, so the mean is length-weighted.
    """
    adv = np.zeros(len(steps), dtype=np.float64)
    for gid in {s["group"] for s in steps}:
        idx = [i for i, s in enumerate(steps) if s["group"] == gid]
        adv[idx] = _norm([steps[i]["ret"] for i in idx], norm)
    return adv


def step_advantage(steps, norm: str):
    """Eq. 3: anchor-state step-level advantage, with the Appendix D singleton fallback.

    Also returns the fraction of steps that landed in a non-singleton anchor
    group (the "step-advantage coverage" diagnostic of Table 7).
    """
    adv = np.zeros(len(steps), dtype=np.float64)
    covered = 0
    for gid in {s["group"] for s in steps}:
        gidx = [i for i, s in enumerate(steps) if s["group"] == gid]
        # Fallback statistics: the prompt group's own step returns.
        fallback = _norm([steps[i]["g_hat"] for i in gidx], norm)
        fb = {i: v for i, v in zip(gidx, fallback)}
        buckets = {}
        for i in gidx:
            buckets.setdefault(steps[i]["anchor"], []).append(i)
        for idx in buckets.values():
            if len(idx) >= 2:
                adv[idx] = _norm([steps[i]["g_hat"] for i in idx], norm)
                covered += len(idx)
            else:
                adv[idx[0]] = fb[idx[0]]
    return adv, covered / max(1, len(steps))


def _loo_nll(steps):
    """Appendix B.4: replace H_t by the leave-one-out mean over its anchor group.

    h_hat_z^(-i) = mean of ell_j over the other visits j to the same anchor
    state. It is conditionally independent of the action this rollout took, so
    the reweighting it drives is an unbiased rescaling of the policy gradient,
    and its variance is Var(ell | z) / (m_z - 1) rather than Var(ell | z). What
    it gives up is within-group resolution: every visit to an anchor gets
    nearly the same score. Singleton anchor groups have no leave-one-out
    average at all, so they keep their own H_t.
    """
    out = {}
    for gid in {s["group"] for s in steps}:
        gidx = [i for i, s in enumerate(steps) if s["group"] == gid]
        buckets = {}
        for i in gidx:
            buckets.setdefault(steps[i]["anchor"], []).append(i)
        for idx in buckets.values():
            if len(idx) >= 2:
                total = sum(steps[i]["nll"] for i in idx)
                for i in idx:
                    out[i] = (total - steps[i]["nll"]) / (len(idx) - 1)
            else:
                out[idx[0]] = steps[idx[0]]["nll"]
    return out


def criticality(steps, alpha_c: float = 1.0, mode: str = "entropy", rng=None):
    """Eqs. 4-6: per-step criticality score c_t, normalized by its trajectory mean.

    mode: "entropy" (default), "uniform" (ablation control: c_t == 1),
    "random" (ablation control: permute the scores inside each trajectory), or
    "loo" (the debiased variant of Appendix B.4).
    """
    c = np.ones(len(steps), dtype=np.float64)
    if mode == "uniform":
        return c
    rng = rng or np.random.default_rng(0)
    nll = _loo_nll(steps) if mode == "loo" else {i: s["nll"] for i, s in enumerate(steps)}
    for key in {(s["group"], s["traj"]) for s in steps}:
        idx = [i for i, s in enumerate(steps) if (s["group"], s["traj"]) == key]
        idx.sort(key=lambda i: steps[i]["t"])
        h = np.array([nll[i] for i in idx], dtype=np.float64)
        e_hat = h / (h.mean() + EPS)  # \\hat e_t = H_t / \\bar H
        if alpha_c >= 1.0:
            score = e_hat
        else:
            g = np.array([steps[i]["g_hat"] for i in idx], dtype=np.float64)
            delta = np.abs(np.diff(g, prepend=g[0]))  # Delta_1 = 0
            d_hat = delta / (delta.mean() + EPS)
            score = alpha_c * e_hat + (1.0 - alpha_c) * d_hat
        if mode == "random":
            score = rng.permutation(score)
        c[idx] = score / (score.mean() + EPS)  # Eq. 5
    return c


def mixing_weight(c, lam0: float, alpha_lam: float, beta_lam: float):
    """Eq. 7: piecewise-linear, clamped mixing law, continuous at c_t = 1."""
    c = np.asarray(c, dtype=np.float64)
    up = np.minimum(lam0 * (1.0 + alpha_lam * (c - 1.0)), 1.0)
    down = np.maximum(lam0 * (1.0 - beta_lam * (1.0 - c)), 0.0)
    return np.where(c >= 1.0, up, down)


def anchor_sizes(steps):
    """m_z for every step: the size of its anchor bucket inside its prompt group."""
    m = np.ones(len(steps), dtype=np.float64)
    for gid in {s["group"] for s in steps}:
        buckets = {}
        for i, s in enumerate(steps):
            if s["group"] == gid:
                buckets.setdefault(s["anchor"], []).append(i)
        for idx in buckets.values():
            m[idx] = len(idx)
    return m


def reliability(lam, m):
    """Group-size correction of the mixing weight (Proposition 3, eq. 18).

    M_S(z) = y_z sigma_z^2 with y_z = m_z / (m_z - 1), so at fixed everything
    else the optimal odds lambda*/(1 - lambda*) = M_E / M_S scale with
    rho_z = 1/y_z = 1 - 1/m_z. Reading the NLL gate's lambda_t as the weight
    it would assign with an infinitely large anchor group (rho = 1) and
    applying that factor to its odds gives

        lambda'_t = lambda_t rho / (lambda_t rho + 1 - lambda_t).

    A singleton step (m_z = 1) has no within-group contrast and gets exactly
    0; large groups are left almost untouched (m=8 -> rho 0.875).
    """
    rho = 1.0 - 1.0 / np.maximum(np.asarray(m, dtype=np.float64), 1.0)
    num = lam * rho
    return np.where(num + (1.0 - lam) > 0, num / np.maximum(num + (1.0 - lam), EPS), 0.0)


def compute_advantages(
    steps,
    method: str = "gaca",
    norm: str = "mean_std",
    w: float = 1.0,
    lam0: float = 0.5,
    alpha_lam: float = 0.5,
    beta_lam: float = 0.5,
    alpha_c: float = 1.0,
    score_mode: str = "entropy",
    rng=None,
    rel: str = "none",
):
    """Per-step advantage for one rollout iteration.

    method:
      "grpo"  -- episode-level advantage only (Eq. 2), broadcast to every step.
      "gigpo" -- w * A^S_t + A^E, one fixed weight, uniform granularity (Eq. 3 + Eq. 2).
      "fixed" -- convex fixed-weight control: lam0 * w * A^S_t + (1-lam0) * A^E.
      "gaca"  -- Eq. 8 with the per-step weight of Eq. 7.
    rel: "none" (the paper) or "msize" (scale the gate's odds by 1 - 1/m_z).
    Returns (advantages, diagnostics).
    """
    if not steps:
        return np.zeros(0), {}
    a_e = episode_advantage(steps, norm)
    diag = {"adv_episode_abs": float(np.abs(a_e).mean())}

    if method == "grpo":
        adv = a_e
        diag.update(step_coverage=0.0, c_mean=1.0, c_std=0.0, lam_mean=0.0, lam_std=0.0, clamped=0.0)
        diag["adv_abs"] = float(np.abs(adv).mean())
        return adv, diag

    a_s, coverage = step_advantage(steps, norm)
    diag["step_coverage"] = coverage
    diag["adv_step_abs"] = float(np.abs(a_s).mean())

    if method == "gigpo":
        adv = w * a_s + a_e
        lam = np.ones(len(steps))  # GiGPO sums the two levels; no mixing weight
        c = np.ones(len(steps))
    elif method == "fixed":
        adv = lam0 * w * a_s + (1.0 - lam0) * a_e
        lam = np.full(len(steps), lam0)
        c = np.ones(len(steps))
    elif method == "gaca":
        c = criticality(steps, alpha_c=alpha_c, mode=score_mode, rng=rng)
        lam = mixing_weight(c, lam0, alpha_lam, beta_lam)
        if rel == "msize":  # estimator-reliability correction, see reliability()
            lam = reliability(lam, anchor_sizes(steps))
        adv = lam * w * a_s + (1.0 - lam) * a_e  # Eq. 8
    else:
        raise ValueError(f"unknown method {method!r}")

    diag.update(
        c_mean=float(c.mean()),
        c_std=float(c.std()),
        lam_mean=float(lam.mean()),
        lam_std=float(lam.std()),
        clamped=float(np.mean((lam <= 0.0) | (lam >= 1.0))) if method == "gaca" else 0.0,
        adv_abs=float(np.abs(adv).mean()),
    )
    return adv, diag


def probe_cuts(L: int, act_start: int, every: int = 16):
    """Cut points for the commitment probe: every `every` tokens through the
    reasoning, plus the start of the <action> tag. Always starts at 0."""
    a0 = max(0, min(int(act_start), L))
    return sorted(set(range(0, a0, every)) | {a0}) if L > 0 else []


def probe_token_weights(L: int, cuts, q, clip=(0.25, 4.0)):
    """Token weights from the commitment probe (docs/IDEAS.md, treatment 1).

    `cuts` are increasing positions starting at 0, `q[k]` the probability of
    the original action given the response's first cuts[k] tokens (then a
    forced close of the reasoning). The response end L is an implicit last cut
    with q = 1 (the original action is certain once fully written). A segment
    [cuts[k], cuts[k+1]) carries the rise of q across it, max(q[k+1]-q[k], 0),
    spread evenly over its tokens. Weights are then clipped to `clip` times
    the mean and renormalized to mean 1, so a step's total credit is
    unchanged -- only where it lands inside the response moves. A flat probe
    (no rise anywhere) returns uniform weights, which is exactly GACA.
    """
    w = np.ones(L, dtype=np.float64)
    if L <= 0 or not cuts:
        return w
    b = list(cuts) + [L]
    qv = list(q) + [1.0]
    dens = np.zeros(L)
    for k in range(len(cuts)):
        a, e = b[k], b[k + 1]
        if e > a:
            dens[a:e] = max(qv[k + 1] - qv[k], 0.0) / (e - a)
    if dens.sum() <= 1e-12:
        return w
    dens = dens / dens.mean()
    lo, hi = clip
    for _ in range(20):  # clip-and-renormalize to a fixed point
        dens = np.clip(dens, lo, hi)
        dens = dens / dens.mean()
        if dens.min() >= lo - 1e-9 and dens.max() <= hi + 1e-9:
            break
    return dens
