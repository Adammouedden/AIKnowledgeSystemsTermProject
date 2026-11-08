"""Checks of the advantage estimators against the equations and the stated
special cases of the GACA paper. Run: pixi run test
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import advantages as A


def mk(group, traj, rets, g_hats, anchors, nlls):
    return [
        {"group": group, "traj": traj, "t": t, "ret": rets, "g_hat": g_hats[t], "anchor": anchors[t], "nll": nlls[t]}
        for t in range(len(g_hats))
    ]


def test_discounted():
    np.testing.assert_allclose(A.discounted_step_returns([0, 0, 1], 0.5), [0.25, 0.5, 1.0])
    np.testing.assert_allclose(A.discounted_step_returns([-0.1, 0, 1.0], 1.0), [0.9, 1.0, 1.0])
    np.testing.assert_allclose(A.discounted_step_returns([], 0.9), [])
    print("ok discounted_step_returns")


def test_episode_rowwise():
    # Two trajectories in one group with unequal lengths: the mean must be
    # length-weighted over step-rows (Appendix D), not a plain per-trajectory mean.
    steps = mk(0, 0, 1.0, [1, 1, 1], ["a", "b", "c"], [1, 1, 1]) + mk(0, 1, 0.0, [0], ["a"], [1])
    a = A.episode_advantage(steps, "mean")
    rows = np.array([1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(a, rows - rows.mean())
    assert abs(a[0] - 0.25) < 1e-12, a  # row-wise mean 0.75, not 0.5
    a2 = A.episode_advantage(steps, "mean_std")
    np.testing.assert_allclose(a2, (rows - rows.mean()) / (rows.std(ddof=1) + A.EPS))
    # Groups are normalized independently.
    steps2 = steps + mk(1, 0, 5.0, [5], ["z"], [1]) + mk(1, 1, 7.0, [7], ["z"], [1])
    a3 = A.episode_advantage(steps2, "mean")
    np.testing.assert_allclose(a3[:4], a)
    np.testing.assert_allclose(a3[4:], [-1.0, 1.0])
    print("ok episode_advantage (row-wise convention)")


def test_step_anchor_grouping():
    # Steps sharing an anchor are contrasted within that anchor group;
    # singletons fall back to prompt-group statistics.
    steps = mk(0, 0, 1.0, [1.0, 0.5], ["A", "B"], [1, 1]) + mk(0, 1, 0.0, [0.0, 0.25], ["A", "C"], [1, 1])
    a, cov = A.step_advantage(steps, "mean")
    # Anchor A holds rows 0 and 2 (g_hat 1.0 and 0.0) -> +-0.5
    assert abs(a[0] - 0.5) < 1e-12 and abs(a[2] + 0.5) < 1e-12, a
    # Rows 1 and 3 are singletons -> prompt-group fallback over all four g_hats
    allg = np.array([1.0, 0.5, 0.0, 0.25])
    assert abs(a[1] - (0.5 - allg.mean())) < 1e-12
    assert abs(a[3] - (0.25 - allg.mean())) < 1e-12
    assert abs(cov - 0.5) < 1e-12, cov
    print("ok step_advantage (anchor grouping + singleton fallback)")


def test_criticality():
    # c_t has mean one within each trajectory, by construction (Eq. 5).
    steps = mk(0, 0, 1.0, [1, 1, 1, 1], list("abcd"), [0.5, 1.0, 1.5, 2.0])
    c = A.criticality(steps)
    np.testing.assert_allclose(c.mean(), 1.0, atol=1e-5)  # exact up to the EPS guard
    np.testing.assert_allclose(c, np.array([0.5, 1.0, 1.5, 2.0]) / 1.25, rtol=1e-5)
    # Monotone in the NLL, and scale invariant.
    assert np.all(np.diff(c) > 0)
    steps10 = mk(0, 0, 1.0, [1, 1, 1, 1], list("abcd"), [5.0, 10.0, 15.0, 20.0])
    np.testing.assert_allclose(c, A.criticality(steps10), rtol=1e-5)
    # Uniform control forces c == 1; random control permutes the same multiset.
    np.testing.assert_allclose(A.criticality(steps, mode="uniform"), np.ones(4))
    cr = A.criticality(steps, mode="random", rng=np.random.default_rng(1))
    np.testing.assert_allclose(np.sort(cr), np.sort(c), rtol=1e-5)
    # Normalization is per trajectory, not global.
    two = mk(0, 0, 1.0, [1, 1], ["a", "b"], [1.0, 3.0]) + mk(0, 1, 1.0, [1, 1], ["a", "b"], [10.0, 30.0])
    c2 = A.criticality(two)
    np.testing.assert_allclose(c2[:2], c2[2:], rtol=1e-5)
    print("ok criticality (Eqs. 4-5, controls)")


def test_criticality_fusion():
    # alpha_c < 1 fuses the reward-change term (Eq. 6).
    g = [0.81, 0.9, 1.0]
    steps = mk(0, 0, 1.0, g, list("abc"), [1.0, 1.0, 1.0])
    c = A.criticality(steps, alpha_c=0.0)
    delta = np.abs(np.diff(np.array(g), prepend=g[0]))
    want = delta / delta.mean()
    np.testing.assert_allclose(c, want / want.mean(), rtol=1e-5)
    half = A.criticality(steps, alpha_c=0.5)
    e = np.ones(3)
    mix = 0.5 * e + 0.5 * want
    np.testing.assert_allclose(half, mix / mix.mean(), rtol=1e-5)
    print("ok criticality fusion (Eq. 6)")


def test_loo_criticality():
    """Appendix B.4: the leave-one-out score averages over the OTHER visits to
    the same anchor state, is independent of this rollout's own action, and has
    lower variance than the single-sample score."""
    # Two trajectories, steps 0 and 1 share anchors A and B across rollouts.
    steps = mk(0, 0, 1.0, [1, 1], ["A", "B"], [0.4, 3.0]) + mk(0, 1, 0.0, [0, 0], ["A", "B"], [2.0, 1.0])
    loo = A._loo_nll(steps)
    # Anchor A holds rows 0 and 2 -> each takes the other's NLL.
    assert abs(loo[0] - 2.0) < 1e-12 and abs(loo[2] - 0.4) < 1e-12, loo
    # Anchor B holds rows 1 and 3.
    assert abs(loo[1] - 1.0) < 1e-12 and abs(loo[3] - 3.0) < 1e-12, loo
    # A row's own NLL never enters its own score (that is the whole point).
    bumped = [dict(s) for s in steps]
    bumped[0]["nll"] = 99.0
    assert abs(A._loo_nll(bumped)[0] - loo[0]) < 1e-12

    # Singletons keep their own H_t, since there is no leave-one-out average.
    solo = mk(0, 0, 1.0, [1], ["Z"], [0.7])
    assert abs(A._loo_nll(solo)[0] - 0.7) < 1e-12

    # Three visits: the average of the other two.
    three = (mk(0, 0, 1.0, [1], ["A"], [1.0]) + mk(0, 1, 1.0, [1], ["A"], [2.0])
             + mk(0, 2, 1.0, [1], ["A"], [6.0]))
    l3 = A._loo_nll(three)
    assert abs(l3[0] - 4.0) < 1e-12 and abs(l3[1] - 3.5) < 1e-12 and abs(l3[2] - 1.5) < 1e-12, l3

    # Lower variance than the single-sample score, as the appendix claims.
    rng = np.random.default_rng(0)
    many = []
    for j in range(8):
        win = 1.0 if j % 2 == 0 else 0.0  # mixed outcomes, else every advantage is 0
        many += mk(0, j, win, list(A.discounted_step_returns([0.0] * 5 + [win], 0.95)),
                   [f"S{t}" for t in range(6)], list(rng.uniform(0.1, 3.0, 6)))
    single = np.array([s["nll"] for s in many])
    loo_v = np.array([A._loo_nll(many)[i] for i in range(len(many))])
    assert loo_v.std() < single.std(), (loo_v.std(), single.std())

    # It routes through compute_advantages and stays finite.
    adv, d = A.compute_advantages(many, "gaca", norm="mean_std", score_mode="loo")
    assert np.all(np.isfinite(adv))
    # And it is a different estimator from the default.
    adv_e, _ = A.compute_advantages(many, "gaca", norm="mean_std", score_mode="entropy")
    assert not np.allclose(adv, adv_e)
    print("ok leave-one-out criticality (Appendix B.4 debiased variant)")


def test_mixing_law():
    # Eq. 7: continuous at c=1, nondecreasing, clamped to [0, 1].
    c = np.linspace(0.0, 4.0, 401)
    lam = A.mixing_weight(c, 0.5, 0.5, 0.5)
    assert lam.min() >= 0.0 and lam.max() <= 1.0
    assert np.all(np.diff(lam) >= -1e-12)
    assert abs(A.mixing_weight([1.0], 0.5, 0.5, 0.5)[0] - 0.5) < 1e-12
    # Figure 3 / Section 4.4: at lam0 = alpha = 0.5 the clamp binds at c = 3.
    assert abs(A.mixing_weight([3.0], 0.5, 0.5, 0.5)[0] - 1.0) < 1e-12
    assert A.mixing_weight([2.999], 0.5, 0.5, 0.5)[0] < 1.0
    # Section 4.4: at alpha = 1 the clamp binds at c = 2 instead.
    assert abs(A.mixing_weight([2.0], 0.5, 1.0, 1.0)[0] - 1.0) < 1e-12
    # Modulation off freezes the weight at lam0.
    np.testing.assert_allclose(A.mixing_weight(c, 0.5, 0.0, 0.0), 0.5)
    # Section 4.4: std(lam) = lam0 * alpha * std(c) while unclamped.
    rng = np.random.default_rng(0)
    cc = 1.0 + 0.31 * rng.standard_normal(20000)
    cc = cc[(cc > 0.2) & (cc < 2.5)]
    l = A.mixing_weight(cc, 0.5, 0.5, 0.5)
    assert abs(l.std() - 0.5 * 0.5 * cc.std()) < 5e-3, (l.std(), 0.25 * cc.std())
    print("ok mixing_weight (Eq. 7)")


def _sample(n_traj=4, T=5, seed=0):
    rng = np.random.default_rng(seed)
    steps = []
    for j in range(n_traj):
        rew = [0.0] * T
        if j % 2 == 0:
            rew[-1] = 1.0
        g = A.discounted_step_returns(rew, 0.95)
        anchors = ["A"] + [f"{j}-{t}" if rng.random() < 0.4 else f"S{t}" for t in range(1, T)]
        nll = list(rng.uniform(0.2, 2.0, T))
        steps += mk(0, j, float(sum(rew)), list(g), anchors, nll)
    return steps


def test_method_special_cases():
    steps = _sample()
    a_e = A.episode_advantage(steps, "mean_std")
    a_s, _ = A.step_advantage(steps, "mean_std")

    grpo, _ = A.compute_advantages(steps, "grpo", norm="mean_std")
    np.testing.assert_allclose(grpo, a_e)

    gigpo, _ = A.compute_advantages(steps, "gigpo", norm="mean_std", w=1.0)
    np.testing.assert_allclose(gigpo, a_s + a_e)

    # "Turning the modulation off freezes lam_t at lam0": GACA with
    # alpha=beta=0 equals the fixed convex control.
    off, doff = A.compute_advantages(steps, "gaca", norm="mean_std", lam0=0.5, alpha_lam=0, beta_lam=0)
    fixed, _ = A.compute_advantages(steps, "fixed", norm="mean_std", lam0=0.5)
    np.testing.assert_allclose(off, fixed)
    # ... and at lam0 = 1/2 that is GiGPO's combination scaled by 1/2.
    np.testing.assert_allclose(off, 0.5 * (a_s + a_e))
    assert abs(doff["lam_mean"] - 0.5) < 1e-12 and doff["lam_std"] < 1e-12

    # The uniform criticality control is the same configuration as modulation-off.
    uni, _ = A.compute_advantages(steps, "gaca", norm="mean_std", score_mode="uniform")
    np.testing.assert_allclose(uni, fixed)

    # lam0 extremes: pure episode-level and pure step-level credit.
    pe, _ = A.compute_advantages(steps, "fixed", norm="mean_std", lam0=0.0)
    ps, _ = A.compute_advantages(steps, "fixed", norm="mean_std", lam0=1.0)
    np.testing.assert_allclose(pe, a_e)
    np.testing.assert_allclose(ps, a_s)

    # GACA is a convex combination, so it never exceeds both constituents.
    gaca, dg = A.compute_advantages(steps, "gaca", norm="mean_std")
    assert np.all(np.abs(gaca) <= np.maximum(np.abs(a_s), np.abs(a_e)) + 1e-9)
    assert not np.allclose(gaca, fixed)  # the weight really moves
    assert abs(dg["lam_mean"] - 0.5) < 0.02  # mean pinned at lam0 (Section 4.4)
    print("ok compute_advantages special cases (GRPO / GiGPO / fixed / GACA)")


def test_gaca_tilts_toward_step_on_uncertain_steps():
    # lam_t must be larger exactly where the NLL is larger within a trajectory.
    steps = _sample(seed=3)
    c = A.criticality(steps)
    lam = A.mixing_weight(c, 0.5, 0.5, 0.5)
    for key in {(s["group"], s["traj"]) for s in steps}:
        idx = [i for i, s in enumerate(steps) if (s["group"], s["traj"]) == key]
        nll = np.array([steps[i]["nll"] for i in idx])
        l = lam[idx]
        assert np.corrcoef(nll, l)[0, 1] > 0.99, np.corrcoef(nll, l)[0, 1]
    print("ok GACA leans on the step signal where NLL is high")


def test_degenerate_inputs():
    # All-equal returns (every rollout failed): zero episode advantage, no NaNs.
    steps = mk(0, 0, 0.0, [0, 0], ["A", "B"], [1, 1]) + mk(0, 1, 0.0, [0, 0], ["A", "B"], [1, 1])
    for m in ("grpo", "gigpo", "fixed", "gaca"):
        adv, d = A.compute_advantages(steps, m, norm="mean_std")
        assert np.all(np.isfinite(adv)), (m, adv)
        np.testing.assert_allclose(adv, 0.0, atol=1e-6)
    # Zero NLL everywhere must not divide by zero.
    steps0 = mk(0, 0, 1.0, [1, 1], ["A", "B"], [0.0, 0.0]) + mk(0, 1, 0.0, [0, 0], ["A", "B"], [0.0, 0.0])
    adv, d = A.compute_advantages(steps0, "gaca", norm="mean_std")
    assert np.all(np.isfinite(adv))
    # A single trajectory in a group: std over rows still defined, no NaN.
    one = mk(0, 0, 1.0, [1, 1], ["A", "B"], [1.0, 2.0])
    for m in ("grpo", "gigpo", "fixed", "gaca"):
        adv, _ = A.compute_advantages(one, m, norm="mean_std")
        assert np.all(np.isfinite(adv)), (m, adv)
    assert A.compute_advantages([], "gaca")[0].size == 0
    print("ok degenerate inputs")


def test_coverage_diagnostic():
    steps = mk(0, 0, 1.0, [1, 1, 1], ["A", "B", "C"], [1, 1, 1]) + mk(0, 1, 0.0, [0, 0, 0], ["A", "B", "X"], [1, 1, 1])
    _, cov = A.step_advantage(steps, "mean")
    assert abs(cov - 4 / 6) < 1e-12, cov
    print("ok step coverage diagnostic")


def test_reliability_gate():
    # Anchor sizes: A shared by both trajectories (m=2), B by both, C and X singletons.
    steps = mk(0, 0, 1.0, [0.81, 0.9, 1], ["A", "B", "C"], [1, 2, 3]) + mk(0, 1, 0.0, [0, 0, 0], ["A", "B", "X"], [3, 2, 1])
    np.testing.assert_allclose(A.anchor_sizes(steps), [2, 2, 1, 2, 2, 1])
    lam = np.array([0.5, 0.9, 0.9, 0.0, 1.0, 0.3])
    m = np.array([2, 8, 1, 4, 1e9, 1])
    out = A.reliability(lam, m)
    # odds scaled by rho = 1 - 1/m: 0.5 at m=2 -> odds 0.5 -> 1/3
    np.testing.assert_allclose(out[0], (0.5 * 0.5) / (0.5 * 0.5 + 0.5))
    np.testing.assert_allclose(out[1], (0.9 * 0.875) / (0.9 * 0.875 + 0.1))
    assert out[2] == 0.0 and out[5] == 0.0, out  # singletons: no contrast, no weight
    assert out[3] == 0.0  # lambda = 0 stays 0
    assert abs(out[4] - 1.0) < 1e-6  # lambda = 1 with a huge group stays 1
    assert np.all(out <= lam + 1e-12), "the correction can only shrink the weight"
    # Wired into compute_advantages: singleton steps get exactly the episode advantage.
    adv, _ = A.compute_advantages(steps, method="gaca", norm="mean", rel="msize")
    a_e = A.episode_advantage(steps, "mean")
    np.testing.assert_allclose(adv[[2, 5]], a_e[[2, 5]])
    adv0, _ = A.compute_advantages(steps, method="gaca", norm="mean")
    assert not np.allclose(adv0[[2, 5]], a_e[[2, 5]])  # the paper's gate does mix them
    print("ok reliability gate (anchor-group size, Prop. 3)")


def test_probe_token_weights():
    L = 40
    # flat probe -> exactly uniform (== GACA)
    w = A.probe_token_weights(L, [0, 16, 32], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(w, 1.0)
    # mean-preserving and clipped
    w = A.probe_token_weights(L, [0, 16, 32], [0.1, 0.2, 0.9])
    assert abs(w.mean() - 1.0) < 1e-9, w.mean()
    assert w.min() >= 0.25 - 1e-6 and w.max() <= 4.0 + 1e-6, (w.min(), w.max())
    # the segment where q jumps (16..32: 0.2 -> 0.9) gets the most weight
    assert w[16:32].mean() > w[0:16].mean() and w[16:32].mean() > w[32:].mean()
    # decreasing q contributes nothing extra (rise is clipped at 0)
    w2 = A.probe_token_weights(L, [0, 16, 32], [0.5, 0.3, 0.3])
    assert w2[0:16].max() <= w2[32:].max() + 1e-9
    # cut grid
    assert A.probe_cuts(50, 35, 16) == [0, 16, 32, 35]
    assert A.probe_cuts(50, 0, 16) == [0]
    assert A.probe_cuts(10, 99, 16) == [0, 10]
    # GACA advantage broadcast through uniform weights is unchanged
    adv = 0.7
    np.testing.assert_allclose(adv * A.probe_token_weights(30, [0], [1.0]), adv)
    print("ok probe token weights (mean-preserving, clipped, uniform == GACA)")


if __name__ == "__main__":
    test_discounted()
    test_episode_rowwise()
    test_step_anchor_grouping()
    test_criticality()
    test_criticality_fusion()
    test_loo_criticality()
    test_mixing_law()
    test_method_special_cases()
    test_gaca_tilts_toward_step_on_uncertain_steps()
    test_degenerate_inputs()
    test_coverage_diagnostic()
    test_reliability_gate()
    test_probe_token_weights()
    print("\nALL ADVANTAGE TESTS PASSED")
