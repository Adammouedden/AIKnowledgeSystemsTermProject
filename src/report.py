"""Build docs/RESULTS.md and the learning-curve figure from the verl-agent arms
(src/runs/verl/) and the offline token studies (src/runs/tokens/).

    python src/report.py
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mcnemar(b, c):
    """Two-sided exact McNemar test on the discordant pairs."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ===================================================================== verl-agent arms
# Experiments of attempt 4 (A100, verl-agent, the paper's reasoning format).
# `python src/report.py` rebuilds docs/RESULTS.md from src/runs/verl/*.

VERL_ARMS = [  # (dir, label, role, what differs from GACA/GRPO)
    ("grpo", "GRPO", "control", "episode-level advantage only (Eq. 2)"),
    ("gaca", "GACA", "control", "the paper's estimator: NLL-gated mix of step and episode advantage (Eq. 8)"),
    ("gaca_msize", "GACA + m_z", "treatment", "GACA, lambda odds scaled by 1 - 1/m_z (anchor-group size; Prop. 3)"),
    ("gaca_loo", "GACA-LOO", "treatment", "GACA, criticality = leave-one-out mean NLL of the other visits (App. B.4)"),
    ("gaca_probe", "GACA + probe (seed 0)", "treatment", "GACA, step advantage spread over tokens by the commitment probe"),
    ("gaca_probe_s1", "GACA + probe (seed 1)", "treatment", "as above, training seed 1, same 128 validation games"),
    ("gaca_probe_c05", "GACA + probe, clip [0.5, 2]", "treatment", "probe with token weights clipped to [0.5, 2] instead of [0.25, 4]"),
    ("gaca_probe_pos", "GACA + probe, positive A only", "treatment", "probe weights only on steps with a positive advantage; others uniform"),
    ("gaca_probe_ref", "GACA + probe, frozen-ref scorer", "treatment", "probe scored by the frozen reference model instead of the policy"),
]
VERL_COLORS = {"grpo": "#2a78d6", "gaca": "#eb6834", "gaca_msize": "#1baf7a", "gaca_loo": "#eda100",
               "gaca_probe": "#e87ba4", "gaca_probe_s1": "#e87ba4", "gaca_probe_c05": "#e87ba4", "gaca_probe_pos": "#e87ba4",
               "gaca_probe_ref": "#e87ba4"}
# validated categorical slots 1-5, fixed order; probe variants keep the probe's hue (colour follows the
# method) and are told apart by line style + legend + direct labels
VERL_STYLE = {"gaca_probe_s1": ":", "gaca_probe_c05": "--", "gaca_probe_pos": "-.", "gaca_probe_ref": (0, (1, 1, 4, 1))}


def _verl_metrics(path):
    import re

    rows = []
    for l in open(path, errors="ignore"):
        if re.search(r"step:\d+ ", l):
            rows.append({k: float(v) for k, v in re.findall(r"([\w/\-\.]+):(-?[\d\.]+(?:e-?\d+)?)", l)})
    return rows


def _verl_episodes(path):
    import collections

    d = collections.defaultdict(dict)
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            d[r["tag"]][r["env"]] = r
    return d


def verl_load():
    import re

    arms = []
    log_lines = open(os.path.join(ROOT, "src/runs/verl/arms.log")).read() if os.path.exists(os.path.join(ROOT, "src/runs/verl/arms.log")) else ""
    for key, label, role, what in VERL_ARMS:
        base = os.path.join(ROOT, "src/runs/verl", key)
        if not os.path.exists(os.path.join(base, "train.log")):
            continue
        rows = _verl_metrics(os.path.join(base, "train.log"))
        eps = _verl_episodes(os.path.join(base, "val_episodes.jsonl"))
        val_iters = [int(r["step"]) for r in rows if "val/success_rate" in r]
        tags = sorted(eps)
        curve = []
        for t, it in zip(tags, val_iters):
            if len(eps[t]) == 128:
                curve.append((it, sum(e["won"] for e in eps[t].values()), eps[t]))
        tr = [r for r in rows if "timing_s/step" in r]
        done = f"end {key}\n" in log_lines + "\n" or bool(re.search(rf"end {key}$", log_lines, re.M))
        arms.append(dict(key=key, label=label, role=role, what=what, rows=tr, curve=curve, done=done,
                         iters=int(tr[-1]["step"]) if tr else 0))
    return arms


def _types(eps):
    import collections

    c = collections.defaultdict(lambda: [0, 0])
    for e in eps.values():
        tt = os.path.basename(os.path.dirname(os.path.dirname(e["gamefile"]))).split("-")[0]
        c[tt][0] += e["won"]
        c[tt][1] += 1
    return c


def verl_plot(arms):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    base = next((a["curve"][0][1] for a in arms if a["curve"]), None)
    if base is not None:
        ax.axhline(100 * base / 128, color=ink2, lw=1.0, ls=(0, (2, 3)), zorder=1)
        ax.annotate("base model", (44, 100 * base / 128), color=ink2, fontsize=8, va="bottom", ha="right")
    ends = []
    for a in arms:
        if len(a["curve"]) < 2:
            continue
        x = [c[0] for c in a["curve"]]
        y = [100 * c[1] / 128 for c in a["curve"]]
        ls = VERL_STYLE.get(a["key"], "-")
        ax.plot(x, y, color=VERL_COLORS[a["key"]], lw=2.0, ls=ls, marker="o", ms=5, mfc=surface, mew=2.0,
                zorder=3, label=a["label"] + ("" if a["done"] else " (running)"))
        if a["done"]:  # running arms are identified by the legend only, so labels never sit on other lines
            ends.append([x[-1], y[-1], a["label"]])
    # direct labels at line ends, nudged apart so they never collide
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 1.6:
            ends[i][1] = ends[i - 1][1] + 1.6
    xl = max((e[0] for e in ends), default=0) + 0.6  # one label column past the last point
    for x, y, lab in ends:
        ax.annotate(f" {lab}", (xl, y), color=ink, fontsize=8.5, va="center", ha="left", annotation_clip=False)
    ax.set_xlabel("Training iteration", color=ink2, fontsize=10)
    ax.set_ylabel("Held-out success (%), 128 games", color=ink2, fontsize=10)
    ax.set_title("ALFWorld, Qwen2.5-1.5B-Instruct, full fine-tuning, one A100 (single seed unless marked)",
                 color=ink, fontsize=10, pad=10)
    ax.grid(True, axis="y", color=grid, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=9, length=0)
    ax.set_xlim(-1, 50)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=ink2)
    fig.tight_layout()
    path = os.path.join(ROOT, "src", "runs", "verl", "learning_curves.png")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return os.path.relpath(path, ROOT)


def _q0_drift(R):
    q = [r["probe/q0_mean"] for r in R if "probe/q0_mean" in r]
    return f"{np.mean(q[:5]):.2f} -> {np.mean(q[-5:]):.2f}" if len(q) >= 10 else "--"


def _cir(R):
    c = [r["probe/cir_mean"] for r in R if "probe/cir_mean" in r]
    return f"{np.mean(c[:5]):.2f} -> {np.mean(c[-5:]):.2f}" if len(c) >= 10 else "--"


def _j(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p)) if os.path.exists(p) else None


def verl_main():
    arms = verl_load()
    out = []
    w = out.append
    w("# Results: GACA vs GRPO, and new credit-assignment treatments, on ALFWorld")
    w("")
    w("Generated by `python src/report.py` from `src/runs/verl/` and `src/runs/tokens/`; do not edit by hand.")
    w("Judgement calls and their measurements are in `DECISIONS.md`; ideas and reasoning in `IDEAS.md`.")
    w("Earlier attempts (RTX 4060: index actions, LoRA, two task types) are in `RESULTS_4060.md`.")
    w("")
    w("## Setup shared by every arm")
    w("")
    w("verl-agent (the paper's pipeline) on one A100 40 GB. Qwen2.5-1.5B-Instruct, full fine-tuning, lr 1e-6,")
    w("the paper's `<think>`/`<action>` prompt with a 512-token response cap, 50 environment steps, group size 8,")
    w("gamma 0.95, KL 0.01 (low-variance), invalid-action penalty 0.1, mean-and-std normalization, PPO mini-batch")
    w("256. **4 tasks x 8 rollouts per iteration (paper: 16 x 8), 34 iterations (paper: 150), one seed (paper: 3)**")
    w("-- ~1,088 trajectories per arm, ~6% of the paper's 19,200. All six ALFWorld task types; every evaluation is")
    w("the **same 128 in-distribution games**, one episode each, sampled at T = 0.4, so arms are compared task by")
    w("task (exact McNemar). Arms differ only in the advantage estimator (`algorithm.gaca.*`). One evaluation game")
    w("is 0.78 points.")
    w("")
    w("## Arms")
    w("")
    w("| arm | role | differs from the controls by | status | iterations | train h | final success | best (iter) |")
    w("|---|---|---|---|---|---|---|---|")
    base = None
    for a in arms:
        if a["curve"]:
            base = a["curve"][0]
            break
    if base:
        w(f"| base model | reference | no training | done | 0 | 0 | **{100 * base[1] / 128:.1f}%** ({base[1]}/128) | -- |")
    for a in arms:
        if not a["curve"]:
            continue
        fin = a["curve"][-1]
        best = max(a["curve"], key=lambda c: c[1])
        hrs = sum(r["timing_s/step"] for r in a["rows"]) / 3600
        st = "done" if a["done"] else "running"
        w(f"| {a['label']} | {a['role']} | {a['what']} | {st} | {a['iters']} | {hrs:.2f} | "
          f"**{100 * fin[1] / 128:.1f}%** ({fin[1]}/128) | {100 * best[1] / 128:.1f}% ({best[0]}) |")
    w("")
    w("Final = the last evaluation (the end-of-training checkpoint). Best is reported for comparison with the")
    w("paper's best-checkpoint convention but is optimistic: it picks the maximum over five noisy evaluations.")
    w("")
    w(f"![Held-out success over training](../{verl_plot(arms)})")
    w("")
    w("### Learning curves (held-out success %, 128 games)")
    w("")
    its = [0, 10, 20, 30]
    w("| arm | " + " | ".join(f"iter {i}" for i in its) + " | final (iter) |")
    w("|---|" + "---|" * (len(its) + 1))
    for a in arms:
        m = {c[0]: c[1] for c in a["curve"]}
        last = a["curve"][-1] if a["curve"] else None
        fin = f"{100 * last[1] / 128:.1f} ({last[0]})" if a["done"] and last and last[0] not in its else ""
        w(f"| {a['label']} | " + " | ".join(f"{100 * m[i] / 128:.1f}" if i in m else "" for i in its) + f" | {fin} |")
    w("")
    fin = [a for a in arms if a["done"] and a["curve"]]
    if len(fin) >= 2:
        w("## Paired comparison of final checkpoints")
        w("")
        w("Exact two-sided McNemar on the 128 shared games; `A only` / `B only` are the games one arm solves and")
        w("the other does not. The base model is each arm's own iteration-0 evaluation.")
        w("")
        w("| A vs B | A | B | A only | B only | both | p |")
        w("|---|---|---|---|---|---|---|")
        pool = [("base model", base[2])] + [(a["label"], a["curve"][-1][2]) for a in fin]
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                (na, ea), (nb, eb) = pool[i], pool[j]
                b = sum(ea[e]["won"] and not eb[e]["won"] for e in ea)
                c = sum(eb[e]["won"] and not ea[e]["won"] for e in ea)
                both = sum(ea[e]["won"] and eb[e]["won"] for e in ea)
                w(f"| {na} vs {nb} | {sum(x['won'] for x in ea.values())} | {sum(x['won'] for x in eb.values())} "
                  f"| {b} | {c} | {both} | {mcnemar(b, c):.3f} |")
        w("")
        w("### Final success by task type")
        w("")
        tts = sorted(_types(base[2]))
        w("| arm | " + " | ".join(t.replace("_", " ") for t in tts) + " |")
        w("|---|" + "---|" * len(tts))
        for name, eps in pool:
            c = _types(eps)
            w(f"| {name} | " + " | ".join(f"{c[t][0]}/{c[t][1]}" for t in tts) + " |")
        w("")
    w("## Training diagnostics (mean over iterations)")
    w("")
    w("| arm | s / iter | probe s / iter | valid actions | response tokens | KL (last 5) | c_t std | lambda std | step coverage | q(0) first 5 -> last 5 | CIR |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        R = a["rows"]
        if not R:
            continue
        mean = lambda k: np.mean([r[k] for r in R if k in r]) if any(k in r for r in R) else float("nan")
        kl5 = np.mean([r.get("actor/kl_loss", np.nan) for r in R[-5:]])
        pr = mean('timing_s/probe')
        pr = f"{pr:.0f}" if np.isfinite(pr) else "--"
        gs = (lambda k, f: "--" if a["key"] == "grpo" else format(mean(k), f))
        w(f"| {a['label']} | {mean('timing_s/step'):.0f} | {pr} | {100 * mean('episode/valid_action_ratio'):.1f}% "
          f"| {mean('response_length/mean'):.0f} | {kl5:.3f} | {gs('gaca/c_std', '.2f')} | {gs('gaca/lam_std', '.3f')} | {gs('gaca/step_coverage', '.2f')} "
          f"| {_q0_drift(R)} | {_cir(R)} |")
    w("")
    w("Paper Table 7 (ALFWorld): c_t std 0.31, lambda std 0.08, clamped < 0.01, step coverage 0.86.")
    w("q(0): probability of the eventual action before any reasoning (probe arms). CIR: Causal Importance of")
    w("Reasoning (Yu et al., 2026), mean JS divergence between the probe at each cut and at the full reasoning;")
    w("logged from `gaca_probe_c05` on. A rising q(0) / falling CIR means the reasoning matters less over training.")
    w("")
    w("## Offline studies (no training)")
    w("")
    e = _j("src/runs/verl/entropy_check_iter1-5.json")
    if e:
        w("### Is GACA's step score noise? (sampled NLL vs exact token entropy)")
        w("")
        w(f"{e['rows']:,} step rows from the first {e['iters']} iterations of a GACA-policy run. Both scores come from the same")
        w("forward pass on the same responses.")
        w("")
        w("| quantity | value |")
        w("|---|---|")
        w(f"| correlation NLL vs exact entropy (Pearson / within-trajectory Spearman) | {e['pearson']:.2f} / {e['within_traj_spearman_mean']:.2f} |")
        w(f"| steps whose lambda moves > 0.1 when NLL is replaced by exact entropy | {100 * e['frac_lambda_diff_gt_0.1']:.1f}% |")
        w(f"| share of the score's variance that is within-state (repeated visits), NLL / entropy | {100 * e['noise_share_nll']:.0f}% / {100 * e['noise_share_ent']:.0f}% |")
        w(f"| share of a response's entropy carried by its top-20% tokens | {100 * e['top20_entropy_share_mean']:.0f}% |")
        w("")
        w("Reading: swapping the step gate's signal changes almost nothing, but most of either score is")
        w("path-dependent noise rather than a property of the state.")
        w("")
    hb, hh, hr = _j("src/runs/tokens/base.analysis.json"), _j("src/runs/tokens/heldout.analysis.json"), _j("src/runs/tokens/heldout.robust.json")
    if hh:
        w("### Which tokens decide the action? (flip-test oracle)")
        w("")
        w("For real steps of the base policy, the response is cut after j tokens and the rest resampled 8 times;")
        w("F(j) = P(final action changes). The drop of F across a segment is that segment's decision importance.")
        w("Token-weighting schemes are scored by how much of their credit lands on decision importance, relative to")
        w("uniform credit (= GACA), and by total-variation distance to importance (lower is better).")
        w("")
        w("| scheme | first run (48 games) credit vs uniform | held-out run (48 new games) credit vs uniform | held-out TV - uniform [95% CI] |")
        w("|---|---|---|---|")
        for k in ("uniform", "entropy", "nll", "action_span", "hybrid", "probe"):
            a1 = f"{hb[k]['credit_on_importance_vs_uniform']:.2f}x" if hb and k in hb else "--"
            a2 = hh[k]
            w(f"| {k} | {a1} | {a2['credit_on_importance_vs_uniform']:.2f}x | {a2['tv_minus_uniform']:+.3f} "
              f"[{a2['tv_minus_uniform_ci'][0]:+.3f}, {a2['tv_minus_uniform_ci'][1]:+.3f}] |")
        w("")
        w(f"Decision importance inside the `<action>` tags: {100 * hh['importance_share_inside_action_tags']:.0f}%; fully resampling a")
        w(f"response changes the action {100 * hh['F0_mean']:.0f}% of the time. The hybrid was designed from the first run and")
        w("did not hold up out of sample. The probe column is unclipped segment mass.")
        w("")
    ht = _j("src/runs/tokens/trained_s1.analysis.json")
    hrt = _j("src/runs/tokens/trained_s1.robust.json")
    if ht and hh:
        w("### Does the probe still work on a trained policy?")
        w("")
        w("The same oracle on the same 48 held-out games, with the final weights of `gaca_probe_s1` (20.3%) in place")
        w("of the base model.")
        w("")
        w("| | base policy | trained (probe seed 1) |")
        w("|---|---|---|")
        w(f"| probe credit on decision importance vs uniform (unclipped) | {hh['probe']['credit_on_importance_vs_uniform']:.2f}x | {ht['probe']['credit_on_importance_vs_uniform']:.2f}x |")
        if hr and hrt:
            a, b = hr["clip"]["0.25-4"], hrt["clip"]["0.25-4"]
            w(f"| same, training configuration (clip [0.25, 4]) [95% CI] | {a['ratio']:.2f}x [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}] | {b['ratio']:.2f}x [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}] |")
        w(f"| probe TV - uniform [95% CI] | {hh['probe']['tv_minus_uniform']:+.3f} | {ht['probe']['tv_minus_uniform']:+.3f} [{ht['probe']['tv_minus_uniform_ci'][0]:+.3f}, {ht['probe']['tv_minus_uniform_ci'][1]:+.3f}] |")
        w(f"| entropy credit vs uniform | {hh['entropy']['credit_on_importance_vs_uniform']:.2f}x | {ht['entropy']['credit_on_importance_vs_uniform']:.2f}x |")
        w(f"| decision importance inside `<action>` | {100 * hh['importance_share_inside_action_tags']:.0f}% | {100 * ht['importance_share_inside_action_tags']:.0f}% |")
        w("")
        w("Training did not degrade the probe as a measurement (on the successful seed; seed 0 kept no weights).")
        w("")
    if hr:
        w("### Robustness of probe weights at the training configuration")
        w("")
        w("Exact training function (`advantages.probe_token_weights`, clip [0.25, 4], mean-preserving) on the held-out run.")
        w("")
        w("| slice | steps | credit on importance vs uniform [95% CI] | steps where probe > uniform |")
        w("|---|---|---|---|")
        for sec in ("spacing", "clip", "task_type", "step_t", "openness_F0", "split_half"):
            for k, x in hr[sec].items():
                w(f"| {sec}: {k} | {x['n']} | {x['ratio']:.2f}x [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}] | {100 * x['frac_steps_better']:.0f}% |")
        w("")
    w("## Runs that are not results")
    w("")
    w("- `src/runs/verl/gaca_random/`: GACA with the criticality score permuted within each trajectory (the paper's")
    w("  Table 2 ablation, which the paper already reports at 88.54 vs 96.88). Stopped at iteration 10 because it")
    w("  duplicates published work; its first-5-iteration step dump is the data of the entropy study above.")
    w("- `src/runs/verl/save_smoke/`: integration test of the final-weights upload (1 iteration, 16-game evaluation).")
    w("- `src/runs/a100/zs_*`: zero-shot prompt-format tests on the HF pipeline (two task types), see DECISIONS.md; code in legacy/4060_hf_pipeline/.")
    w("")
    w("## Caveats")
    w("")
    w("- **One seed per arm** (two for the probe once seed 1 completes). Earlier on this benchmark two seeds of one")
    w("  method differed by more than most gaps here; read p-values, not point differences.")
    w("- **~6% of the paper's training budget.** Every curve is still rising at the end; absolute numbers are not")
    w("  comparable to the paper's (96.88% for GACA at 1.5B). See DECISIONS.md for why the budget is what it is.")
    w("- **Matched on iterations (rollout budget), not wall-clock.** The probe arm takes ~50% longer per iteration.")
    w("  GACA + m_z and GACA-LOO ran under the 3.75 h wall-clock cap and stopped at 33 iterations, one short of")
    w("  the controls' 34 (~3% less data); the probe arms are fixed at 34.")
    w("- Offline token studies use the **base** policy (no trained checkpoint existed then).")
    w("")
    text = "\n".join(out)
    open(os.path.join(ROOT, "docs", "RESULTS.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    verl_main()
