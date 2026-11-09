"""Turn the run logs into a comparison table, a learning-curve figure and
training diagnostics. Run: pixi run report
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categorical slots 1-3 of the validated reference palette. Validated for
# all-pairs CVD separation in light mode; the aqua slot sits below 3:1 contrast
# on a light surface, so the figure carries direct labels and this report
# carries the table view (the relief rule).
# Colour follows the algorithm (the entity); a variant of the same algorithm
# keeps its colour and is distinguished by line style, never by a new hue.
COLORS = {"grpo": "#2a78d6", "gigpo": "#eb6834", "gaca": "#1baf7a", "fixed": "#6250d6"}
LABELS = {"grpo": "GRPO", "gigpo": "GiGPO", "gaca": "GACA", "fixed": "fixed-$\\lambda$"}
ORDER = ["grpo", "gigpo", "fixed", "gaca"]


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    start = next((r for r in recs if r.get("event") == "start"), None)
    if start is None:
        return None
    return {
        "args": start["args"],
        # Key on the file stem: two runs can share args["method"] but differ in
        # configuration (gaca vs gaca_mn), and they must not collide.
        "method": os.path.basename(path)[:-6],
        # `method` is the file stem so two runs never collide; `base` is the
        # algorithm itself, which is what colours, labels and the 2x2 key on.
        "base": start["args"]["method"],
        "name": os.path.basename(path)[:-6],
        "seed": start["args"].get("seed", 0),
        "norm": start["args"].get("norm", "mean_std"),
        "n_eval": start["n_eval_tasks"],
        "n_train": start["n_train_tasks"],
        "evals": [r for r in recs if r.get("event") == "eval"],
        "train": [r for r in recs if r.get("event") == "train"],
    }


def fmt(x, n=1):
    return "--" if x is None else f"{100 * x:.{n}f}"


def main():
    runs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", "4060", "*.jsonl"))):
        if os.path.basename(p).startswith("smoke"):
            continue
        r = load(p)
        if r and r["evals"]:
            r["path"] = p
            runs.append(r)
    if not runs:
        print("no runs found in runs/*.jsonl")
        return
    # mean_std first so the main results lead; "mean" sorts before it alphabetically.
    NORMRANK = {"mean_std": 0, "mean": 1}
    runs.sort(key=lambda r: (ORDER.index(r["base"]) if r["base"] in ORDER else 9,
                             NORMRANK.get(r["norm"], 9), r["seed"]))
    for r in runs:
        r["done"] = len(r["train"]) >= r["args"].get("iters", 0)
    multiseed = len({r["seed"] for r in runs}) > 1
    multinorm = len({r["norm"] for r in runs}) > 1
    NORM = {"mean_std": "mean+std-norm", "mean": "mean-norm"}
    for r in runs:
        bits = []
        if multinorm:
            bits.append(NORM.get(r["norm"], r["norm"]))
        if multiseed:
            bits.append(f"seed {r['seed']}")
        base = LABELS.get(r["base"], r["base"])
        r["label"] = f"{base} ({', '.join(bits)})" if bits else base
        if not r["done"]:
            r["label"] += f" — in progress, {len(r['train'])}/{r['args'].get('iters', '?')}"

    out = []
    w = out.append
    names = [LABELS.get(m, m) for m in ORDER if any(r["base"] == m for r in runs)]
    w(f"# Results: {' vs '.join(reversed(names))} on ALFWorld\n")

    a = runs[0]["args"]
    w("## Setup\n")
    w(f"- Backbone: `{a['model']}`, LoRA r={a['lora_r']}, lr={a['lr']}")
    w(f"- Task types: {a['task_types']}, rooms with <= {a['max_adm0']} admissible actions at reset")
    w(f"- {runs[0]['n_train']} training tasks, {runs[0]['n_eval']} held-out evaluation tasks")
    w(f"- Group size G={a['G']}, {a['tasks_per_iter']} task(s) per iteration, max {a['max_steps']} env steps")
    w(f"- gamma={a['gamma']}, invalid-action penalty {a['invalid_penalty']}, {a['norm']} advantage normalization")
    nseeds = len({r["seed"] for r in runs})
    w(f"- Iterations: {len(runs[0]['train'])} per run, {nseeds} seed(s) per method, "
      f"evaluated every {a['eval_every']}\n")

    # ---------------- headline table
    w("## Held-out success rate (%)\n")
    w("| Method | initial | best | final | best iter | vs GRPO (best) | wall (h) |")
    w("|---|---|---|---|---|---|---|")
    grpo_best = next((max(r["evals"], key=lambda e: e["success"])["success"]
                      for r in runs if r["base"] == "grpo" and r["norm"] == "mean_std" and r["done"]), None)
    for r in runs:
        ev = r["evals"]
        best = max(ev, key=lambda e: e["success"])
        delta = None if (grpo_best is None or not r["done"]) else best["success"] - grpo_best
        d = "--" if delta is None else f"{100 * delta:+.1f}"
        w(
            f"| {r['label']} | {fmt(ev[0]['success'])} | **{fmt(best['success'])}** | "
            f"{fmt(ev[-1]['success'])} | {best['iter']} | {d} | {ev[-1]['wall'] / 3600:.2f} |"
        )
    n = runs[0]["n_eval"]
    w("")
    w(f"One episode per evaluation task, {n} tasks, so one task is {100 / n:.1f} points and the binomial")
    w(f"standard error at a 20% success rate is {100 * (0.2 * 0.8 / n) ** 0.5:.1f} points. Differences smaller")
    w("than roughly twice that are not resolvable with this eval set and a single seed.\n")

    # ---------------- training-side numbers
    w("## Training-side behaviour\n")
    w("| Method | train success | iters with a gradient | mean \\|A\\| | episode steps | invalid actions |")
    w("|---|---|---|---|---|---|")
    for r in runs:
        t = r["train"]
        if not t:
            continue
        used = sum(1 for x in t if not x.get("skipped"))
        w(
            f"| {r['label']} | {fmt(np.mean([x['train_success'] for x in t]))} | "
            f"{used}/{len(t)} | {np.mean([x['adv_abs'] for x in t]):.3f} | "
            f"{np.mean([x['ep_steps'] for x in t]):.1f} | {fmt(np.mean([x['invalid'] for x in t]))} |"
        )
    w("")

    # ---------------- GACA diagnostics, mirroring the paper's Table 7
    gaca = [r for r in runs if r["base"] in ("gaca", "gigpo", "fixed")]
    if gaca:
        w("## Credit-assignment diagnostics\n")
        w("Compare with Table 7 of the paper (its ALFWorld column in brackets).\n")
        w("| Method | c_t mean/std [1.00/0.31] | lambda_t mean/std [0.50/0.08] | clamped [<0.01] | step coverage [0.86] |")
        w("|---|---|---|---|---|")
        for r in gaca:
            t = r["train"]
            if not t:
                continue
            g = lambda k: np.mean([x[k] for x in t])
            w(
                f"| {r['label']} | {g('c_mean'):.2f} / {g('c_std'):.2f} | "
                f"{g('lam_mean'):.2f} / {g('lam_std'):.2f} | {g('clamped'):.3f} | {g('step_coverage'):.2f} |"
            )
        w("")

    # ---------------- cost
    w("## Cost per iteration (seconds)\n")
    w("| Method | rollout | update | total |")
    w("|---|---|---|---|")
    for r in runs:
        t = r["train"]
        if not t:
            continue
        ro = np.mean([x["t_rollout"] for x in t])
        up = np.mean([x["t_update"] for x in t])
        w(f"| {r['label']} | {ro:.1f} | {up:.1f} | {ro + up:.1f} |")
    w("")

    # ---------------- attempt 1: the reward-shaping failure
    a1 = os.path.join(ROOT, "runs", "4060", "attempt1_penalty0.1", "grpo.jsonl")
    if os.path.exists(a1):
        r1 = load(a1)
        if r1 and r1["train"]:
            t1, e1 = r1["train"], r1["evals"]
            cur = next((r for r in runs if r["method"] == "grpo"), None)
            w("## Why the invalid-action penalty was removed\n")
            w("A first attempt kept the paper's 0.1 penalty for an inadmissible action. It did not")
            w("plateau, it degraded. At a ~5% success rate almost no group of 8 rollouts contains a")
            w("win, so the group-relative advantage was set almost entirely by which rollout made")
            w("fewer invalid actions. The policy optimized that nuisance signal and solved fewer")
            w("tasks. Under index-based action selection an inadmissible action is nearly")
            w("impossible to emit in the first place, so the penalty the paper needs for free-form")
            w("text actions became the only thing the gradient could see.\n")
            w("| | attempt 1 (penalty 0.1) | attempt 2 (penalty 0.0) |")
            w("|---|---|---|")
            blk = lambda t, k, lo, hi: np.mean([x[k] for x in t[lo:hi]])
            n1 = len(t1)
            w(f"| best held-out success | {fmt(max(e['success'] for e in e1))} | "
              f"{fmt(max(e['success'] for e in cur['evals'])) if cur else '--'} |")
            w(f"| mean abs advantage, first -> last third | {blk(t1,'adv_abs',0,n1//3):.2f} -> {blk(t1,'adv_abs',-n1//3,None):.2f} | "
              + (f"{blk(cur['train'],'adv_abs',0,len(cur['train'])//3):.2f} -> {blk(cur['train'],'adv_abs',-len(cur['train'])//3,None):.2f} |" if cur and len(cur['train'])>=3 else "-- |"))
            w(f"| action NLL, first -> last third | {blk(t1,'nll_mean',0,n1//3):.2f} -> {blk(t1,'nll_mean',-n1//3,None):.2f} | "
              + (f"{blk(cur['train'],'nll_mean',0,len(cur['train'])//3):.2f} -> {blk(cur['train'],'nll_mean',-len(cur['train'])//3,None):.2f} |" if cur and len(cur['train'])>=3 else "-- |"))
            w(f"| invalid-action rate, first -> last third | {100*blk(t1,'invalid',0,n1//3):.1f}% -> {100*blk(t1,'invalid',-n1//3,None):.1f}% | "
              + (f"{100*blk(cur['train'],'invalid',0,len(cur['train'])//3):.1f}% -> {100*blk(cur['train'],'invalid',-len(cur['train'])//3,None):.1f}% |" if cur and len(cur['train'])>=3 else "-- |"))
            w(f"| gradient norm (mean) | {np.mean([x['grad_norm'] for x in t1]):.1f} | "
              + (f"{np.mean([x['grad_norm'] for x in cur['train']]):.1f} |" if cur else "-- |"))
            w("")
            w(f"Attempt 1 ran {n1} GRPO iterations and is kept in `runs/attempt1_penalty0.1/`.\n")

    # ---------------- paired per-task comparison
    det, lab = {}, {}
    for r in runs:
        q = os.path.join(ROOT, "runs", "4060", f"{r['name']}_paired.eval.json")
        if os.path.exists(q):
            d = json.load(open(q))
            if d.get("per_task"):
                det[r["name"]] = [bool(x["won"]) for x in d["per_task"]]
                lab[r["name"]] = r["label"]
    if len(det) >= 2:
        w("## Paired comparison on the held-out tasks\n")
        w("Each method's best checkpoint re-evaluated greedily on the *same* 64 tasks in the")
        w("same order, so the methods can be compared task by task. McNemar's exact test uses")
        w("only the tasks the two methods disagree on, which is far more sensitive at n=64")
        w("than comparing two independent rates.\n")
        w("| A vs B | A only | B only | both | neither | McNemar p (2-sided) |")
        w("|---|---|---|---|---|---|")
        ms = list(det)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = det[ms[i]], det[ms[j]]
                n = min(len(a), len(b))
                a, b = a[:n], b[:n]
                only_a = sum(1 for x, y in zip(a, b) if x and not y)
                only_b = sum(1 for x, y in zip(a, b) if y and not x)
                both = sum(1 for x, y in zip(a, b) if x and y)
                neither = n - only_a - only_b - both
                w(f"| {lab[ms[i]]} vs {lab[ms[j]]} | {only_a} | {only_b} | {both} | {neither} | "
                  f"{mcnemar(only_a, only_b):.3f} |")
        w("")
        if len(ms) > 2:
            ps, ovl = [], []
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = det[ms[i]], det[ms[j]]
                    n = min(len(a), len(b))
                    ps.append(mcnemar(sum(1 for x, y in zip(a[:n], b[:n]) if x and not y),
                                      sum(1 for x, y in zip(a[:n], b[:n]) if y and not x)))
                    ovl.append(sum(1 for x, y in zip(a[:n], b[:n]) if x and y))
            w(f"No pair is distinguishable: the smallest p over all {len(ps)} comparisons is")
            w(f"{min(ps):.3f}, and the number of tasks any two runs both solve never exceeds")
            w(f"{max(ovl)}. Whatever these runs differ in, it is not something this evaluation can")
            w("resolve.\n")

        # The overlap is the informative number, not the rate gap. Compare the
        # two methods at a matched seed, not two seeds of the same method.
        pair = None
        for r0 in runs:
            if r0["base"] != "grpo" or r0["norm"] != "mean_std" or r0["name"] not in det:
                continue
            for r1 in runs:
                if (r1["base"] == "gaca" and r1["norm"] == "mean_std"
                        and r1["seed"] == r0["seed"] and r1["name"] in det):
                    pair = (r0["name"], r1["name"])
                    break
            if pair:
                break
        if pair is None and len(ms) >= 2:
            pair = (ms[0], ms[1])
        if pair:
            ms = list(pair)
            a, b = det[ms[0]], det[ms[1]]
            n = min(len(a), len(b))
            na, nb = sum(a[:n]), sum(b[:n])
            both = sum(1 for x, y in zip(a[:n], b[:n]) if x and y)
            exp = na * nb / n
            w("### What the overlap says\n")
            w(f"{lab[ms[0]]} solves {na} tasks and {lab[ms[1]]} solves {nb}, but they agree on only")
            w(f"**{both}**. If the two sets of outcomes were statistically independent the expected")
            w(f"overlap would be {na}x{nb}/{n} = {exp:.1f}, which is what we observe.\n")
            w("That is the result. The two trained policies are not converging on a stable set of")
            w("solvable tasks and then differing at the margin; they succeed on almost disjoint")
            w("task sets at nearly the same rate. Which tasks a run ends up solving is dominated by")
            w("run-to-run variance, not by the credit-assignment scheme. A 1-task gap between the")
            w("best checkpoints carries no information against that background.\n")

    # ---------------- normalization 2x2 (paper Remark 3)
    best_of = {}
    for r in runs:
        if r["done"]:
            best_of[(r["base"], r["norm"], r["seed"])] = max(e["success"] for e in r["evals"])
    norms = [n for n in ("mean_std", "mean") if any(k[1] == n for k in best_of)]
    if len(norms) >= 2 and all(("grpo", n, 0) in best_of and ("gaca", n, 0) in best_of for n in norms):
        NORMLBL = {"mean_std": "mean-and-std-norm", "mean": "mean-norm"}
        w("## Does normalization change GACA's margin? (paper Remark 3)\n")
        w("Remark 3 argues that mean-and-std normalization rescales each estimator by its own")
        w("group standard deviation and so \"removes exactly that information, driving lambda*_z")
        w("toward 1/2 uniformly in z\", which by Proposition 2 shrinks the gain from adapting. The")
        w("paper observes larger gains on its mean-norm benchmark than on its mean-and-std-norm")
        w("one, but says the two benchmarks \"differ in far more than their normalization, so this")
        w("is a consistency check and not a test\". Holding the benchmark fixed and varying only")
        w("the normalization is that test.\n")
        w("| Normalization | GRPO | GACA | GACA - GRPO |")
        w("|---|---|---|---|")
        margins = {}
        for n in norms:
            g, a_ = best_of[("grpo", n, 0)], best_of[("gaca", n, 0)]
            margins[n] = a_ - g
            w(f"| {NORMLBL[n]} | {fmt(g)} | {fmt(a_)} | {100 * (a_ - g):+.1f} |")
        w("")
        if len(margins) == 2:
            d = margins["mean"] - margins["mean_std"]
            w(f"Remark 3 predicts the mean-norm margin should be the larger of the two. It is")
            w(f"{'larger' if d > 0 else 'smaller'} by {abs(100 * d):.1f} points "
              f"({100 * margins['mean']:+.1f} against {100 * margins['mean_std']:+.1f}).\n")
            if margins["mean"] * margins["mean_std"] < 0:
                w("The more telling detail is that **the margin changes sign**: GACA is ahead by one")
                w("task under one normalization and behind by one task under the other. A quantity")
                w("that flips sign when an arbitrary normalization choice is changed, by one task")
                w("each way, is measuring noise rather than an effect of the estimator. That is the")
                w("same conclusion the paired tests reach, arrived at independently.\n")
            w("**A confound that has to be stated.** Dropping the standard deviation also changes the")
            w("*scale* of the advantage, not just its per-state structure. Measured over training:")
            w("mean absolute advantage 0.061 under mean-norm against 0.234 under mean-and-std-norm,")
            w("and mean gradient norm 0.88 against 3.07 — roughly a fourfold smaller effective step.")
            w("Both arms of the mean-norm column take that smaller step, so the comparison of")
            w("*margins* is still matched, but a smaller mean-norm margin could reflect less")
            w("effective training rather than less to gain from adapting. Separating the two would")
            w("need the mean-norm arms re-run at a compensating learning rate, which was beyond the")
            w("compute available here.\n")
            w("This should not be over-read. Each margin is a difference of two single runs on 64")
            w("tasks, so one task is 1.6 points, and the paired results below show the methods'")
            w("success sets overlapping no more than chance predicts. A difference of a few points")
            w("between two such margins is well inside that noise.\n")

    # ---------------- seed variance: the noise floor for every other number
    byms = {}
    for r in runs:
        if r["done"]:
            byms[(r["base"], r["norm"], r["seed"])] = max(e["success"] for e in r["evals"])
    methods = sorted({(m, nm) for m, nm, _ in byms})
    seeds = sorted({sd for _, _, sd in byms})
    if len(seeds) >= 2 and any(all((m, nm, sd) in byms for sd in seeds) for m, nm in methods):
        w("## Seed variance: the noise floor\n")
        w("Each method re-run with a different sampling seed. Task selection and ordering go")
        w("through `random.Random(1234)` independently of `--seed`, so these runs see the same")
        w("tasks in the same order and differ only in sampling randomness. The spread between two")
        w("seeds of the *same* method is what any difference *between* methods has to beat.\n")
        head = " | ".join(f"seed {sd}" for sd in seeds)
        w(f"| Method | {head} | spread |")
        w("|---" * (len(seeds) + 2) + "|")
        spreads = []
        for m, nm in methods:
            if not all((m, nm, sd) in byms for sd in seeds):
                continue
            vals = [byms[(m, nm, sd)] for sd in seeds]
            sp = max(vals) - min(vals)
            spreads.append(sp)
            cells = " | ".join(fmt(v) for v in vals)
            w(f"| {LABELS.get(m, m)} | {cells} | {100 * sp:.1f} |")
        w("")
        gaps = []
        for sd in seeds:
            for nm in {nm for _, nm in methods}:
                if ("grpo", nm, sd) in byms and ("gaca", nm, sd) in byms:
                    gaps.append(byms[("gaca", nm, sd)] - byms[("grpo", nm, sd)])
        if spreads and gaps:
            ms_ = 100 * float(np.mean(spreads))
            mg = 100 * float(np.mean([abs(g) for g in gaps]))
            signs = {int(np.sign(g)) for g in gaps if g != 0}
            w(f"Mean within-method spread between seeds: **{ms_:.1f} points**. Mean absolute")
            w(f"GACA-minus-GRPO difference at a matched seed: **{mg:.1f} points** "
              f"({', '.join(f'{100 * g:+.1f}' for g in gaps)}).\n")
            if mg <= ms_:
                w("The method effect is no larger than the effect of changing the random seed, so")
                w("these runs cannot attribute any of it to the estimator.\n")
            elif len(signs) > 1:
                w("The difference does not even keep its sign across seeds, so it is not a method")
                w("effect.\n")
            else:
                w("The difference exceeds the seed spread and keeps its sign, which is suggestive")
                w("but still rests on two seeds; the paired tests below are the stronger evidence.\n")

    # ---------------- learning curve
    try:
        fig = plot(runs)
        w("## Learning curves\n")
        w(f"![Held-out success rate over training](../{fig})\n")
    except Exception as e:  # pragma: no cover
        w(f"(figure not produced: {e})\n")

    w("## Verdict\n")
    w("**No detectable difference between GACA and GRPO at this training budget.**")
    w("GACA's learning curve sits consistently at or above GRPO's -- it led at 3 of 5 matched")
    w("checkpoints, tied at 1, trailed at 1, and its mean over the post-baseline checkpoints is")
    w("10.9% against 8.4% -- but every one of those gaps is inside the noise, the best-checkpoint")
    w("difference is a single task, and the paired test returns p = 1.000.\n")
    w("What this experiment does establish:\n")
    w("1. The implementation is faithful where it can be checked. Anchor-state step coverage is")
    w("   0.82-0.84 against the paper's 0.86, and every equation is pinned by unit tests.")
    w("2. GACA's overhead really is negligible: 111.7 s/iteration against GRPO's 118.1 s, with")
    w("   GACA performing *more* optimizer steps (56 vs 50 non-skipped iterations).")
    w("3. Reward shaping dominates credit assignment at low success rates. Removing the")
    w("   invalid-action penalty moved the best result from 4.7% to 14.1%; switching estimator")
    w("   moved it by 1.5 points. That ordering is the most useful thing here.\n")
    w("What it cannot establish: whether granularity-adaptive credit assignment helps. 75")
    w("iterations on 150 task-groups at a ~10% success rate, with one seed and 64 evaluation")
    w("tasks, is one to two orders of magnitude short of the paper's budget. The honest")
    w("conclusion is a null result with wide error bars, not a refutation.\n")
    w("## What this does and does not show\n")
    w("**Does.** The GACA estimator is implemented to the paper's equations and its mechanism")
    w("operates in roughly the intended range: the anchor-state grouping gives a genuine")
    w("within-group contrast on ~82% of steps against the paper's 86%, the criticality score has")
    w("mean one by construction, and the realized mixing weight averages lambda_0, so credit is")
    w("redistributed across steps rather than scaled up on average. Both methods learn: held-out")
    w("success roughly quadruples from the 3.1% zero-shot baseline.\n")
    w("**Does not.** It does not reproduce the paper's *ordering*. GACA is not shown to beat GRPO")
    w("here, and with one seed, 75 iterations and 64 evaluation tasks this setup could not have")
    w("detected the paper's ALFWorld margin reliably even if it were present. The paper trains 150")
    w("epochs over three seeds at a success rate an order of magnitude higher, where a group of 8")
    w("rollouts usually contains both a success and a failure; here a third of iterations carry no")
    w("gradient at all because every rollout failed. Credit assignment can only matter once there")
    w("is credit to assign, and that is the regime this hardware could not reach.\n")
    w("**Two differences from the paper worth naming.** The criticality score is three times more")
    w("dispersed than the paper reports (c_t std ~1.0 against 0.31), because index-selected actions")
    w("are 1-3 tokens long and their mean per-token NLL is not smoothed by averaging over a long")
    w("action; the gate therefore spends more time near its clamps (~5% of steps against <1%). And")
    w("the paper's invalid-action penalty had to be removed, for the reason given above.\n")
    w("## Raw logs\n")
    for r in runs:
        w(f"- `{os.path.relpath(r['path'], ROOT)}`")
    w("")

    out[0] = ("# Results (attempts 1-3, RTX 4060): GACA vs GRPO on ALFWorld — index actions, LoRA, two task types\n\n"
              "> Superseded setup. Kept as history; the current experiments are in `RESULTS.md`.")
    text = "\n".join(out)
    open(os.path.join(ROOT, "docs", "RESULTS_4060.md"), "w").write(text)
    print(text)


def mcnemar(b, c):
    """Two-sided exact McNemar test on the discordant pairs."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def plot(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = "#fcfcfb"
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e3e2dd"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    xmax = 0
    for r in runs:
        ev = sorted(r["evals"], key=lambda e: e["iter"])
        x = [e["iter"] for e in ev]
        y = [100 * e["success"] for e in ev]
        xmax = max(xmax, max(x))
        c = COLORS.get(r["base"], ink2)
        # Colour carries the method; dash carries the variant (normalization
        # first, then seed), so a variant never reads as a third method.
        style = "-" if r["norm"] == "mean_std" else "--"
        if r["seed"] != 0:
            style = ":"
        ax.plot(x, y, color=c, lw=2.0, marker="o", ms=5, mfc=surface, mew=2.0, zorder=3,
                ls=style, label=r["label"])
        # Direct label at the line end: identity is never colour-alone, and it
        # is the relief the contrast WARN on the aqua slot requires.
        if len(runs) > 2 and r["norm"] != "mean_std":
            continue
        ax.annotate(
            f"  {r['label'].split(' (')[0]} {y[-1]:.0f}%",
            (x[-1], y[-1]), color=ink, fontsize=9, va="center", ha="left",
            annotation_clip=False, zorder=4,
        )

    ax.set_xlabel("Training iteration", color=ink2, fontsize=10)
    ax.set_ylabel("Held-out success rate (%)", color=ink2, fontsize=10)
    ax.set_title("Credit assignment on ALFWorld (Qwen2.5-1.5B-Instruct + LoRA)", color=ink, fontsize=11, pad=12)
    ax.grid(True, axis="y", color=grid, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=9, length=0)
    ax.set_xlim(-1, xmax * 1.28)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=ink2)
    fig.tight_layout()
    path = os.path.join(ROOT, "runs", "4060", "learning_curves.png")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return os.path.relpath(path, ROOT)


# ===================================================================== verl-agent arms
# Experiments of attempt 4 (A100, verl-agent, the paper's reasoning format).
# `python src/report.py verl` rebuilds docs/RESULTS.md from runs/verl/*.

VERL_ARMS = [  # (dir, label, role, what differs from GACA/GRPO)
    ("grpo", "GRPO", "control", "episode-level advantage only (Eq. 2)"),
    ("gaca", "GACA", "control", "the paper's estimator: NLL-gated mix of step and episode advantage (Eq. 8)"),
    ("gaca_msize", "GACA + m_z", "treatment", "GACA, lambda odds scaled by 1 - 1/m_z (anchor-group size; Prop. 3)"),
    ("gaca_loo", "GACA-LOO", "treatment", "GACA, criticality = leave-one-out mean NLL of the other visits (App. B.4)"),
    ("gaca_probe", "GACA + probe (seed 0)", "treatment", "GACA, step advantage spread over tokens by the commitment probe"),
    ("gaca_probe_s1", "GACA + probe (seed 1)", "treatment", "as above, training seed 1, same 128 validation games"),
]
VERL_COLORS = {"grpo": "#2a78d6", "gaca": "#eb6834", "gaca_msize": "#1baf7a", "gaca_loo": "#eda100",
               "gaca_probe": "#e87ba4", "gaca_probe_s1": "#e87ba4"}  # validated categorical slots 1-5, fixed order


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
    log_lines = open(os.path.join(ROOT, "runs/verl/arms.log")).read() if os.path.exists(os.path.join(ROOT, "runs/verl/arms.log")) else ""
    for key, label, role, what in VERL_ARMS:
        base = os.path.join(ROOT, "runs/verl", key)
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
        ls = ":" if a["key"].endswith("_s1") else "-"
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
    ax.set_xlim(-1, 44)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=ink2)
    fig.tight_layout()
    path = os.path.join(ROOT, "runs", "verl", "learning_curves.png")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return os.path.relpath(path, ROOT)


def _j(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p)) if os.path.exists(p) else None


def verl_main():
    arms = verl_load()
    out = []
    w = out.append
    w("# Results: GACA vs GRPO, and new credit-assignment treatments, on ALFWorld")
    w("")
    w("Generated by `python src/report.py verl` from `runs/verl/` and `runs/tokens/`; do not edit by hand.")
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
    w("| arm | s / iter | probe s / iter | valid actions | response tokens | KL (last 5) | c_t std | lambda std | step coverage |")
    w("|---|---|---|---|---|---|---|---|---|")
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
          f"| {mean('response_length/mean'):.0f} | {kl5:.3f} | {gs('gaca/c_std', '.2f')} | {gs('gaca/lam_std', '.3f')} | {gs('gaca/step_coverage', '.2f')} |")
    w("")
    w("Paper Table 7 (ALFWorld): c_t std 0.31, lambda std 0.08, clamped < 0.01, step coverage 0.86.")
    w("")
    w("## Offline studies (no training)")
    w("")
    e = _j("runs/verl/entropy_check_iter1-5.json")
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
    hb, hh, hr = _j("runs/tokens/base.analysis.json"), _j("runs/tokens/heldout.analysis.json"), _j("runs/tokens/heldout.robust.json")
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
    w("- `runs/verl/gaca_random/`: GACA with the criticality score permuted within each trajectory (the paper's")
    w("  Table 2 ablation, which the paper already reports at 88.54 vs 96.88). Stopped at iteration 10 because it")
    w("  duplicates published work; its first-5-iteration step dump is the data of the entropy study above.")
    w("- `runs/verl/smoke/`, `probe_smoke/`, `save_smoke/`: integration tests (1-2 iterations, 16-game evaluations).")
    w("- `runs/a100/zs_*`: zero-shot prompt-format tests on the HF pipeline (two task types), see DECISIONS.md.")
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
    if len(sys.argv) > 1 and sys.argv[1] == "verl":
        verl_main()
    else:
        main()
