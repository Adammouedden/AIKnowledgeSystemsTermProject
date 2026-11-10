"""Figures and the attention analysis for the paper (docs/paper/).

    python src/paper_figures.py attention   # CPU: action->reasoning attention vs measured decision importance
    python src/paper_figures.py figs        # all figures -> docs/paper/figs/*.pdf

Run with the verl-agent venv (transformers, matplotlib). Data come from src/runs/verl/ (training arms)
and src/runs/tokens/ (flip-test oracle). Colours: the validated categorical palette used by
src/report.py; colour follows the method, variants of the probe keep its hue and change line style.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import advantages as ADV  # noqa: E402
import report as R  # noqa: E402

FIG = os.path.join(ROOT, "docs", "paper", "figs")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2dd", "#ffffff"
TOK = os.path.join(ROOT, "src", "runs", "tokens")


# ----------------------------------------------------------------- attention analysis (CPU)


def _segments(r):
    """Fine flip-test segments of one oracle record: bounds and measured importance shares."""
    L, cuts = r["L"], r["cuts"]
    F = [r["F"][str(c)] for c in cuts] + [0.0]
    b = cuts + [L]
    imp = np.array([max(F[k] - F[k + 1], 0.0) for k in range(len(cuts))])
    return b, imp


def attention(n=64, seed=0):
    """For n held-out oracle steps (base model, the same responses the flip test resampled):
    attention mass that the action tokens (from the <action> tag on) place on each reasoning segment,
    averaged over heads and over the last third of the layers. Compared against the measured decision
    importance and against the probe weights, all on the same segments. Thought Anchors (Bogdan et al.)
    use attention between sentences as their white-box importance signal; this asks whether it agrees
    with the counterfactual one here."""
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(8)
    recs = [json.loads(l) for l in open(os.path.join(TOK, "heldout.jsonl"))]
    rng = np.random.default_rng(seed)
    pick = [i for i in rng.permutation(len(recs)) if recs[i]["act_start"] and recs[i]["act_start"] > 32][:n]
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32,
                                             attn_implementation="eager").eval()
    nl = m.config.num_hidden_layers
    layers = list(range(nl - nl // 3, nl))
    out = []
    for k, i in enumerate(pick):
        r = recs[i]
        P, L, a0 = len(r["prompt"]), r["L"], r["act_start"]
        ids = torch.tensor([r["prompt"] + r["resp"][:L]])
        with torch.no_grad():
            att = m(ids, output_attentions=True).attentions  # per layer: (1, heads, T, T)
        A = torch.stack([att[l][0].mean(0) for l in layers]).mean(0)  # (T, T)
        rows = A[P + a0 : P + L, P : P + L].mean(0).numpy()  # action queries -> response keys
        b, imp = _segments(r)
        seg_att = np.array([rows[b[s] : b[s + 1]].sum() for s in range(len(b) - 1)])
        reason = [s for s in range(len(b) - 1) if b[s + 1] <= a0]
        qv = [float(np.exp(r["q"][str(c)])) for c in r["cuts"] if c <= a0 and str(c) in r["q"]]
        sub = [c for c in r["cuts"] if c <= a0 and str(c) in r["q"]]
        w = ADV.probe_token_weights(L, sub, qv)
        seg_w = np.array([w[b[s] : b[s + 1]].sum() for s in range(len(b) - 1)])
        seg_len = np.array([b[s + 1] - b[s] for s in range(len(b) - 1)], float)
        out.append({"i": int(i), "att": seg_att.tolist(), "imp": imp.tolist(), "probe": seg_w.tolist(),
                    "len": seg_len.tolist(), "reason": reason, "att_tokens": rows.tolist(), "a0": a0, "L": L})
        print(f"{k + 1}/{len(pick)}", flush=True)
    json.dump(out, open(os.path.join(TOK, "attention.json"), "w"))
    summarize_attention()


def summarize_attention():
    """Credit on decision importance (reasoning segments only), relative to uniform, for attention
    and probe weights; plus per-step Spearman of attention vs importance."""
    d = json.load(open(os.path.join(TOK, "attention.json")))
    ratios = {"attention": [], "probe": [], "uniform": []}
    rho = []
    for e in d:
        rs = e["reason"]
        if len(rs) < 3:
            continue
        imp = np.array(e["imp"])[rs]
        if imp.sum() <= 0:
            continue
        imp = imp / imp.sum()
        for k, v in (("attention", e["att"]), ("probe", e["probe"]), ("uniform", e["len"])):
            v = np.array(v)[rs]
            ratios[k].append(float((v / v.sum() * imp).sum()))
        a, i = np.array(e["att"])[rs] / np.array(e["len"])[rs], imp / np.array(e["len"])[rs]
        ra, ri = np.argsort(np.argsort(a)), np.argsort(np.argsort(i))
        if np.std(ra) > 0 and np.std(ri) > 0:
            rho.append(float(np.corrcoef(ra, ri)[0, 1]))
    u = np.mean(ratios["uniform"])
    rng = np.random.default_rng(0)
    rep = {"steps": len(ratios["uniform"])}
    for k in ("attention", "probe"):
        x, y = np.array(ratios[k]), np.array(ratios["uniform"])
        boots = [x[ix].mean() / y[ix].mean() for ix in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
        rep[k] = {"ratio": float(x.mean() / u), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    rep["spearman_attention_vs_importance_density"] = float(np.mean(rho)) if rho else None
    json.dump(rep, open(os.path.join(TOK, "attention.summary.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


# ----------------------------------------------------------------- figures


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7, length=0)
    ax.grid(True, axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "legend.fontsize": 6.5, "pdf.fonttype": 42})
    return plt


ARM_ORDER = ["grpo", "gaca", "gaca_msize", "gaca_loo", "commit_s1", "commit_c05", "commit_pos"]


def fig_curves(arms):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    fig.patch.set_facecolor(SURF)
    _style(ax)
    base = arms[0]["curve"][0][1]
    ax.axhline(100 * base / 128, color=INK2, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax.annotate("base model", (17, 100 * base / 128 - 0.3), color=INK2, fontsize=6, va="top", ha="center")
    ends = []
    for a in arms:
        if len(a["curve"]) < 2:
            continue
        x = [c[0] for c in a["curve"]]
        y = [100 * c[1] / 128 for c in a["curve"]]
        ls = R.VERL_STYLE.get(a["key"], "-")
        ax.plot(x, y, color=R.VERL_COLORS[a["key"]], lw=1.4, ls=ls, marker="o", ms=3, mfc=SURF, mew=1.2,
                zorder=3, label=a["label"])
        ends.append([x[-1], y[-1], a["label"]])
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 1.25:
            ends[i][1] = ends[i - 1][1] + 1.25
    xl = max(e[0] for e in ends) + 0.8
    for x, y, lab in ends:
        ax.annotate(lab, (xl, y), color=INK, fontsize=5.6, va="center", ha="left", annotation_clip=False)
    ax.set_xlim(-1, 56)
    ax.set_xticks([0, 10, 20, 30, 34])
    # compact key for the line styles (identity is also carried by the direct labels)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=INK2, lw=1.2, ls=ls) for ls in ("-", "--", "-.")]
    ax.legend(keys, ["method", "probe, clip [0.5, 2]", "probe, positive $A_t$ only"], frameon=False,
              loc="upper left", fontsize=5.5, handlelength=2.4)
    ax.set_ylim(0, None)
    ax.set_xlabel("training iteration", color=INK2)
    ax.set_ylabel("held-out success (%)", color=INK2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "curves.pdf"))
    plt.close(fig)


def fig_oracle():
    """Credit on decision importance vs uniform, by weighting scheme: base vs trained policy."""
    plt = _plt()
    hb = json.load(open(os.path.join(TOK, "heldout.analysis.json")))
    ht = json.load(open(os.path.join(TOK, "trained_s1.analysis.json")))
    att = json.load(open(os.path.join(TOK, "attention.summary.json"))) if os.path.exists(os.path.join(TOK, "attention.summary.json")) else None
    keys = [("uniform", "uniform\n(GACA)"), ("entropy", "token\nentropy"), ("nll", "sampled\nNLL"),
            ("action_span", "2x action\ntokens"), ("probe", "commitment\nprobe")]
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    fig.patch.set_facecolor(SURF)
    _style(ax)
    x = np.arange(len(keys))
    wdt = 0.36
    vb = [hb[k]["credit_on_importance_vs_uniform"] for k, _ in keys]
    vt = [ht[k]["credit_on_importance_vs_uniform"] for k, _ in keys]
    ax.bar(x - wdt / 2 - 0.01, vb, wdt, color="#2a78d6", label="base policy", zorder=3)
    ax.bar(x + wdt / 2 + 0.01, vt, wdt, color="#eb6834", label="trained (COMMIT)", zorder=3)
    for xi, a, b in zip(x, vb, vt):
        ax.annotate(f"{a:.2f}", (xi - wdt / 2, a), ha="center", va="bottom", fontsize=5.5, color=INK)
        ax.annotate(f"{b:.2f}", (xi + wdt / 2, b), ha="center", va="bottom", fontsize=5.5, color=INK)
    ax.axhline(1.0, color=INK2, lw=0.7, ls=(0, (2, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in keys], fontsize=6)
    ax.set_ylabel("credit on decisive\nsegments / uniform", color=INK2, fontsize=6.5, labelpad=2)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(vb + vt) * 1.18)
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(FIG, "oracle.pdf"))
    plt.close(fig)


def fig_deciles():
    """Where the decision is made vs where the entropy is, by response decile (held-out oracle)."""
    plt = _plt()
    recs = [json.loads(l) for l in open(os.path.join(TOK, "heldout.jsonl"))]
    dec, ent = np.zeros(10), np.zeros(10)
    for r in recs:
        b, imp = _segments(r)
        if imp.sum() <= 0:
            continue
        imp = imp / imp.sum()
        e = np.array(r["ent"][: r["L"]])
        for k in range(len(b) - 1):
            d = min(9, int((b[k] + b[k + 1]) / 2 / r["L"] * 10))
            dec[d] += imp[k]
            ent[d] += e[b[k] : b[k + 1]].sum() / max(e.sum(), 1e-9)
    dec, ent = dec / dec.sum(), ent / ent.sum()
    fig, ax = plt.subplots(figsize=(3.35, 1.7))
    fig.patch.set_facecolor(SURF)
    _style(ax)
    x = np.arange(10)
    ax.bar(x - 0.19, 100 * dec, 0.36, color="#2a78d6", label="decision importance (flip test)", zorder=3)
    ax.bar(x + 0.19, 100 * ent, 0.36, color="#1baf7a", label="token entropy", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{10 * i}-{10 * i + 10}" for i in x], fontsize=5.5)
    ax.set_xlabel("position in the response (% of tokens)", color=INK2)
    ax.set_ylabel("share (%)", color=INK2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "deciles.pdf"))
    plt.close(fig)


def fig_example():
    """One held-out step: per-token credit under uniform / probe weights, and measured importance,
    plus the action tokens' attention to the reasoning."""
    from transformers import AutoTokenizer

    plt = _plt()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    d = json.load(open(os.path.join(TOK, "attention.json")))
    recs = [json.loads(l) for l in open(os.path.join(TOK, "heldout.jsonl"))]
    # the step whose importance is most concentrated in the reasoning, among moderately short ones
    pool = [e for e in d if 50 <= e["L"] <= 110 and e["reason"]] or [e for e in d if e["reason"]]
    best = max(pool, key=lambda e: max(np.array(e["imp"])[e["reason"]]))
    r = recs[best["i"]]
    L, a0 = r["L"], r["a0"] if "a0" in r else best["a0"]
    b, imp = _segments(r)
    sub = [c for c in r["cuts"] if c <= best["a0"] and str(c) in r["q"]]
    w = ADV.probe_token_weights(L, sub, [float(np.exp(r["q"][str(c)])) for c in sub])
    imp_tok = np.zeros(L)
    for s in range(len(b) - 1):
        imp_tok[b[s] : b[s + 1]] = imp[s] / max(b[s + 1] - b[s], 1)
    att = np.zeros(L)
    att[: len(best["att_tokens"])] = best["att_tokens"]
    att[best["a0"] :] = np.nan  # the action attends to itself trivially; show reasoning only
    pieces = [tok.decode([t]) for t in r["resp"][:L]]
    rows = [("measured decision importance", imp_tok / max(imp_tok.max(), 1e-9), "Blues"),
            ("probe token weight", (w - w.min()) / max(w.max() - w.min(), 1e-9), "RdPu"),
            ("action->token attention", att / np.nanmax(att), "Greens")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(6.9, 1.9), sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, (name, v, cmap) in zip(axes, rows):
        ax.imshow(np.nan_to_num(v[None, :], nan=0.0), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks([0])
        ax.set_yticklabels([name], fontsize=6.5, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axvline(best["a0"] - 0.5, color=INK, lw=0.6)
    step = max(1, L // 18)
    axes[-1].set_xticks(range(0, L, step))
    axes[-1].set_xticklabels([pieces[i].strip()[:9] or "_" for i in range(0, L, step)], fontsize=5, rotation=50, ha="right")
    axes[0].annotate("<action> ->", (best["a0"], -0.9), fontsize=5.5, color=INK, ha="left", annotation_clip=False)
    fig.tight_layout(pad=0.2, h_pad=0.1)
    fig.savefig(os.path.join(FIG, "example.pdf"))
    plt.close(fig)
    text = "".join(pieces)
    json.dump({"i": best["i"], "action": r["action"], "text": text}, open(os.path.join(FIG, "example.json"), "w"), indent=1)


def fig_robust():
    plt = _plt()
    hr = json.load(open(os.path.join(TOK, "heldout.robust.json")))
    items = [("spacing x1", hr["spacing"]["x1"]), ("spacing x3", hr["spacing"]["x3"]),
             ("clip [0.5, 2]", hr["clip"]["0.5-2"]), ("clip [0.25, 4] (used)", hr["clip"]["0.25-4"]),
             ("clip none", hr["clip"]["none"])]
    tnames = {"pick_and_place_simple": "pick & place", "look_at_obj_in_light": "examine in light",
              "pick_clean_then_place_in_recep": "clean & place", "pick_heat_then_place_in_recep": "heat & place",
              "pick_cool_then_place_in_recep": "cool & place", "pick_two_obj_and_place": "pick two & place"}
    items += [(tnames.get(k, k), v) for k, v in hr["task_type"].items()]
    items += [("decision settled", hr["openness_F0"]["settled"]), ("decision open", hr["openness_F0"]["open"]),
              ("game half A", hr["split_half"]["half0"]), ("game half B", hr["split_half"]["half1"])]
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    fig.patch.set_facecolor(SURF)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.grid(False, axis="y")
    y = np.arange(len(items))[::-1]
    for yi, (lab, v) in zip(y, items):
        ax.plot(v["ci"], [yi, yi], color="#e87ba4", lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot([v["ratio"]], [yi], "o", color="#e87ba4", ms=3.5, mfc=SURF, mew=1.2, zorder=4)
    ax.axvline(1.0, color=INK2, lw=0.7, ls=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([l for l, _ in items], fontsize=6)
    ax.set_xlabel("credit on decisive segments / uniform", color=INK2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "robust.pdf"))
    plt.close(fig)


def figs():
    os.makedirs(FIG, exist_ok=True)
    arms = [a for a in R.verl_load() if a["curve"] and a["done"] and a["key"] in ARM_ORDER]
    arms.sort(key=lambda a: ARM_ORDER.index(a["key"]) if a["key"] in ARM_ORDER else 99)
    fig_curves(arms)
    fig_oracle()
    fig_deciles()
    fig_robust()
    if os.path.exists(os.path.join(TOK, "attention.json")):
        fig_example()
    print("figures ->", FIG)


# ----------------------------------------------------------------- tables


def tables():
    """docs/paper/tables/results.tex from the arms' logs (final checkpoints, paired McNemar)."""
    os.makedirs(os.path.join(ROOT, "docs", "paper", "tables"), exist_ok=True)
    arms = {a["key"]: a for a in R.verl_load() if a["curve"] and a["done"] and a["key"] in ARM_ORDER}
    order = [k for k in ARM_ORDER if k in arms]
    fin = {k: arms[k]["curve"][-1][2] for k in order}
    base = arms["grpo"]["curve"][0][2]

    def pair(X, Y):
        b = sum(X[e]["won"] and not Y[e]["won"] for e in X)
        c = sum(Y[e]["won"] and not X[e]["won"] for e in X)
        return b, c, R.mcnemar(b, c)

    def cell(X, Y):
        b, c, p = pair(X, Y)
        ps = "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
        bold = p < 0.05
        txt = f"{b}/{c} ({ps})"
        return f"\\textbf{{{txt}}}" if bold else txt

    names = {"grpo": "GRPO~\\cite{shao2024deepseekmath}", "gaca": "GACA~\\cite{liang2026gaca}",
             "gaca_msize": "GACA + $m_z$ gate", "gaca_loo": "GACA-LOO (App.~B.4 of~\\cite{liang2026gaca})",
             "commit_s1": "\\textbf{COMMIT (ours)}", "commit_c05": "\\quad clip $[0.5, 2]$",
             "commit_pos": "\\quad positive $A_t$ only", "commit_ref": "\\quad frozen-reference scorer"}
    lines = ["\\begin{tabular}{@{}lrrcc@{}}", "\\toprule",
             "Method & Final & Best & vs.\\ GRPO & vs.\\ GACA \\\\", "\\midrule",
             f"Base model (no training) & {100 * sum(e['won'] for e in base.values()) / 128:.1f} & -- & -- & -- \\\\"]
    for k in order:
        w = sum(e["won"] for e in fin[k].values())
        best = max(c[1] for c in arms[k]["curve"])
        vg = "--" if k == "grpo" else cell(fin["grpo"], fin[k])
        va = "--" if k == "gaca" else cell(fin["gaca"], fin[k])
        if k == "gaca_msize":
            lines.append("\\midrule")
        lines.append(f"{names[k]} & {100 * w / 128:.1f} & {100 * best / 128:.1f} & {vg} & {va} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(ROOT, "docs", "paper", "tables", "results.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"attention": attention, "figs": figs, "summary": summarize_attention, "tables": tables}[sys.argv[1]]()
