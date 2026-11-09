"""Offline measurement studies (no training).

    python src/measure.py entropy src/runs/verl/<arm>/steps.jsonl      # GACA's NLL vs exact token entropy
    python src/measure.py tokens --out src/runs/tokens/<name> ...       # token flip-test oracle (GPU, vLLM)
    python src/measure.py tokens_analyze src/runs/tokens/<name>.jsonl   # score token-weighting schemes
    python src/measure.py tokens_robust src/runs/tokens/<name>.jsonl    # robustness of the probe weights

Run with the verl-agent venv (vLLM, transformers, alfworld); see docs/HANDOFF.md.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advantages as ADV  # noqa: E402

# ----------------------------------------------------------------- NLL vs exact entropy


def entropy_check(path, out=None):
    """How much do GACA's sampled-token NLL and the exact mean token entropy
    disagree per step, and what would the disagreement do to lambda_t?

    Input: the per-step dump of a verl-agent run (GACA_STEP_DUMP). Both scores
    come from the same forward pass on the same responses, so every difference
    is the single-sample noise of the NLL, not a different policy."""
    rows = [json.loads(l) for l in open(path)]
    nll = np.array([r["nll"] for r in rows])
    ent = np.array([r["ent"] for r in rows])
    rep = {"rows": len(rows), "iters": len({r["it"] for r in rows})}

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    rep["pearson"] = float(np.corrcoef(nll, ent)[0, 1])
    rep["spearman"] = spearman(nll, ent)
    rep["mean_nll"], rep["mean_ent"] = float(nll.mean()), float(ent.mean())

    # GACA normalizes the score within each trajectory (Eq. 5) and feeds it to Eq. 7.
    by_traj = {}
    for i, r in enumerate(rows):
        by_traj.setdefault((r["it"], r["traj"]), []).append(i)
    lam_n, lam_e, within_rho, topq_agree = np.zeros(len(rows)), np.zeros(len(rows)), [], []
    for idx in by_traj.values():
        idx = np.array(idx)
        cn = nll[idx] / (nll[idx].mean() + ADV.EPS)
        ce = ent[idx] / (ent[idx].mean() + ADV.EPS)
        lam_n[idx] = ADV.mixing_weight(cn, 0.5, 0.5, 0.5)
        lam_e[idx] = ADV.mixing_weight(ce, 0.5, 0.5, 0.5)
        if len(idx) >= 4:
            within_rho.append(spearman(nll[idx], ent[idx]))
            k = max(1, len(idx) // 4)  # the steps each score calls most critical
            tn, te = set(idx[np.argsort(-nll[idx])[:k]]), set(idx[np.argsort(-ent[idx])[:k]])
            topq_agree.append(len(tn & te) / k)
    d = np.abs(lam_n - lam_e)
    rep["within_traj_spearman_mean"] = float(np.mean(within_rho))
    rep["top_quartile_overlap"] = float(np.mean(topq_agree))
    rep["lambda_absdiff_mean"] = float(d.mean())
    rep["lambda_absdiff_p90"] = float(np.percentile(d, 90))
    rep["frac_lambda_diff_gt_0.05"] = float(np.mean(d > 0.05))
    rep["frac_lambda_diff_gt_0.1"] = float(np.mean(d > 0.1))
    rep["lambda_std_nll"], rep["lambda_std_ent"] = float(lam_n.std()), float(lam_e.std())
    rep["clamped_nll"] = float(np.mean((lam_n <= 0) | (lam_n >= 1)))
    rep["clamped_ent"] = float(np.mean((lam_e <= 0) | (lam_e >= 1)))

    # Noise share: visits to the *same* anchor state in the same group differ only
    # by which response was drawn, so within-state variance / total variance is
    # the fraction of each score that is single-sample noise rather than state.
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["it"], r["uid"], r["anchor"]), []).append(i)
    multi = [np.array(g) for g in groups.values() if len(g) >= 2]
    rep["anchor_groups_m>=2"] = len(multi)
    for name, x in (("nll", nll), ("ent", ent)):
        within = np.concatenate([x[g] - x[g].mean() for g in multi])
        pooled = np.concatenate([x[g] for g in multi])
        rep[f"noise_share_{name}"] = float(np.sum(within ** 2) / np.sum((pooled - pooled.mean()) ** 2))
    rep["top20_entropy_share_mean"] = float(np.mean([r["top20"] for r in rows]))
    rep["resp_len_mean"] = float(np.mean([r["len"] for r in rows]))
    print(json.dumps(rep, indent=1))
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    return rep


# ----------------------------------------------------------------- token decision relevance


def token_oracle(args):
    """Which tokens of a reasoning-plus-action response decide the action?

    For real steps sampled from the policy, cut the response after j tokens,
    resample the remainder K times and record F(j) = P(final action differs
    from the original). F falls from F(0) (resample everything) to F(L) = 0;
    the drop across a segment, F(j_a) - F(j_b), is the share of the decision
    settled inside it -- its decision importance. The environment is
    deterministic, so an unchanged action means an identical next state, which
    makes F an exact step-level relevance measure. Token-weighting schemes are
    then scored by how well the credit they assign matches that importance.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import env as E

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(args.model, gpu_memory_utilization=0.55, max_model_len=4096, seed=args.seed, enable_prefix_caching=True)
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=512, seed=None)

    def ids_of(msgs):
        return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    # ---- A. real steps from policy rollouts
    games = E.list_games("train")
    random.Random(1234).shuffle(games)
    games = games[args.task_offset : args.task_offset + args.n_games]
    envs = [E.AlfEnv(g["gamefile"], args.max_steps) for g in games]
    for e in envs:
        e.reset()
    hist = [[] for _ in envs]
    alive = list(range(len(envs)))
    steps = []
    while alive:
        prompts = [ids_of(E.build_prompt(envs[i], hist[i], mode="think")) for i in alive]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        for i, p, o in zip(list(alive), prompts, outs):
            text = o.outputs[0].text
            act, valid = E.parse_think_action(text, envs[i].admissible)
            if valid and act in envs[i].admissible:
                steps.append({"game": i, "t": envs[i].t, "prompt": p, "resp": list(o.outputs[0].token_ids),
                              "action": act, "admissible": list(envs[i].admissible)})
            ob, done, won, _ = envs[i].step(act)
            hist[i].append((act, ob))
            if done:
                alive.remove(i)
    print(f"collected {len(steps)} valid steps from {len(envs)} games")
    rng.shuffle(steps)
    steps = steps[: args.n_steps]

    # ---- B. exact per-token entropy and sampled NLL (one forward pass per step)
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    for s in steps:
        ids = torch.tensor([s["prompt"] + s["resp"]], device="cuda")
        with torch.no_grad():
            lg = hf(ids).logits[0, len(s["prompt"]) - 1 : -1].float()
        lp = torch.log_softmax(lg, -1)
        s["ent"] = (-(lp.exp() * lp).sum(-1)).cpu().tolist()
        s["nll"] = (-lp.gather(-1, torch.tensor(s["resp"], device="cuda")[:, None])[:, 0]).cpu().tolist()
    del hf
    torch.cuda.empty_cache()

    # ---- C. flip test at cut points
    reqs, owners = [], []
    for si, s in enumerate(steps):
        resp = s["resp"]
        stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
        L = len(resp)
        while L and resp[L - 1] in stop:
            L -= 1
        txt_ids = resp[:L]
        cuts = {0} | {round(L * k / args.n_cuts) for k in range(1, args.n_cuts)}
        # where the <action> tag starts, in tokens
        for j in range(L):
            if "<action>" in tok.decode(txt_ids[: j + 1]):
                s["act_start"] = j
                cuts |= {j, min(L, j + 1)}
                break
        s["L"], s["cuts"] = L, sorted(c for c in cuts if c < L)
        for c in s["cuts"]:
            for _ in range(args.K):
                reqs.append(TokensPrompt(prompt_token_ids=s["prompt"] + txt_ids[:c]))
                owners.append((si, c))
    # ---- B2. commitment probe: q(c) = P(original action | prompt + resp[:c] + forced close).
    # One teacher-forced forward per cut, no generation. Before the <action>
    # tag the reasoning is closed by force; from the tag on, the original
    # tokens up to and including the action are scored as they are.
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    close = tok("</think>\n<action>", add_special_tokens=False).input_ids
    for s in steps:
        act_ids = tok(s["action"] + "</action>", add_special_tokens=False).input_ids
        a0 = s.get("act_start", s["L"])
        q = {}
        for c in s["cuts"] + [s["L"]]:
            if c <= a0:
                ctx, tgt = s["prompt"] + s["resp"][:c] + close, act_ids
            else:
                ctx, tgt = s["prompt"] + s["resp"][:c], s["resp"][c : s["L"]]
            if not tgt:
                q[c] = 0.0
                continue
            ids = torch.tensor([ctx + tgt], device="cuda")
            with torch.no_grad():
                lp = torch.log_softmax(hf(ids).logits[0, len(ctx) - 1 : -1].float(), -1)
            q[c] = float(lp.gather(-1, torch.tensor(tgt, device="cuda")[:, None]).sum())
        s["q"] = {str(k): v for k, v in q.items()}
    del hf
    torch.cuda.empty_cache()
    print(f"flip test: {len(reqs)} continuations")
    t0 = time.time()
    outs = llm.generate(reqs, sp, use_tqdm=False)
    print(f"  done in {time.time() - t0:.0f}s")
    flips = {}
    for (si, c), o in zip(owners, outs):
        s = steps[si]
        text = tok.decode(s["resp"][:c]) + o.outputs[0].text
        a, valid = E.parse_think_action(text, s["admissible"])
        flips.setdefault((si, c), []).append(0.0 if (valid and a == s["action"]) else 1.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".jsonl", "w") as f:
        for si, s in enumerate(steps):
            F = {c: float(np.mean(flips[(si, c)])) for c in s["cuts"]}
            f.write(json.dumps({k: s[k] for k in ("game", "t", "action", "L", "cuts", "ent", "nll", "q", "prompt", "resp")}
                               | {"act_start": s.get("act_start"), "F": {str(c): v for c, v in F.items()}}) + "\n")
    token_analyze(args.out + ".jsonl")


def token_analyze(path, n_boot=2000):
    """Score token-weighting schemes against measured decision importance."""
    recs = [json.loads(l) for l in open(path)]
    schemes = ("uniform", "entropy", "nll", "action_span", "hybrid", "probe")
    per = []  # per step: dict scheme -> (TV distance, credit on importance)
    act_share, f0 = [], []
    for r in recs:
        L, cuts = r["L"], r["cuts"]
        F = [r["F"][str(c)] for c in cuts] + [0.0]
        b = cuts + [L]
        imp = np.array([max(F[k] - F[k + 1], 0.0) for k in range(len(cuts))])
        f0.append(F[0])
        if imp.sum() < 1e-9:
            continue
        imp = imp / imp.sum()
        ent, nll = np.array(r["ent"][:L]), np.array(r["nll"][:L])
        a0 = r["act_start"] if r["act_start"] is not None else L
        span = np.array([2.0 if j >= a0 else 1.0 for j in range(L)])
        # hybrid (pre-registered from the first oracle run): entropy on the
        # reasoning, but action-span tokens get the mean reasoning entropy
        # instead of their own near-zero entropy.
        hyb = ent.copy()
        if a0 < L:
            hyb[a0:] = ent[:a0].mean() if a0 > 0 else 1.0
        w = {"uniform": np.ones(L), "entropy": ent, "nll": nll, "action_span": span, "hybrid": hyb}
        seg_mass = {}
        if "q" in r:
            # probe: how much P(original action) rises across the segment
            # (log-prob gain, clipped at 0) -- a segment-level score directly
            qv = [np.exp(r["q"][str(c)]) for c in cuts] + [np.exp(r["q"][str(L)]) if str(L) in r["q"] else 1.0]
            seg_mass["probe"] = np.array([max(qv[i + 1] - qv[i], 0.0) for i in range(len(cuts))]) + 1e-6
        out = {}
        for k in schemes:
            if k == "probe":
                if "probe" not in seg_mass:
                    continue
                m = seg_mass["probe"]
            else:
                m = np.array([w[k][b[i] : b[i + 1]].sum() for i in range(len(cuts))])
            m = m / max(m.sum(), 1e-12)
            # TV distance between credit and importance; and importance-weighted credit
            # relative to uniform (>1 = credit concentrates where the decision is made)
            out[k] = (0.5 * np.abs(m - imp).sum(), float((m * imp).sum()))
        per.append(out)
        act_share.append(float(sum(imp[i] for i in range(len(cuts)) if b[i] >= a0)))
    rng = np.random.default_rng(0)
    rep = {"steps": len(recs), "steps_with_decision_mass": len(per), "F0_mean": float(np.mean(f0)),
           "importance_share_inside_action_tags": float(np.mean(act_share))}
    uni = np.array([p["uniform"][1] for p in per])
    for k in schemes:
        if any(k not in p for p in per):
            continue
        tv = np.array([p[k][0] for p in per])
        cr = np.array([p[k][1] for p in per])
        dtv = np.array([p[k][0] - p["uniform"][0] for p in per])
        boots = [dtv[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
        rep[k] = {"tv_to_importance": float(tv.mean()), "credit_on_importance_vs_uniform": float(cr.mean() / uni.mean()),
                  "tv_minus_uniform": float(dtv.mean()),
                  "tv_minus_uniform_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.splitext(path)[0] + ".analysis.json", "w"), indent=1)
    return rep


def token_robust(path, task_offset=400, n_games=48, n_boot=2000):
    """Robustness of probe-weighted token credit, using the exact training
    function (advantages.probe_token_weights) on flip-test data.

    For each step the fine flip-test grid gives the measured decision
    importance per segment. Probe weights are built from q at a (possibly
    coarser) subset of the reasoning cuts, clipped, mean-preserving, then
    integrated over the fine segments. Metric: credit on importance relative to
    uniform (sum_seg credit_share * importance_share, divided by uniform's),
    >1 means the weights move credit toward where the decision is made."""
    import env as E

    games = E.list_games("train")
    random.Random(1234).shuffle(games)
    gtype = {i: g["task_type"] for i, g in enumerate(games[task_offset : task_offset + n_games])}
    recs = [json.loads(l) for l in open(path)]

    def score(r, spacing=1, clip=(0.25, 4.0)):
        L, cuts = r["L"], r["cuts"]
        F = [r["F"][str(c)] for c in cuts] + [0.0]
        b = cuts + [L]
        imp = np.array([max(F[k] - F[k + 1], 0.0) for k in range(len(cuts))])
        if imp.sum() < 1e-9:
            return None
        imp = imp / imp.sum()
        a0 = r["act_start"] if r["act_start"] is not None else L
        reason = [c for c in cuts if c <= a0]
        sub = sorted(set(reason[::spacing]) | {0, min(a0, L)} & set(cuts))
        sub = [c for c in sub if str(c) in r["q"]]
        qv = [float(np.exp(r["q"][str(c)])) for c in sub]
        w = ADV.probe_token_weights(L, sub, qv, clip=clip) if clip else _noclip(L, sub, qv)
        m = np.array([w[b[i] : b[i + 1]].sum() for i in range(len(cuts))])
        u = np.array([b[i + 1] - b[i] for i in range(len(cuts))], dtype=float)
        m, u = m / m.sum(), u / u.sum()
        return float((m * imp).sum()), float((u * imp).sum())

    def _noclip(L, sub, qv):
        return ADV.probe_token_weights(L, sub, qv, clip=(0.0, 1e9))

    rng = np.random.default_rng(0)

    def summarize(pairs):
        pairs = [p for p in pairs if p]
        a = np.array([p[0] for p in pairs]); u = np.array([p[1] for p in pairs])
        boots = []
        for _ in range(n_boot):
            ix = rng.integers(0, len(a), len(a))
            boots.append(a[ix].mean() / u[ix].mean())
        return {"n": len(a), "ratio": float(a.mean() / u.mean()),
                "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "frac_steps_better": float(np.mean(a > u))}

    rep = {"spacing": {}, "clip": {}, "task_type": {}, "step_t": {}, "openness_F0": {}, "split_half": {}}
    for sp in (1, 2, 3):
        rep["spacing"][f"x{sp}"] = summarize([score(r, sp) for r in recs])
    for name, cl in (("none", None), ("0.1-10", (0.1, 10.0)), ("0.25-4", (0.25, 4.0)), ("0.5-2", (0.5, 2.0))):
        rep["clip"][name] = summarize([score(r, 2, cl) for r in recs])
    for tt in sorted(set(gtype.values())):
        rep["task_type"][tt] = summarize([score(r, 2) for r in recs if gtype.get(r["game"]) == tt])
    ts = np.array([r["t"] for r in recs]); tm = np.median(ts)
    rep["step_t"]["early(t<=%d)" % tm] = summarize([score(r, 2) for r in recs if r["t"] <= tm])
    rep["step_t"]["late"] = summarize([score(r, 2) for r in recs if r["t"] > tm])
    f0 = np.array([r["F"][str(r["cuts"][0])] for r in recs]); q1, q2 = np.quantile(f0, [1 / 3, 2 / 3])
    for lab, lo, hi in (("settled", -1, q1), ("middle", q1, q2), ("open", q2, 2)):
        rep["openness_F0"][lab] = summarize([score(r, 2) for r, f in zip(recs, f0) if lo < f <= hi])
    g = sorted(gtype)
    for h, half in enumerate((set(g[::2]), set(g[1::2]))):
        rep["split_half"][f"half{h}"] = summarize([score(r, 2) for r in recs if r["game"] in half])
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.splitext(path)[0] + ".robust.json", "w"), indent=1)
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["entropy", "tokens", "tokens_analyze", "tokens_robust"])
    p.add_argument("paths", nargs="*")
    p.add_argument("--out", default="src/runs/tokens/oracle")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--task_offset", type=int, default=256, help="first training-split game index to draw from")
    p.add_argument("--K", type=int, default=8, help="resamples per cut point")
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_games", type=int, default=48)
    p.add_argument("--n_steps", type=int, default=320)
    p.add_argument("--n_cuts", type=int, default=16)
    args = p.parse_args()
    if args.cmd == "tokens":
        token_oracle(args)
    elif args.cmd == "tokens_robust":
        token_robust(args.paths[0], task_offset=args.task_offset)
    elif args.cmd == "tokens_analyze":
        token_analyze(args.paths[0])
    elif args.cmd == "entropy":
        entropy_check(args.paths[0], os.path.splitext(args.paths[0])[0] + ".entropy.json")


if __name__ == "__main__":
    main()
