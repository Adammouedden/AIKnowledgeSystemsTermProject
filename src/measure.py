"""Measurement study: which mixing gate best recovers the true step advantage?

No training. For groups of G rollouts collected exactly as in training, every
step visit i gets a Monte-Carlo oracle for its true advantage

    A_i = Q(h_i, a_i) - V(h_i),

where h_i is the visit's full interaction history (the policy conditions on
it) and both terms are averages of discounted step returns over K
continuations sampled from the same policy. TextWorld is deterministic, so a
state is restored exactly by replaying the visit's action prefix. The
continuations are independent of the G observed rollouts, so the oracle is
cross-fitted: its noise is uncorrelated with every estimator's error and adds
the same constant to all of their errors.

Each gate lambda_t then gives an advantage lambda_t A^S + (1 - lambda_t) A^E
built from the observed rollouts only, and is scored by its alignment with the
oracle on the same steps -- a paired comparison, so it resolves differences
that a training run at this budget cannot.

    python src/measure.py collect --out runs/measure/base --n_groups 32
    python src/measure.py analyze runs/measure/base.jsonl
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

# ----------------------------------------------------------------- collection


TIMES = {"prompt": 0.0, "gen": 0.0, "env": 0.0, "calls": 0}


def continuations(model, tok, pool, specs, args):
    """Value of each spec {gamefile, prefix}: E[gamma**(T_end - len(prefix)) * won].

    The prefix is replayed verbatim inside a worker, then the policy samples
    until the episode ends. Slots are refilled from the queue as soon as their
    episode finishes, so generation always runs at full batch.
    """
    import torch

    import env as E
    import train as T

    out = [0.0] * len(specs)
    eos_ids = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id})
    queue = list(range(len(specs)))[::-1]
    slot = {}  # runner -> [spec index, hist, last step index]
    free = list(range(len(pool)))

    def finish(si, last, won):
        out[si] = float(args.gamma ** (last - len(specs[si]["prefix"]))) if won else 0.0

    def refill():
        while free and queue:
            take = []
            while free and queue:
                take.append((free.pop(), queue.pop()))
            res = pool.restore([(r, specs[si]["gamefile"], specs[si]["prefix"]) for r, si in take])
            for (r, si), (obs, done, won) in zip(take, res):
                pre = specs[si]["prefix"]
                if done:  # the prefix itself ended the episode
                    finish(si, len(obs) - 1, won)
                    free.append(r)
                else:
                    slot[r] = [si, list(zip(pre, obs)), len(pre) - 1]

    tm = TIMES
    t = time.time()
    refill()
    tm["env"] += time.time() - t
    while slot:
        t = time.time()
        batch = sorted(slot)[: args.gen_batch]
        views = pool.views
        texts = [
            tok.apply_chat_template(
                E.build_prompt(views[r], slot[r][1], history_len=args.history_len, mode=args.action_mode),
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in batch
        ]
        enc, cache = T.encode_prompts(model, tok, texts, args)
        torch.cuda.synchronize()
        tm["prompt"] += time.time() - t
        t = time.time()
        with torch.no_grad(), T.amp(model):
            o = model.generate(
                **enc, past_key_values=cache, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id, eos_token_id=eos_ids, use_cache=True,
            )
        reps = tok.batch_decode(o[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        tm["gen"] += time.time() - t
        tm["calls"] += 1
        t = time.time()
        if args.action_mode == "text":
            acts = [E.parse_text_action(rp, views[r].admissible)[0] for r, rp in zip(batch, reps)]
        else:
            acts = [E.parse_action(rp, views[r].admissible) for r, rp in zip(batch, reps)]
        for r, a, (ob, d, w, _) in zip(batch, acts, pool.step(batch, acts)):
            slot[r][1].append((a, ob))
            slot[r][2] += 1
            if d:
                finish(slot[r][0], slot[r][2], w)
                del slot[r]
                free.append(r)
        refill()
        tm["env"] += time.time() - t
    return out


def collect(args):
    import torch

    import env as E
    import train as T

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    types = args.task_types.split(",")
    pool_games = E.list_games("train", task_types=types)
    random.Random(1234).shuffle(pool_games)
    # Tasks *after* the training slice, so the study never touches a task a
    # trained policy has seen.
    tasks = E.probe_games(pool_games[args.task_offset : args.task_offset + 4 * args.n_groups])
    if args.max_adm0:
        tasks = [g for g in tasks if g["n_adm0"] <= args.max_adm0]

    tok, model = T.build_model(args, lora=False, weights=args.adapter)
    pool = E.make_pool(args.pool, tasks[0]["gamefile"], args.max_steps, workers=args.workers)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log = open(args.out + ".jsonl", "a", buffering=1)
    log.write(json.dumps({"event": "start", "args": vars(args), "time": time.time()}) + "\n")
    n_inf, t0 = 0, time.time()
    for gi, g in enumerate(tasks):
        if n_inf >= args.n_groups:
            break
        steps, eps = T.rollout(model, tok, pool, [(gi, g["gamefile"])] * args.G, args, train=True)
        wins = sum(e["won"] for e in eps)
        rec = {"event": "group", "group": gi, "gamefile": g["gamefile"], "task_type": g["task_type"],
               "wins": wins, "n_steps": len(steps)}
        if 0 < wins < args.G:  # an all-win or all-fail group gives every estimator exactly zero
            # Deterministic transitions and a terminal-only reward give
            # Q(h_t, a_t) = gamma * V(h_{t+1}) exactly, so one value per unique
            # action prefix serves both: V_t = val(a_<t), Q_t = gamma val(a_<=t).
            # Identical prefixes (every trajectory's step 0, for one) share it.
            acts = {}
            for s in steps:
                acts.setdefault(s["traj"], {})[s["t"]] = s["action"]
            ends = {}  # full trajectories need no simulation: a win at the last step, or a capped failure
            for e_i, e in enumerate(eps):
                ends[json.dumps([acts[e_i][k] for k in range(e["steps"])])] = 1.0 / args.gamma if e["won"] else 0.0
            keys = set()
            for s in steps:
                pre = [acts[s["traj"]][k] for k in range(s["t"])]
                s["kV"], s["kQ"] = json.dumps(pre), json.dumps(pre + [s["action"]])
                keys.update([s["kV"], s["kQ"]])
            klist = sorted(k for k in keys if k not in ends)
            specs = [{"gamefile": g["gamefile"], "prefix": json.loads(k)} for k in klist for _ in range(args.K)]
            vals = continuations(model, tok, pool, specs, args)
            val = dict(ends)
            val.update({k: float(np.mean(vals[j * args.K : (j + 1) * args.K])) for j, k in enumerate(klist)})
            rec["steps"] = [
                {k: s[k] for k in ("group", "traj", "t", "anchor", "nll", "ret", "g_hat", "action")}
                | {"Q": args.gamma * val[s["kQ"]], "V": val[s["kV"]]}
                for s in steps
            ]
            rec["n_specs"] = len(specs)
            n_inf += 1
        rec["elapsed"] = time.time() - t0
        rec["times"] = dict(TIMES)
        log.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "steps"}), flush=True)
    pool.close()


# ----------------------------------------------------------------- analysis


def gates(steps, norm, lam0=0.5, rng=None):
    """Every gate's advantage on one group's steps (all built from the observed rollouts)."""
    a_e = ADV.episode_advantage(steps, norm)
    a_s, _ = ADV.step_advantage(steps, norm)
    m = ADV.anchor_sizes(steps)
    c = ADV.criticality(steps, mode="entropy")
    c_loo = ADV.criticality(steps, mode="loo")
    lam_nll = ADV.mixing_weight(c, lam0, 0.5, 0.5)
    lam_loo = ADV.mixing_weight(c_loo, lam0, 0.5, 0.5)
    lam_fix = np.full(len(steps), lam0)
    mix = lambda lam: lam * a_s + (1 - lam) * a_e
    return {
        "GRPO (A^E only)": a_e,
        "GiGPO (A^S + A^E)": a_s + a_e,
        "fixed lambda_0": mix(lam_fix),
        "GACA (NLL)": mix(lam_nll),
        "GACA-LOO (B.4)": mix(lam_loo),
        "fixed + msize": mix(ADV.reliability(lam_fix, m)),
        "GACA + msize": mix(ADV.reliability(lam_nll, m)),
        "LOO + msize": mix(ADV.reliability(lam_loo, m)),
    }, {"a_e": a_e, "a_s": a_s, "m": m, "c": c, "c_loo": c_loo}


def _cos(x, y):
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def analyze(paths, n_boot=2000):
    groups = []
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if r.get("event") == "group" and r.get("steps"):
                groups.append(r["steps"])
    print(f"{len(groups)} informative groups, {sum(len(g) for g in groups)} step visits")
    rng = np.random.default_rng(0)
    report = {}
    for norm in ("mean", "mean_std"):
        per = []  # per group: dict gate -> (dot, |x|^2), and |y|^2
        feats = []
        for st in groups:
            y = np.array([s["Q"] - s["V"] for s in st])
            gs, f = gates(st, norm)
            per.append(({k: (float(v @ y), float(v @ v)) for k, v in gs.items()}, float(y @ y)))
            feats.append((f, y, st))
        names = list(per[0][0])

        def score(idx):
            # Pooled cosine over the selected groups: sum dot / sqrt(sum |x|^2 sum |y|^2)
            yy = sum(per[i][1] for i in idx)
            return {k: sum(per[i][0][k][0] for i in idx) / np.sqrt(sum(per[i][0][k][1] for i in idx) * yy + 1e-12)
                    for k in names}

        full = score(range(len(per)))
        boots = [score(rng.integers(0, len(per), len(per))) for _ in range(n_boot)]
        ref = "GACA-LOO (B.4)"
        rows = []
        for k in names:
            b = np.array([x[k] for x in boots])
            d = np.array([x[k] - x[ref] for x in boots])
            rows.append({"gate": k, "cos": full[k], "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                         "vs_loo": full[k] - full[ref],
                         "vs_loo_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         "p_better": float(np.mean(d > 0))})
        report[norm] = {"gates": rows, "binned": binned(feats)}
    return report, len(groups), sum(len(g) for g in groups)


def binned(feats):
    """Least-squares optimal lambda per bin: fit y ~ b_S A^S + b_E A^E, lambda* = b_S / (b_S + b_E).

    Scale-free (the two coefficients absorb any global scale), so it answers
    the question Proposition 3 poses -- where should the weight sit? -- by
    regressing the oracle on the two estimators inside each bin."""
    rows = {"m_z": [], "c_t": [], "spread": []}
    A_S, A_E, Y, M, C, SP = [], [], [], [], [], []
    for f, y, st in feats:
        # Within-bucket spread of the observed step returns (the measured C_z + sigma^2).
        b = {}
        for i, s in enumerate(st):
            b.setdefault(s["anchor"], []).append(s["g_hat"])
        sp = [float(np.std(b[s["anchor"]])) if len(b[s["anchor"]]) > 1 else np.nan for s in st]
        A_S += list(f["a_s"]); A_E += list(f["a_e"]); Y += list(y); M += list(f["m"]); C += list(f["c"]); SP += sp
    A_S, A_E, Y, M, C, SP = map(np.array, (A_S, A_E, Y, M, C, SP))

    def fit(mask):
        X = np.stack([A_S[mask], A_E[mask]], 1)
        if mask.sum() < 20 or np.linalg.matrix_rank(X) < 2:
            return None
        bS, bE = np.linalg.lstsq(X, Y[mask], rcond=None)[0]
        return {"n": int(mask.sum()), "b_S": float(bS), "b_E": float(bE),
                "lam_star": float(bS / (bS + bE)) if abs(bS + bE) > 1e-9 else None,
                "r2": float(1 - np.sum((Y[mask] - X @ [bS, bE]) ** 2) / max(np.sum(Y[mask] ** 2), 1e-12))}

    for lo, hi, lab in [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 1e9, "9+")]:
        rows["m_z"].append({"bin": lab, **(fit((M >= lo) & (M <= hi)) or {})})
    qs = np.quantile(C, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for a, b in zip(qs[:-1], qs[1:]):
        rows["c_t"].append({"bin": f"{a:.2f}-{b:.2f}", **(fit((C >= a) & (C <= b)) or {})})
    ok = ~np.isnan(SP)
    rows["spread"].append({"bin": "zero", **(fit(ok & (SP == 0)) or {})})
    pos = ok & (SP > 0)
    if pos.sum():
        med = np.median(SP[pos])
        rows["spread"].append({"bin": f"(0,{med:.2f}]", **(fit(pos & (SP <= med)) or {})})
        rows["spread"].append({"bin": f">{med:.2f}", **(fit(pos & (SP > med)) or {})})
    rows["all"] = fit(np.ones(len(Y), bool))
    return rows


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
            f.write(json.dumps({k: s[k] for k in ("game", "t", "action", "L", "cuts", "ent", "nll")}
                               | {"act_start": s.get("act_start"), "F": {str(c): v for c, v in F.items()}}) + "\n")
    token_analyze(args.out + ".jsonl")


def token_analyze(path, n_boot=2000):
    """Score token-weighting schemes against measured decision importance."""
    recs = [json.loads(l) for l in open(path)]
    schemes = ("uniform", "entropy", "nll", "action_span")
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
        w = {"uniform": np.ones(L), "entropy": ent, "nll": nll, "action_span": span}
        out = {}
        for k in schemes:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["collect", "analyze", "entropy", "tokens", "tokens_analyze"])
    p.add_argument("paths", nargs="*")
    p.add_argument("--out", default="runs/measure/base")
    p.add_argument("--adapter", default="base")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n_groups", type=int, default=32, help="informative groups to measure")
    p.add_argument("--task_offset", type=int, default=256, help="first task index (after the training slice)")
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--K", type=int, default=4, help="continuations per Q and per V estimate")
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--invalid_penalty", type=float, default=0.0)
    p.add_argument("--action_mode", default="text", choices=["index", "text", "think"])
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--max_prompt_len", type=int, default=4096)
    p.add_argument("--history_len", type=int, default=10)
    p.add_argument("--gen_batch", type=int, default=128)
    p.add_argument("--pool", type=int, default=128)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--task_types", default="pick_and_place_simple,look_at_obj_in_light")
    p.add_argument("--max_adm0", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--n_games", type=int, default=48)
    p.add_argument("--n_steps", type=int, default=320)
    p.add_argument("--n_cuts", type=int, default=16)
    args = p.parse_args()
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "tokens":
        token_oracle(args)
    elif args.cmd == "tokens_analyze":
        token_analyze(args.paths[0])
    elif args.cmd == "entropy":
        entropy_check(args.paths[0], os.path.splitext(args.paths[0])[0] + ".entropy.json")
    else:
        rep, ng, ns = analyze(args.paths)
        print(json.dumps(rep, indent=1))
        out = os.path.splitext(args.paths[0])[0] + ".analysis.json"
        json.dump({"groups": ng, "visits": ns, **rep}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
