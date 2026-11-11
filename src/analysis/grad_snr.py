"""Does token weighting make the policy gradient less noisy? (DECISIONS.md, 2026-11-10)

Hypothesis: concentrating a step's credit on its decision tokens raises the
gradient's signal-to-noise ratio, which is what lets COMMIT gain from a larger
learning rate where uniform credit (GACA) degrades. Training runs give one noisy
number per configuration; this measures the mechanism directly, with confidence
intervals, at a fixed policy and on identical rollouts -- only the credit changes.

Stage 1 (`rollouts`, vLLM): sample groups of G rollouts per training game at
T = 1.0, and keep the groups whose rollouts disagree on winning (a group where all
8 lose or all 8 win has zero advantage and no gradient; those are what dynamic
sampling drops).

Stage 2 (`grads`, HF): per kept group, score each step (token NLL for GACA's
criticality, the commitment probe for COMMIT), compute GACA's step advantage with
src/advantages.py (the training code), then the policy gradient of that group,
    g = sum_steps sum_tokens  A_step * w_tok * grad log pi(tok)
under three token-weight schemes on the same rollouts:
    uniform  w = 1                          (GACA)
    action   <action> span 2x, reasoning 1x, mean 1   (gaca_actw)
    probe    COMMIT's commitment-probe weights, clip [0.25, 4]   (commit_*)
At PPO's first mini-batch the ratio is 1, so this is the direction the update takes.
Gradients are stored on a fixed strided 1/128 subsample of every parameter tensor
(inner products and cosines are estimated without bias up to a constant scale).

Stage 3 (`report`): per scheme, across groups i != j,
    coherence     mean cos(g_i, g_j)
    batch cosine  cos(mean of 4 groups, mean of 4 other groups); 4 groups = one
                  training batch (TRAIN_BS=4), so this is how reproducible one
                  iteration's update direction is
    noise scale   B = tr(Sigma) / |mu|^2 in groups (McCandlish et al., 2018): the
                  batch size beyond which larger steps stop paying off; lower = a
                  cleaner signal
with jackknife (leave-one-group-out) 95% intervals, and paired differences
between schemes (same groups, so the comparison is within-sample).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "demo"))
import advantages as ADV  # noqa: E402
import env as E  # noqa: E402
import make_trace as MT  # noqa: E402

SCHEMES = ("uniform", "action", "probe")
STRIDE = 128


# ---------------------------------------------------------------- stage 1
def rollouts(args):
    games = E.list_games("train")
    random.Random(args.seed).shuffle(games)
    games = games[args.game_offset:]
    if args.device == "cuda":
        gen = MT.Gen(args.model, "cuda", args.seed, 1.0, gpu_util=0.6)
        tok = gen.tok
        generate = gen.generate
    else:  # plumbing smoke test: canned well-formed responses, no sampling
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        gen = None
    kept, tried, t0 = [], 0, time.time()
    # One fixed pool, reloaded every wave. Each new TextWorld env copies fast-downward's 32 MB library into
    # /tmp and keeps the deleted copy mapped until exit: fresh envs per wave filled the 32 GB disk.
    pool = [E.AlfEnv(games[0]["gamefile"], args.max_steps) for _ in range(args.wave * args.group)]
    while len(kept) < args.target and tried < args.max_games:
        wave = games[tried: tried + args.wave]
        tried += len(wave)
        envs = [(gi, k, pool[gi * args.group + k]) for gi, g in enumerate(wave) for k in range(args.group)]
        for gi, k, e in envs:
            e.load(wave[gi]["gamefile"])
        hist = {(gi, k): [] for gi, k, _ in envs}
        steps = {(gi, k): [] for gi, k, _ in envs}
        won = {(gi, k): False for gi, k, _ in envs}
        alive = list(range(len(envs)))
        while alive:
            ids = []
            for j in alive:
                gi, k, e = envs[j]
                msgs = E.build_think_prompt(e, hist[(gi, k)])
                ids.append(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            if gen is not None:
                outs = generate(ids)
            else:
                outs = [tok(f"<think>I should look around first.</think>\n<action>{envs[j][2].admissible[0]}</action>",
                            add_special_tokens=False).input_ids for j in alive]
            nxt = []
            for j, p, r in zip(alive, ids, outs):
                gi, k, e = envs[j]
                text = tok.decode(r, skip_special_tokens=True)
                act, valid = E.parse_think_action(text, e.admissible)
                rec = {"t": e.t, "anchor": hashlib.md5(e.obs.encode()).hexdigest()[:12],  # verl-agent anchors on the observation text
                       "prompt_ids": p,
                       "resp_ids": list(r), "text": text, "valid": bool(valid)}
                ob, done, w, _ = e.step(act if valid else "noop")
                rec["reward"] = (MT.WIN if w else 0.0) - (0.0 if valid else MT.PENALTY)
                steps[(gi, k)].append(rec)
                hist[(gi, k)].append((act, ob))
                if done:
                    won[(gi, k)] = bool(w)
                else:
                    nxt.append(j)
            alive = nxt
        for gi, g in enumerate(wave):
            w = [won[(gi, k)] for k in range(args.group)]
            if args.fake_wins:
                w = [k % 2 == 0 for k in range(args.group)]
                for k in range(args.group):
                    steps[(gi, k)][-1]["reward"] += MT.WIN * w[k]
            if 0 < sum(w) < args.group:
                kept.append({"game": g["gamefile"], "trajs": [{"traj": k, "won": w[k], "steps": steps[(gi, k)]}
                                                             for k in range(args.group)]})
        print(f"[rollouts] {tried} games tried, {len(kept)} groups kept, {time.time() - t0:.0f}s", flush=True)
    kept = kept[: args.target]
    json.dump({"model": args.model, "tried": tried, "max_steps": args.max_steps, "groups": kept}, open(args.out, "w"))
    print(f"[rollouts] wrote {args.out}: {len(kept)} groups from {tried} games", flush=True)


# ---------------------------------------------------------------- stage 2
def _hidden(model, seqs, device):
    """Right-padded batch -> last hidden states (B, T, H)."""
    import torch
    T = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), T, dtype=torch.long, device=device)
    att = torch.zeros(len(seqs), T, dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = torch.tensor(s, device=device)
        att[b, : len(s)] = 1
    return model.model(input_ids=ids, attention_mask=att).last_hidden_state


def _logp_at(model, h, seq_len_ctx, tgt):
    """Log-probs of tgt tokens following a context of length seq_len_ctx, from hidden states h (T, H)."""
    import torch
    pos = torch.arange(seq_len_ctx - 1, seq_len_ctx - 1 + len(tgt), device=h.device)
    lp = torch.log_softmax(model.lm_head(h[pos]).float(), -1)
    return lp.gather(-1, torch.tensor(tgt, device=h.device)[:, None])[:, 0]


def score_group(model, tok, grp, device, every=16, clip=(0.25, 4.0), k_action=2.0):
    """NLL, probe weights and action-span weights for every step of one group."""
    import torch
    close = tok(MT.CLOSE, add_special_tokens=False).input_ids
    for tr in grp["trajs"]:
        for s in tr["steps"]:
            p, r = s["prompt_ids"], s["resp_ids"]
            L = len(r)
            m = E._ACT_RE.findall(s["text"])
            a0 = MT._act_start(tok, r)
            ok = "<think>" in s["text"] and "</think>" in s["text"] and bool(m) and a0 is not None and L > 0
            seqs = [p + r]
            cuts, tgt = [], None
            if ok:
                tgt = tok(m[-1] + "</action>", add_special_tokens=False).input_ids
                cuts = ADV.probe_cuts(L, a0, every)
                seqs += [p + r[:c] + close + tgt for c in cuts]
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                h = _hidden(model, seqs, device)
                tl = _logp_at(model, h[0], len(p), r) if L else torch.zeros(0)
                q = [float(torch.exp(_logp_at(model, h[1 + b], len(p) + c + len(close), tgt).sum())) for b, c in enumerate(cuts)]
            s["nll"] = float(-tl.mean()) if L else 0.0
            if ok:
                s["w_probe"] = ADV.probe_token_weights(L, cuts, q, clip).tolist()
                wa = np.ones(L)
                wa[a0:] = k_action
                s["w_action"] = (wa / wa.mean()).tolist()
            else:
                s["w_probe"] = s["w_action"] = [1.0] * L
            s["w_uniform"] = [1.0] * L
    MT.advantages(grp["trajs"])  # GACA step advantage A_gaca (+ intermediates), training code


def grads(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    data = json.load(open(args.inp))
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = args.device
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    offs = np.random.RandomState(0).randint(0, STRIDE, size=len(params))
    out = {s: [] for s in SCHEMES}
    t0 = time.time()
    for gi, grp in enumerate(data["groups"]):
        model.eval()
        score_group(model, tok, grp, dev)
        rows = [s for tr in grp["trajs"] for s in tr["steps"] if abs(s["A_gaca"]) > 0 and len(s["resp_ids"])]
        for sch in SCHEMES:
            model.zero_grad(set_to_none=True)
            model.train()
            for s in rows:
                p, r = s["prompt_ids"], s["resp_ids"]
                coef = torch.tensor(np.asarray(s[f"w_{sch}"]) * s["A_gaca"], dtype=torch.float32, device=dev)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    h = _hidden(model, [p + r], dev)[0]
                    lp = _logp_at(model, h, len(p), r)
                (-(coef * lp).sum()).backward()  # descent on -objective: grad = -g; sign is irrelevant to every statistic below
            with torch.no_grad():
                v = torch.cat([q.grad.reshape(-1)[o::STRIDE].float() if q.grad is not None
                               else torch.zeros_like(q.reshape(-1)[o::STRIDE], dtype=torch.float32)
                               for q, o in zip(params, offs)]).cpu().numpy()
            out[sch].append(v)
        n_act = sum(1 for s in rows if s["w_probe"] != s["w_uniform"])
        print(f"[grads] group {gi + 1}/{len(data['groups'])}: {len(rows)} step rows ({n_act} well-formed), "
              f"{time.time() - t0:.0f}s", flush=True)
    np.savez(args.out, **{s: np.stack(v) for s, v in out.items()})
    print(f"[grads] wrote {args.out}", flush=True)


# ---------------------------------------------------------------- stage 3
def _stats(G, batch=4, n_split=400, seed=0):
    """G: (n, d) per-group gradients. Statistics from the Gram matrix."""
    K = G @ G.T
    n = len(K)
    nrm = np.sqrt(np.diag(K))
    off = ~np.eye(n, dtype=bool)
    coh = float((K / np.outer(nrm, nrm))[off].mean())
    mu2 = float(K[off].mean())                  # unbiased |mu|^2
    tr = float(np.diag(K).mean() - mu2)          # unbiased tr(Sigma)
    B = tr / mu2 if mu2 > 0 else float("inf")
    rng = np.random.RandomState(seed)
    cs = []
    if n >= 2 * batch:
        for _ in range(n_split):
            idx = rng.permutation(n)
            a, b = idx[:batch], idx[batch: 2 * batch]
            ga, gb = K[np.ix_(a, a)].sum(), K[np.ix_(b, b)].sum()
            cs.append(K[np.ix_(a, b)].sum() / math.sqrt(ga * gb))
    return {"coherence": coh, "batch_cosine": float(np.mean(cs)) if cs else float("nan"), "noise_scale": B}


def _jack(Gs, f):
    """Jackknife estimate and 95% interval of f over groups (leave one group out)."""
    n = len(Gs[0])
    full = f(*Gs)
    loo = np.array([f(*[np.delete(G, i, 0) for G in Gs]) for i in range(n)])
    se = math.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())
    return full, full - 1.96 * se, full + 1.96 * se


def report(args):
    lines = ["| checkpoint | groups | scheme | coherence (mean pairwise cos) | 4-group batch cosine | noise scale B (groups) |",
             "|---|---|---|---|---|---|"]
    diffs = ["| checkpoint | comparison | coherence diff [95% CI] | batch cosine diff [95% CI] |", "|---|---|---|---|"]
    res = {}
    for name, path in zip(args.names, args.npz):
        z = np.load(path)
        n = len(z["uniform"])
        res[name] = {}
        for sch in SCHEMES:
            G = z[sch]
            r = {k: _jack([G], lambda g, k=k: _stats(g)[k]) for k in ("coherence", "batch_cosine", "noise_scale")}
            res[name][sch] = r
            fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
            lines.append(f"| {name} | {n} | {sch} | {fmt(r['coherence'])} | {fmt(r['batch_cosine'])} | {fmt(r['noise_scale'])} |")
        for sch in ("action", "probe"):
            d = {k: _jack([z[sch], z["uniform"]], lambda a, b, k=k: _stats(a)[k] - _stats(b)[k])
                 for k in ("coherence", "batch_cosine")}
            res[name][f"{sch}_minus_uniform"] = d
            fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
            diffs.append(f"| {name} | {sch} - uniform | {fmt(d['coherence'])} | {fmt(d['batch_cosine'])} |")
    txt = "\n".join(lines + [""] + diffs) + "\n"
    open(args.out + ".md", "w").write(txt)
    json.dump(res, open(args.out + ".json", "w"), indent=1)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("rollouts")
    a.add_argument("--model", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cuda")
    a.add_argument("--group", type=int, default=8)
    a.add_argument("--max-steps", type=int, default=30)
    a.add_argument("--wave", type=int, default=16)  # 16 games x 8 = 128 envs, ~4 GB of /tmp
    a.add_argument("--target", type=int, default=16)
    a.add_argument("--max-games", type=int, default=160)
    a.add_argument("--game-offset", type=int, default=0)
    a.add_argument("--seed", type=int, default=11)
    a.add_argument("--fake-wins", action="store_true", help="smoke test only")
    b = sp.add_parser("grads")
    b.add_argument("--model", required=True)
    b.add_argument("--inp", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--device", default="cuda")
    c = sp.add_parser("report")
    c.add_argument("--npz", nargs="+", required=True)
    c.add_argument("--names", nargs="+", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"rollouts": rollouts, "grads": grads, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
