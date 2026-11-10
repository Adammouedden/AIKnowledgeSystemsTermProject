"""Record one full RL iteration of the trained GACA + probe policy for the demo UI.

One ALFWorld task, a group of G rollouts at the training temperature, then
everything the trainer computes from them: rewards, returns, the GRPO, GiGPO,
GACA and probe-weighted (ours) advantages, and the per-token credit. All the
estimator math goes through src/advantages.py, the same code the training runs
called, so the numbers in the demo are the numbers training would use.

    python src/demo/make_trace.py --model models/ckpt/gaca_probe_s1 --out src/demo/trace.json
    python src/demo/make_trace.py --device cpu --group 2 --max-steps 2 ...   # plumbing smoke test
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import advantages as ADV  # noqa: E402
import env as E  # noqa: E402

GAMMA, PENALTY, WIN = 0.95, 0.1, 10.0  # training config: gamma, invalid-action penalty, success reward
CLOSE = "</think>\n<action>"


class Gen:
    """Sampling backend: vLLM on GPU, HF generate on CPU (smoke tests only)."""

    def __init__(self, model, device, seed, temperature):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.device, self.temperature = device, temperature
        if device == "cuda":
            from vllm import LLM, SamplingParams
            self.llm = LLM(model, gpu_memory_utilization=0.45, max_model_len=4096, seed=seed, enable_prefix_caching=True)
            self.sp = SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, max_tokens=512)
        else:
            import torch
            from transformers import AutoModelForCausalLM
            torch.manual_seed(seed)
            self.hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()

    def generate(self, prompts):
        if self.device == "cuda":
            from vllm.inputs import TokensPrompt
            outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], self.sp, use_tqdm=False)
            return [list(o.outputs[0].token_ids) for o in outs]
        import torch
        res = []
        for p in prompts:
            with torch.no_grad():
                o = self.hf.generate(torch.tensor([p]), do_sample=True, temperature=self.temperature, top_p=1.0,
                                     top_k=0, max_new_tokens=512, pad_token_id=self.tok.eos_token_id)
            res.append(o[0, len(p):].tolist())
        return res


def rollout_group(gen, gamefile, G, max_steps):
    """G rollouts of one game, stepped in lockstep like the trainer's env batch."""
    tok = gen.tok
    envs = [E.AlfEnv(gamefile, max_steps) for _ in range(G)]
    for e in envs:
        e.reset()
    hist = [[] for _ in range(G)]
    trajs = [{"traj": i, "steps": [], "won": False} for i in range(G)]
    alive = list(range(G))
    while alive:
        msgs = {i: E.build_think_prompt(envs[i], hist[i]) for i in alive}
        ids = {i: tok.apply_chat_template(msgs[i], tokenize=True, add_generation_prompt=True) for i in alive}
        outs = gen.generate([ids[i] for i in alive])
        for i, resp in zip(list(alive), outs):
            e = envs[i]
            text = tok.decode(resp, skip_special_tokens=True)
            act, valid = E.parse_think_action(text, e.admissible)
            rec = {"t": e.t, "obs": e.obs, "task": e.task, "admissible": list(e.admissible),
                   "anchor": hashlib.md5(e.obs.encode()).hexdigest()[:12],  # verl-agent anchors on the text observation
                   "prompt": msgs[i][0]["content"], "prompt_ids": ids[i], "resp_ids": resp, "text": text,
                   "action": act, "valid": bool(valid)}
            ob, done, won, _ = e.step(act if valid else "noop")
            rec["next_obs"] = ob
            rec["reward"] = (WIN if won else 0.0) - (0.0 if valid else PENALTY)
            rec["done"] = bool(done)
            trajs[i]["steps"].append(rec)
            hist[i].append((act, ob))
            if done:
                trajs[i]["won"] = bool(won)
                alive.remove(i)
    return trajs


def score(tok, model, device, trajs, every=16, clip=(0.25, 4.0)):
    """Per-token log-probs (NLL for GACA) and the commitment probe for every step."""
    import torch
    from transformers import AutoModelForCausalLM
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    hf = AutoModelForCausalLM.from_pretrained(model, dtype=dt).to(device).eval()
    close = tok(CLOSE, add_special_tokens=False).input_ids

    def logp(ctx, tgt):
        ids = torch.tensor([ctx + tgt], device=device)
        with torch.no_grad():
            lp = torch.log_softmax(hf(ids).logits[0, len(ctx) - 1: -1].float(), -1)
        return lp.gather(-1, torch.tensor(tgt, device=device)[:, None])[:, 0].cpu().numpy()

    for tr in trajs:
        for s in tr["steps"]:
            p, r = s["prompt_ids"], s["resp_ids"]
            tl = logp(p, r)
            s["tok_nll"] = [float(-x) for x in tl]
            s["nll"] = float(-tl.mean())
            s["tokens"] = [tok.decode([t]) for t in r]
            # probe, exactly as gigpo/core_probe.build_probe_requests: only for well-formed responses
            L = len(r)
            m = E._ACT_RE.findall(s["text"])
            a0 = _act_start(tok, r)
            if "<think>" in s["text"] and "</think>" in s["text"] and m and a0 is not None:
                tgt = tok(m[-1] + "</action>", add_special_tokens=False).input_ids
                cuts = ADV.probe_cuts(L, a0, every)
                q = [float(math.exp(logp(p + r[:c] + close, tgt).sum())) for c in cuts]
                s["probe"] = {"cuts": cuts, "q": q, "act_start": a0}
                s["w"] = ADV.probe_token_weights(L, cuts, q, clip).tolist()
            else:
                s["probe"] = None
                s["w"] = [1.0] * L
    del hf


def _act_start(tok, ids):
    text, starts = "", []
    for t in ids:
        starts.append(len(text))
        text += tok.decode([t])
    k = text.find("<action>")
    if k < 0:
        return None
    for j in range(len(ids)):
        if starts[j] <= k < (starts[j + 1] if j + 1 < len(ids) else len(text) + 1):
            return j
    return None


def advantages(trajs):
    """All four estimators on the group's step rows, plus the intermediates the UI shows."""
    rows = []
    for tr in trajs:
        rew = [s["reward"] for s in tr["steps"]]
        g = ADV.discounted_step_returns(rew, GAMMA)
        ret = float(sum(rew))
        tr["ret"] = ret
        for s, gh in zip(tr["steps"], g):
            s["ret"], s["g_hat"] = ret, float(gh)
            rows.append({"group": 0, "traj": tr["traj"], "t": s["t"], "ret": ret, "g_hat": float(gh),
                         "anchor": s["anchor"], "nll": s["nll"]})
    a_e = ADV.episode_advantage(rows, "mean_std")
    a_s, cov = ADV.step_advantage(rows, "mean_std")
    c = ADV.criticality(rows, mode="entropy")
    lam = ADV.mixing_weight(c, 0.5, 0.5, 0.5)
    m = ADV.anchor_sizes(rows)
    out = {name: ADV.compute_advantages(rows, method=name)[0] for name in ("grpo", "gigpo", "gaca")}
    assert np.allclose(out["gaca"], lam * a_s + (1 - lam) * a_e)
    rets = np.array([r["ret"] for r in rows])
    k = 0
    for tr in trajs:
        for s in tr["steps"]:
            s.update(A_ep=float(a_e[k]), A_st=float(a_s[k]), c=float(c[k]), lam=float(lam[k]), m_z=int(m[k]),
                     A_grpo=float(out["grpo"][k]), A_gigpo=float(out["gigpo"][k]), A_gaca=float(out["gaca"][k]))
            k += 1
    # anchor-group statistics, for showing the GiGPO contrast
    groups = {}
    for tr in trajs:
        for s in tr["steps"]:
            groups.setdefault(s["anchor"], []).append([tr["traj"], s["t"], s["g_hat"]])
    return {"ret_mean": float(rets.mean()), "ret_std": float(rets.std(ddof=1)), "n_rows": len(rows),
            "step_coverage": float(cov), "anchor_groups": {a: v for a, v in groups.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ckpt/gaca_probe_s1")
    ap.add_argument("--out", default="src/demo/trace.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tries", type=int, default=24)
    ap.add_argument("--min-wins", type=int, default=2)
    ap.add_argument("--max-wins", type=int, default=6)
    ap.add_argument("--game-offset", type=int, default=0)
    args = ap.parse_args()

    gen = Gen(args.model, args.device, args.seed, args.temperature)
    games = E.list_games("train")
    random.Random(7).shuffle(games)
    best = None
    for gi, g in enumerate(games[args.game_offset: args.game_offset + args.tries]):
        t0 = time.time()
        trajs = rollout_group(gen, g["gamefile"], args.group, args.max_steps)
        wins = sum(t["won"] for t in trajs)
        print(f"game {gi}: {wins}/{args.group} wins, {sum(len(t['steps']) for t in trajs)} steps, {time.time() - t0:.0f}s", flush=True)
        if args.min_wins <= wins <= args.max_wins:
            best = (g, trajs)
            break
        if best is None or (0 < wins < args.group):
            best = (g, trajs)
    g, trajs = best
    if args.device == "cuda":
        del gen.llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    score(gen.tok, args.model, args.device, trajs)
    stats = advantages(trajs)
    for tr in trajs:
        for s in tr["steps"]:
            for k in ("prompt_ids", "resp_ids"):
                s.pop(k)
            s["tok_nll"] = [round(x, 3) for x in s["tok_nll"]]
            s["w"] = [round(x, 3) for x in s["w"]]
    meta = {"model": os.path.basename(args.model.rstrip("/")), "gamefile": "/".join(g["gamefile"].split("/")[-3:-1]),
            "task": trajs[0]["steps"][0]["task"], "group": args.group, "temperature": args.temperature,
            "gamma": GAMMA, "penalty": PENALTY, "win_reward": WIN, "lam0": 0.5, "alpha": 0.5, "beta": 0.5,
            "probe_every": 16, "clip": [0.25, 4.0]} | stats
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"meta": meta, "trajs": trajs}, open(args.out, "w"))
    print("wrote", args.out, f"{os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
