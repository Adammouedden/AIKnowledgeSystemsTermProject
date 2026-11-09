"""GRPO / GiGPO / GACA on ALFWorld with a Qwen2.5 backbone and LoRA.

One iteration:
  1. roll out `tasks_per_iter` tasks x `G` trajectories, in lockstep batches
  2. build per-step records (return, discounted step return, anchor, action NLL)
  3. turn them into per-step advantages (advantages.py)
  4. one optimizer step on the clipped surrogate of Eq. 1, broadcasting each
     step's advantage to the tokens of that step's action

Because each batch of rollouts is consumed by exactly one optimizer step, the
update is on-policy: the importance ratio is 1 and the clip is inactive. It is
written out in full anyway so the loss is literally Eq. 1.
"""

import argparse
import json
import os
import random
import shutil
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import advantages as ADV
import env as E


# ----------------------------------------------------------------- rollout


def amp(model):
    """bf16 autocast for full fine-tuning, whose master weights are fp32 (see
    build_model); a no-op for the LoRA path, whose weights are bf16 already."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(getattr(model, "_amp", False)))


_TURN = "<|im_start|>user\n"


def encode_prompts(model, tok, texts, args):
    """Tokenize a batch of chat prompts, sharing the KV cache of their common prefix.

    In text mode every prompt starts with the same system message and few-shot
    turns (~1700 tokens), and only the final user turn differs. Re-prefilling
    that prefix for every sequence at every step dominated generation time, so
    it is run once at batch 1 and its cache is repeated across the batch. The
    layout is [prefix][left pad][suffix]; generate() derives position ids from
    the attention-mask cumsum, so the hole leaves every position exactly as in
    an unpadded prompt (pinned by test_prefix_cache_matches).

    Returns (enc, cache): enc has input_ids / attention_mask over the full
    sequence; cache is None when there is no shared prefix.
    """
    dev = model.device
    cut = texts[0].rfind(_TURN)
    prefix = texts[0][:cut] if cut > 0 else ""
    if not getattr(args, "prefix_cache", True) or not prefix or any(not t.startswith(prefix) for t in texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_prompt_len).to(dev)
        return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}, None
    pre = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    suf = tok([t[len(prefix) :] for t in texts], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.no_grad(), amp(model):
        cache = model(input_ids=pre, use_cache=True).past_key_values
    cache.batch_repeat_interleave(len(texts))
    B, P = len(texts), pre.shape[1]
    ids = torch.cat([pre.expand(B, P), suf.input_ids], dim=1)
    mask = torch.cat([torch.ones(B, P, dtype=suf.attention_mask.dtype, device=dev), suf.attention_mask], dim=1)
    assert ids.shape[1] <= args.max_prompt_len, (ids.shape, args.max_prompt_len)
    return {"input_ids": ids, "attention_mask": mask}, cache


def rollout(model, tok, pool, jobs, args, train: bool):
    """Run one lockstep batch of episodes.

    `jobs` is a list of (group_id, gamefile), at most one per runner in `pool`.
    All live episodes are stepped together so generation stays batched.

    Returns (steps, episodes). `steps` are the per-step records consumed by
    advantages.compute_advantages, each also carrying the token ids and the
    action NLL. `episodes` are per-episode summaries.
    """
    assert len(jobs) <= len(pool), (len(jobs), len(pool))
    envs = pool.load([gf for _, gf in jobs])
    hists = [[] for _ in jobs]
    rows = [[] for _ in jobs]  # per-episode list of step records
    rewards = [[] for _ in jobs]
    won = [False] * len(jobs)
    n_invalid = [0] * len(jobs)
    alive = list(range(len(jobs)))
    eos_ids = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id})

    while alive:
        batch = alive[: args.gen_batch]
        texts = [
            tok.apply_chat_template(
                E.build_prompt(envs[i], hists[i], history_len=args.history_len, mode=args.action_mode),
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in batch
        ]
        enc, cache = encode_prompts(model, tok, texts, args)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad(), amp(model):
            out = model.generate(
                **enc,
                past_key_values=cache,
                max_new_tokens=args.max_new_tokens,
                do_sample=train,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=train,
            )
        comp = out.sequences[:, plen:]
        if train:
            # temperature 1 / no truncation warpers => scores are the raw logits,
            # so this is exactly log pi_theta_old(a_t^j | s_t, a_t^<j).
            logits = torch.stack(out.scores, dim=1).float()
            logp = torch.log_softmax(logits, dim=-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        texts_out = tok.batch_decode(comp, skip_special_tokens=True)

        acts = []
        for k, i in enumerate(batch):
            ids = comp[k].tolist()
            # Keep tokens up to and including the first terminator.
            cut = len(ids)
            for j, tid in enumerate(ids):
                if tid in eos_ids:
                    cut = j + 1
                    break
            if args.action_mode == "think":
                act, _ = E.parse_think_action(texts_out[k], envs[i].admissible)
            elif args.action_mode == "text":
                act, _ = E.parse_text_action(texts_out[k], envs[i].admissible)
            else:
                act = E.parse_action(texts_out[k], envs[i].admissible)
            anchor = envs[i].anchor()
            prompt_ids = [t for t, m in zip(enc["input_ids"][k].tolist(), enc["attention_mask"][k].tolist()) if m]
            rec = {
                "group": jobs[i][0],
                "traj": i,
                "t": envs[i].t,
                "anchor": anchor,
                "prompt_ids": prompt_ids,
                "completion_ids": ids[:cut],
                "action": act,
            }
            if train:
                rec["nll"] = float(-logp[k, :cut].mean())
            rows[i].append(rec)
            acts.append(act)

        results = pool.step(batch, acts)
        envs = pool.views[: len(jobs)]
        for i, act, (obs, done, w, valid) in zip(batch, acts, results):
            hists[i].append((act, obs))
            won[i] = won[i] or w
            if not valid:
                n_invalid[i] += 1
            r = -args.invalid_penalty if not valid else 0.0
            if done and won[i]:
                r += 1.0
            rewards[i].append(r)
            if done:
                alive.remove(i)

    steps, episodes = [], []
    for i in range(len(jobs)):
        g_hat = ADV.discounted_step_returns(rewards[i], args.gamma)
        ret = float(sum(rewards[i]))
        for k, rec in enumerate(rows[i]):
            rec["ret"] = ret
            rec["g_hat"] = float(g_hat[k])
            steps.append(rec)
        episodes.append(
            {
                "group": jobs[i][0],
                "won": bool(won[i]),
                "steps": len(rows[i]),
                "ret": ret,
                "invalid": n_invalid[i] / max(1, len(rows[i])),
                "tokens": float(np.mean([len(r["completion_ids"]) for r in rows[i]])) if rows[i] else 0.0,
            }
        )
    return steps, episodes


# ----------------------------------------------------------------- update


def _pad_batch(recs, pad_id, device):
    """[left-pad prompt][right-pad completion] so every completion starts at the
    same offset; then only the last max_Lc+1 logits are needed."""
    lp = max(len(r["prompt_ids"]) for r in recs)
    lc = max(len(r["completion_ids"]) for r in recs)
    ids, mask, cmask = [], [], []
    for r in recs:
        p, c = r["prompt_ids"], r["completion_ids"]
        ids.append([pad_id] * (lp - len(p)) + p + c + [pad_id] * (lc - len(c)))
        mask.append([0] * (lp - len(p)) + [1] * len(p) + [1] * len(c) + [0] * (lc - len(c)))
        cmask.append([1] * len(c) + [0] * (lc - len(c)))
    t = lambda x, d=torch.long: torch.tensor(x, dtype=d, device=device)
    return t(ids), t(mask), t(cmask, torch.float32), lp, lc


def _token_logps(model, ids, mask, pos, lp, lc):
    with amp(model):
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=lc + 1, use_cache=False)
    tgt = ids[:, lp : lp + lc]
    return torch.log_softmax(out.logits[:, :-1].float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)


def surrogate_loss(model, recs, adv, pad_id, device, clip_eps, beta_kl=0.0):
    """Token-level clipped surrogate of Eq. 1 for one microbatch of step rows."""
    ids, mask, cmask, lp, lc = _pad_batch(recs, pad_id, device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)  # left padding must not shift RoPE
    logp = _token_logps(model, ids, mask, pos, lp, lc)
    a = torch.as_tensor(adv, dtype=torch.float32, device=device).unsqueeze(1)
    ratio = torch.exp(logp - logp.detach())  # == 1 on-policy; keeps the clip exact
    per_tok = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    kl_val = 0.0
    if beta_kl > 0:
        # Reference policy = this model with the LoRA adapter switched off, so
        # it costs one extra forward pass and no extra parameters. k3 estimator
        # (Schulman): exp(d) - d - 1 with d = log pi_ref - log pi_theta, which
        # is non-negative and lower-variance than the plain log-ratio.
        # Full fine-tuning has no adapter to switch off and keeps a frozen bf16
        # copy of the initial weights instead (build_model).
        ref = model.__dict__.get("_ref")
        with torch.no_grad():
            if ref is not None:
                ref_logp = _token_logps(ref, ids, mask, pos, lp, lc)
            else:
                with model.disable_adapter():
                    ref_logp = _token_logps(model, ids, mask, pos, lp, lc)
        d = (ref_logp - logp).clamp(-10.0, 10.0)
        kl = torch.exp(d) - d - 1.0
        per_tok = per_tok + beta_kl * kl
        kl_val = float(((kl * cmask).sum(1) / cmask.sum(1).clamp(min=1)).mean().detach())
    per_row = (per_tok * cmask).sum(1) / cmask.sum(1).clamp(min=1)  # (1/|a_t|) sum_j
    mean_logp = float(((logp * cmask).sum(1) / cmask.sum(1).clamp(min=1)).mean().detach())
    return per_row.sum(), int(len(recs)), mean_logp, kl_val


def update(model, opt, steps, adv, args, device, pad_id):
    """One optimizer step over the collected step rows, microbatched.

    Advantages are always computed from *all* the steps of the iteration, so the
    credit assignment is unaffected; `rows_per_update` only caps how many rows
    enter the gradient, which keeps the cost per iteration bounded. The subsample
    is uniform over rows, so the gradient stays an unbiased estimate of the one
    over the full batch.
    """
    # Rows whose advantage is exactly zero contribute exactly zero gradient.
    # Under a pure sparse reward that is every row of a group in which no
    # rollout succeeded, so dropping them before subsampling doubles the number
    # of informative rows in the update rather than spending half the budget on
    # rows that cannot move the policy.
    order = [i for i in range(len(steps)) if abs(float(adv[i])) > 1e-8]
    random.shuffle(order)
    if args.rows_per_update and len(order) > args.rows_per_update:
        order = order[: args.rows_per_update]
    total = len(order)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    model.train()
    opt.zero_grad(set_to_none=True)
    loss_sum, lp_sum, kl_sum = 0.0, 0.0, 0.0
    for s in range(0, total, args.micro_bs):
        chunk = order[s : s + args.micro_bs]
        loss, n, mean_logp, kl = surrogate_loss(
            model, [steps[i] for i in chunk], [adv[i] for i in chunk], pad_id, device, args.clip_eps, args.beta_kl
        )
        (loss / total).backward()
        loss_sum += loss.item()
        lp_sum += mean_logp * n
        kl_sum += kl * n
    gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.eval()
    return loss_sum / max(1, total), float(gn), lp_sum / max(1, total), kl_sum / max(1, total)


# ----------------------------------------------------------------- checkpoints


def save_ckpt(path, model, opt, rng, it, cursor, best, elapsed):
    """Write a resumable checkpoint: adapter, optimizer, cursor and RNG state.

    Written to a temporary directory and swapped into place, so a crash during
    the write leaves the previous checkpoint intact.
    """
    tmp, old = path + ".tmp", path + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    if _is_full(model):  # fp32 master weights, so a resume is exact
        torch.save(model.state_dict(), os.path.join(tmp, "weights.pt"))
    else:
        model.save_pretrained(tmp)
    torch.save(
        {
            "opt": opt.state_dict(),
            "iter": it,
            "cursor": cursor,
            "best": best,
            "elapsed": elapsed,
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_gen": rng.bit_generator.state,
        },
        os.path.join(tmp, "state.pt"),
    )
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(path):
        os.rename(path, old)
    os.rename(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def _is_full(model):
    return bool(getattr(model, "_amp", False))


def save_policy(model, path):
    """Policy weights for evaluation: the adapter (LoRA) or a bf16 copy of the
    full model (3 GB rather than 6 GB of fp32 master weights; the disk is 32 GB)."""
    if _is_full(model):
        model.save_pretrained(path, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    else:
        model.save_pretrained(path)


def load_ckpt(path, model, opt, rng):
    if _is_full(model):
        model.load_state_dict(torch.load(os.path.join(path, "weights.pt"), map_location="cuda"))
    else:
        from peft import load_peft_weights, set_peft_model_state_dict

        set_peft_model_state_dict(model, load_peft_weights(path))
    st = torch.load(os.path.join(path, "state.pt"), weights_only=False, map_location="cpu")
    opt.load_state_dict(st["opt"])
    random.setstate(st["rng_py"])
    np.random.set_state(st["rng_np"])
    torch.set_rng_state(st["rng_torch"].cpu() if hasattr(st["rng_torch"], "cpu") else st["rng_torch"])
    rng.bit_generator.state = st["rng_gen"]
    return st["iter"], st["cursor"], st["best"], st.get("elapsed", 0.0)


# ----------------------------------------------------------------- main


def build_model(args, lora: bool = True, weights: str = ""):
    """lora=True: the trainable policy (LoRA or full, per args.finetune).
    lora=False: a frozen bf16 policy for evaluation, from `weights` if given
    (a full-model directory or a LoRA adapter directory) else the backbone."""
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    full = lora and getattr(args, "finetune", "lora") == "full"
    src = args.model
    if weights and weights != "base" and not os.path.exists(os.path.join(weights, "adapter_config.json")):
        src = weights  # a saved full model
    # Full fine-tuning keeps fp32 master weights and runs under bf16 autocast:
    # at the paper's lr of 1e-6 an update is ~1e-4 of a typical weight, below
    # bf16's ~4e-3 resolution, so pure-bf16 weights would round most updates away.
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.float32 if full else torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    # Qwen ships a sampling preset (temperature 0.7, top_p 0.8, top_k 20,
    # repetition_penalty 1.1). Those are logits processors: leaving them on
    # would mean the recorded log-probs are not the policy's and that sampling
    # is off-policy. Pin everything to the identity so out.scores are raw logits.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    model.generation_config.repetition_penalty = 1.0
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.renormalize_logits = False
    model.config.use_cache = True
    if not lora:
        if weights and os.path.exists(os.path.join(weights, "adapter_config.json")):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, weights)
            model.config.use_cache = True
        model.eval()
        return tok, model
    if full:
        model._amp = True
        ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
        ref = ref.to("cuda").eval().requires_grad_(False)
        model.__dict__["_ref"] = ref  # bypass nn.Module registration: not a submodule, never saved
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = True
        model.eval()
        return tok, model
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = True  # checkpointing turns this off; generation needs it
    model.eval()
    return tok, model


def evaluate(model, tok, pool, tasks, args, detail=False):
    jobs = [(i, g["gamefile"]) for i, g in enumerate(tasks)]
    eps = []
    for s in range(0, len(jobs), args.eval_batch):
        _, e = rollout(model, tok, pool, jobs[s : s + args.eval_batch], args, train=False)
        eps += e
    out = {
        "success": float(np.mean([x["won"] for x in eps])),
        "steps": float(np.mean([x["steps"] for x in eps])),
        "invalid": float(np.mean([x["invalid"] for x in eps])),
        "tokens_per_step": float(np.mean([x["tokens"] for x in eps])),
        "n": len(eps),
    }
    if detail:
        # Per-task outcomes on a fixed task order let the methods be compared
        # pairwise (McNemar) instead of as two independent rates.
        out["per_task"] = [
            {"task_type": t["task_type"], "won": bool(e["won"]), "steps": e["steps"]}
            for t, e in zip(tasks, eps)
        ]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="gaca", choices=["grpo", "gigpo", "fixed", "gaca"])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="runs/run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--tasks_per_iter", type=int, default=2)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--invalid_penalty", type=float, default=0.0)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--norm", default="mean_std", choices=["mean", "mean_std"])
    # GACA hyperparameters (Table 4)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lam0", type=float, default=0.5)
    p.add_argument("--alpha_lam", type=float, default=0.5)
    p.add_argument("--beta_lam", type=float, default=0.5)
    p.add_argument("--alpha_c", type=float, default=1.0)
    p.add_argument("--score_mode", default="entropy", choices=["entropy", "uniform", "random", "loo"])
    # optimization
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--finetune", default="lora", choices=["lora", "full"])
    p.add_argument("--micro_bs", type=int, default=8)
    p.add_argument("--rows_per_update", type=int, default=96, help="cap on step rows per optimizer step (0=all)")
    p.add_argument("--gen_batch", type=int, default=16)
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=0, help="0 = 8 for index actions, 24 for text")
    p.add_argument("--action_mode", default="index", choices=["index", "text", "think"])
    p.add_argument("--max_prompt_len", type=int, default=0, help="0 = 2048 for index actions, 4096 for text (few-shot turns)")
    p.add_argument("--history_len", type=int, default=10)
    # data
    p.add_argument("--task_types", default="pick_and_place_simple,look_at_obj_in_light")
    p.add_argument("--max_adm0", type=int, default=0, help="keep tasks with at most this many actions at reset (0=all)")
    p.add_argument("--n_train", type=int, default=256)
    p.add_argument("--n_eval", type=int, default=48)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--time_budget", type=float, default=0.0, help="stop after this many hours (0=off)")
    p.add_argument("--eval_only", default="", help="path to a saved LoRA adapter: evaluate it and exit")
    p.add_argument("--ckpt_every", type=int, default=10, help="write a resumable checkpoint every N iterations")
    p.add_argument("--resume", action="store_true", help="continue from <out>_ckpt if it exists")
    args = p.parse_args()
    if not args.max_new_tokens:
        args.max_new_tokens = {"think": 512, "text": 24}.get(args.action_mode, 8)
    if not args.max_prompt_len:
        args.max_prompt_len = 4096 if args.action_mode == "text" else 2048  # think: the paper's 2048

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    types = args.task_types.split(",")
    # A bounded task pool: TextWorld leaks about 1 MB per distinct game loaded,
    # so the number of distinct games a run touches has to stay bounded.
    train_pool = E.list_games("train", task_types=types)
    random.Random(1234).shuffle(train_pool)
    train_tasks = E.probe_games(train_pool[: args.n_train])
    eval_all = E.list_games("valid_seen", task_types=types) + E.list_games("valid_unseen", task_types=types)
    random.Random(1234).shuffle(eval_all)
    eval_tasks = E.probe_games(eval_all)  # probe all, then filter, then take n_eval
    if args.max_adm0:
        train_tasks = [g for g in train_tasks if g["n_adm0"] <= args.max_adm0]
        eval_tasks = [g for g in eval_tasks if g["n_adm0"] <= args.max_adm0]
    eval_tasks = eval_tasks[: args.n_eval]

    if args.eval_only:
        # "base" evaluates the untrained backbone (zero-shot baseline).
        tok, model = build_model(args, lora=False, weights=args.eval_only)
        pool = E.make_pool(args.eval_batch, eval_tasks[0]["gamefile"], args.max_steps)
        ev = evaluate(model, tok, pool, eval_tasks, args, detail=True)
        ev.update(event="eval_only", adapter=args.eval_only, method=args.method)
        with open(args.out + ".eval.json", "w") as f:
            json.dump(ev, f, indent=1)
        print(json.dumps({k: v for k, v in ev.items() if k != "per_task"}))
        return

    tok, model = build_model(args)
    pool = E.make_pool(max(args.tasks_per_iter * args.G, args.eval_batch), train_tasks[0]["gamefile"], args.max_steps)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.finetune == "full":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    log_path = args.out + ".jsonl"
    log = open(log_path, "a", buffering=1)
    meta = {
        "event": "start",
        "args": vars(args),
        "n_train_tasks": len(train_tasks),
        "n_eval_tasks": len(eval_tasks),
        "trainable_params": trainable,
        "time": time.time(),
    }
    log.write(json.dumps(meta) + "\n")
    print(json.dumps(meta), flush=True)

    t_start = time.time()
    cursor = 0
    best = -1.0
    start_it = 0
    ckpt_dir = args.out + "_ckpt"
    if args.resume and os.path.isdir(ckpt_dir):
        start_it, cursor, best, elapsed = load_ckpt(ckpt_dir, model, opt, rng)
        t_start -= elapsed  # keep --time_budget measuring total training time
        msg = {"event": "resume", "iter": start_it, "cursor": cursor, "best": best, "elapsed_h": elapsed / 3600}
        log.write(json.dumps(msg) + "\n")
        print(json.dumps(msg), flush=True)

    for it in range(start_it, args.iters):
        if args.time_budget and (time.time() - t_start) / 3600 > args.time_budget:
            log.write(json.dumps({"event": "time_budget_reached", "iter": it}) + "\n")
            break
        if it % args.eval_every == 0:
            t0 = time.time()
            ev = evaluate(model, tok, pool, eval_tasks, args)
            ev.update(event="eval", iter=it, wall=time.time() - t_start, eval_secs=time.time() - t0)
            log.write(json.dumps(ev) + "\n")
            print(json.dumps(ev), flush=True)
            if ev["success"] > best:
                best = ev["success"]
                save_policy(model, args.out + "_best")

        jobs = []
        for k in range(args.tasks_per_iter):
            g = train_tasks[(cursor + k) % len(train_tasks)]
            jobs += [(it * 100 + k, g["gamefile"])] * args.G
        cursor += args.tasks_per_iter

        t0 = time.time()
        steps, eps = rollout(model, tok, pool, jobs, args, train=True)
        t_roll = time.time() - t0

        t0 = time.time()
        adv, diag = ADV.compute_advantages(
            steps,
            method=args.method,
            norm=args.norm,
            w=args.w,
            lam0=args.lam0,
            alpha_lam=args.alpha_lam,
            beta_lam=args.beta_lam,
            alpha_c=args.alpha_c,
            score_mode=args.score_mode,
            rng=rng,
        )
        # An iteration in which every advantage is exactly zero (e.g. a GRPO
        # group where all rollouts failed identically) carries no gradient at
        # all, so the update is skipped. Methods are matched on iterations --
        # the rollout budget -- not on wall-clock, so this only saves time.
        if float(np.abs(adv).max()) < 1e-8:
            loss, gnorm, logp_mean, kl, skipped = 0.0, 0.0, 0.0, 0.0, 1
        else:
            loss, gnorm, logp_mean, kl = update(model, opt, steps, adv, args, model.device, tok.pad_token_id)
            skipped = 0
        t_upd = time.time() - t0

        rec = {
            "event": "train",
            "iter": it,
            "wall": time.time() - t_start,
            "t_rollout": t_roll,
            "t_update": t_upd,
            "n_steps": len(steps),
            "rows_used": 0 if skipped else min(int(np.sum(np.abs(adv) > 1e-8)), args.rows_per_update or len(steps)),
            "rows_informative": int(np.sum(np.abs(adv) > 1e-8)),
            "skipped": skipped,
            "train_success": float(np.mean([e["won"] for e in eps])),
            "train_ret": float(np.mean([e["ret"] for e in eps])),
            "ep_steps": float(np.mean([e["steps"] for e in eps])),
            "invalid": float(np.mean([e["invalid"] for e in eps])),
            "nll_mean": float(np.mean([s["nll"] for s in steps])),
            "loss": loss,
            "grad_norm": gnorm,
            "nll_update": -logp_mean,
            "kl_ref": kl,
            "vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            **{k: float(v) for k, v in diag.items()},
        }
        log.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

        if args.ckpt_every and (it + 1) % args.ckpt_every == 0:
            save_ckpt(ckpt_dir, model, opt, rng, it + 1, cursor, best, time.time() - t_start)

    ev = evaluate(model, tok, pool, eval_tasks, args)
    ev.update(event="eval", iter=args.iters, wall=time.time() - t_start, final=True)
    log.write(json.dumps(ev) + "\n")
    print(json.dumps(ev), flush=True)
    if ev["success"] > best:
        save_policy(model, args.out + "_best")
    save_policy(model, args.out + "_last")
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # run complete: the resume state is dead weight on a 32 GB disk
    log.close()


if __name__ == "__main__":
    main()
