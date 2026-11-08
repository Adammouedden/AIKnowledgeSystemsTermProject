"""End-to-end checks of the rollout/update machinery on the real model.

The important one is test_logprob_alignment: it asserts that the log-probs the
update recomputes from the stored token ids match, to within bf16 noise, the
ones the rollout recorded while sampling. A mistake in the left/right padding,
the position ids or the `logits_to_keep` offset shows up here immediately.

Run: pixi run test-train
"""

import os, sys, random, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import advantages as ADV
import env as E
import train as T


class A:  # minimal stand-in for the argparse namespace
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    max_steps = 6
    gamma = 0.95
    invalid_penalty = 0.1
    gen_batch = 8
    eval_batch = 8
    max_new_tokens = 8
    max_prompt_len = 2048
    history_len = 10
    clip_eps = 0.2
    micro_bs = 4
    max_grad_norm = 1.0
    lora_r = 8
    lr = 1e-5
    rows_per_update = 0
    beta_kl = 0.0


def test_checkpoint_roundtrip():
    """Checkpoint save/load must restore adapter weights, optimizer state and
    RNG exactly. Runs on CPU with a tiny randomly-initialised model so it never
    competes with a training job for the GPU."""
    import copy
    import shutil
    import tempfile

    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    rng = np.random.default_rng(7)

    def step():
        ids = torch.randint(0, 64, (2, 8))
        loss = model(input_ids=ids, labels=ids).loss
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(3):
        step()
    d = tempfile.mkdtemp()
    path = os.path.join(d, "ck")
    T.save_ckpt(path, model, opt, rng, it=3, cursor=11, best=0.25, elapsed=123.0)
    assert os.path.exists(os.path.join(path, "state.pt")), "state.pt missing"

    # The reference trajectory: what the next two steps produce from here.
    want_w = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items() if "lora" in k})
    for _ in range(2):
        step()
    after_w = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k}
    want_draw = rng.random(3).tolist()
    want_py = [random.random() for _ in range(3)]

    # Perturb everything, then restore.
    for _ in range(5):
        step()
    rng.random(10); [random.random() for _ in range(10)]
    it, cursor, best, elapsed = T.load_ckpt(path, model, opt, rng)
    assert (it, cursor, best, elapsed) == (3, 11, 0.25, 123.0), (it, cursor, best, elapsed)
    got_w = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k}
    assert want_w.keys() and all(torch.allclose(want_w[k], got_w[k]) for k in want_w), "adapter weights not restored"
    assert rng.random(3).tolist() == want_draw, "numpy Generator state not restored"
    assert [random.random() for _ in range(3)] == want_py, "python RNG state not restored"

    # Optimizer moments restored too: replaying the same two steps from the
    # restored state must land on exactly the reference weights. step() draws
    # its inputs from the global torch RNG, which load_ckpt also restores, so
    # the replay sees the same batches.
    for _ in range(2):
        step()
    replay = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k}
    assert all(torch.allclose(after_w[k], replay[k], atol=1e-6) for k in after_w), \
        "replay after resume diverged -> optimizer state not restored"
    shutil.rmtree(d, ignore_errors=True)
    print("ok checkpoint save/load restores weights, optimizer moments and RNG")


def _setup():
    tok, model = T.build_model(A)
    games = E.list_games("train", task_types=["pick_and_place_simple"])
    random.Random(0).shuffle(games)
    pool = E.make_pool(8, games[0]["gamefile"], A.max_steps)
    return tok, model, games, pool


def test_rollout_shape(tok, model, games, pool):
    jobs = [(0, games[0]["gamefile"])] * 4 + [(1, games[1]["gamefile"])] * 4
    t0 = time.time()
    steps, eps = T.rollout(model, tok, pool, jobs, A, train=True)
    print(f"  rollout of {len(jobs)} episodes x <= {A.max_steps} steps in {time.time() - t0:.1f}s")
    assert len(eps) == 8
    assert all(1 <= e["steps"] <= A.max_steps for e in eps)
    assert len(steps) == sum(e["steps"] for e in eps)
    for s in steps:
        assert s["prompt_ids"] and s["completion_ids"]
        assert np.isfinite(s["nll"]) and s["nll"] >= 0
        assert len(s["anchor"]) == 16
        assert s["group"] in (0, 1)
    # Discounted step returns must be consistent with the stored trajectory return.
    for key in {(s["group"], s["traj"]) for s in steps}:
        tr = sorted([s for s in steps if (s["group"], s["traj"]) == key], key=lambda s: s["t"])
        assert [s["t"] for s in tr] == list(range(len(tr)))
        assert all(abs(s["ret"] - tr[0]["ret"]) < 1e-12 for s in tr)
    # All rollouts of a group start from the same state, so they share step-0 anchor.
    for gid in (0, 1):
        a0 = {s["anchor"] for s in steps if s["group"] == gid and s["t"] == 0}
        assert len(a0) == 1, a0
    print("ok rollout structure")
    return steps, eps


def test_logprob_alignment(tok, model, steps):
    """Recomputed log-probs must match the ones recorded while sampling.

    bf16 kernels are not bit-reproducible across different batch shapes, so the
    tolerance is loose; what pins the alignment down is that a deliberate
    one-token shift is an order of magnitude worse, and that the unpadded
    single-row path agrees with the padded batch.
    """
    recs = random.Random(0).sample(steps, min(12, len(steps)))
    want = np.array([r["nll"] for r in recs])

    def padded(shift=0, extra=0):
        ids, mask, cmask, lp, lc = T._pad_batch(recs, tok.pad_token_id, model.device)
        if extra:  # same tokens, different kernel shapes -> isolates bf16 noise
            pad = torch.full((ids.shape[0], extra), tok.pad_token_id, device=ids.device, dtype=ids.dtype)
            ids = torch.cat([pad, ids], 1)
            mask = torch.cat([torch.zeros_like(pad), mask], 1)
            lp += extra
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=lc + 2, use_cache=False)
            lg = out.logits[:, 1 - shift : -1 - shift] if shift else out.logits[:, 1:-1]
            logp = torch.log_softmax(lg.float(), -1).gather(-1, ids[:, lp : lp + lc].unsqueeze(-1)).squeeze(-1)
        return ((-logp * cmask).sum(1) / cmask.sum(1)).cpu().numpy()

    got = padded()
    err = np.abs(got - want)
    r = np.corrcoef(got, want)[0, 1]
    print(f"  padded batch vs sampled:   max |err| {err.max():.4f}  mean {err.mean():.4f}  corr {r:.5f}")

    # bf16 floor: identical tokens, different padding width, so any difference
    # here is pure kernel non-determinism and nothing to do with alignment.
    noise = np.abs(padded(extra=64) - got)
    print(f"  bf16 floor (repad by 64):  max |err| {noise.max():.4f}  mean {noise.mean():.4f}")
    assert err.mean() < max(0.1, 4 * noise.mean()), (err.mean(), noise.mean())
    assert r > 0.9, r  # reported as a diagnostic; the noise-floor check above is the real test

    shifted = padded(shift=1)
    serr = np.abs(shifted - want)
    print(f"  one-token-shifted control: mean |err| {serr.mean():.3f} (must be far worse)")
    assert serr.mean() > 8 * err.mean(), (serr.mean(), err.mean())

    # Unpadded, one row at a time: no left or right padding anywhere.
    single = []
    for rec in recs:
        one = torch.tensor([rec["prompt_ids"] + rec["completion_ids"]], device=model.device)
        with torch.no_grad():
            lg = model(input_ids=one, use_cache=False).logits[0, len(rec["prompt_ids"]) - 1 : -1]
            l = torch.log_softmax(lg.float(), -1).gather(
                -1, torch.tensor(rec["completion_ids"], device=lg.device).unsqueeze(-1)
            )
        single.append(float(-l.mean()))
    d = np.abs(np.array(single) - got)
    print(f"  unpadded single row vs padded batch: max |err| {d.max():.4f} mean {d.mean():.4f}")
    assert d.max() < 0.3, list(zip(single, got.tolist()))
    print("ok log-prob alignment (padding, position ids and logit offset)")


def test_loss_sign_and_grad(tok, model, steps):
    """Positive advantage must raise the log-prob of the taken action."""
    recs = steps[:4]
    model.train()  # gradient checkpointing is a no-op in eval mode
    # On-policy the ratio is exactly 1, so the clipped surrogate's *value* is
    # -sum_i A_i (its gradient is what carries grad log pi). Check both the
    # value and that flipping every advantage flips it exactly.
    vals = []
    for sign in (+1.0, -1.0):
        loss, n, _, _ = T.surrogate_loss(model, recs, [sign] * len(recs), tok.pad_token_id, model.device, A.clip_eps)
        vals.append(loss.item())
    assert abs(vals[0] + vals[1]) < 1e-3, vals
    assert abs(vals[0] - (-1.0 * len(recs))) < 1e-3, vals
    # Zero advantage gives zero loss and zero gradient.
    loss, _, _, _ = T.surrogate_loss(model, recs, [0.0] * len(recs), tok.pad_token_id, model.device, A.clip_eps)
    assert abs(loss.item()) < 1e-6
    # Gradients reach the LoRA parameters and nothing else.
    model.zero_grad(set_to_none=True)
    loss, _, _, _ = T.surrogate_loss(model, recs, [1.0] * len(recs), tok.pad_token_id, model.device, A.clip_eps)
    loss.backward()
    tr = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert tr, "no trainable parameters"
    assert all("lora" in n.lower() for n, _ in tr), [n for n, _ in tr][:3]
    assert all(p.grad is not None for _, p in tr), "some LoRA tensor is outside the graph"
    nz = [n for n, p in tr if p.grad.abs().sum() > 0]
    print(f"  {len(nz)}/{len(tr)} LoRA tensors received a non-zero gradient")
    # lora_B starts at zero, so on the very first backward only lora_B has a
    # non-zero gradient; lora_A gets one as soon as B has moved off zero.
    assert all("lora_B" in n for n in nz), nz[:3]
    assert len(nz) == sum(1 for n, _ in tr if "lora_B" in n), (len(nz), len(tr))
    model.zero_grad(set_to_none=True)
    model.eval()
    print("ok loss sign, zero-advantage no-op, gradient flow")


def test_update_direction(tok, model, steps):
    """A positive advantage must make the taken action more likely, and a
    negative one less likely. Checked by overfitting one step row, which keeps
    the effect far above the bf16 noise floor."""
    import copy
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    saved = copy.deepcopy(get_peft_model_state_dict(model))
    rec = [steps[0]]
    ids, mask, cmask, lp, lc = T._pad_batch(rec, tok.pad_token_id, model.device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)

    def nll():
        model.eval()
        with torch.no_grad():
            o = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=lc + 1, use_cache=False)
            l = (
                torch.log_softmax(o.logits[:, :-1].float(), -1)
                .gather(-1, ids[:, lp : lp + lc].unsqueeze(-1))
                .squeeze(-1)
            )
        return float(((-l * cmask).sum(1) / cmask.sum(1)).mean())

    out = {}
    for sign in (+1.0, -1.0):
        set_peft_model_state_dict(model, copy.deepcopy(saved))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        n0 = nll()
        for _ in range(5):
            T.update(model, opt, rec, np.array([sign]), A, model.device, tok.pad_token_id)
        out[sign] = (n0, nll())
    set_peft_model_state_dict(model, saved)
    print(f"  A=+1: NLL {out[+1.0][0]:.4f} -> {out[+1.0][1]:.4f}   (must fall)")
    print(f"  A=-1: NLL {out[-1.0][0]:.4f} -> {out[-1.0][1]:.4f}   (must rise)")
    assert out[+1.0][1] < out[+1.0][0] - 0.01, out[+1.0]
    assert out[-1.0][1] > out[-1.0][0] + 0.01, out[-1.0]
    # Restoring the adapter must restore the policy.
    assert abs(nll() - out[+1.0][0]) < 0.05, (nll(), out[+1.0][0])
    print("ok update moves the policy in the direction of the advantage")


def test_eval_is_greedy(tok, model, games, pool):
    """train=False must be deterministic."""
    A2 = type("A2", (A,), {"max_steps": 4})
    jobs = [(i, games[10 + i]["gamefile"]) for i in range(3)]
    a = T.rollout(model, tok, pool, jobs, A2, train=False)[0]
    b = T.rollout(model, tok, pool, jobs, A2, train=False)[0]
    assert len(a) == len(b)
    assert [s["completion_ids"] for s in a] == [s["completion_ids"] for s in b]
    assert "nll" not in a[0]
    print("ok evaluation rollouts are deterministic")


def test_kl_penalty(tok, model, steps):
    """KL to the reference policy must be exactly zero while the adapter is the
    identity (LoRA B is zero-initialised), and strictly positive once it is not.
    This is the sharpest available check that `disable_adapter()` really gives
    the reference policy and that the k3 estimator is wired up correctly."""
    import copy
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    recs = steps[:4]
    model.train()
    saved = copy.deepcopy(get_peft_model_state_dict(model))

    _, _, _, kl0 = T.surrogate_loss(model, recs, [0.0] * len(recs), tok.pad_token_id, model.device, 0.2, 0.01)
    print(f"  KL at adapter init (must be 0): {kl0:.3e}")
    assert kl0 == 0.0, kl0

    # beta_kl = 0 must skip the reference pass entirely and report nothing.
    _, _, _, kloff = T.surrogate_loss(model, recs, [0.0] * len(recs), tok.pad_token_id, model.device, 0.2, 0.0)
    assert kloff == 0.0

    # Move the adapter off the identity; the KL must become strictly positive,
    # and the loss at zero advantage must then be exactly beta * sum of KLs.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for _ in range(3):
        T.update(model, opt, recs, np.ones(len(recs)), A, model.device, tok.pad_token_id)
    model.train()  # update() leaves the model in eval mode, where checkpointing is off
    loss, _, _, kl1 = T.surrogate_loss(model, recs, [0.0] * len(recs), tok.pad_token_id, model.device, 0.2, 0.01)
    print(f"  KL after moving the adapter:    {kl1:.4f}  (loss at A=0: {loss.item():.6f})")
    assert kl1 > 1e-4, kl1
    assert abs(loss.item() - 0.01 * kl1 * len(recs)) < 1e-3 * max(1.0, abs(loss.item())), (loss.item(), kl1)

    set_peft_model_state_dict(model, saved)
    model.train()
    _, _, _, kl2 = T.surrogate_loss(model, recs, [0.0] * len(recs), tok.pad_token_id, model.device, 0.2, 0.01)
    assert kl2 == 0.0, kl2
    model.eval()
    print("ok KL penalty (zero at identity, positive after drift, restores)")


def test_advantages_on_real_rollout(steps):
    # Give half the trajectories a success so the advantages are not all zero,
    # while keeping the real anchor structure and real action NLLs.
    import copy

    steps = copy.deepcopy(steps)
    keys = sorted({(s["group"], s["traj"]) for s in steps})
    winners = set(keys[::2])
    for key in keys:
        tr = sorted([s for s in steps if (s["group"], s["traj"]) == key], key=lambda s: s["t"])
        rew = [0.0] * len(tr)
        if key in winners:
            rew[-1] = 1.0
        g = ADV.discounted_step_returns(rew, 0.95)
        for k, s_ in enumerate(tr):
            s_["ret"] = float(sum(rew))
            s_["g_hat"] = float(g[k])
    for m in ("grpo", "gigpo", "fixed", "gaca"):
        adv, d = ADV.compute_advantages(steps, m, norm="mean_std")
        assert adv.shape == (len(steps),) and np.all(np.isfinite(adv))
        assert d["adv_abs"] > 1e-3, (m, d)
        print(f"  {m:6s} |A|={d['adv_abs']:.3f} coverage={d['step_coverage']:.2f} "
              f"c={d['c_mean']:.2f}+-{d['c_std']:.2f} lam={d['lam_mean']:.2f}+-{d['lam_std']:.2f}")
    d = ADV.compute_advantages(steps, "gaca", norm="mean_std")[1]
    assert d["step_coverage"] > 0.2, d  # anchor grouping must actually fire
    print("ok advantages on a real rollout")


def _mem(tag):
    torch.cuda.empty_cache()
    print(f"    [{tag}] peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    test_checkpoint_roundtrip()
    if len(sys.argv) > 1 and sys.argv[1] == "ckpt":
        raise SystemExit(0)
    tok, model, games, pool = _setup()
    _mem("model loaded")
    steps, eps = test_rollout_shape(tok, model, games, pool)
    _mem("rollout")
    test_logprob_alignment(tok, model, steps)
    _mem("alignment")
    test_loss_sign_and_grad(tok, model, steps)
    _mem("loss+backward")
    test_kl_penalty(tok, model, steps)
    _mem("kl penalty")
    test_advantages_on_real_rollout(steps)
    test_eval_is_greedy(tok, model, games, pool)
    _mem("eval rollout")
    test_update_direction(tok, model, steps)
    _mem("full update")
    print("ALL TRAINER TESTS PASSED")
