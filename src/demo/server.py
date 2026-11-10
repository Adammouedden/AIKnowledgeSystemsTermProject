"""Live Credit Debugger: runs a fresh rollout group of the COMMIT policy on demand.

The checkpoint stays loaded (vLLM for sampling, HF for per-token scores and the
probe). GET /api/run streams every lockstep environment step to the page as
server-sent events, then scores the group, computes the GRPO, GiGPO, GACA and
COMMIT advantages with src/advantages.py and sends the finished trace. One run at
a time; the page is the same debugger as the static demo, with a live bar.

    python src/demo/server.py --model models/ckpt/commit_s1 --port 17100
"""

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import make_trace as MT  # noqa: E402
import env as E  # noqa: E402

from fastapi import FastAPI, Query  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402

TYPES = {"pick_and_place_simple": "Put", "look_at_obj_in_light": "Examine in light",
         "pick_clean_then_place_in_recep": "Clean", "pick_heat_then_place_in_recep": "Heat",
         "pick_cool_then_place_in_recep": "Cool", "pick_two_obj_and_place": "Put two"}

app = FastAPI()
S = {}  # model state, filled in main()
LOCK = threading.Lock()


def label(gamefile):
    folder = gamefile.split("/")[-3]
    parts = folder.split("-")
    kind = TYPES.get(parts[0], parts[0])
    obj, recep = parts[1], parts[3] if len(parts) > 3 else ""
    return f"{kind}: {obj} → {recep}" if kind != "Examine in light" else f"Examine {obj} under a lamp"


def public(rec):
    return {k: v for k, v in rec.items() if k not in ("prompt_ids", "resp_ids")}


def finalize(gamefile, trajs, G, temperature, model_name):
    stats = MT.advantages(trajs)
    for tr in trajs:
        for s in tr["steps"]:
            for k in ("prompt_ids", "resp_ids"):
                s.pop(k, None)
            s["tok_nll"] = [round(x, 3) for x in s["tok_nll"]]
            s["w"] = [round(x, 3) for x in s["w"]]
    meta = {"model": model_name, "gamefile": "/".join(gamefile.split("/")[-3:-1]),
            "task": trajs[0]["steps"][0]["task"], "group": G, "temperature": temperature,
            "gamma": MT.GAMMA, "penalty": MT.PENALTY, "win_reward": MT.WIN, "lam0": 0.5, "alpha": 0.5, "beta": 0.5,
            "probe_every": 16, "clip": [0.25, 4.0]} | stats
    return {"meta": meta, "trajs": trajs}


def run_job(q, game, G, max_steps, temperature):
    gen = S["gen"]
    try:
        g = S["games"][game]["gamefile"]
        from vllm import SamplingParams
        gen.sp = SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, max_tokens=512)
        q.put({"type": "start", "game": game, "label": label(g), "group": G, "max_steps": max_steps,
               "temperature": temperature, "model": S["name"]})
        t0 = time.time()

        def on_step(t, recs):
            q.put({"type": "step", "t": t, "steps": [{"traj": i, **public(r), "won": bool(r["reward"] > 0)} for i, r in recs],
                   "elapsed": round(time.time() - t0, 1)})

        trajs = MT.rollout_group(gen, g, G, max_steps, on_step=on_step)
        q.put({"type": "phase", "name": "score", "wins": sum(t["won"] for t in trajs),
               "steps": sum(len(t["steps"]) for t in trajs), "elapsed": round(time.time() - t0, 1)})
        last = [0.0]

        def progress(k, n):
            if time.time() - last[0] > 0.5 or k == n:
                last[0] = time.time()
                q.put({"type": "progress", "done": k, "total": n})

        MT.score(gen.tok, S["model"], "cuda", trajs, hf=S["hf"], progress=progress)
        q.put({"type": "done", "trace": finalize(g, trajs, G, temperature, S["name"]), "elapsed": round(time.time() - t0, 1)})
    except Exception as e:  # report to the page instead of dying silently
        traceback.print_exc()
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        q.put(None)


@app.get("/", response_class=HTMLResponse)
def index():
    page = open(os.path.join(HERE, "template.html")).read()
    trace = open(os.path.join(HERE, "trace.json")).read().replace("</", "<\\/")
    return page.replace("/*TRACE*/", trace).replace("/*LIVE*/", "window.LIVE = true;")


@app.get("/api/games")
def games():
    return JSONResponse([{"idx": i, "label": label(g["gamefile"])} for i, g in enumerate(S["games"][:60])])


@app.get("/api/health")
def health():
    return {"ok": True, "busy": LOCK.locked(), "model": S.get("name")}


@app.get("/api/run")
def run(game: int = Query(-1), group: int = Query(8, ge=2, le=8), max_steps: int = Query(30, ge=5, le=50),
        temperature: float = Query(1.0, ge=0.1, le=1.5)):
    if not LOCK.acquire(blocking=False):
        return JSONResponse({"error": "A run is already in progress. Wait for it to finish."}, status_code=409)
    if game < 0 or game >= len(S["games"]):
        game = random.randrange(60)
    q = queue.Queue()

    def work():
        try:
            run_job(q, game, group, max_steps, temperature)
        finally:
            LOCK.release()

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            try:
                ev = q.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if ev is None:
                return
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ckpt/commit_s1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=17100)
    ap.add_argument("--gpu-util", type=float, default=0.45)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM
    import uvicorn
    S["model"], S["name"] = args.model, os.path.basename(args.model.rstrip("/"))
    S["gen"] = MT.Gen(args.model, "cuda", 0, 1.0, gpu_util=args.gpu_util)
    S["hf"] = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    games = E.list_games("train")
    random.Random(7).shuffle(games)
    S["games"] = games
    print(f"ready on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
