"""One-line progress report for the running training jobs."""

import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["grpo", "gaca", "grpo_mn", "gaca_mn"]


def main():
    now = time.strftime("%H:%M")
    parts = []
    running = None
    for m in ORDER:
        p = os.path.join(ROOT, "runs", "4060", f"{m}.jsonl")
        if not os.path.exists(p):
            parts.append(f"{m}:-")
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        tr = [r for r in recs if r.get("event") == "train"]
        ev = [r for r in recs if r.get("event") == "eval"]
        total = next((r["args"]["iters"] for r in recs if r.get("event") == "start"), 90)
        it = (tr[-1]["iter"] + 1) if tr else 0
        best = max((e["success"] for e in ev), default=0.0)
        fresh = os.path.getmtime(p) > time.time() - 900
        if fresh:
            running = m
        tag = f"{m}:{it}/{total} best_eval={100 * best:.1f}%"
        if tr:
            # Marginal rate over the recent iterations, not wall/iters: the
            # latter stays inflated for a long time by the initial evaluation
            # sitting in the numerator, which made early ETAs 2-4x too high.
            k = min(10, len(tr) - 1)
            if k > 0:
                rate = (tr[-1]["wall"] - tr[-1 - k]["wall"]) / k
            else:
                rate = tr[-1]["wall"]
            every = next((r["args"]["eval_every"] for r in recs if r.get("event") == "start"), 15)
            evals_left = max(0, (total - it) // max(1, every)) + 1
            eval_s = np.mean([e.get("eval_secs", 335) for e in ev if e.get("eval_secs")]) if ev else 335
            left = ((total - it) * rate + evals_left * eval_s) / 3600
            tag += f" {rate:.0f}s/it"
            if fresh:
                tag += f" ETA {left:.1f}h"
        parts.append(tag)
    done = all(os.path.exists(os.path.join(ROOT, "runs", "4060", f"{m}.jsonl")) for m in ORDER) and running is None
    print(f"[{now}] " + " | ".join(parts) + ("  ALL IDLE/DONE" if done else ""))


if __name__ == "__main__":
    main()
