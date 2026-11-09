# GACA vs GRPO on ALFWorld, and new credit-assignment treatments

Reproduction of **GACA** (Granularity-Adaptive Credit Assignment, ICLR 2026) against vanilla **GRPO** on
ALFWorld with Qwen2.5-1.5B-Instruct, followed by new treatments for adaptive credit assignment -- most
importantly **probe-weighted token credit**, which spreads each step's advantage over the tokens of its
reasoning according to where the decision was actually made.

**Start here:** `docs/RESULTS.md` (all results, generated), `docs/IDEAS.md` (the ideas and offline
evidence), `docs/DECISIONS.md` (every judgement call with its measurement), `docs/HANDOFF.md` (state).

```
docs/                 RESULTS.md (generated), IDEAS.md, DECISIONS.md, HANDOFF.md, RESULTS_4060.md (history)
src/advantages.py     GRPO / GiGPO / GACA estimators, reliability gate, probe token weights (unit-tested)
src/env.py            ALFWorld wrapper and prompts (used by the offline token studies)
src/measure.py        offline studies: NLL-vs-entropy check, token flip-test oracle, probe robustness
src/report.py         builds docs/RESULTS.md and the learning-curve figure
src/test_advantages.py  estimator unit tests
src/verl/             arm launchers, queue runner, HF backup, verl-agent.patch (our changes) + its base commit
src/runs/verl/        verl-agent arms: train.log, per-episode validation outcomes, per-step dumps, queue.txt
src/runs/tokens/      token decision-relevance oracle data and analyses
src/runs/a100/, src/runs/4060/   zero-shot format tests; attempts 1-3 logs
legacy/               code no longer used (4060 HF/LoRA pipeline, retired launchers); see legacy/README.md
references/           the GACA paper and the project proposal
```

## Running

Experiments run on [verl-agent](https://github.com/langfengQ/verl-agent) at the commit in
`src/verl/VERL_AGENT_COMMIT` with `src/verl/verl-agent.patch` applied (it adds the `gaca` advantage estimator,
which calls `src/advantages.py`, plus probe token credit, episode dumps and a final-weights upload).

```bash
src/verl/run_arms.sh grpo gaca gaca_probe    # arms one after another, 34 iterations each
python src/report.py                     # rebuild docs/RESULTS.md
python src/test_advantages.py            # estimator unit tests (CPU)
```

Every arm shares the paper's recipe (reasoning-plus-action prompt, 512-token responses, 50 steps, group
size 8, full fine-tuning, lr 1e-6, KL 0.01, invalid-action penalty 0.1) and differs only in
`algorithm.gaca.*`; all are evaluated on the same 128 games. The training budget is ~6% of the paper's;
see `docs/DECISIONS.md`.
