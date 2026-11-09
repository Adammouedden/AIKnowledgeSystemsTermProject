# GACA vs GRPO on ALFWorld, and new credit-assignment treatments

Reproduction of **GACA** (Granularity-Adaptive Credit Assignment, ICLR 2026) against vanilla **GRPO** on
ALFWorld with Qwen2.5-1.5B-Instruct, followed by new treatments for adaptive credit assignment -- most
importantly **probe-weighted token credit**, which spreads each step's advantage over the tokens of its
reasoning according to where the decision was actually made.

**Start here:** `docs/RESULTS.md` (all results, generated), `docs/IDEAS.md` (the ideas and offline
evidence), `docs/DECISIONS.md` (every judgement call with its measurement), `docs/HANDOFF.md` (state).

```
docs/            RESULTS.md (generated), IDEAS.md, DECISIONS.md, HANDOFF.md, RESULTS_4060.md (history)
src/advantages.py  GRPO / GiGPO / GACA estimators, reliability gate, probe token weights (unit-tested)
src/measure.py     offline studies: MC advantage oracle, NLL-vs-entropy check, token flip-test oracle
src/report.py      builds docs/RESULTS.md and the figures (`python src/report.py verl`)
src/env.py, src/train.py   the original HF/LoRA pipeline (attempts 1-3, and zero-shot format tests)
src/test_*.py      test suites
verl/              experiment launchers for verl-agent + verl-agent.patch (our changes) + backup scripts
runs/verl/         verl-agent arms: train.log, per-episode validation outcomes, per-step dumps
runs/tokens/       token decision-relevance oracle data and analyses
runs/a100/, runs/measure/   zero-shot format tests, MC-oracle pilot
runs/4060/, legacy/4060/    attempts 1-3 (RTX 4060) logs and retired launchers
references/        the GACA paper and the project proposal
```

## Running

Experiments run on [verl-agent](https://github.com/langfengQ/verl-agent) at the commit in
`verl/VERL_AGENT_COMMIT` with `verl/verl-agent.patch` applied (it adds the `gaca` advantage estimator,
which calls `src/advantages.py`, plus probe token credit, episode dumps and a final-weights upload).

```bash
verl/run_arms.sh grpo gaca gaca_probe    # arms one after another, 34 iterations each
python src/report.py verl                # rebuild docs/RESULTS.md
python src/test_advantages.py            # estimator unit tests (CPU)
```

Every arm shares the paper's recipe (reasoning-plus-action prompt, 512-token responses, 50 steps, group
size 8, full fine-tuning, lr 1e-6, KL 0.01, invalid-action penalty 0.1) and differs only in
`algorithm.gaca.*`; all are evaluated on the same 128 games. The training budget is ~6% of the paper's;
see `docs/DECISIONS.md`.
