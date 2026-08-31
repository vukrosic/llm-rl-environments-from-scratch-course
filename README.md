# Road to OpenAI: Building an LLM RL Environment From Scratch

A complete MacBook-sized research tutorial: generate an executable Sokoban
world, hide an exact verifier behind it, build a 19,876-parameter
autoregressive decoder-only Transformer, create trajectories, warm-start the
policy with generated demonstrations, fine-tune it with online reinforcement
learning, and evaluate it on frozen unseen states.

No model weights, tokenizer, or dataset are downloaded. The HTML presentation
used in the tutorial is the repository's `index.html`.

## Measured result

The frozen five-seed comparison used the same behavior-cloned checkpoint for
both paired conditions:

| Condition | Mean solve rate on 90 unseen states |
|---|---:|
| SFT warm start | 39.3% |
| SFT + 80 RL updates | **52.0%** |
| Paired difference | **+12.7 percentage points** |

Every paired seed improved: +7.8, +16.7, +20.0, +8.9, and +10.0 percentage
points. A small-sample t-based 95% interval for the mean difference is +6.0 to
+19.3 points. The selected checkpoint also solves the excluded title board in
the BFS-optimal six actions, `DDLDRR`, with every state, probability, and
reward retained in `artifacts/title_board_replay.json`.

This result is deliberately narrow. It covers one fixed wall topology with one
box and symbolic observations; it is not evidence of general Sokoban or
natural-language reasoning.

## Run it

Python 3.11+ and PyTorch are required.

```bash
python -m pip install -e '.[dev]'  # only if PyTorch/pytest are not installed
```

```bash
python -m pytest
python scripts/reproduce.py --preset quick
python scripts/reproduce.py --preset final
python scripts/analyze.py
```

Evaluate the selected checkpoint and print its complete title replay:

```bash
python scripts/evaluate.py artifacts/tiny_sokoban_model.pt
```

The recorded run used CPU on an Apple M4. The one-seed pilot took about 15
seconds of model time; the five final seeds used about 95 cumulative seconds.
For this very small workload, CPU was faster than Apple MPS.

## What is being trained?

The policy receives a 41-token causal sequence encoding the current 6 x 6
board and predicts one of four action tokens: `U`, `D`, `L`, or `R`. Python—not
the model—owns movement, pushing, termination, and reward.

Training has two stages:

1. Exact BFS generates shortest-action pairs from training states. Thirty
   local behavior-cloning epochs give the random model enough navigation skill
   to produce useful trajectories.
2. The policy creates 5,120 fresh executable rollouts per seed. REINFORCE uses
   return-to-go and a hidden progress reward based on the change in BFS
   distance. The policy receives the scalar reward, never the solution or
   distance itself.

Three retained feasibility pilots show why the warm start was added: random
weights plus sparse or shaped online RL collapsed to repetitive actions and
did not generalize under the MacBook budget. See `PILOT.md`.

## Evidence map

- `EXPERIMENT_PROTOCOL.md` — question, controls, hashes, metrics, and claim
  boundary frozen before confirmation outcomes
- `configs/final.json` — exact final architecture and optimizer settings
- `results/final/summary.json` — aggregate and per-seed results
- `results/final/analysis.json` — paired interval, distance breakdown, and
  automated receipt checks
- `results/final/per_seed_results.csv` — compact paired table
- `results/final/trajectories/` — all 910 held-out evaluation trajectories
- `artifacts/tiny_sokoban_model.pt` — selected approximately 100 KB checkpoint
- `artifacts/title_board_replay.json` — complete recorded six-step replay
- `SOURCE_AUDIT.md` — exact relationship to RAGEN

Intermediate final checkpoints and update logs are reproducible but ignored by
Git to keep the public repository small.

## Repository map

- `tiny_sokoban/environment.py` — reset/step contract and Sokoban rules
- `tiny_sokoban/bfs.py` — exact hidden solver and distance verifier
- `tiny_sokoban/generator.py` — deterministic task enumeration and splits
- `tiny_sokoban/tokenizer.py` — 17-symbol local tokenizer
- `tiny_sokoban/model.py` — causal decoder-only Transformer
- `tiny_sokoban/sft.py` — generated behavior-cloning warm start
- `tiny_sokoban/rollout.py` — batched executable trajectories
- `tiny_sokoban/reinforce.py` — on-policy REINFORCE updates
- `tiny_sokoban/evaluation.py` — frozen greedy metrics
- `scripts/reproduce.py` — complete pilot/final pipeline
- `scripts/analyze.py` — evidence and hash audit
- `tests/` — mechanics, solver, tokenizer, causal mask, SFT, and RL checks

## Presentation

Open `index.html` in Chrome. Use Left/Right, A/D, Page Up/Page Down, or Space.
Add `?all=1` when rendering static frames.

```bash
npm install       # optional; only needed for slide rendering
npm run render    # 1920 x 1080 PNGs + contact sheets when ImageMagick exists
npm run audit     # viewport and overflow checks
```

## Research boundary

This project is inspired by
[RAGEN](https://arxiv.org/abs/2504.20073) and its
[official implementation](https://github.com/mll-lab-nu/RAGEN). RAGEN uses a
pretrained Qwen model and a much larger StarPO/PPO stack. This repository is a
bounded, independently implemented mechanism study and teaching artifact—not a
numerical replication of the paper's reported scores.
