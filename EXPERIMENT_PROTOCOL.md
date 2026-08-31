# Frozen experiment protocol

Frozen before the five-seed confirmation outcomes were inspected.

## Question

After random-initialized sparse RL failed the feasibility gate, the bounded
confirmation question is: does 80 updates of executable RL fine-tuning improve
greedy solve rate on unseen one-box Sokoban states beyond the identical
BFS-imitation warm start?

This is a MacBook-scale mechanism study inspired by RAGEN. It is not a
numerical reproduction of RAGEN's pretrained model, optimizer, infrastructure,
or reported results.

## Environment and hidden verifier

- Board: the fixed 6 x 6 wall topology drawn on the title slide.
- State: player position, one box position, one goal position, and fixed walls.
- Actions: `U`, `D`, `L`, `R`.
- Transition: deterministic Sokoban movement and pushing.
- Base reward: `-0.1` per attempted action, `+1` when the box reaches the goal,
  and `+10` when solved.
- Termination: solved; truncation after 20 actions; deadlocks are recorded.
- Generator: enumerate solvable configurations and label shortest-path length
  with exact BFS.

The BFS oracle is not included in the policy input. It has three explicit
roles: generate shortest-action imitation examples from training states,
measure puzzle difficulty, and provide the RL-only scalar potential difference
`0.5 * (distance_before - distance_after)`. The board solution and distance are
never exposed as observations during RL interaction.

The exact title start state is never a training example. An earlier deck draft
used a legal but non-optimal 10-action illustration; BFS finds the 6-action
solution `DDLDRR`, which the final deck replays from the trained checkpoint.
Because the title state was inspected during pilots, it is a qualitative replay
only and is excluded from aggregate confirmation metrics.

## Policy

- Autoregressive causal decoder-only Transformer initialized from scratch.
- 19,876 trainable parameters: two blocks, width 32, four heads, feed-forward
  width 64.
- Input: row-major board tokens and an action-prediction token. No action
  history is retained, so the policy is Markov and each decision is state-only.
- Output: a distribution over the four action tokens.
- No pretrained checkpoint, external tokenizer, dataset, or model download.

## Paired conditions

For every seed, both reported conditions share the exact same random
initialization and the exact same behavior-cloned checkpoint.

1. `sft`: 30 epochs of cross-entropy on shortest-action pairs generated only
   from training states with solution length at most 10.
2. `sft_plus_rl`: start from that same SFT checkpoint, then perform 80 batches
   of 64 executable on-policy rollouts. Tasks are sampled uniformly by shortest
   distance 1-10. REINFORCE uses return-to-go, a batch baseline, an entropy
   coefficient of 0.005, and the hidden BFS-potential reward.

SFT supplies initial behavioral coverage; only the second condition actually
collects fresh trajectories from the environment and updates from their
rewards. This replacement was frozen after three random-policy RL pilots failed
to generalize and before the confirmation test outcomes were read.

## Confirmation freeze

- Config: `configs/final.json`
- Config SHA-256: `a29bba5a676e035d7da5c78ac6f17408fcb8143a73baa5b9831816ff8ac92c9c`
- Split seed: `314159`, not used by the pilots.
- Split SHA-256: `012b3f7886449d3073d365beb4851973d6f1508acc2ff9bb316f4f3c931e6098`
- Paired model seeds: `2001, 2002, 2003, 2004, 2005`.
- Test identifiers are disjoint from training and validation identifiers.

## Metrics and receipts

- Primary: greedy solve rate on all unseen test puzzles with optimal length
  1-10, excluding the title state.
- Secondary: overall solve rate including the one length-11 state; solve rate
  by distance; invalid-action rate; deadlock rate; excess steps when solved;
  and the predesignated title replay.
- Receipts: canonical config and split hashes, per-seed CSV/JSON, raw JSONL
  trajectories, training logs, runtime/device metadata, selected checkpoint
  hash, and serialized title replay.

## Allowed conclusion

The strongest allowed positive conclusion is that RL fine-tuning improved the
measured held-out solve rate for this fixed-wall, one-box puzzle family under
the frozen tiny-policy setup. The experiment cannot establish general Sokoban
competence, natural-language reasoning, transfer to new wall layouts, or
equivalence to frontier-scale agent training.
