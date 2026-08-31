# Source and fidelity audit

## Source system

- Wang et al., **RAGEN: Understanding Self-Evolution in LLM Agents via
  Multi-Turn Reinforcement Learning**, arXiv:2504.20073:
  https://arxiv.org/abs/2504.20073
- Official repository: https://github.com/mll-lab-nu/RAGEN

The 6 x 6 one-box Sokoban setting and multi-turn policy/environment loop are
inspired by RAGEN's Figure 9 and Appendix C.1. The implementation, generated
state family, small model, experiment, and measured results in this repository
are independent.

## What is retained

- An autoregressive token policy alternates with an executable environment.
- The environment owns state transitions and rule-based reward.
- Training uses complete on-policy trajectories.
- Initial-state difficulty and sampling are explicit.
- Sokoban is represented as a small 6 x 6 one-box world.
- Held-out trajectories are evaluated greedily and retained.

## Intentional deviations

- 19,876-parameter random-initialized decoder-only Transformer, not
  Qwen2.5-0.5B-Instruct or another pretrained language model.
- A 17-symbol local vocabulary, not a natural-language tokenizer.
- Generated BFS behavior cloning before RL, used as a tiny local substitute for
  the behavioral coverage a pretrained instruction model supplies.
- REINFORCE with return-to-go, a batch baseline, entropy bonus, and hidden
  potential reward; not StarPO/PPO/GAE/veRL.
- One fixed wall topology with varied player, box, and goal positions.
- Local CPU execution, not distributed GPU training.

## Measured local result

The confirmation setup was frozen before outcomes at config hash
`a29bba5a676e035d7da5c78ac6f17408fcb8143a73baa5b9831816ff8ac92c9c`
and split hash
`012b3f7886449d3073d365beb4851973d6f1508acc2ff9bb316f4f3c931e6098`.

Across five paired seeds, SFT-only solved 39.3% and SFT+RL solved 52.0% of the
90 prespecified unseen distance-1–10 states, a +12.7 percentage-point mean
difference. Every paired seed improved. The title state was inspected during
pilots and is therefore excluded from this metric; it remains a qualitative
artifact check, solved in the six-step optimum by all final checkpoints.

## Claim boundary

This is a bounded mechanism reproduction and teaching artifact, not a
numerical replication of RAGEN. The result does not establish general Sokoban
competence, transfer to new wall layouts, natural-language reasoning, or a
causal explanation for all of the improvement. Exact receipts and uncertainty
are in `results/final/analysis.json`.
