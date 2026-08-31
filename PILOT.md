# Fail-fast pilot receipt

## Original sparse-reward pilot

- Config hash: `4b04746bb5784ce68e3679310054bb86c99126e19b5283d4f10299c8b82bec00`
- Split hash: `2eb0458bc90786a217850e698a38f2750118439bbc18057fd0c893f9b79ea555`
- Seed: `101`
- Device: Apple MPS
- Model: 101,380 parameters
- Budget: 40 updates x 32 episodes for each condition
- Raw local receipt: `results/pilot_sparse_sampling_failure/`

The curriculum produced training successes and advanced from shortest-path
boundary 2 to 3, but neither final greedy policy solved any held-out 8-10-step
puzzle. Uniform sampling solved 2.17% overall and curriculum solved 0% overall;
both scored 0% on the prespecified primary band. The title-board policies
collapsed to repeated single-direction actions.

This is a feasibility failure, not the tutorial's intended scientific result.
It shows that task sampling alone did not provide enough fine-grained credit for
this tiny random-initialized policy under the pilot budget. Before any final
confirmation run, the experiment was therefore revised to isolate executable
progress reward: sparse curriculum versus BFS-potential-shaped curriculum. Both
conditions retain identical task sampling, model initialization, optimizer,
and update budget. The BFS distance stays hidden and only contributes a scalar
potential difference to reward.

MPS also required roughly 54-58 seconds per 40-update condition. The revised
pilot benchmarks CPU before the final device is frozen.

## Reward-shaping feasibility pilots

The first CPU reward comparison used the same 101,380-parameter random policy,
80 updates x 32 episodes, and one seed. Sparse reward reached 3.26% overall
held-out solve rate and shaped reward reached 0%; both were 0% on 8-10-step
puzzles. The receipt is preserved at `results/pilot_reward_first/` with config
hash `cfaa8aae6ee2d8e9f128b6d098b761dea6e06e66ac875ae6bf931945de98a47d`.

A tuned shaped-only run increased the budget to 250 updates x 64 episodes and
lowered entropy pressure. It still solved 0% of held-out puzzles and collapsed
to a repeated action, despite successes inside sampled training episodes. The
receipt is preserved at `results/pilot_reward_tuned_failure/` with config hash
`747b09852fdae1610b859c4b7ee48f8f317d21d44241819ec5bf3c2ba990f79a`.

Together these pilots reject the tutorial claim that a tiny random policy can
reliably discover the behavior from sparse or shaped on-policy reward alone
under a convenient MacBook budget. They motivated a smaller state-only policy
and a generated imitation warm start, analogous in purpose—not scale—to
starting agent RL from a pretrained instruction model.

## Behavior-cloning + RL pilot

- Config hash: `b6b4e27254b7f709b2dcd423232afc19dfed12736f00b466c0ec70b017ded813`
- Split hash: `2eb0458bc90786a217850e698a38f2750118439bbc18057fd0c893f9b79ea555`
- Seed: `101`
- Device: CPU
- Model: 19,876 parameters
- SFT: 1,347 BFS-generated training examples, 30 epochs
- RL: 80 updates x 64 executable rollouts
- Raw local receipt: `results/pilot/`

The SFT checkpoint solved 57.53% of all 73 validation puzzles. RL fine-tuning
raised this to 73.97%, a +16.44 percentage-point pilot difference. Both stages
solved the excluded title state with the exact 6-action path `DDLDRR`. Runtime
was 4.75 seconds for SFT and 10.24 additional seconds for RL.

The pilot was used only to select architecture and budget. Confirmation uses a
new split seed (`314159`) and five new paired model seeds; its hashes and
metrics were frozen in `EXPERIMENT_PROTOCOL.md` before those test outcomes were
inspected.
