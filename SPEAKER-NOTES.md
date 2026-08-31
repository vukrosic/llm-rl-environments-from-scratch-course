# Speaker notes

## 01 — Road to OpenAI

Open on the result, not a promise. This is the recorded greedy replay from the
checkpoint in the repository: six actions, optimal according to exact BFS, with
roughly 99% policy probability on each chosen action. The title state is a
qualitative artifact check, not part of the aggregate metric. [Source: S21]

## 02 — What the roles request

OpenAI's posting emphasizes RL environments, measurement reliability, and
continuous evaluation. Anthropic's emphasizes agentic worlds, rigorous
evaluation, and production training. We build a tiny public version of that
artifact; this is not their private stack and does not imply hiring. [Sources:
S12–S13]

## 03 — Six reproducible worlds

Puzzle, phone, customer service, code, science, and browser tasks share the
same abstract loop: state, action, transition, feedback, verifier. We choose
Sokoban because it is visual, executable, and cheap enough for repeated local
training. [Sources: S08, S14–S18]

## 04 — Recorded title replay

This is no longer an illustrative ten-action sequence. It is the serialized
seed-2001 policy replay: down, down, left, down, right, right. Every state,
four-way action distribution, reward, push flag, and terminal flag is in the
JSON artifact. [Source: S21]

## 05 — Phone environment

ToolSandbox shows that an environment need not look like a game. The agent
queries a contact, hits a real state-dependent error, enables cellular service,
and retries. Hidden milestones grade state changes, not eloquent prose. This is
a reconstruction of the paper example, not a live trajectory. [Source: S14]

## 06 — Harnessed agentic RL

The harness builds context, executes actions, and returns observations. The
trainer observes the resulting model calls and verifier outcomes. This
separation lets the same executable world support evaluation, trajectory
collection, and RL. [Source: S19]

## 07 — Capability boundary

Tasks always solved are already mastered; tasks always failed offer no positive
behavior to reinforce. Mixed-success tasks often provide the richest local
learning signal. The rows are explanatory examples. [Source: S19]

## 08 — Environment factory

EvoEnv shifts synthetic data from individual questions to code that can sample,
render, solve, and score unlimited instances. Admission checks execution,
meaning, difficulty, and novelty. [Source: S20]

## 09 — One generated episode

The model sees a subset-sum prompt while hidden code retains the planted answer
and checks the response. This solve–verify asymmetry is the same design pattern
we use with BFS in Sokoban. [Source: S20]

## 10 — What we built

Name the complete stack: world, oracle, policy, rollout recorder, and updater.
Nothing is downloaded except the already required PyTorch package. This is
small by design so viewers can inspect every layer. [Source: S21]

## 11 — Environment contract

Python owns the state transition. The model cannot declare that it moved or
solved the board; it can only emit one of four actions. BFS remains hidden and
acts as verifier and reward instrument. [Source: S21]

## 12 — Generate and split

Enumerating valid player, box, and goal positions produces 530 solvable starts.
We stratify by exact shortest distance and freeze disjoint train, validation,
and test identifiers. After pilots, confirmation moves to a new split seed.
[Source: S21]

## 13 — Tiny policy

The network is genuinely decoder-only: strict causal self-attention over a
41-token board prompt. “Language model” here means an autoregressive token
policy, not a pretrained English model. State-only input also removes a failure
mode where action history encouraged repetition. [Source: S21]

## 14 — Random-RL failure

Do not hide the failed approach. Three pilots showed that random weights plus
sparse or shaped online reward found training successes but collapsed on
held-out states. The lesson is behavioral coverage: reward cannot reinforce a
useful path the policy almost never produces. [Source: S21]

## 15 — SFT warm start

BFS generates local demonstrations only from training-state identifiers. SFT
teaches basic navigation but solves just 39.3% of frozen test states on average.
That leaves a measurable gap for online interaction. [Source: S21]

## 16 — Executable RL

Each seed adds 5,120 fresh rollouts. The hidden potential reward gives positive
credit when exact distance falls and negative credit when it rises. REINFORCE
updates action probabilities from returns; the policy never reads the answer.
[Source: S21]

## 17 — Frozen result

Across the same 90 unseen states, SFT averages 39.3% and SFT plus RL 52.0%.
Every paired seed improves. The interval is useful but small-n; the defensible
claim is limited to this setup. [Source: S21]

## 18 — Difficulty result

The largest descriptive gains occur at 8–10 optimal moves. Do not call this a
proven mechanism: there are only a few unique states in the longest bands, and
distance was not randomized as a causal treatment. [Source: S21]

## 19 — Replay receipt

Show that “trajectory” means more than an action string. The artifact includes
before/after state, all action probabilities, validity, push status, reward,
deadlock, and solved status for every step. [Source: S21]

## 20 — Reproduce on Mac

Tests, quick pilot, final run, and receipt audit are separate commands. The
quick path is about 15 seconds of model time; five final seeds total about 95
seconds on the development M4 CPU. No dataset or model download occurs.
[Source: S21]

## 21 — Next research

Change one axis at a time: topology, observation representation, reward, or
sampling. Preserve the split and evidence discipline. The portfolio signal is
not a flashy solve; it is a rebuildable environment plus measurements another
researcher can attack. [Source: S21]
