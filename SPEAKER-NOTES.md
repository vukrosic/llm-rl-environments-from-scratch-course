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

## 10 — Our experiment

The left side separates the two training stages. BFS first supplies imitation
examples from training states so a random 19,876-parameter causal Transformer
can produce useful behavior. The treatment then creates 5,120 executable
on-policy episodes per seed and uses return-to-go REINFORCE with a batch
baseline, entropy bonus, and hidden BFS-distance shaping. The model receives
only board tokens and scalar reward—not the distance or answer path. Against
the identical SFT checkpoint, RL raises greedy solve rate on 90 unseen states
from 39.3% to 52.0%, with all five paired seeds improving. This is a bounded
mechanism study, not PPO, GRPO, or a full RAGEN replication. [Source: S21]
