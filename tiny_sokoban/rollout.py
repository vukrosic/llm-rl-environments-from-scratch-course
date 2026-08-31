"""Batched autoregressive interaction between the policy and Sokoban."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Categorical

from .bfs import shortest_distance
from .environment import ACTIONS, Puzzle, SokobanEnv, State, state_dict
from .model import TinyCausalPolicy
from .tokenizer import SokobanTokenizer


@dataclass
class StepRecord:
    state: dict[str, list[int]]
    action: str
    action_probabilities: dict[str, float]
    base_reward: float
    shaping_reward: float
    reward: float
    next_state: dict[str, list[int]]
    valid: bool
    pushed: bool
    solved: bool
    deadlock: bool


@dataclass
class EpisodeTrace:
    puzzle_id: str
    optimal_steps: int
    initial_state: dict[str, list[int]]
    steps: list[StepRecord] = field(default_factory=list)
    log_probs: list[Tensor] = field(default_factory=list, repr=False)
    entropies: list[Tensor] = field(default_factory=list, repr=False)
    truncated: bool = False

    @property
    def actions(self) -> list[str]:
        return [step.action for step in self.steps]

    @property
    def rewards(self) -> list[float]:
        return [step.reward for step in self.steps]

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def solved(self) -> bool:
        return bool(self.steps and self.steps[-1].solved)

    @property
    def invalid_actions(self) -> int:
        return sum(not step.valid for step in self.steps)

    @property
    def hit_deadlock(self) -> bool:
        return any(step.deadlock for step in self.steps)

    def json_dict(self) -> dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "optimal_steps": self.optimal_steps,
            "initial_state": self.initial_state,
            "actions": self.actions,
            "total_reward": self.total_reward,
            "solved": self.solved,
            "truncated": self.truncated,
            "invalid_actions": self.invalid_actions,
            "hit_deadlock": self.hit_deadlock,
            "steps": [
                {
                    "state": step.state,
                    "action": step.action,
                    "action_probabilities": step.action_probabilities,
                    "base_reward": step.base_reward,
                    "shaping_reward": step.shaping_reward,
                    "reward": step.reward,
                    "next_state": step.next_state,
                    "valid": step.valid,
                    "pushed": step.pushed,
                    "solved": step.solved,
                    "deadlock": step.deadlock,
                }
                for step in self.steps
            ],
        }


def rollout_batch(
    model: TinyCausalPolicy,
    puzzles: list[Puzzle] | tuple[Puzzle, ...],
    tokenizer: SokobanTokenizer,
    *,
    device: str | torch.device,
    max_steps: int = 20,
    greedy: bool = False,
    reward_mode: str = "sparse",
    shaping_scale: float = 0.5,
) -> list[EpisodeTrace]:
    if reward_mode not in {"sparse", "progress"}:
        raise ValueError("reward_mode must be sparse or progress")
    if not puzzles:
        return []
    envs = [SokobanEnv(puzzle, max_steps=max_steps) for puzzle in puzzles]
    states: list[State] = [env.reset() for env in envs]
    histories: list[list[str]] = [[] for _ in envs]
    traces = [
        EpisodeTrace(
            puzzle_id=puzzle.puzzle_id,
            optimal_steps=puzzle.optimal_steps,
            initial_state=state_dict(puzzle.initial_state),
        )
        for puzzle in puzzles
    ]
    active = list(range(len(envs)))

    while active:
        sequences = [tokenizer.encode(states[index], histories[index]) for index in active]
        batch = tokenizer.pad_batch(sequences, device=device)
        logits = model(batch.input_ids, batch.attention_mask)
        distribution = Categorical(logits=logits)
        action_indices = logits.argmax(dim=-1) if greedy else distribution.sample()
        log_probs = distribution.log_prob(action_indices)
        entropies = distribution.entropy()
        probabilities = logits.softmax(dim=-1).detach().cpu()

        next_active: list[int] = []
        for local_index, env_index in enumerate(active):
            action_index = int(action_indices[local_index].item())
            action = ACTIONS[action_index]
            before = states[env_index]
            after, base_reward, terminated, truncated, info = envs[env_index].step(action)
            shaping_reward = 0.0
            if reward_mode == "progress":
                before_distance = shortest_distance(before)
                after_distance = shortest_distance(after)
                if before_distance is not None:
                    effective_after = after_distance if after_distance is not None else max_steps + 5
                    shaping_reward = shaping_scale * (before_distance - effective_after)
            reward = float(base_reward) + shaping_reward
            traces[env_index].log_probs.append(log_probs[local_index])
            traces[env_index].entropies.append(entropies[local_index])
            traces[env_index].steps.append(
                StepRecord(
                    state=state_dict(before),
                    action=action,
                    action_probabilities={
                        candidate: float(probabilities[local_index, candidate_index].item())
                        for candidate_index, candidate in enumerate(ACTIONS)
                    },
                    base_reward=float(base_reward),
                    shaping_reward=float(shaping_reward),
                    reward=float(reward),
                    next_state=state_dict(after),
                    valid=bool(info["valid"]),
                    pushed=bool(info["pushed"]),
                    solved=bool(info["solved"]),
                    deadlock=bool(info["deadlock"]),
                )
            )
            states[env_index] = after
            histories[env_index].append(action)
            traces[env_index].truncated = bool(truncated)
            if not (terminated or truncated):
                next_active.append(env_index)
        active = next_active
    return traces
