"""Minimal on-policy REINFORCE and task samplers."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .environment import Puzzle
from .model import PolicyConfig, TinyCausalPolicy
from .rollout import EpisodeTrace, rollout_batch
from .tokenizer import SokobanTokenizer


@dataclass(frozen=True)
class TrainConfig:
    updates: int = 300
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    entropy_coefficient: float = 0.02
    gamma: float = 1.0
    shaping_scale: float = 0.5
    gradient_clip: float = 1.0
    max_steps: int = 20
    max_train_difficulty: int = 10
    initial_boundary: int = 2
    curriculum_threshold: float = 0.60
    curriculum_window_updates: int = 5
    curriculum_replay_fraction: float = 0.20
    log_every: int = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


class PuzzleSampler:
    def __init__(
        self,
        puzzles: list[Puzzle] | tuple[Puzzle, ...],
        *,
        condition: str,
        seed: int,
        config: TrainConfig,
    ) -> None:
        if condition not in {"uniform", "curriculum"}:
            raise ValueError("condition must be uniform or curriculum")
        self.condition = condition
        self.random = random.Random(seed)
        self.config = config
        self.by_distance: dict[int, list[Puzzle]] = {}
        for puzzle in puzzles:
            if puzzle.optimal_steps <= config.max_train_difficulty:
                self.by_distance.setdefault(puzzle.optimal_steps, []).append(puzzle)
        if not self.by_distance:
            raise ValueError("no puzzles fall within max_train_difficulty")
        self.distances = sorted(self.by_distance)
        self.minimum = min(self.distances)
        self.maximum = max(self.distances)
        self.boundary = min(max(config.initial_boundary, self.minimum), self.maximum)
        self.recent_solve_rates: list[float] = []

    def sample(self, count: int) -> list[Puzzle]:
        selected: list[Puzzle] = []
        for _ in range(count):
            if self.condition == "uniform":
                distance = self.random.choice(self.distances)
            else:
                near = [
                    distance
                    for distance in self.distances
                    if max(self.minimum, self.boundary - 1) <= distance <= self.boundary
                ]
                replay = [distance for distance in self.distances if distance < self.boundary - 1]
                if replay and self.random.random() < self.config.curriculum_replay_fraction:
                    distance = self.random.choice(replay)
                else:
                    distance = self.random.choice(near)
            selected.append(self.random.choice(self.by_distance[distance]))
        return selected

    def observe(self, traces: list[EpisodeTrace]) -> bool:
        if self.condition != "curriculum" or self.boundary >= self.maximum:
            return False
        solve_rate = sum(trace.solved for trace in traces) / len(traces)
        self.recent_solve_rates.append(solve_rate)
        window = self.config.curriculum_window_updates
        self.recent_solve_rates = self.recent_solve_rates[-window:]
        if (
            len(self.recent_solve_rates) == window
            and sum(self.recent_solve_rates) / window >= self.config.curriculum_threshold
        ):
            self.boundary = min(self.boundary + 1, self.maximum)
            self.recent_solve_rates.clear()
            return True
        return False


def returns_to_go(rewards: list[float], gamma: float) -> list[float]:
    running = 0.0
    result: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        result.append(running)
    return list(reversed(result))


def train_policy(
    *,
    policy_config: PolicyConfig,
    initial_state_dict: dict[str, torch.Tensor],
    tokenizer: SokobanTokenizer,
    train_puzzles: list[Puzzle] | tuple[Puzzle, ...],
    condition: str,
    sampler_condition: str,
    reward_mode: str,
    seed: int,
    config: TrainConfig,
    device: str | torch.device,
    log_path: str | Path | None = None,
) -> tuple[TinyCausalPolicy, dict[str, Any]]:
    set_seed(seed)
    device = torch.device(device)
    model = TinyCausalPolicy(policy_config).to(device)
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sampler = PuzzleSampler(
        train_puzzles,
        condition=sampler_condition,
        seed=seed + 7919,
        config=config,
    )
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for update in range(1, config.updates + 1):
        puzzles = sampler.sample(config.batch_size)
        traces = rollout_batch(
            model,
            puzzles,
            tokenizer,
            device=device,
            max_steps=config.max_steps,
            greedy=False,
            reward_mode=reward_mode,
            shaping_scale=config.shaping_scale,
        )
        flat_returns = [
            value
            for trace in traces
            for value in returns_to_go(trace.rewards, config.gamma)
        ]
        returns = torch.tensor(
            flat_returns,
            dtype=torch.float32,
            device=device,
        )
        advantages = returns - returns.mean()
        scale = advantages.std(unbiased=False)
        if float(scale.item()) > 1e-8:
            advantages = advantages / scale
        else:
            advantages = torch.zeros_like(advantages)
        flat_log_probs = torch.cat([torch.stack(trace.log_probs) for trace in traces])
        flat_entropies = torch.cat([torch.stack(trace.entropies) for trace in traces])
        policy_loss = -(advantages.detach() * flat_log_probs).mean()
        entropy = flat_entropies.mean()
        loss = policy_loss - config.entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        advanced = sampler.observe(traces)

        row = {
            "update": update,
            "condition": condition,
            "sampler_condition": sampler_condition,
            "reward_mode": reward_mode,
            "seed": seed,
            "loss": float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "gradient_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
            "mean_return_to_go": float(returns.mean().detach().cpu().item()),
            "mean_episode_return": sum(trace.total_reward for trace in traces) / len(traces),
            "solve_rate": sum(trace.solved for trace in traces) / len(traces),
            "mean_actions": sum(len(trace.steps) for trace in traces) / len(traces),
            "boundary": sampler.boundary,
            "boundary_advanced": advanced,
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in ("loss", "entropy", "gradient_norm", "mean_return_to_go")
        ):
            raise FloatingPointError(f"non-finite training row: {row}")
        history.append(row)

    runtime = time.perf_counter() - started
    summary = {
        "condition": condition,
        "sampler_condition": sampler_condition,
        "reward_mode": reward_mode,
        "seed": seed,
        "runtime_seconds": runtime,
        "updates": config.updates,
        "episodes": config.updates * config.batch_size,
        "final_boundary": sampler.boundary,
        "train_config": asdict(config),
        "history": history,
    }
    if log_path is not None:
        destination = Path(log_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
            encoding="utf-8",
        )
    return model, summary
