"""Frozen greedy evaluation and trajectory serialization."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .environment import Puzzle
from .model import TinyCausalPolicy
from .rollout import EpisodeTrace, rollout_batch
from .tokenizer import SokobanTokenizer


def _metric_block(traces: list[EpisodeTrace]) -> dict[str, float | int]:
    if not traces:
        return {
            "count": 0,
            "solve_rate": 0.0,
            "mean_steps": 0.0,
            "mean_excess_steps_solved": 0.0,
            "invalid_action_rate": 0.0,
            "deadlock_rate": 0.0,
        }
    solved = [trace for trace in traces if trace.solved]
    total_actions = sum(len(trace.steps) for trace in traces)
    return {
        "count": len(traces),
        "solve_rate": sum(trace.solved for trace in traces) / len(traces),
        "mean_steps": sum(len(trace.steps) for trace in traces) / len(traces),
        "mean_excess_steps_solved": (
            sum(len(trace.steps) - trace.optimal_steps for trace in solved) / len(solved)
            if solved
            else 0.0
        ),
        "invalid_action_rate": (
            sum(trace.invalid_actions for trace in traces) / total_actions
            if total_actions
            else 0.0
        ),
        "deadlock_rate": sum(trace.hit_deadlock for trace in traces) / len(traces),
    }


def evaluate_model(
    model: TinyCausalPolicy,
    puzzles: list[Puzzle] | tuple[Puzzle, ...],
    tokenizer: SokobanTokenizer,
    *,
    device: str | torch.device,
    max_steps: int = 20,
    batch_size: int = 128,
    primary_min_steps: int = 8,
    primary_max_steps: int = 10,
) -> tuple[dict[str, Any], list[EpisodeTrace]]:
    model.eval()
    traces: list[EpisodeTrace] = []
    with torch.no_grad():
        for start in range(0, len(puzzles), batch_size):
            traces.extend(
                rollout_batch(
                    model,
                    list(puzzles[start : start + batch_size]),
                    tokenizer,
                    device=device,
                    max_steps=max_steps,
                    greedy=True,
                )
            )
    by_distance: dict[int, list[EpisodeTrace]] = defaultdict(list)
    for trace in traces:
        by_distance[trace.optimal_steps].append(trace)
    primary = [
        trace
        for trace in traces
        if primary_min_steps <= trace.optimal_steps <= primary_max_steps
    ]
    metrics: dict[str, Any] = {
        "overall": _metric_block(traces),
        "primary": {
            "min_optimal_steps": primary_min_steps,
            "max_optimal_steps": primary_max_steps,
            **_metric_block(primary),
        },
        "by_optimal_steps": {
            str(distance): _metric_block(items)
            for distance, items in sorted(by_distance.items())
        },
    }
    return metrics, traces
