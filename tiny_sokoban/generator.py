"""Deterministic puzzle enumeration, stratification, and split manifests."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .bfs import shortest_solution
from .environment import FREE_CELLS, TITLE_STATE, Puzzle, State


def puzzle_id(state: State) -> str:
    return (
        f"p{state.player[0]}{state.player[1]}-"
        f"b{state.box[0]}{state.box[1]}-g{state.goal[0]}{state.goal[1]}"
    )


def enumerate_puzzles(*, min_steps: int = 1, max_steps: int = 12) -> list[Puzzle]:
    puzzles: list[Puzzle] = []
    for goal in FREE_CELLS:
        for box in FREE_CELLS:
            if box == goal:
                continue
            for player in FREE_CELLS:
                if player == box:
                    continue
                state = State(player=player, box=box, goal=goal)
                solution = shortest_solution(state)
                if solution is None or not (min_steps <= len(solution) <= max_steps):
                    continue
                puzzles.append(
                    Puzzle(
                        puzzle_id=puzzle_id(state),
                        initial_state=state,
                        optimal_actions=solution,
                    )
                )
    puzzles.sort(key=lambda item: (item.optimal_steps, item.puzzle_id))
    return puzzles


@dataclass(frozen=True)
class PuzzleSplits:
    train: tuple[Puzzle, ...]
    validation: tuple[Puzzle, ...]
    test: tuple[Puzzle, ...]

    def by_name(self, name: str) -> tuple[Puzzle, ...]:
        if name not in {"train", "validation", "test"}:
            raise KeyError(name)
        return getattr(self, name)


def split_puzzles(
    puzzles: list[Puzzle],
    *,
    split_seed: int = 1729,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> PuzzleSplits:
    by_distance: dict[int, list[Puzzle]] = {}
    for puzzle in puzzles:
        by_distance.setdefault(puzzle.optimal_steps, []).append(puzzle)

    train: list[Puzzle] = []
    validation: list[Puzzle] = []
    test: list[Puzzle] = []
    title_id = puzzle_id(TITLE_STATE)
    for distance, bucket in sorted(by_distance.items()):
        local = [item for item in bucket if item.puzzle_id != title_id]
        random.Random(split_seed + distance * 1009).shuffle(local)
        train_end = int(len(local) * train_fraction)
        validation_end = train_end + int(len(local) * validation_fraction)
        train.extend(local[:train_end])
        validation.extend(local[train_end:validation_end])
        test.extend(local[validation_end:])

    title = next((item for item in puzzles if item.puzzle_id == title_id), None)
    if title is None:
        raise RuntimeError("title puzzle was not generated")
    test.append(title)
    for split in (train, validation, test):
        split.sort(key=lambda item: (item.optimal_steps, item.puzzle_id))
    return PuzzleSplits(tuple(train), tuple(validation), tuple(test))


def build_default_splits(*, split_seed: int = 1729, max_steps: int = 12) -> PuzzleSplits:
    return split_puzzles(
        enumerate_puzzles(min_steps=1, max_steps=max_steps),
        split_seed=split_seed,
    )


def split_manifest(splits: PuzzleSplits) -> dict[str, object]:
    payload: dict[str, object] = {"format_version": 1, "splits": {}}
    split_data: dict[str, object] = {}
    for name in ("train", "validation", "test"):
        items = splits.by_name(name)
        split_data[name] = [
            {"puzzle_id": item.puzzle_id, "optimal_steps": item.optimal_steps}
            for item in items
        ]
    payload["splits"] = split_data
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_split_manifest(path: str | Path, splits: PuzzleSplits) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = split_manifest(splits)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(payload["sha256"])
