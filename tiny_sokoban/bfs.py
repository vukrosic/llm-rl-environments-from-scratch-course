"""Exact shortest-path solver used only by generation and evaluation."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

from .environment import ACTIONS, State, is_deadlock, transition_state


@lru_cache(maxsize=None)
def shortest_solution(state: State, max_depth: int = 40) -> tuple[str, ...] | None:
    if state.box == state.goal:
        return ()
    if is_deadlock(state):
        return None
    queue: deque[tuple[State, tuple[str, ...]]] = deque([(state, ())])
    visited = {(state.player, state.box)}
    while queue:
        current, actions = queue.popleft()
        if len(actions) >= max_depth:
            continue
        for action in ACTIONS:
            successor, valid, _ = transition_state(current, action)
            if not valid:
                continue
            key = successor.player, successor.box
            if key in visited:
                continue
            path = actions + (action,)
            if successor.box == successor.goal:
                return path
            visited.add(key)
            if not is_deadlock(successor):
                queue.append((successor, path))
    return None


def shortest_distance(state: State, max_depth: int = 40) -> int | None:
    solution = shortest_solution(state, max_depth=max_depth)
    return None if solution is None else len(solution)
