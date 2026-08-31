"""A dependency-free 6 x 6 one-box Sokoban environment.

The wall topology and title state match the animated puzzle on the first slide.
The API intentionally resembles Gymnasium without requiring Gymnasium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

Coord = tuple[int, int]
BOARD_SIZE: Final = 6
ACTIONS: Final = ("U", "D", "L", "R")
DELTAS: Final[dict[str, Coord]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

TITLE_WALLS: Final[frozenset[Coord]] = frozenset(
    {(0, col) for col in range(BOARD_SIZE)}
    | {(BOARD_SIZE - 1, col) for col in range(BOARD_SIZE)}
    | {(row, 0) for row in range(BOARD_SIZE)}
    | {(row, BOARD_SIZE - 1) for row in range(BOARD_SIZE)}
    | {(1, 4), (2, 4), (3, 4)}
)
FREE_CELLS: Final[tuple[Coord, ...]] = tuple(
    (row, col)
    for row in range(BOARD_SIZE)
    for col in range(BOARD_SIZE)
    if (row, col) not in TITLE_WALLS
)


@dataclass(frozen=True)
class State:
    player: Coord
    box: Coord
    goal: Coord

    def __post_init__(self) -> None:
        occupied = (self.player, self.box, self.goal)
        if any(cell in TITLE_WALLS for cell in occupied):
            raise ValueError("player, box, and goal must be on floor cells")
        if self.player == self.box:
            raise ValueError("player and box cannot occupy the same cell")


TITLE_STATE: Final = State(player=(1, 2), box=(2, 2), goal=(4, 4))
TITLE_ACTIONS: Final = tuple("DLDRURDLDR")


@dataclass(frozen=True)
class Puzzle:
    puzzle_id: str
    initial_state: State
    optimal_actions: tuple[str, ...]

    @property
    def optimal_steps(self) -> int:
        return len(self.optimal_actions)


def add_coord(left: Coord, right: Coord) -> Coord:
    return left[0] + right[0], left[1] + right[1]


def transition_state(state: State, action: str) -> tuple[State, bool, bool]:
    """Apply one action and return ``(next_state, valid, pushed)``."""

    if action not in DELTAS:
        raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")
    delta = DELTAS[action]
    destination = add_coord(state.player, delta)
    if destination in TITLE_WALLS:
        return state, False, False

    box = state.box
    pushed = False
    if destination == box:
        box_destination = add_coord(box, delta)
        if box_destination in TITLE_WALLS:
            return state, False, False
        box = box_destination
        pushed = True
    return State(player=destination, box=box, goal=state.goal), True, pushed


def is_deadlock(state: State) -> bool:
    """Detect irreversible non-goal corner deadlocks for the one-box board."""

    if state.box == state.goal:
        return False
    row, col = state.box
    up = (row - 1, col) in TITLE_WALLS
    down = (row + 1, col) in TITLE_WALLS
    left = (row, col - 1) in TITLE_WALLS
    right = (row, col + 1) in TITLE_WALLS
    return (up or down) and (left or right)


def state_dict(state: State) -> dict[str, list[int]]:
    return {
        "player": list(state.player),
        "box": list(state.box),
        "goal": list(state.goal),
    }


def render_ascii(state: State) -> str:
    rows: list[str] = []
    for row in range(BOARD_SIZE):
        cells: list[str] = []
        for col in range(BOARD_SIZE):
            cell = (row, col)
            if cell in TITLE_WALLS:
                symbol = "#"
            elif cell == state.box and cell == state.goal:
                symbol = "*"
            elif cell == state.player and cell == state.goal:
                symbol = "+"
            elif cell == state.player:
                symbol = "P"
            elif cell == state.box:
                symbol = "B"
            elif cell == state.goal:
                symbol = "G"
            else:
                symbol = "."
            cells.append(symbol)
        rows.append("".join(cells))
    return "\n".join(rows)


class SokobanEnv:
    """Deterministic one-box Sokoban with executable reward."""

    def __init__(self, puzzle: Puzzle, *, max_steps: int = 20) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.puzzle = puzzle
        self.max_steps = max_steps
        self.state = puzzle.initial_state
        self.steps = 0
        self.done = False

    def reset(self) -> State:
        self.state = self.puzzle.initial_state
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action: str) -> tuple[State, float, bool, bool, dict[str, object]]:
        if self.done:
            raise RuntimeError("episode is finished; call reset before stepping again")
        next_state, valid, pushed = transition_state(self.state, action)
        self.state = next_state
        self.steps += 1
        solved = self.state.box == self.state.goal
        reward = -0.1 + (1.0 if solved else 0.0) + (10.0 if solved else 0.0)
        terminated = solved
        truncated = self.steps >= self.max_steps and not solved
        self.done = terminated or truncated
        info: dict[str, object] = {
            "valid": valid,
            "pushed": pushed,
            "solved": solved,
            "deadlock": is_deadlock(self.state),
            "steps": self.steps,
        }
        return self.state, reward, terminated, truncated, info

    def render(self) -> str:
        return render_ascii(self.state)
