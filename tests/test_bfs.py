from tiny_sokoban.bfs import shortest_solution
from tiny_sokoban.environment import TITLE_ACTIONS, TITLE_STATE
from tiny_sokoban.generator import build_default_splits, puzzle_id


def test_bfs_matches_title_solution_length() -> None:
    solution = shortest_solution(TITLE_STATE)
    assert solution is not None
    assert len(solution) == 6
    assert len(TITLE_ACTIONS) == 10


def test_title_state_is_test_only() -> None:
    splits = build_default_splits()
    title_id = puzzle_id(TITLE_STATE)
    assert title_id not in {item.puzzle_id for item in splits.train}
    assert title_id not in {item.puzzle_id for item in splits.validation}
    assert title_id in {item.puzzle_id for item in splits.test}
