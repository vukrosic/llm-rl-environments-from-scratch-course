from tiny_sokoban.environment import TITLE_ACTIONS, TITLE_STATE, Puzzle, SokobanEnv, transition_state


def test_title_trajectory_solves_in_ten_steps() -> None:
    puzzle = Puzzle("title", TITLE_STATE, TITLE_ACTIONS)
    env = SokobanEnv(puzzle)
    total = 0.0
    for action in TITLE_ACTIONS:
        _, reward, terminated, _, _ = env.step(action)
        total += reward
    assert terminated
    assert env.state.box == env.state.goal
    assert env.steps == 10
    assert abs(total - 10.0) < 1e-9


def test_invalid_wall_move_keeps_state() -> None:
    next_state, valid, pushed = transition_state(TITLE_STATE, "U")
    assert next_state == TITLE_STATE
    assert not valid
    assert not pushed
