from tiny_sokoban.environment import TITLE_STATE
from tiny_sokoban.tokenizer import SokobanTokenizer


def test_encoding_contains_board_history_and_action_prompt() -> None:
    tokenizer = SokobanTokenizer(history_limit=20)
    ids = tokenizer.encode(TITLE_STATE, ["D", "L"])
    tokens = [tokenizer.id_to_token[index] for index in ids]
    assert tokens[:2] == ["<bos>", "<state>"]
    assert tokens[-4:] == ["<history>", "D", "L", "<action>"]
    assert tokens.count("P") == 1
    assert tokens.count("B") == 1
    assert tokens.count("G") == 1


def test_state_only_encoding_drops_action_history() -> None:
    tokenizer = SokobanTokenizer()
    ids = tokenizer.encode(TITLE_STATE, ["D", "L"])
    tokens = [tokenizer.id_to_token[index] for index in ids]
    assert tokens[-2:] == ["<history>", "<action>"]
