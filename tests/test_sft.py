from tiny_sokoban.generator import build_default_splits
from tiny_sokoban.sft import build_sft_examples
from tiny_sokoban.tokenizer import SokobanTokenizer


def test_sft_examples_remain_inside_training_state_split() -> None:
    splits = build_default_splits()
    tokenizer = SokobanTokenizer(history_limit=0)
    examples = build_sft_examples(
        splits.train,
        tokenizer,
        max_puzzle_steps=3,
    )
    train_ids = {puzzle.puzzle_id for puzzle in splits.train}
    assert examples
    assert {example.puzzle_id for example in examples} <= train_ids
