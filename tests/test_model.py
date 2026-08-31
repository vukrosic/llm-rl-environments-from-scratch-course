import torch

from tiny_sokoban.model import PolicyConfig, TinyCausalPolicy
from tiny_sokoban.tokenizer import SokobanTokenizer


def test_model_is_tiny_and_has_four_action_logits() -> None:
    tokenizer = SokobanTokenizer()
    model = TinyCausalPolicy(PolicyConfig(vocab_size=len(tokenizer)))
    logits = model(torch.tensor([[1, 2, 3, 5]], dtype=torch.long))
    assert logits.shape == (1, 4)
    assert model.parameter_count() == 19_876


def test_future_tokens_do_not_change_past_logits() -> None:
    tokenizer = SokobanTokenizer()
    model = TinyCausalPolicy(PolicyConfig(vocab_size=len(tokenizer), dropout=0.0)).eval()
    left = torch.tensor([[1, 2, 7, 5]], dtype=torch.long)
    right = torch.tensor([[1, 2, 7, 5, 13]], dtype=torch.long)
    with torch.no_grad():
        left_logits = model(left, return_sequence=True)
        right_logits = model(right, return_sequence=True)
    assert torch.allclose(left_logits[:, :4], right_logits[:, :4], atol=1e-6)
