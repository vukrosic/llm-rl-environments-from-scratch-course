import copy

import torch

from tiny_sokoban.generator import build_default_splits
from tiny_sokoban.model import PolicyConfig, TinyCausalPolicy
from tiny_sokoban.reinforce import TrainConfig, train_policy
from tiny_sokoban.tokenizer import SokobanTokenizer


def test_one_update_is_finite_and_changes_parameters() -> None:
    tokenizer = SokobanTokenizer()
    policy_config = PolicyConfig(
        vocab_size=len(tokenizer),
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
    )
    base = TinyCausalPolicy(policy_config)
    initial = copy.deepcopy(base.state_dict())
    config = TrainConfig(updates=1, batch_size=8, max_train_difficulty=2)
    trained, summary = train_policy(
        policy_config=policy_config,
        initial_state_dict=initial,
        tokenizer=tokenizer,
        train_puzzles=build_default_splits().train,
        condition="shaped",
        sampler_condition="curriculum",
        reward_mode="progress",
        seed=7,
        config=config,
        device=torch.device("cpu"),
    )
    assert summary["updates"] == 1
    assert all(torch.isfinite(value).all() for value in trained.state_dict().values())
    assert any(not torch.equal(initial[key], trained.state_dict()[key]) for key in initial)
