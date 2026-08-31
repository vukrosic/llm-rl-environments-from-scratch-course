"""Fixed vocabulary for board observations and autoregressive action history."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .environment import ACTIONS, BOARD_SIZE, TITLE_WALLS, State

TOKENS = (
    "<pad>",
    "<bos>",
    "<state>",
    "</state>",
    "<history>",
    "<action>",
    "#",
    ".",
    "P",
    "B",
    "G",
    "*",
    "+",
    "U",
    "D",
    "L",
    "R",
)


@dataclass(frozen=True)
class EncodedBatch:
    input_ids: Tensor
    attention_mask: Tensor


class SokobanTokenizer:
    def __init__(self, *, history_limit: int = 0) -> None:
        if history_limit < 0:
            raise ValueError("history_limit must be non-negative")
        self.history_limit = history_limit
        self.token_to_id = {token: index for index, token in enumerate(TOKENS)}
        self.id_to_token = dict(enumerate(TOKENS))
        self.pad_token_id = self.token_to_id["<pad>"]
        self.action_token_ids = tuple(self.token_to_id[action] for action in ACTIONS)

    def __len__(self) -> int:
        return len(TOKENS)

    def board_symbols(self, state: State) -> list[str]:
        symbols: list[str] = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell = row, col
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
                symbols.append(symbol)
        return symbols

    def encode(self, state: State, history: list[str] | tuple[str, ...]) -> list[int]:
        if any(action not in ACTIONS for action in history):
            raise ValueError("history contains an unknown action")
        retained_history = list(history[-self.history_limit :]) if self.history_limit else []
        tokens = (
            ["<bos>", "<state>"]
            + self.board_symbols(state)
            + ["</state>", "<history>"]
            + retained_history
            + ["<action>"]
        )
        return [self.token_to_id[token] for token in tokens]

    def pad_batch(
        self,
        sequences: list[list[int]],
        *,
        device: str | torch.device = "cpu",
    ) -> EncodedBatch:
        if not sequences:
            raise ValueError("cannot pad an empty batch")
        max_length = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), max_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention_mask[index, :length] = True
        return EncodedBatch(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def action_from_index(index: int) -> str:
        return ACTIONS[index]
