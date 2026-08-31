"""A tiny random-initialized causal Transformer policy for Sokoban.

The model consumes a tokenized board plus optional action history and predicts exactly
one of the four environment actions.  It deliberately has no dependency on
Hugging Face or pretrained weights so the complete experiment can run on a
MacBook with only PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PolicyConfig:
    """Architecture parameters saved alongside every checkpoint."""

    vocab_size: int
    max_seq_len: int = 64
    num_actions: int = 4
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 64
    dropout: float = 0.0
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be at least two")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


class TinyCausalPolicy(nn.Module):
    """Decoder-only Transformer with a four-way action head.

    ``nn.TransformerEncoder`` is used only as a convenient stack of masked
    self-attention blocks.  The strict upper-triangular attention mask makes it
    a decoder-only causal network: token ``t`` can attend only to ``<= t``.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.action_head = nn.Linear(config.d_model, config.num_actions)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def causal_mask(length: int, device: torch.device | str) -> Tensor:
        """Return a boolean mask where ``True`` blocks future attention."""

        return torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def _validate_inputs(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, time], got {input_ids.shape}")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if not attention_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one non-padding token")
        return attention_mask

    @staticmethod
    def _apply_legal_mask(logits: Tensor, legal_action_mask: Tensor | None) -> Tensor:
        if legal_action_mask is None:
            return logits
        mask = legal_action_mask.to(device=logits.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != logits.shape:
            raise ValueError(
                "legal_action_mask must have shape [num_actions] or "
                f"[batch, num_actions], got {mask.shape} for logits {logits.shape}"
            )
        if not mask.any(dim=-1).all():
            raise ValueError("every state must expose at least one legal action")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        legal_action_mask: Tensor | None = None,
        *,
        return_sequence: bool = False,
    ) -> Tensor:
        """Return action logits for the final observed token or every token.

        ``legal_action_mask`` applies to final-token logits.  Sequence logits
        are intentionally unmasked so they can be used to test causal behavior.
        """

        attention_mask = self._validate_inputs(input_ids, attention_mask)
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.blocks(
            hidden,
            mask=self.causal_mask(seq_len, input_ids.device),
            src_key_padding_mask=~attention_mask,
            is_causal=True,
        )
        sequence_logits = self.action_head(hidden)
        if return_sequence:
            return sequence_logits

        final_positions = attention_mask.long().sum(dim=1) - 1
        final_logits = sequence_logits[
            torch.arange(batch_size, device=input_ids.device), final_positions
        ]
        return self._apply_legal_mask(final_logits, legal_action_mask)

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def checkpoint_payload(self, **metadata: Any) -> dict[str, Any]:
        """Create a portable checkpoint dictionary containing architecture metadata."""

        return {
            "format_version": 1,
            "policy_config": asdict(self.config),
            "model_state_dict": self.state_dict(),
            "metadata": metadata,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | dict[str, Any],
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["TinyCausalPolicy", dict[str, Any]]:
        payload = (
            torch.load(checkpoint, map_location=map_location, weights_only=False)
            if isinstance(checkpoint, str)
            else checkpoint
        )
        model = cls(PolicyConfig(**payload["policy_config"]))
        model.load_state_dict(payload["model_state_dict"])
        return model, payload.get("metadata", {})


# A shorter alias keeps tutorial snippets readable.
CausalPolicy = TinyCausalPolicy
