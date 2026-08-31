#!/usr/bin/env python3
"""Evaluate a saved tiny Sokoban policy on the frozen test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tiny_sokoban.evaluation import evaluate_model
from tiny_sokoban.environment import TITLE_STATE
from tiny_sokoban.generator import build_default_splits, puzzle_id
from tiny_sokoban.model import TinyCausalPolicy
from tiny_sokoban.reinforce import resolve_device
from tiny_sokoban.tokenizer import SokobanTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, metadata = TinyCausalPolicy.from_checkpoint(str(args.checkpoint), map_location=device)
    model.to(device)
    final_config = json.loads((ROOT / "configs" / "final.json").read_text())
    split_seed = args.split_seed if args.split_seed is not None else int(final_config["split_seed"])
    tokenizer = SokobanTokenizer(
        history_limit=int(metadata.get("tokenizer_history_limit", final_config["history_limit"]))
    )
    splits = build_default_splits(split_seed=split_seed)
    title_id = puzzle_id(TITLE_STATE)
    measured_test = [item for item in splits.test if item.puzzle_id != title_id]
    title = next(item for item in splits.test if item.puzzle_id == title_id)
    metrics, _ = evaluate_model(
        model,
        measured_test,
        tokenizer,
        device=device,
        primary_min_steps=1,
        primary_max_steps=10,
    )
    _, title_traces = evaluate_model(model, [title], tokenizer, device=device)
    print(
        json.dumps(
            {
                "metadata": metadata,
                "split_seed": split_seed,
                "metrics": metrics,
                "title_replay": title_traces[0].json_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
