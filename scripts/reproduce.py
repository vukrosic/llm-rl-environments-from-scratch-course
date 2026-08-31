#!/usr/bin/env python3
"""Run the tiny Sokoban behavior-cloning + RL experiment end to end."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tiny_sokoban.environment import TITLE_STATE
from tiny_sokoban.evaluation import evaluate_model
from tiny_sokoban.generator import (
    build_default_splits,
    puzzle_id,
    write_split_manifest,
)
from tiny_sokoban.model import PolicyConfig, TinyCausalPolicy
from tiny_sokoban.reinforce import TrainConfig, resolve_device, set_seed, train_policy
from tiny_sokoban.sft import SFTConfig, build_sft_examples, train_sft
from tiny_sokoban.tokenizer import SokobanTokenizer


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def select_evaluation_puzzles(
    puzzles: tuple,
    *,
    per_distance: int,
    seed: int,
) -> list:
    buckets: dict[int, list] = defaultdict(list)
    for puzzle in puzzles:
        buckets[puzzle.optimal_steps].append(puzzle)
    selected: list = []
    for distance, items in sorted(buckets.items()):
        local = sorted(items, key=lambda item: item.puzzle_id)
        random.Random(seed + distance * 3571).shuffle(local)
        chosen = local[:per_distance]
        selected.extend(chosen)
    selected.sort(key=lambda item: (item.optimal_steps, item.puzzle_id))
    return selected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(
    rows: list[dict[str, Any]],
    *,
    treatment: str,
    control: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    result: dict[str, Any] = {}
    for condition, items in sorted(grouped.items()):
        result[condition] = {
            "seeds": [item["seed"] for item in items],
            "mean_initial_primary_solve_rate": mean(
                item["initial_primary_solve_rate"] for item in items
            ),
            "mean_final_primary_solve_rate": mean(
                item["final_primary_solve_rate"] for item in items
            ),
            "mean_final_overall_solve_rate": mean(
                item["final_overall_solve_rate"] for item in items
            ),
            "mean_runtime_seconds": mean(item["runtime_seconds"] for item in items),
            "mean_final_boundary": mean(item["final_boundary"] for item in items),
        }
    if control in grouped and treatment in grouped:
        by_condition = {
            condition: {item["seed"]: item for item in items}
            for condition, items in grouped.items()
        }
        common = sorted(set(by_condition[control]) & set(by_condition[treatment]))
        differences = [
            by_condition[treatment][seed]["final_primary_solve_rate"]
            - by_condition[control][seed]["final_primary_solve_rate"]
            for seed in common
        ]
        result["paired_primary_difference"] = {
            "treatment": treatment,
            "control": control,
            "treatment_minus_control_by_seed": differences,
            "mean": mean(differences) if differences else 0.0,
        }
    return result


def learning_curve_svg(training_summaries: list[dict[str, Any]], path: Path) -> None:
    width, height = 1100, 560
    left, right, top, bottom = 90, 40, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    histories: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for summary in training_summaries:
        histories[summary["condition"]].append(summary["history"])
    max_update = max(len(history) for group in histories.values() for history in group)
    palette = ["#ffb347", "#51e2c2", "#ff765f", "#8da7ff"]
    colors = {
        condition: palette[index % len(palette)]
        for index, condition in enumerate(sorted(histories))
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="28" fill="#07151d"/>',
        '<text x="90" y="34" fill="#e9f7f3" font-family="system-ui" font-size="24" font-weight="800">Training solve rate</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_h * (1 - value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#27434f"/>')
        lines.append(f'<text x="{left-18}" y="{y+6:.1f}" text-anchor="end" fill="#8ba2ad" font-family="monospace" font-size="16">{value:.1f}</text>')
    for condition, group in sorted(histories.items()):
        points: list[str] = []
        for update in range(max_update):
            values = [history[update]["solve_rate"] for history in group if update < len(history)]
            if not values:
                continue
            x = left + plot_w * update / max(1, max_update - 1)
            y = top + plot_h * (1 - mean(values))
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline fill="none" stroke="{colors[condition]}" stroke-width="5" points="{" ".join(points)}"/>'
        )
    lines.extend(
        [
            f'<text x="{left}" y="{height-26}" fill="#8ba2ad" font-family="monospace" font-size="17">policy updates →</text>',
            *[
                f'<circle cx="{width-330 + index*145}" cy="32" r="7" fill="{colors[condition]}"/><text x="{width-315 + index*145}" y="38" fill="#e9f7f3" font-family="system-ui" font-size="17">{condition}</text>'
                for index, condition in enumerate(sorted(histories))
            ],
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = canonical_sha256(config)
    output_dir = ROOT / config["output_dir"]
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "config.json", config)
    (output_dir / "config.sha256").write_text(config_hash + "\n", encoding="utf-8")

    torch.set_num_threads(int(config.get("torch_num_threads", torch.get_num_threads())))
    tokenizer = SokobanTokenizer(history_limit=int(config.get("history_limit", 0)))
    splits = build_default_splits(split_seed=int(config["split_seed"]), max_steps=12)
    split_hash = write_split_manifest(output_dir / "split_manifest.json", splits)
    eval_config = config["evaluation"]
    evaluation_split = str(eval_config.get("split", "test"))
    title_id = puzzle_id(TITLE_STATE)
    title_puzzle = next(item for item in splits.test if item.puzzle_id == title_id)
    metric_candidates = tuple(
        item for item in splits.by_name(evaluation_split) if item.puzzle_id != title_id
    )
    evaluation_puzzles = select_evaluation_puzzles(
        metric_candidates,
        per_distance=int(eval_config["max_puzzles_per_distance"]),
        seed=int(config["split_seed"]) + 1,
    )
    device = resolve_device(config["device"])
    policy_config = PolicyConfig(vocab_size=len(tokenizer), **config["model"])
    train_config = TrainConfig(**config["training"])
    sft_config = SFTConfig(**config["sft"])
    sft_examples = build_sft_examples(
        splits.train,
        tokenizer,
        max_puzzle_steps=sft_config.max_puzzle_steps,
    )
    per_seed_rows: list[dict[str, Any]] = []
    training_summaries: list[dict[str, Any]] = []
    artifact_checkpoint: Path | None = None

    def evaluate_stage(model: TinyCausalPolicy) -> tuple[dict[str, Any], list, Any]:
        metrics, traces = evaluate_model(
            model,
            evaluation_puzzles,
            tokenizer,
            device=device,
            max_steps=int(eval_config["max_steps"]),
            batch_size=int(eval_config["batch_size"]),
            primary_min_steps=int(eval_config["primary_min_steps"]),
            primary_max_steps=int(eval_config["primary_max_steps"]),
        )
        _, title_traces = evaluate_model(
            model,
            [title_puzzle],
            tokenizer,
            device=device,
            max_steps=int(eval_config["max_steps"]),
            batch_size=1,
            primary_min_steps=1,
            primary_max_steps=12,
        )
        return metrics, traces, title_traces[0]

    def save_stage(
        *,
        model: TinyCausalPolicy,
        condition: str,
        seed: int,
        metrics: dict[str, Any],
        traces: list,
        title_trace: Any,
    ) -> Path:
        write_json(output_dir / f"{condition}_seed{seed}_metrics.json", metrics)
        write_jsonl(
            output_dir / "trajectories" / f"{condition}_seed{seed}.jsonl",
            [trace.json_dict() for trace in traces],
        )
        write_json(
            output_dir / f"{condition}_seed{seed}_title_trajectory.json",
            title_trace.json_dict(),
        )
        checkpoint = output_dir / "checkpoints" / f"{condition}_seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.checkpoint_payload(
                condition=condition,
                seed=seed,
                config_sha256=config_hash,
                split_sha256=split_hash,
                tokenizer_history_limit=tokenizer.history_limit,
            ),
            checkpoint,
        )
        return checkpoint

    for seed in config["seeds"]:
        set_seed(int(seed))
        base_model = TinyCausalPolicy(policy_config).to(device)
        initial_metrics, _, initial_title = evaluate_stage(base_model)
        write_json(
            output_dir / f"initial_seed{seed}_metrics.json",
            initial_metrics,
        )
        write_json(
            output_dir / f"initial_seed{seed}_title_trajectory.json",
            initial_title.json_dict(),
        )

        sft_summary = train_sft(
            base_model,
            sft_examples,
            tokenizer,
            config=sft_config,
            device=device,
            seed=int(seed),
        )
        write_json(output_dir / "training" / f"sft_seed{seed}.json", sft_summary)
        sft_metrics, sft_traces, sft_title = evaluate_stage(base_model)
        sft_checkpoint = save_stage(
            model=base_model,
            condition="sft",
            seed=int(seed),
            metrics=sft_metrics,
            traces=sft_traces,
            title_trace=sft_title,
        )
        per_seed_rows.append(
            {
                "seed": int(seed),
                "condition": "sft",
                "evaluation_split": evaluation_split,
                "initial_primary_solve_rate": initial_metrics["primary"]["solve_rate"],
                "final_primary_solve_rate": sft_metrics["primary"]["solve_rate"],
                "final_overall_solve_rate": sft_metrics["overall"]["solve_rate"],
                "runtime_seconds": sft_summary["runtime_seconds"],
                "sft_runtime_seconds": sft_summary["runtime_seconds"],
                "rl_runtime_seconds": 0.0,
                "final_boundary": 0,
                "title_solved": sft_title.solved,
                "title_actions": "".join(sft_title.actions),
            }
        )
        if config["artifact_condition"] == "sft" and int(seed) == int(config["artifact_seed"]):
            artifact_checkpoint = sft_checkpoint
            write_json(output_dir / "title_board_replay.json", sft_title.json_dict())

        sft_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in base_model.state_dict().items()
        }

        for condition_spec in config["conditions"]:
            condition = condition_spec["name"]
            model, training = train_policy(
                policy_config=policy_config,
                initial_state_dict=sft_state_dict,
                tokenizer=tokenizer,
                train_puzzles=splits.train,
                condition=condition,
                sampler_condition=condition_spec["sampler"],
                reward_mode=condition_spec["reward"],
                seed=int(seed),
                config=train_config,
                device=device,
                log_path=output_dir / "training" / f"{condition}_seed{seed}.jsonl",
            )
            training_summaries.append(training)
            metrics, traces, title_trace = evaluate_stage(model)
            checkpoint = save_stage(
                model=model,
                condition=condition,
                seed=int(seed),
                metrics=metrics,
                traces=traces,
                title_trace=title_trace,
            )
            if condition == config["artifact_condition"] and int(seed) == int(config["artifact_seed"]):
                artifact_checkpoint = checkpoint
                write_json(output_dir / "title_board_replay.json", title_trace.json_dict())

            per_seed_rows.append(
                {
                    "seed": int(seed),
                    "condition": condition,
                    "evaluation_split": evaluation_split,
                    "initial_primary_solve_rate": initial_metrics["primary"]["solve_rate"],
                    "final_primary_solve_rate": metrics["primary"]["solve_rate"],
                    "final_overall_solve_rate": metrics["overall"]["solve_rate"],
                    "runtime_seconds": sft_summary["runtime_seconds"] + training["runtime_seconds"],
                    "sft_runtime_seconds": sft_summary["runtime_seconds"],
                    "rl_runtime_seconds": training["runtime_seconds"],
                    "final_boundary": training["final_boundary"],
                    "title_solved": title_trace.solved,
                    "title_actions": "".join(title_trace.actions),
                }
            )

    csv_path = output_dir / "per_seed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(per_seed_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(per_seed_rows)
    comparison = config["primary_comparison"]
    aggregate_metrics = aggregate(
        per_seed_rows,
        treatment=comparison["treatment"],
        control=comparison["control"],
    )
    if training_summaries:
        learning_curve_svg(training_summaries, output_dir / "learning_curve.svg")

    checkpoint_hash = None
    if config.get("publish_artifact") and artifact_checkpoint is not None:
        artifact_dir = ROOT / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        destination = artifact_dir / "tiny_sokoban_model.pt"
        shutil.copy2(artifact_checkpoint, destination)
        checkpoint_hash = checkpoint_sha256(destination)
        shutil.copy2(output_dir / "title_board_replay.json", artifact_dir / "title_board_replay.json")

    summary = {
        "experiment": config["name"],
        "config_sha256": config_hash,
        "split_sha256": split_hash,
        "evaluation_split": evaluation_split,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_parameters": TinyCausalPolicy(policy_config).parameter_count(),
        "tokenizer_history_limit": tokenizer.history_limit,
        "sft_examples": len(sft_examples),
        "sft_config": config["sft"],
        "train_puzzles": len(splits.train),
        "validation_puzzles": len(splits.validation),
        "test_puzzles_total": len(splits.test),
        "evaluation_puzzles_total": len(metric_candidates),
        "evaluation_puzzles_evaluated": len(evaluation_puzzles),
        "title_excluded_from_aggregate_metrics": True,
        "artifact_checkpoint_sha256": checkpoint_hash,
        "aggregate": aggregate_metrics,
        "per_seed": per_seed_rows,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("quick", "final"), default="quick")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / f"{args.preset}.json"
    summary = run(config_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
