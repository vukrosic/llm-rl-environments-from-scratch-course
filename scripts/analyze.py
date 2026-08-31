#!/usr/bin/env python3
"""Audit the frozen run and derive compact paired statistics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce import canonical_sha256
from tiny_sokoban.environment import TITLE_STATE
from tiny_sokoban.generator import puzzle_id


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result_dir = ROOT / "results" / "final"
    summary = read_json(result_dir / "summary.json")
    config = read_json(result_dir / "config.json")
    manifest = read_json(result_dir / "split_manifest.json")
    rows = list(csv.DictReader((result_dir / "per_seed_results.csv").open()))

    by_condition: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][int(row["seed"])] = row
    seeds = sorted(set(by_condition["sft"]) & set(by_condition["sft_plus_rl"]))
    differences = [
        float(by_condition["sft_plus_rl"][seed]["final_primary_solve_rate"])
        - float(by_condition["sft"][seed]["final_primary_solve_rate"])
        for seed in seeds
    ]
    mean_difference = statistics.mean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    # Student-t critical value for a prespecified five-pair, two-sided 95% interval.
    t_critical_df4 = 2.7764451051977987

    by_distance: dict[str, dict[str, float]] = {}
    distances: set[str] = set()
    metric_payloads: dict[str, list[dict]] = defaultdict(list)
    for condition in ("sft", "sft_plus_rl"):
        for seed in seeds:
            payload = read_json(result_dir / f"{condition}_seed{seed}_metrics.json")
            metric_payloads[condition].append(payload)
            distances.update(payload["by_optimal_steps"])
    for distance in sorted(distances, key=int):
        by_distance[distance] = {
            condition: statistics.mean(
                payload["by_optimal_steps"][distance]["solve_rate"]
                for payload in metric_payloads[condition]
            )
            for condition in ("sft", "sft_plus_rl")
        }

    split_ids = {
        name: {row["puzzle_id"] for row in manifest["splits"][name]}
        for name in ("train", "validation", "test")
    }
    title_id = puzzle_id(TITLE_STATE)
    artifact = ROOT / "artifacts" / "tiny_sokoban_model.pt"
    trajectory_files = sorted((result_dir / "trajectories").glob("*.jsonl"))
    trajectory_line_counts = {
        path.name: sum(1 for line in path.open(encoding="utf-8") if line.strip())
        for path in trajectory_files
    }

    checks = {
        "config_hash_matches": canonical_sha256(config) == summary["config_sha256"],
        "config_hash_file_matches": (result_dir / "config.sha256").read_text().strip()
        == summary["config_sha256"],
        "split_hash_matches": manifest["sha256"] == summary["split_sha256"],
        "splits_are_pairwise_disjoint": not (
            split_ids["train"] & split_ids["validation"]
            or split_ids["train"] & split_ids["test"]
            or split_ids["validation"] & split_ids["test"]
        ),
        "title_is_test_only": title_id in split_ids["test"]
        and title_id not in split_ids["train"]
        and title_id not in split_ids["validation"],
        "title_excluded_from_metrics": summary["evaluation_puzzles_evaluated"]
        == len(split_ids["test"]) - 1,
        "all_ten_metric_trajectory_files_have_91_rows": len(trajectory_line_counts) == 10
        and set(trajectory_line_counts.values()) == {91},
        "artifact_hash_matches": file_sha256(artifact)
        == summary["artifact_checkpoint_sha256"],
        "all_reported_title_replays_solved": all(
            row["title_solved"].lower() == "true" for row in rows
        ),
        "all_reported_title_paths_are_optimal": all(
            row["title_actions"] == "DDLDRR" for row in rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"receipt audit failed: {checks}")

    analysis = {
        "paired_primary_result": {
            "seeds": seeds,
            "sft_mean": summary["aggregate"]["sft"]["mean_final_primary_solve_rate"],
            "sft_plus_rl_mean": summary["aggregate"]["sft_plus_rl"][
                "mean_final_primary_solve_rate"
            ],
            "difference_by_seed": differences,
            "mean_difference": mean_difference,
            "sample_standard_deviation": statistics.stdev(differences),
            "t_95_percent_interval": [
                mean_difference - t_critical_df4 * standard_error,
                mean_difference + t_critical_df4 * standard_error,
            ],
            "all_five_differences_positive": all(value > 0 for value in differences),
            "exact_sign_test_one_sided_p": 0.5 ** len(differences),
            "exact_sign_test_two_sided_p": min(1.0, 2 * 0.5 ** len(differences)),
        },
        "mean_solve_rate_by_optimal_steps": by_distance,
        "receipt_checks": checks,
        "trajectory_line_counts": trajectory_line_counts,
    }
    destination = result_dir / "analysis.json"
    destination.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
