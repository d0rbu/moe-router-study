#!/usr/bin/env python3
"""
Script to analyze accuracy statistics from delphi score files.

Takes an experiment directory, looks in delphi/scores subdirectory,
and computes accuracy statistics for each latent file.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def calculate_accuracy(score_data: list[dict[str, Any]]) -> float | None:
    """Calculate accuracy from score data."""
    total = len(score_data)

    assert total > 0, "No score data"

    correct = sum(item.get("correct", False) for item in score_data)

    return correct / total


def analyze_experiment(experiment_dir: Path) -> dict[str, Any]:
    """Analyze all latent score files in an experiment directory."""
    scores_dir = experiment_dir / "delphi" / "scores"

    assert scores_dir.exists(), f"Scores directory does not exist: {scores_dir}"

    # Find all files matching *_latent*.txt pattern
    latent_files = list(scores_dir.glob("*_latent*.txt"))

    assert latent_files, f"No files matching *_latent*.txt found in {scores_dir}"

    per_latent_accuracies = {}
    total_correct = 0
    total_samples = 0

    print(f"Found {len(latent_files)} latent score files")

    for latent_file in sorted(latent_files):
        with open(latent_file) as f:
            score_data = json.load(f)

        if not score_data:
            print(f"Warning: Empty or invalid file: {latent_file.name}")
            continue

        accuracy = calculate_accuracy(score_data)
        num_correct = sum(
            1 for item in score_data if item.get("correct", False) is True
        )
        num_total = len(score_data)

        per_latent_accuracies[latent_file.name] = {
            "accuracy": accuracy,
            "correct": num_correct,
            "total": num_total,
        }

        total_correct += num_correct
        total_samples += num_total

        print(f"  {latent_file.name}: {accuracy:.4f} ({num_correct}/{num_total})")

    overall_accuracy = total_correct / total_samples if total_samples > 0 else None

    return {
        "per_latent": per_latent_accuracies,
        "overall": {
            "accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "num_latents": len(per_latent_accuracies),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze accuracy statistics from delphi score files"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (e.g., out/olmoe-i_lmsys_batch_size=4096_num_epochs=1_steps=119000/4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file to save results to",
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")

    print(f"Analyzing experiment directory: {experiment_dir}")
    print()

    results = analyze_experiment(experiment_dir)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Number of latents analyzed: {results['overall']['num_latents']}")
    print(f"Total samples: {results['overall']['total_samples']}")
    print(f"Total correct: {results['overall']['total_correct']}")
    if results["overall"]["accuracy"] is not None:
        print(
            f"Overall accuracy: {results['overall']['accuracy']:.4f} ({results['overall']['accuracy'] * 100:.2f}%)"
        )
    else:
        print("Overall accuracy: N/A")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
