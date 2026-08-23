from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Displays a sample of segmentation prediction data."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=project_root / "data" / "segmentation_prediction.npz",
        help="NPZ file to inspect.",
    )
    parser.add_argument("--rows", type=int, default=5, help="Number of sample rows.")
    parser.add_argument("--columns", type=int, default=8, help="Number of sample columns.")
    return parser.parse_args()


def show_prediction_sample(input_path: Path, rows: int, columns: int) -> None:
    if rows < 1 or columns < 1:
        raise ValueError("--rows and --columns must be greater than zero.")

    with np.load(input_path, allow_pickle=False) as archive:
        required_fields = {"class_ids", "confidence"}
        missing_fields = required_fields.difference(archive.files)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required fields: {missing}")

        class_ids = archive["class_ids"]
        confidence = archive["confidence"]
        probabilities = archive.get("probabilities")
        sample_rows = min(rows, class_ids.shape[0])
        sample_columns = min(columns, class_ids.shape[1])

        print(f"File: {input_path}")
        displayed_fields = "class_ids, confidence"
        if probabilities is not None:
            displayed_fields += ", probabilities"
        print(f"Displayed fields: {displayed_fields}")
        print(f"class_ids: shape={class_ids.shape}, dtype={class_ids.dtype}")
        print(f"confidence: shape={confidence.shape}, dtype={confidence.dtype}")
        if probabilities is not None:
            print(
                f"probabilities: shape={probabilities.shape}, "
                f"dtype={probabilities.dtype}"
            )
        print(f"\nSample ({sample_rows}x{sample_columns}) of class_ids:")
        print(class_ids[:sample_rows, :sample_columns])
        print(f"\nSample ({sample_rows}x{sample_columns}) of confidence:")
        print(confidence[:sample_rows, :sample_columns])
        if probabilities is not None:
            print(f"\nSample ({sample_rows}x{sample_columns}) of probabilities:")
            print(probabilities[:sample_rows, :sample_columns])


def main() -> None:
    arguments = parse_arguments()
    show_prediction_sample(arguments.input, arguments.rows, arguments.columns)


if __name__ == "__main__":
    main()