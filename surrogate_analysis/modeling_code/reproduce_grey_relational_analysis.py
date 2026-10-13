"""Reproduce the 5 x 4 grey relational matrix used in manuscript Fig. 5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd




def minmax(values: np.ndarray) -> np.ndarray:
    lower = values.min(axis=0)
    span = values.max(axis=0) - lower
    if np.any(span == 0):
        raise ValueError("A constant column cannot be min-max normalized")
    return (values - lower) / span


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--distinguishing-coefficient", type=float, default=0.5)
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    reference_path = args.reference.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("Dataset or reference matrix is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    zeta = args.distinguishing_coefficient
    if not (0.0 < zeta <= 1.0):
        raise ValueError("The distinguishing coefficient must lie in (0, 1]")

    dataset = pd.read_csv(dataset_path)
    if dataset.shape != (3125, 9):
        raise ValueError(f"Expected a 3125 x 9 dataset, found {dataset.shape}")
    normalized = minmax(dataset.to_numpy(dtype=float))

    matrix = np.empty((5, 4), dtype=float)
    for i in range(5):
        for j in range(4):
            difference = np.abs(normalized[:, i] - normalized[:, 5 + j])
            delta_min = float(difference.min())
            delta_max = float(difference.max())
            coefficient = (delta_min + zeta * delta_max) / (
                difference + zeta * delta_max
            )
            matrix[i, j] = float(coefficient.mean())

    result = pd.DataFrame(matrix, index=dataset.columns[:5], columns=dataset.columns[5:])
    result.to_csv(output_dir / "fig05_grey_relational_matrix_reproduced.csv")

    reference = pd.read_csv(reference_path, index_col=0)
    reference = reference.loc[result.index, result.columns]
    difference = result - reference
    difference.to_csv(output_dir / "fig05_reproduction_difference.csv")
    max_abs_difference = float(np.abs(difference.to_numpy()).max())
    if max_abs_difference > 1e-12:
        raise AssertionError(f"Reproduction mismatch: {max_abs_difference}")

    long_rows: list[dict[str, object]] = []
    for row_name in result.index:
        for column_name in result.columns:
            value = float(result.loc[row_name, column_name])
            if value >= 0.9:
                level = "very high"
            elif value >= 0.8:
                level = "high"
            elif value >= 0.6:
                level = "moderate"
            else:
                level = "low"
            long_rows.append(
                {
                    "mesoscopic_parameter": row_name,
                    "macroscopic_response": column_name,
                    "grey_relational_degree": value,
                    "classification": level,
                }
            )
    long = pd.DataFrame(long_rows).sort_values(
        ["macroscopic_response", "grey_relational_degree"], ascending=[True, False]
    )
    long.to_csv(output_dir / "fig05_grey_relational_degrees_long.csv", index=False)



    print(result.to_csv())
    print(f"max_abs_difference={max_abs_difference:.3e}")


if __name__ == "__main__":
    main()
