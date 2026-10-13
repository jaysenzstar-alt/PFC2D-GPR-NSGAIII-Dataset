"""Reproduce the archived run-best-plus-tolerance dispersion analysis.

For the current target-common filter, use analysis_code/reanalyze_round3.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PARAMETERS = {
    "pb_emod_gpa": (15.0, 75.0),
    "pb_kra": (1.0, 3.0),
    "pb_ten_mpa": (15.0, 95.0),
    "pb_coh_mpa": (15.0, 35.0),
    "pb_fa_deg": (15.0, 75.0),
}

REQUIRED_COLUMNS = {
    "sample",
    "algorithm",
    "run",
    "mean_relative_error",
    *PARAMETERS.keys(),
}




def validate_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def rankdata_average(values: pd.Series) -> pd.Series:
    return values.rank(method="average")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--algorithm", default="NSGAIII")
    parser.add_argument("--expected-targets", type=int, default=12)
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.005,
        help="Absolute mean-relative-error band above each run's best value.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not (0.0 < args.epsilon < 0.1):
        raise ValueError("epsilon must lie in (0, 0.1)")
    validate_output_dir(output_dir)

    if args.expected_targets < 1 or args.expected_runs < 1:
        raise ValueError("expected-targets and expected-runs must be positive")
    files = sorted(input_dir.glob(f"pareto_sample*_{args.algorithm}_run*.csv"))
    expected_files = args.expected_targets * args.expected_runs
    if len(files) != expected_files:
        raise ValueError(
            f"Expected {expected_files} {args.algorithm} population files, found {len(files)}"
        )

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for path in files:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        if set(frame["algorithm"].astype(str)) != {args.algorithm}:
            raise ValueError(f"Unexpected algorithm value in {path.name}")
        frames.append(frame)
        manifest_rows.append(
            {
                "file": path.name,
                "rows": len(frame),
            }
        )

    data = pd.concat(frames, ignore_index=True)
    if sorted(data["sample"].unique().tolist()) != list(range(1, args.expected_targets + 1)):
        raise ValueError(f"Expected samples 1-{args.expected_targets}")
    if sorted(data["run"].unique().tolist()) != list(range(1, args.expected_runs + 1)):
        raise ValueError(f"Expected runs 1-{args.expected_runs}")

    pd.DataFrame(manifest_rows).to_csv(output_dir / "input_manifest.csv", index=False)

    inventory = (
        data.groupby(["sample", "run"], as_index=False)
        .agg(
            population_size=("mean_relative_error", "size"),
            best_mean_relative_error=("mean_relative_error", "min"),
            median_mean_relative_error=("mean_relative_error", "median"),
        )
        .sort_values(["sample", "run"])
    )
    inventory.to_csv(output_dir / "run_population_inventory.csv", index=False)

    best_by_run = (
        data.loc[data.groupby(["sample", "run"])["mean_relative_error"].idxmin()]
        .sort_values(["sample", "run"])
        .reset_index(drop=True)
    )
    best_by_run.to_csv(output_dir / "best_compromise_solution_by_run.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    count_by_run_rows: list[dict[str, object]] = []
    near_frames: list[pd.DataFrame] = []
    spearman_rows: list[dict[str, object]] = []

    for sample, sample_data in data.groupby("sample", sort=True):
        pooled_best = float(sample_data["mean_relative_error"].min())
        near_by_run: list[pd.DataFrame] = []
        for run, run_data in sample_data.groupby("run", sort=True):
            run_best = float(run_data["mean_relative_error"].min())
            threshold = run_best + args.epsilon
            run_near = run_data.loc[
                run_data["mean_relative_error"] <= threshold
            ].copy()
            run_near["run_best_mean_relative_error"] = run_best
            run_near["near_best_threshold"] = threshold
            near_by_run.append(run_near)
            count_by_run_rows.append(
                {
                    "sample": int(sample),
                    "algorithm": args.algorithm,
                    "run": int(run),
                    "population_size": len(run_data),
                    "run_best_mean_relative_error": run_best,
                    "near_equivalent_epsilon": args.epsilon,
                    "near_equivalent_threshold": threshold,
                    "near_equivalent_count": len(run_near),
                }
            )
        near = pd.concat(near_by_run, ignore_index=True)
        near_frames.append(near)
        count_rows.append(
            {
                "sample": int(sample),
                "algorithm": args.algorithm,
                "pooled_population_size": len(sample_data),
                "pooled_best_mean_relative_error": pooled_best,
                "near_equivalent_epsilon": args.epsilon,
                "near_equivalent_count": len(near),
                "runs_represented": int(near["run"].nunique()),
            }
        )

        for parameter, (lower, upper) in PARAMETERS.items():
            values = near[parameter].astype(float)
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            span = upper - lower
            summary_rows.append(
                {
                    "sample": int(sample),
                    "algorithm": args.algorithm,
                    "parameter": parameter,
                    "domain_lower": lower,
                    "domain_upper": upper,
                    "n_near_equivalent": len(values),
                    "median": float(values.median()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "normalized_iqr": (q3 - q1) / span,
                    "normalized_range": (float(values.max()) - float(values.min())) / span,
                }
            )

        ranked = near[list(PARAMETERS)].apply(rankdata_average)
        corr = ranked.corr(method="pearson")
        names = list(PARAMETERS)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                spearman_rows.append(
                    {
                        "sample": int(sample),
                        "algorithm": args.algorithm,
                        "parameter_1": left,
                        "parameter_2": right,
                        "spearman_rho": float(corr.loc[left, right]),
                        "n_near_equivalent": len(near),
                    }
                )

    near_all = pd.concat(near_frames, ignore_index=True)
    near_all.to_csv(output_dir / "near_equivalent_solutions.csv", index=False)
    pd.DataFrame(count_rows).to_csv(output_dir / "near_equivalent_counts.csv", index=False)
    pd.DataFrame(count_by_run_rows).to_csv(
        output_dir / "near_equivalent_counts_by_run.csv", index=False
    )

    parameter_summary = pd.DataFrame(summary_rows)
    parameter_summary.to_csv(output_dir / "near_equivalent_parameter_summary.csv", index=False)

    weak_summary = (
        parameter_summary.groupby(["algorithm", "parameter"], as_index=False)
        .agg(
            mean_normalized_iqr=("normalized_iqr", "mean"),
            median_normalized_iqr=("normalized_iqr", "median"),
            mean_normalized_range=("normalized_range", "mean"),
            median_normalized_range=("normalized_range", "median"),
        )
        .sort_values("mean_normalized_iqr", ascending=False)
    )
    weak_summary.insert(0, "constraint_rank_weakest_first", np.arange(1, len(weak_summary) + 1))
    weak_summary.to_csv(output_dir / "decision_variable_constraint_ranking.csv", index=False)

    spearman = pd.DataFrame(spearman_rows)
    spearman.to_csv(output_dir / "near_equivalent_parameter_spearman.csv", index=False)



    print(f"files={len(files)}")
    print(f"pooled_solutions={len(data)}")
    print(f"near_equivalent_solutions={len(near_all)}")
    print(weak_summary.to_csv(index=False))


if __name__ == "__main__":
    main()
