from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input_data"
MODEL_DIR = ROOT / "trained_models"
RESULT_DIR = ROOT / "modeling_results"

INPUT = RESULT_DIR / "forward_check" / "forward_check_predictions.csv"
OUTPUT_DIR = RESULT_DIR / "reproduced_figures"
STEM = OUTPUT_DIR / "Fig08_forward_check_round2_v3"




def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [STEM.with_suffix(suffix) for suffix in (".pdf", ".tif", ".png")]
    outputs.append(OUTPUT_DIR / "Fig08_forward_check_round2_v3_manifest.json")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing figure files: {existing}")

    frame = pd.read_csv(INPUT)
    if frame["sample"].tolist() != [1, 2, 3, 4, 5, 6, 8]:
        raise ValueError("Expected retained in-domain samples 1-6 and 8")

    panels = [
        ("actual_peak_stress_mpa", "pred_peak_stress_mpa", "Peak stress (MPa)"),
        (
            "actual_peak_strain_permille",
            "pred_peak_strain_permille",
            "Peak strain (‰)",
        ),
        (
            "actual_elastic_modulus_gpa",
            "pred_elastic_modulus_gpa",
            "Elastic modulus (GPa)",
        ),
        ("actual_poisson_ratio", "pred_poisson_ratio", "Poisson's ratio"),
    ]

    style = {
        "font.family": "Arial",
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    with mpl.rc_context(style):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(190 / 25.4, 140 / 25.4),
            layout="constrained",
        )
        label_offsets = {
            0: {1: (5, 2), 3: (-10, 4)},
            1: {3: (-9, 5), 4: (4, 3)},
            3: {1: (-10, 5), 8: (4, -8)},
        }
        for panel_index, (label, ax, (actual_col, predicted_col, title)) in enumerate(zip(
            "abcd", axes.flat, panels
        )):
            actual = frame[actual_col].astype(float)
            predicted = frame[predicted_col].astype(float)
            low = min(actual.min(), predicted.min())
            high = max(actual.max(), predicted.max())
            span = high - low
            margin = 0.07 * span if span > 0 else 0.05 * max(abs(high), 1.0)
            limits = (low - margin, high + margin)

            ax.plot(
                limits,
                limits,
                color="#4D4D4D",
                linestyle="--",
                linewidth=1.0,
                label="1:1 line",
                zorder=1,
            )
            ax.scatter(
                actual,
                predicted,
                s=27,
                marker="o",
                facecolor="#0072B2",
                edgecolor="black",
                linewidth=0.55,
                label="In-domain case",
                zorder=2,
            )
            for sample, x, y in zip(frame["sample"], actual, predicted):
                offset = label_offsets.get(panel_index, {}).get(int(sample), (3, 3))
                ax.annotate(
                    str(int(sample)),
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=5.8,
                )

            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"PFC2D target {title}")
            ax.set_ylabel(f"GPR-predicted {title}")
            ax.grid(True, color="#D9D9D9", linewidth=0.45, zorder=0)
            ax.text(
                0.02,
                0.98,
                label,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )

        fig.savefig(STEM.with_suffix(".pdf"), format="pdf", bbox_inches=None)
        temporary_tiff = OUTPUT_DIR / "Fig08_forward_check_round2_v3_rgba_temp.tif"
        fig.savefig(
            temporary_tiff,
            format="tiff",
            dpi=1010,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches=None,
        )
        with Image.open(temporary_tiff) as image:
            image.convert("RGB").save(
                STEM.with_suffix(".tif"),
                format="TIFF",
                compression="tiff_lzw",
                dpi=(1010, 1010),
            )
        temporary_tiff.unlink()
        fig.savefig(STEM.with_suffix(".png"), format="png", dpi=300, bbox_inches=None)
        plt.close(fig)

    manifest = {
        "figure": "Fig. 8 round-2 forward check v3",
        "audience": "Applied Soft Computing revised manuscript",
        "source_csv": INPUT.relative_to(ROOT).as_posix(),
        "n_cases": 7,
        "included_samples": [1, 2, 3, 4, 5, 6, 8],
        "excluded_submitted_samples": [7, 9, 10, 11, 12, 13, 14, 15, 16],
        "exclusion_reason": "one or more mesoscopic inputs lie outside the Table 1 training bounds",
        "transformations": ["none; plotted actual and production-GPR predictions directly"],
        "uncertainty": "not applicable; deterministic forward cases, no replicate estimator shown",
        "dimensions_mm": [190, 140],
        "formats": [path.suffix.lstrip(".") for path in outputs[:3]],
        "tiff_export": "RGB, LZW, 1010 dpi requested to clear the 1000-dpi effective-width threshold",
        "matplotlib_version": mpl.__version__,
        "alt_text": (
            "Four actual-versus-predicted scatter panels for peak stress, peak strain, elastic modulus, "
            "and Poisson's ratio. Seven in-domain cases lie close to the dashed one-to-one lines; "
            "elastic modulus shows the largest visible departures."
        ),
    }
    outputs[3].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
