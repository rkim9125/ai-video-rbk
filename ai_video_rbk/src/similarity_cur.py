import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def hms_to_seconds(hms: str) -> float:
    parts = hms.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"Invalid HH:MM:SS format: {hms}")


def load_gt_from_txt(gt_txt_path: str):
    lines = Path(gt_txt_path).read_text(encoding="utf-8").splitlines()
    boundaries = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$", line)
        if not m:
            continue

        time_str = m.group(1)
        title = m.group(2).strip()

        boundaries.append({
            "time": hms_to_seconds(time_str),
            "time_str": time_str,
            "title": title
        })

    return boundaries


def load_similarities(sim_path: str):
    return json.loads(Path(sim_path).read_text(encoding="utf-8"))


def load_pred_boundaries(pred_path: str):
    return json.loads(Path(pred_path).read_text(encoding="utf-8"))


def seconds_to_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def plot_similarity_with_boundaries(similarities, gt_boundaries, pred_boundaries, output_path):
    # Kept for backward compatibility: a single overview plot.
    plot_overview(similarities, gt_boundaries, pred_boundaries, output_path, show=False)


def plot_overview(similarities, gt_boundaries, pred_boundaries, output_path, show: bool = False, invert: bool = False):
    x = [item["right_start"] for item in similarities]
    y = [item["similarity"] for item in similarities]
    if invert:
        y = [1.0 - v for v in y]

    plt.figure(figsize=(18, 6))
    plt.plot(x, y, linewidth=1.2, label="Semantic similarity (or 1-sim)")

    # GT boundaries
    for i, gt in enumerate(gt_boundaries):
        plt.axvline(
            x=gt["time"],
            linestyle="--",
            alpha=0.7,
            label="GT boundary" if i == 0 else None,
            color="tab:green",
        )

    # Pred boundaries
    for i, pred in enumerate(pred_boundaries):
        plt.axvline(
            x=pred["boundary_time"],
            linestyle=":",
            alpha=0.6,
            label="Predicted boundary" if i == 0 else None,
            color="tab:red",
        )

    # annotate GT times only (avoid clutter)
    y_text = max(y) if y else 1.0
    for gt in gt_boundaries:
        plt.text(
            gt["time"],
            y_text,
            gt["time_str"],
            rotation=90,
            verticalalignment="bottom",
            fontsize=8,
            color="tab:green",
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Similarity" + (" (inverted)" if invert else ""))
    plt.title("Semantic Similarity Curve with GT and Predicted Boundaries")
    if not invert:
        plt.ylim(0, 1.05)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved plot to: {output_path}")


def plot_zoom_around_gt(
    similarities,
    gt_boundaries,
    pred_boundaries,
    output_path: str,
    window_seconds: float = 120.0,
    invert: bool = False,
    show: bool = False,
):
    """
    More readable visualization:
    - One subplot per GT boundary
    - Zooms in around each GT time (±window_seconds)
    """
    x = [item["right_start"] for item in similarities]
    y_raw = [item["similarity"] for item in similarities]
    y = ([1.0 - v for v in y_raw] if invert else y_raw)

    # Grid sizing: simple 3x3 when <=9 GT, otherwise ceil.
    n = len(gt_boundaries)
    cols = 3
    rows = (n + cols - 1) // cols if n else 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.0, rows * 3.6), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for gi, gt in enumerate(gt_boundaries):
        ax = axes_flat[gi]
        t = float(gt["time"])
        xmin, xmax = t - window_seconds, t + window_seconds

        # plot whole curve but with limited view
        ax.plot(x, y, linewidth=1.0, color="tab:blue")
        ax.set_xlim(xmin, xmax)

        # GT line + label
        ax.axvline(t, linestyle="--", alpha=0.9, color="tab:green")
        ax.text(
            t,
            ax.get_ylim()[1],
            gt["time_str"],
            rotation=90,
            verticalalignment="bottom",
            fontsize=8,
            color="tab:green",
        )

        # predicted boundaries in this local window
        local_preds = [p for p in pred_boundaries if xmin <= float(p["boundary_time"]) <= xmax]
        for pi, p in enumerate(local_preds):
            ax.axvline(float(p["boundary_time"]), linestyle=":", alpha=0.55, color="tab:red")

        # annotate nearest dip to GT (by minimum similarity in local window)
        local_pairs = [
            (xj, yj) for xj, yj in zip(x, y) if xmin <= xj <= xmax
        ]
        if local_pairs:
            x_min, y_min = min(local_pairs, key=lambda pair: pair[1])
            ax.scatter([x_min], [y_min], color="black", s=18, zorder=5)
            ax.text(
                x_min,
                y_min,
                f"min@{x_min:.1f}s",
                fontsize=7,
                color="black",
                ha="left",
                va="bottom",
            )

        ax.grid(True, alpha=0.25)
        title = gt["time_str"]
        if gt.get("title"):
            # keep it short
            title = f"{gt['time_str']}"
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("1-sim" if invert else "sim")

    # turn off unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Similarity Curve Zoomed Around GT Boundaries", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved zoom plot to: {output_path}")


def print_nearest_similarity_to_gt(similarities, gt_boundaries):
    print("\n=== Nearest similarity dip around each GT boundary ===")
    for gt in gt_boundaries:
        nearest = min(similarities, key=lambda s: abs(s["right_start"] - gt["time"]))
        print(
            f"GT {gt['time_str']} ({gt['title']}) | "
            f"nearest similarity point at {nearest['right_start']:.3f}s "
            f"(sim={nearest['similarity']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot semantic similarity curve with GT and predicted boundaries."
    )
    parser.add_argument(
        "--sim",
        required=True,
        help="Path to similarities.json"
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to lecture boundary txt"
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to predicted boundaries json"
    )
    parser.add_argument(
        "--out",
        default="similarity_overlay.png",
        help="Output image path"
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["overview", "zoom", "both"],
        help="Visualization mode: overview (single plot), zoom (GT-local subplots), or both (default).",
    )
    parser.add_argument(
        "--zoom-window-seconds",
        type=float,
        default=120.0,
        help="Half-width of zoom window around each GT boundary in seconds (default: 120).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Plot 1-sim instead of sim to make dips appear as peaks.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (default: save+close).",
    )
    args = parser.parse_args()

    similarities = load_similarities(args.sim)
    gt_boundaries = load_gt_from_txt(args.gt)
    pred_boundaries = load_pred_boundaries(args.pred)

    print(f"Loaded similarities: {len(similarities)}")
    print(f"Loaded GT boundaries: {len(gt_boundaries)}")
    print(f"Loaded predicted boundaries: {len(pred_boundaries)}")

    print_nearest_similarity_to_gt(similarities, gt_boundaries)
    if args.mode in ["overview", "both"]:
        overview_out = args.out
        # If user requested zoom-only, keep out as-is.
        plot_overview(
            similarities,
            gt_boundaries,
            pred_boundaries,
            overview_out,
            show=args.show,
            invert=args.invert,
        )

    if args.mode in ["zoom", "both"]:
        zoom_out = Path(args.out).with_name(f"{Path(args.out).stem}_zoom.png")
        plot_zoom_around_gt(
            similarities,
            gt_boundaries,
            pred_boundaries,
            str(zoom_out),
            window_seconds=args.zoom_window_seconds,
            invert=args.invert,
            show=args.show,
        )


if __name__ == "__main__":
    main()