import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def plot_summary_metrics(summary_rows: List[Dict[str, str]], out_path: Path) -> None:
    methods = [r["Representation"] for r in summary_rows]
    precision = [to_float(r, "Precision") for r in summary_rows]
    recall = [to_float(r, "Recall") for r in summary_rows]
    f1 = [to_float(r, "F1") for r in summary_rows]

    x = list(range(len(methods)))
    w = 0.22

    plt.figure(figsize=(9, 5))
    plt.bar([i - w for i in x], precision, width=w, label="Precision")
    plt.bar(x, recall, width=w, label="Recall")
    plt.bar([i + w for i in x], f1, width=w, label="F1")

    plt.xticks(x, methods)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Experiment A - Macro Average Metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_lecture_f1(lecture_rows: List[Dict[str, str]], out_path: Path) -> None:
    sentence = [r for r in lecture_rows if r["Method"] == "Sentence-based"]
    time_based = [r for r in lecture_rows if r["Method"] == "Time-based"]

    lecture_ids = [r["Lecture"] for r in sentence]
    sentence_f1 = [to_float(r, "F1") for r in sentence]
    time_f1 = [to_float(r, "F1") for r in time_based]

    x = list(range(len(lecture_ids)))
    w = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - w / 2 for i in x], sentence_f1, width=w, label="Sentence-based")
    plt.bar([i + w / 2 for i in x], time_f1, width=w, label="Time-based (10s)")

    plt.xticks(x, lecture_ids)
    plt.ylim(0, 1.0)
    plt.ylabel("F1")
    plt.title("Experiment A - Lecture-level F1 Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Experiment A summary tables.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tables_dir = repo_root / "thesis_project" / "tables"
    figs_dir = tables_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = tables_dir / "expA_summary_table.csv"
    lecture_csv = tables_dir / "expA_lecture_level_table.csv"

    summary_rows = load_csv(summary_csv)
    lecture_rows = load_csv(lecture_csv)

    out_summary = figs_dir / "expA_macro_metrics.png"
    out_f1 = figs_dir / "expA_lecture_f1.png"

    plot_summary_metrics(summary_rows, out_summary)
    plot_lecture_f1(lecture_rows, out_f1)

    print(f"Saved: {out_summary}")
    print(f"Saved: {out_f1}")


if __name__ == "__main__":
    main()
