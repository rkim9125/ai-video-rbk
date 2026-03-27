import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_summary(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def build_row(summary: Dict) -> Dict[str, object]:
    setting = summary["setting"]
    method = "A1" if setting.startswith("sentence_") else "A2"

    representation = "Sentence-based" if summary["representation"] == "sentence" else "Time-based"
    window_unit = summary.get("window_unit", "")
    window_size = summary.get("window_size", "")

    ctrl = summary.get("controlled_variables", {})
    threshold = ctrl.get("threshold", "")
    local_minima = to_yes_no(bool(ctrl.get("local_minima", False)))
    min_distance = f"{int(ctrl.get('min_distance_sec', 0))}s" if ctrl.get("min_distance_sec") is not None else ""

    macro = summary.get("macro_average", {})
    precision = macro.get("precision", "")
    recall = macro.get("recall", "")
    f1 = macro.get("f1", "")

    return {
        "Method": method,
        "Representation": representation,
        "Window Unit": window_unit,
        "Window Size": window_size,
        "Threshold": threshold,
        "Local Minima": local_minima,
        "Min-Distance": min_distance,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Experiment A summary table CSV.")
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path.",
    )
    parser.add_argument(
        "--out",
        default="thesis_project/tables/expA_summary_table.csv",
        help="Output CSV path relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    expa_root = repo_root / "thesis_project" / "results" / "expA_representation"
    out_path = repo_root / args.out

    summary_paths = sorted(expa_root.glob("*/evaluation_summary.json"))
    rows: List[Dict[str, object]] = []
    for summary_path in summary_paths:
        rows.append(build_row(load_summary(summary_path)))

    # Sort so sentence-based appears before time-based.
    rows.sort(key=lambda r: r["Method"])

    header = [
        "Method",
        "Representation",
        "Window Unit",
        "Window Size",
        "Threshold",
        "Local Minima",
        "Min-Distance",
        "Precision",
        "Recall",
        "F1",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
