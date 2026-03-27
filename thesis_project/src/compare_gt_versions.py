import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def hms_to_seconds(hms: str) -> float:
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_gt_from_txt(gt_txt_path: Path) -> List[Dict[str, object]]:
    lines = gt_txt_path.read_text(encoding="utf-8").splitlines()
    boundaries: List[Dict[str, object]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$", line)
        if not m:
            continue
        time_str, title = m.group(1), m.group(2).strip()
        boundaries.append({"time": hms_to_seconds(time_str), "time_str": time_str, "title": title})
    return boundaries


def load_pred(pred_path: Path) -> List[Dict[str, object]]:
    return json.loads(pred_path.read_text(encoding="utf-8"))


def match_boundaries(gt: List[Dict[str, object]], pred: List[Dict[str, object]], tolerance: float) -> Tuple[List[Dict], set, set]:
    gt_used = set()
    pred_used = set()
    matches = []

    for pi, p in enumerate(pred):
        best_gi = None
        best_diff = None
        p_time = float(p["boundary_time"])
        for gi, g in enumerate(gt):
            if gi in gt_used:
                continue
            diff = abs(p_time - float(g["time"]))
            if diff <= tolerance and (best_diff is None or diff < best_diff):
                best_diff = diff
                best_gi = gi
        if best_gi is not None:
            gt_used.add(best_gi)
            pred_used.add(pi)
            matches.append({"gt_index": best_gi, "pred_index": pi, "abs_error_sec": best_diff})
    return matches, gt_used, pred_used


def compute_metrics(gt: List[Dict[str, object]], pred: List[Dict[str, object]], tolerance: float) -> Dict[str, float]:
    matches, gt_used, _ = match_boundaries(gt, pred, tolerance=tolerance)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - len(gt_used)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "gt_count": len(gt),
        "pred_count": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def discover_prediction_files(results_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in results_root.glob("**/lecture*_boundaries.json"):
        # keep only per-lecture files in lecture subfolders (avoid top-level copies)
        if p.parent.name.startswith("lecture"):
            files.append(p)
    return sorted(files)


def write_csv(path: Path, rows: List[Dict[str, object]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation metrics between original GT and corrected GT.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--results-subdirs", nargs="*", default=["expA_representation", "expB_window_size"])
    parser.add_argument("--orig-annotations-dir", default="ai_video_rbk/annotations")
    parser.add_argument("--corrected-annotations-dir", default="ai_video_rbk/annotations_corrected")
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results_base = repo_root / "thesis_project" / "results"
    orig_ann_dir = (repo_root / args.orig_annotations_dir).resolve()
    corr_ann_dir = (repo_root / args.corrected_annotations_dir).resolve()
    tables_dir = repo_root / "thesis_project" / "tables"

    detail_rows: List[Dict[str, object]] = []
    for subdir in args.results_subdirs:
        pred_files = discover_prediction_files(results_base / subdir)
        for pred_file in pred_files:
            # .../results/<subdir>/<setting>/<lecture>/<lectureX_boundaries.json>
            parts = pred_file.parts
            setting = parts[-3]
            lecture = parts[-2]

            orig_gt = orig_ann_dir / f"{lecture}_boundaries.txt"
            corr_gt = corr_ann_dir / f"{lecture}_boundaries.txt"
            if not orig_gt.exists() or not corr_gt.exists():
                continue

            pred = load_pred(pred_file)
            m_orig = compute_metrics(load_gt_from_txt(orig_gt), pred, args.tolerance_seconds)
            m_corr = compute_metrics(load_gt_from_txt(corr_gt), pred, args.tolerance_seconds)

            detail_rows.append(
                {
                    "experiment_family": subdir,
                    "setting": setting,
                    "lecture": lecture,
                    "pred_count": m_orig["pred_count"],
                    "gt_count_orig": m_orig["gt_count"],
                    "gt_count_corrected": m_corr["gt_count"],
                    "precision_orig": round(m_orig["precision"], 4),
                    "recall_orig": round(m_orig["recall"], 4),
                    "f1_orig": round(m_orig["f1"], 4),
                    "precision_corrected": round(m_corr["precision"], 4),
                    "recall_corrected": round(m_corr["recall"], 4),
                    "f1_corrected": round(m_corr["f1"], 4),
                    "delta_precision": round(m_corr["precision"] - m_orig["precision"], 4),
                    "delta_recall": round(m_corr["recall"] - m_orig["recall"], 4),
                    "delta_f1": round(m_corr["f1"] - m_orig["f1"], 4),
                }
            )

    detail_header = [
        "experiment_family",
        "setting",
        "lecture",
        "pred_count",
        "gt_count_orig",
        "gt_count_corrected",
        "precision_orig",
        "recall_orig",
        "f1_orig",
        "precision_corrected",
        "recall_corrected",
        "f1_corrected",
        "delta_precision",
        "delta_recall",
        "delta_f1",
    ]
    detail_csv = tables_dir / "gt_comparison_detail.csv"
    write_csv(detail_csv, detail_rows, detail_header)

    # setting-level aggregation
    by_setting: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for r in detail_rows:
        key = (str(r["experiment_family"]), str(r["setting"]))
        by_setting.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, object]] = []
    for (exp_family, setting), rows in sorted(by_setting.items()):
        n = len(rows)
        summary_rows.append(
            {
                "experiment_family": exp_family,
                "setting": setting,
                "n_lectures": n,
                "avg_f1_orig": round(sum(float(r["f1_orig"]) for r in rows) / n, 4),
                "avg_f1_corrected": round(sum(float(r["f1_corrected"]) for r in rows) / n, 4),
                "delta_avg_f1": round(sum(float(r["delta_f1"]) for r in rows) / n, 4),
                "avg_precision_orig": round(sum(float(r["precision_orig"]) for r in rows) / n, 4),
                "avg_precision_corrected": round(sum(float(r["precision_corrected"]) for r in rows) / n, 4),
                "avg_recall_orig": round(sum(float(r["recall_orig"]) for r in rows) / n, 4),
                "avg_recall_corrected": round(sum(float(r["recall_corrected"]) for r in rows) / n, 4),
            }
        )

    summary_header = [
        "experiment_family",
        "setting",
        "n_lectures",
        "avg_f1_orig",
        "avg_f1_corrected",
        "delta_avg_f1",
        "avg_precision_orig",
        "avg_precision_corrected",
        "avg_recall_orig",
        "avg_recall_corrected",
    ]
    summary_csv = tables_dir / "gt_comparison_summary.csv"
    write_csv(summary_csv, summary_rows, summary_header)

    print(f"Saved: {detail_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
