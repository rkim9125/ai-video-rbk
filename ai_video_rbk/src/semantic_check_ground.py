import argparse
import json
import re
from pathlib import Path


def hms_to_seconds(hms: str) -> float:
    parts = hms.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    raise ValueError(f"Invalid HH:MM:SS format: {hms}")


def load_gt_from_txt(gt_txt_path: str):
    """
    Reads lines like:
    00:00:12 Introduction and Administrative Overview
    00:05:36 Technical Infrastructure and Prerequisite Review
    """
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


def load_pred(pred_path: str):
    data = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    return data


def match_boundaries(gt, pred, tolerance=30.0):
    """
    One-to-one greedy matching.
    Each GT boundary can match at most one predicted boundary.
    """
    gt_used = set()
    pred_used = set()
    matches = []

    for pi, p in enumerate(pred):
        best_gi = None
        best_diff = None

        for gi, g in enumerate(gt):
            if gi in gt_used:
                continue

            diff = abs(p["boundary_time"] - g["time"])
            if diff <= tolerance:
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_gi = gi

        if best_gi is not None:
            gt_used.add(best_gi)
            pred_used.add(pi)
            matches.append({
                "gt_index": best_gi,
                "gt_time": gt[best_gi]["time"],
                "gt_time_str": gt[best_gi]["time_str"],
                "gt_title": gt[best_gi]["title"],
                "pred_index": pi,
                "pred_time": pred[pi]["boundary_time"],
                "pred_similarity": pred[pi]["similarity"],
                "abs_error_sec": best_diff
            })

    return matches, gt_used, pred_used


def compute_metrics(gt, pred, tolerance=30.0):
    matches, gt_used, pred_used = match_boundaries(gt, pred, tolerance)

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - len(gt_used)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tolerance_sec": tolerance,
        "gt_count": len(gt),
        "pred_count": len(pred),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches
    }


def print_report(gt, pred, metrics):
    print("=== Evaluation Report ===")
    print(f"GT boundaries: {metrics['gt_count']}")
    print(f"Predicted boundaries: {metrics['pred_count']}")
    print(f"Tolerance: ±{metrics['tolerance_sec']} sec")
    print()
    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print()

    print("=== Matched Boundaries ===")
    if not metrics["matches"]:
        print("No matches found.")
    else:
        for m in metrics["matches"]:
            print(
                f"GT {m['gt_time_str']} ({m['gt_title']}) "
                f"<--> Pred {m['pred_time']:.3f}s "
                f"(sim={m['pred_similarity']:.4f}, error={m['abs_error_sec']:.3f}s)"
            )

    matched_gt_indices = {m["gt_index"] for m in metrics["matches"]}
    matched_pred_indices = {m["pred_index"] for m in metrics["matches"]}

    print()
    print("=== Unmatched GT Boundaries ===")
    for i, g in enumerate(gt):
        if i not in matched_gt_indices:
            print(f"GT {g['time_str']} ({g['title']})")

    print()
    print("=== Unmatched Predicted Boundaries ===")
    for i, p in enumerate(pred):
        if i not in matched_pred_indices:
            print(f"Pred {p['boundary_time']:.3f}s (sim={p['similarity']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted boundary times against ground-truth with precision/recall/F1."
    )
    parser.add_argument(
        "--gt",
        default="lecture1_boundaries.txt",
        help="Path to ground-truth boundaries txt",
    )
    parser.add_argument(
        "--pred",
        default="boundaries_ws5_t055_d20.json",
        help="Path to predicted boundaries json",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=30.0,
        help="Matching tolerance in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--report-out",
        default="evaluation_report.json",
        help="Where to save evaluation_report.json (default: evaluation_report.json)",
    )
    args = parser.parse_args()

    gt = load_gt_from_txt(args.gt)
    pred = load_pred(args.pred)

    metrics = compute_metrics(gt, pred, tolerance=args.tolerance)
    print_report(gt, pred, metrics)

    Path(args.report_out).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nSaved: {args.report_out}")


if __name__ == "__main__":
    main()