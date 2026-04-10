"""
Quantify GT–prediction alignment before changing GT or the task definition.

1) Per-GT: count predictions in ±window_sec (default = eval tolerance).
2) Per-FP: distance to nearest GT; bucket into redundant-near-GT vs farther spurious-style FPs.

Reuses matching logic from semantic_check_ground.compute_metrics.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _import_ground_module(repo_root: Path):
    src = repo_root / "ai_video_rbk" / "src"
    sys.path.insert(0, str(src))
    import semantic_check_ground as scg  # type: ignore

    return scg


def preds_in_window(
    pred: List[dict], center_sec: float, half_width: float
) -> List[Tuple[int, dict]]:
    out: List[Tuple[int, dict]] = []
    for i, p in enumerate(pred):
        t = float(p["boundary_time"])
        if abs(t - center_sec) <= half_width:
            out.append((i, p))
    return out


def nearest_gt_distance(pred_time: float, gt: List[dict]) -> float:
    return min(abs(pred_time - float(g["time"])) for g in gt)


def fp_bucket(nearest_dist: float, tol: float, mid_max: float) -> str:
    """After FP is known: how far is nearest GT?"""
    if nearest_dist <= tol:
        return "redundant_near_gt"
    if nearest_dist <= mid_max:
        return "offset_near_miss"
    return "spurious_far"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GT-centered prediction density + FP typing (alignment diagnostic)."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--lecture", default="lecture1", help="Lecture id (e.g. lecture1).")
    parser.add_argument(
        "--gt",
        default="",
        help="GT txt path. Default: ai_video_rbk/annotations/{lecture}_boundaries.txt",
    )
    parser.add_argument(
        "--pred",
        default="",
        help="Predicted boundaries JSON. Default: expE md30 run for this lecture.",
    )
    parser.add_argument(
        "--density-window",
        type=float,
        default=30.0,
        help="Half-width (seconds) for 'predictions near this GT' (default: 30, same as typical tolerance).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=30.0,
        help="Matching tolerance for TP/FP (default: 30).",
    )
    parser.add_argument(
        "--mid-band-max",
        type=float,
        default=120.0,
        help="Upper bound (sec) for 'offset_near_miss' FP bucket above tolerance (default: 120).",
    )
    parser.add_argument(
        "--out-dir",
        default="thesis_project/tables/analysis_alignment",
        help="Output directory relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    scg = _import_ground_module(repo_root)

    lec = args.lecture
    gt_path = Path(args.gt) if args.gt else repo_root / "ai_video_rbk" / "annotations" / f"{lec}_boundaries.txt"
    pred_path = (
        Path(args.pred)
        if args.pred
        else repo_root
        / "thesis_project/results/expE_prediction_pruning/sentence_w3_t055_md30"
        / lec
        / f"{lec}_boundaries.json"
    )

    gt = scg.load_gt_from_txt(str(gt_path))
    pred = scg.load_pred(str(pred_path))
    metrics = scg.compute_metrics(gt, pred, tolerance=args.tolerance)

    matched_pred: Set[int] = {m["pred_index"] for m in metrics["matches"]}
    half_w = float(args.density_window)
    tol = float(args.tolerance)
    mid_max = float(args.mid_band_max)

    out_dir = (repo_root / args.out_dir).resolve() / lec
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Analysis A: density per GT ---
    density_rows: List[Dict[str, object]] = []
    for gi, g in enumerate(gt):
        gtime = float(g["time"])
        in_win = preds_in_window(pred, gtime, half_w)
        density_rows.append(
            {
                "gt_index": gi,
                "gt_time_sec": round(gtime, 3),
                "gt_time_str": g["time_str"],
                "gt_title": g["title"],
                f"preds_in_pm{int(half_w)}s": len(in_win),
                "pred_indices_in_window": ";".join(str(i) for i, _ in in_win),
                "matched_tp": 1 if any(gi == m["gt_index"] for m in metrics["matches"]) else 0,
            }
        )

    density_csv = out_dir / "gt_prediction_density.csv"
    with density_csv.open("w", newline="", encoding="utf-8") as f:
        hdr = [
            "gt_index",
            "gt_time_sec",
            "gt_time_str",
            "gt_title",
            f"preds_in_pm{int(half_w)}s",
            "pred_indices_in_window",
            "matched_tp",
        ]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for row in density_rows:
            w.writerow(row)

    # --- Analysis B: FP typing ---
    fp_rows: List[Dict[str, object]] = []
    bucket_counts: Dict[str, int] = {"redundant_near_gt": 0, "offset_near_miss": 0, "spurious_far": 0}

    for pi, p in enumerate(pred):
        if pi in matched_pred:
            continue
        pt = float(p["boundary_time"])
        d_near = nearest_gt_distance(pt, gt)
        bucket = fp_bucket(d_near, tol, mid_max)
        bucket_counts[bucket] += 1
        fp_rows.append(
            {
                "pred_index": pi,
                "pred_time_sec": round(pt, 3),
                "similarity": round(float(p.get("similarity", 0.0)), 6),
                "nearest_gt_dist_sec": round(d_near, 3),
                "fp_bucket": bucket,
            }
        )

    fp_csv = out_dir / "fp_classification.csv"
    with fp_csv.open("w", newline="", encoding="utf-8") as f:
        hdr = ["pred_index", "pred_time_sec", "similarity", "nearest_gt_dist_sec", "fp_bucket"]
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for row in fp_rows:
            w.writerow(row)

    n_gt = len(gt)
    mean_preds_per_gt_window = sum(float(r[f"preds_in_pm{int(half_w)}s"]) for r in density_rows) / n_gt if n_gt else 0.0
    max_preds_single_gt_window = max((int(r[f"preds_in_pm{int(half_w)}s"]) for r in density_rows), default=0)

    n_fp = len(fp_rows)
    summary = {
        "lecture": lec,
        "gt_path": str(gt_path.relative_to(repo_root)),
        "pred_path": str(pred_path.relative_to(repo_root)),
        "density_window_pm_sec": half_w,
        "match_tolerance_sec": tol,
        "fp_mid_band_max_sec": mid_max,
        "metrics": {
            "gt_count": metrics["gt_count"],
            "pred_count": metrics["pred_count"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "analysis_A_gt_windows": {
            "mean_preds_per_gt_in_window": round(mean_preds_per_gt_window, 3),
            "max_preds_in_any_single_gt_window": max_preds_single_gt_window,
        },
        "analysis_B_fp_buckets": {
            **bucket_counts,
            "fp_total": n_fp,
            "pct_redundant_near_gt": round(100.0 * bucket_counts["redundant_near_gt"] / n_fp, 2) if n_fp else 0.0,
            "pct_offset_near_miss": round(100.0 * bucket_counts["offset_near_miss"] / n_fp, 2) if n_fp else 0.0,
            "pct_spurious_far": round(100.0 * bucket_counts["spurious_far"] / n_fp, 2) if n_fp else 0.0,
        },
        "interpretation_hint": (
            "If redundant_near_gt dominates FP and mean_preds_per_gt_in_window is high, "
            "predictions cluster around true regions but over-fire (granularity/pruning issue). "
            "If spurious_far dominates, many predictions are far from any GT (signal quality or task mismatch)."
        ),
    }

    summary_path = out_dir / "alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote:\n- {density_csv}\n- {fp_csv}\n- {summary_path}")


if __name__ == "__main__":
    main()
