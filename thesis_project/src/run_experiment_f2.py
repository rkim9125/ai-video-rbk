"""
Experiment F2 — Prominence filtering on top of the best F1 marker-gated setup.

Fixed (F1 best): marker window ±15s, semantic confirm similarity < 0.60, local minima on,
min-distance 30s, sentence w=3.

Varies: minimum similarity prominence only (single-factor experiment).

Prominence (method 1): for dip at index i,
  prominence = min(left_avg - dip, right_avg - dip)
where left_avg = mean(sim[i-span:i]), right_avg = mean(sim[i+1:i+1+span]).
Default span = 1 (one adjacent similarity on each side).

prominence_min = 0 reproduces F1 best (no extra filtering beyond local minimum structure).
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_expf1_helpers():
    p = Path(__file__).resolve().parent / "run_experiment_f1.py"
    spec = importlib.util.spec_from_file_location("expf1_helpers", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_experiment_f1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def similarity_prominence(sim_values: List[float], idx: int, span: int) -> float:
    """min(left_avg - dip, right_avg - dip); span = number of points on each side."""
    dip = sim_values[idx]
    left_part = sim_values[max(0, idx - span) : idx]
    right_part = sim_values[idx + 1 : idx + 1 + span]
    if not left_part or not right_part:
        return 0.0
    left_avg = sum(left_part) / len(left_part)
    right_avg = sum(right_part) / len(right_part)
    return min(left_avg - dip, right_avg - dip)


def detect_with_prominence(
    similarities: List[Dict[str, Any]],
    sc_mod: Any,
    confirm_threshold: float,
    prominence_min: float,
    context_span: int,
) -> List[Dict[str, Any]]:
    sim_values = [float(x["similarity"]) for x in similarities]
    boundaries: List[Dict[str, Any]] = []

    for i, item in enumerate(similarities):
        if not sc_mod.is_local_minimum(sim_values, i):
            continue
        if sim_values[i] >= confirm_threshold:
            continue
        prom = similarity_prominence(sim_values, i, context_span)
        if prom < prominence_min - 1e-12:
            continue

        boundaries.append(
            {
                "boundary_index": len(boundaries),
                "between_windows": [item["left_window_id"], item["right_window_id"]],
                "boundary_time": float(item["right_start"]),
                "similarity": float(item["similarity"]),
                "left_window_end": float(item["left_end"]),
                "right_window_start": float(item["right_start"]),
                "prominence": round(prom, 6),
                "reason": "marker_gated_prominence_semantic",
            }
        )
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment F2: prominence on F1-best marker gating.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--marker-window-pm", type=float, default=15.0, help="±seconds around marker (F1 best).")
    parser.add_argument("--confirm-threshold", type=float, default=0.60)
    parser.add_argument("--min-distance-final", type=float, default=30.0)
    parser.add_argument("--marker-dedupe-sec", type=float, default=20.0)
    parser.add_argument("--context-span", type=int, default=1, help="Adjacent similarity steps per side.")
    parser.add_argument(
        "--prominence-mins",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.04, 0.06, 0.08],
    )
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument("--annotations-dir", default="ai_video_rbk/annotations")
    parser.add_argument("--lectures", nargs="*", default=[])
    args = parser.parse_args()

    f1 = load_expf1_helpers()
    repo_root = Path(args.repo_root).resolve()
    ann_dir = (repo_root / args.annotations_dir).resolve()
    lecture_ids = args.lectures if args.lectures else f1.discover_lectures(repo_root, ann_dir)
    if not lecture_ids:
        raise RuntimeError("No lectures found.")

    sc_mod = f1.load_semantic_check_module(repo_root)
    scg = f1.load_ground_module(repo_root)

    marker_patterns = [
        r"\bnow\b",
        r"\bnext\b",
        r"\bmove on\b",
        r"\banother important\b",
        r"\btoday we (will|are going to)\b",
        r"\blet'?s move on\b",
    ]

    cache_root = repo_root / "thesis_project" / "results" / "expF1_marker_gated" / "_similarity_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    all_summary: List[Dict[str, object]] = []
    master_rows: List[Dict[str, str]] = []
    all_lecture_rows: List[Dict[str, object]] = []

    for pmin in args.prominence_mins:
        ptag = f"p{str(pmin).replace('.', '')}"
        setting = (
            f"F2_pm{int(args.marker_window_pm)}_c{str(args.confirm_threshold).replace('.', '')}_"
            f"{ptag}_md{int(args.min_distance_final)}_s{args.context_span}"
        )
        out_root = repo_root / "thesis_project" / "results" / "expF2_prominence" / setting
        lect_rows: List[Dict[str, object]] = []

        for lecture_id in lecture_ids:
            lec_cache = cache_root / f"sentence_w{args.window_size}" / lecture_id
            sim_path = f1.ensure_similarities_cached(repo_root, lecture_id, lec_cache, args.window_size, sc_mod)
            similarities = json.loads(sim_path.read_text(encoding="utf-8"))

            raw = detect_with_prominence(
                similarities,
                sc_mod,
                confirm_threshold=float(args.confirm_threshold),
                prominence_min=float(pmin),
                context_span=int(args.context_span),
            )

            vtt_path = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lecture_id}.vtt"
            markers = f1.marker_times_from_vtt(vtt_path, marker_patterns, dedupe_min_distance=args.marker_dedupe_sec)

            gated = f1.gate_semantic_by_markers(raw, markers, delta_sec=float(args.marker_window_pm))
            for b in gated:
                b["reason"] = "marker_gated_prominence_semantic"

            final_bounds = sc_mod.filter_boundaries_by_min_distance(
                gated,
                min_distance_seconds=float(args.min_distance_final),
            )
            for idx, b in enumerate(final_bounds):
                b["boundary_index"] = idx

            out_dir = out_root / lecture_id
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / f"{lecture_id}_boundaries.json"
            # strip prominence for evaluator if needed — ground script only needs boundary_time, similarity
            pred_path.write_text(json.dumps(final_bounds, indent=2, ensure_ascii=False), encoding="utf-8")

            gt_path = ann_dir / f"{lecture_id}_boundaries.txt"
            m = f1.evaluate_pred_json(repo_root, gt_path, pred_path, args.tolerance_seconds, scg)

            lect_rows.append(
                {
                    "Setting": setting,
                    "Lecture": lecture_id,
                    "marker_window_pm_sec": int(args.marker_window_pm),
                    "confirm_threshold": args.confirm_threshold,
                    "prominence_min": pmin,
                    "context_span": args.context_span,
                    "min_distance_sec": args.min_distance_final,
                    "Precision": round(float(m["precision"]), 4),
                    "Recall": round(float(m["recall"]), 4),
                    "F1": round(float(m["f1"]), 4),
                    "Predicted Boundaries": int(m["pred_count"]),
                    "GT Boundaries": int(m["gt_count"]),
                    "semantic_after_prominence": len(raw),
                }
            )
            all_lecture_rows.append(dict(lect_rows[-1]))

            master_rows.append(
                {
                    "experiment_id": "expF2_prominence",
                    "setting": setting,
                    "lecture_id": lecture_id,
                    "pred_count": str(m["pred_count"]),
                    "gt_count": str(m["gt_count"]),
                    "tp": str(m["TP"]),
                    "fp": str(m["FP"]),
                    "fn": str(m["FN"]),
                    "precision": f"{float(m['precision']):.4f}",
                    "recall": f"{float(m['recall']):.4f}",
                    "f1": f"{float(m['f1']):.4f}",
                    "tolerance_sec": str(m["tolerance_sec"]),
                    "notes": f"prom_min={pmin} span={args.context_span} on F1-best gate",
                }
            )

        n = len(lect_rows)
        all_summary.append(
            {
                "Setting": setting,
                "marker_window_pm_sec": int(args.marker_window_pm),
                "confirm_threshold": args.confirm_threshold,
                "prominence_min": pmin,
                "context_span": args.context_span,
                "min_distance_sec": int(args.min_distance_final),
                "Precision": round(sum(float(r["Precision"]) for r in lect_rows) / n, 4),
                "Recall": round(sum(float(r["Recall"]) for r in lect_rows) / n, 4),
                "F1": round(sum(float(r["F1"]) for r in lect_rows) / n, 4),
                "Predicted Boundaries": round(sum(float(r["Predicted Boundaries"]) for r in lect_rows) / n, 2),
            }
        )

        f1.write_csv(
            out_root / "evaluation_table.csv",
            lect_rows,
            list(lect_rows[0].keys()) if lect_rows else [],
        )
        (out_root / "evaluation_summary.json").write_text(
            json.dumps({"experiment": "expF2_prominence", "setting": setting, "macro": all_summary[-1], "lectures": lect_rows}, indent=2),
            encoding="utf-8",
        )

    tables = repo_root / "thesis_project" / "tables"
    f1.write_csv(
        tables / "expF2_prominence_summary.csv",
        all_summary,
        [
            "Setting",
            "marker_window_pm_sec",
            "confirm_threshold",
            "prominence_min",
            "context_span",
            "min_distance_sec",
            "Precision",
            "Recall",
            "F1",
            "Predicted Boundaries",
        ],
    )

    # Lecture-level wide F1 matrix (columns = prominence_min); prom0 = F1-best equivalent
    baseline_by_lecture: Dict[str, float] = {}
    for row in all_lecture_rows:
        if abs(float(row["prominence_min"])) < 1e-15:
            baseline_by_lecture[str(row["Lecture"])] = float(row["F1"])

    # Simpler lecture F1 matrix
    prom_levels = sorted(set(float(r["prominence_min"]) for r in all_lecture_rows))
    wide_header = ["Lecture"] + [f"prominence_min_{p}" for p in prom_levels]
    wide_rows: List[Dict[str, object]] = []
    for lec in lecture_ids:
        wr: Dict[str, object] = {"Lecture": lec}
        for row in all_lecture_rows:
            if row["Lecture"] == lec:
                wr[f"prominence_min_{row['prominence_min']}"] = row["F1"]
        wide_rows.append(wr)
    f1.write_csv(tables / "expF2_prominence_lecture_f1.csv", wide_rows, wide_header)

    imp_rows: List[Dict[str, object]] = []
    for pmin in prom_levels:
        if abs(pmin) < 1e-15:
            continue
        imp = sum(
            1
            for lec in lecture_ids
            for row in all_lecture_rows
            if row["Lecture"] == lec
            and abs(float(row["prominence_min"]) - pmin) < 1e-15
            and float(row["F1"]) > baseline_by_lecture.get(lec, 0.0) + 1e-6
        )
        worse = sum(
            1
            for lec in lecture_ids
            for row in all_lecture_rows
            if row["Lecture"] == lec
            and abs(float(row["prominence_min"]) - pmin) < 1e-15
            and float(row["F1"]) < baseline_by_lecture.get(lec, 0.0) - 1e-6
        )
        imp_rows.append(
            {
                "prominence_min": pmin,
                "lectures_improved_vs_prom0": imp,
                "lectures_worse_vs_prom0": worse,
                "lectures_unchanged": len(lecture_ids) - imp - worse,
            }
        )
    if imp_rows:
        f1.write_csv(
            tables / "expF2_prominence_lecture_improvement_vs_prom0.csv",
            imp_rows,
            list(imp_rows[0].keys()),
        )

    master_csv = repo_root / "thesis_project" / "tables" / "experiment_master_table.csv"
    header = [
        "experiment_id",
        "setting",
        "lecture_id",
        "pred_count",
        "gt_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "tolerance_sec",
        "notes",
    ]
    with master_csv.open("r", newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    filtered = [r for r in existing if r.get("experiment_id") != "expF2_prominence"]
    filtered.extend(master_rows)
    f1.write_csv(master_csv, filtered, header)

    print("Saved:")
    print(f"  {tables / 'expF2_prominence_summary.csv'}")
    print(f"  {tables / 'expF2_prominence_lecture_f1.csv'}")
    if imp_rows:
        print(f"  {tables / 'expF2_prominence_lecture_improvement_vs_prom0.csv'}")
    print(f"  {master_csv}")


if __name__ == "__main__":
    main()
