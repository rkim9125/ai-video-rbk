"""
Experiment F3 — Marker-gated boundaries with slide transitions as auxiliary signal.

Fixed (F1 best, no F2 prominence): sentence w=3, semantic confirm similarity < 0.60,
local minima, marker window ±15 s, min-distance 30 s.

Varies: slide proximity window (±seconds) and decision rule:
  F3-OR:  marker_near AND (semantic_confirm OR slide_near)
  F3-AND: marker_near AND semantic_confirm AND slide_near

Semantic path matches F1: local minimum + similarity < threshold, then marker gate.
OR adds slide transition times s with marker_near(s) (slide_near(s) is trivially true).
AND keeps only semantic dips whose boundary time is within ±slide_window of some slide time.
"""

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_expf1_helpers():
    p = Path(__file__).resolve().parent / "run_experiment_f1.py"
    spec = importlib.util.spec_from_file_location("expf1_helpers", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_experiment_f1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hms_to_seconds(hms: str) -> float:
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_slide_times(slide_path: Path) -> List[float]:
    """Same parsing as Experiment D (`slide_candidates`)."""
    if not slide_path.exists():
        return []
    times: List[float] = []
    for line in slide_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d{2}:\d{2}:\d{2})", line)
        if m:
            times.append(hms_to_seconds(m.group(1)))
    return sorted(times)


def is_marker_near(t: float, markers: List[float], window_sec: float) -> bool:
    return any(abs(float(m) - t) <= window_sec for m in markers)


def is_slide_near(t: float, slides: List[float], window_sec: float) -> bool:
    return any(abs(float(s) - t) <= window_sec for s in slides)


def merge_or_rule(
    semantic_marker_gated: List[Dict[str, Any]],
    slide_times: List[float],
    markers: List[float],
    marker_pm: float,
    slide_window_sec: float,
) -> List[Dict[str, Any]]:
    """
    marker AND (semantic OR slide_near): semantic dips already satisfy semantic;
    add slide transition times s with marker_near(s). For t=s, slide_near(t) is true
    for any slide_window_sec > 0, so OR predictions do not depend on slide_window_sec
    (sw20 vs sw30 stay identical unless candidate generation changes).
    """
    _ = slide_window_sec  # kept for CLI/grid parity with AND rule and thesis table
    out: List[Dict[str, Any]] = []
    for b in semantic_marker_gated:
        bb = dict(b)
        bb["reason"] = "marker_gated_semantic_or_slide"
        out.append(bb)

    for s in slide_times:
        if not is_marker_near(float(s), markers, marker_pm):
            continue
        out.append(
            {
                "boundary_index": len(out),
                "between_windows": [-1, -1],
                "boundary_time": float(s),
                "similarity": 1.0,
                "left_window_end": float(s),
                "right_window_start": float(s),
                "reason": "marker_slide_aux",
            }
        )

    out.sort(key=lambda x: float(x["boundary_time"]))
    return out


def merge_and_rule(
    semantic_marker_gated: List[Dict[str, Any]],
    slide_times: List[float],
    slide_window: float,
) -> List[Dict[str, Any]]:
    """marker AND semantic AND slide_near: filter semantic dips by slide proximity only."""
    out: List[Dict[str, Any]] = []
    for b in semantic_marker_gated:
        t = float(b["boundary_time"])
        if not is_slide_near(t, slide_times, slide_window):
            continue
        bb = dict(b)
        bb["reason"] = "marker_gated_semantic_and_slide"
        out.append(bb)
    return out


def run_grid() -> List[Tuple[str, str, float]]:
    """(setting_id, rule, slide_window_sec)."""
    return [
        ("F3_OR_sw20", "OR", 20.0),
        ("F3_OR_sw30", "OR", 30.0),
        ("F3_AND_sw20", "AND", 20.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment F3: marker + semantic + slide.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--marker-window-pm", type=float, default=15.0)
    parser.add_argument("--confirm-threshold", type=float, default=0.60)
    parser.add_argument("--min-distance-final", type=float, default=30.0)
    parser.add_argument("--marker-dedupe-sec", type=float, default=20.0)
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument("--annotations-dir", default="ai_video_rbk/annotations")
    parser.add_argument("--slides-dir", default="thesis_project/data/slide_transitions")
    parser.add_argument("--lectures", nargs="*", default=[])
    args = parser.parse_args()

    f1 = load_expf1_helpers()
    repo_root = Path(args.repo_root).resolve()
    ann_dir = (repo_root / args.annotations_dir).resolve()
    slides_dir = (repo_root / args.slides_dir).resolve()
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

    grid = run_grid()
    all_summary: List[Dict[str, object]] = []
    all_lecture_rows: List[Dict[str, object]] = []
    master_rows: List[Dict[str, str]] = []

    for setting_id, rule, slide_w in grid:
        setting = (
            f"{setting_id}_pm{int(args.marker_window_pm)}_c{str(args.confirm_threshold).replace('.', '')}_"
            f"md{int(args.min_distance_final)}"
        )
        out_root = repo_root / "thesis_project" / "results" / "expF3_slide" / setting
        lect_rows: List[Dict[str, object]] = []

        for lecture_id in lecture_ids:
            lec_cache = cache_root / f"sentence_w{args.window_size}" / lecture_id
            sim_path = f1.ensure_similarities_cached(repo_root, lecture_id, lec_cache, args.window_size, sc_mod)
            similarities = json.loads(sim_path.read_text(encoding="utf-8"))

            semantic_raw = sc_mod.detect_boundaries(
                similarities,
                threshold=float(args.confirm_threshold),
                require_local_minimum=True,
            )

            vtt_path = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lecture_id}.vtt"
            markers = f1.marker_times_from_vtt(vtt_path, marker_patterns, dedupe_min_distance=args.marker_dedupe_sec)

            gated = f1.gate_semantic_by_markers(
                semantic_raw,
                markers,
                delta_sec=float(args.marker_window_pm),
            )

            slide_times = load_slide_times(slides_dir / f"{lecture_id}_slides.txt")

            if rule == "OR":
                merged = merge_or_rule(
                    gated,
                    slide_times,
                    markers,
                    float(args.marker_window_pm),
                    float(slide_w),
                )
            else:
                merged = merge_and_rule(gated, slide_times, slide_w)

            final_bounds = sc_mod.filter_boundaries_by_min_distance(
                merged,
                min_distance_seconds=float(args.min_distance_final),
            )
            for idx, b in enumerate(final_bounds):
                b["boundary_index"] = idx

            n_sem_after_gate = len(gated)
            n_slide_aux = len([b for b in merged if b.get("reason") == "marker_slide_aux"])
            n_merged_before_md = len(merged)

            out_dir = out_root / lecture_id
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / f"{lecture_id}_boundaries.json"
            pred_path.write_text(json.dumps(final_bounds, indent=2, ensure_ascii=False), encoding="utf-8")

            gt_path = ann_dir / f"{lecture_id}_boundaries.txt"
            m = f1.evaluate_pred_json(repo_root, gt_path, pred_path, args.tolerance_seconds, scg)

            lect_rows.append(
                {
                    "Setting": setting,
                    "Rule": rule,
                    "slide_window_sec": slide_w,
                    "Lecture": lecture_id,
                    "marker_window_pm_sec": int(args.marker_window_pm),
                    "confirm_threshold": args.confirm_threshold,
                    "min_distance_sec": args.min_distance_final,
                    "Precision": round(float(m["precision"]), 4),
                    "Recall": round(float(m["recall"]), 4),
                    "F1": round(float(m["f1"]), 4),
                    "Predicted Boundaries": int(m["pred_count"]),
                    "GT Boundaries": int(m["gt_count"]),
                    "semantic_after_marker_gate": n_sem_after_gate,
                    "slide_aux_candidates": n_slide_aux,
                    "candidates_before_min_distance": n_merged_before_md,
                    "slide_times_count": len(slide_times),
                }
            )
            all_lecture_rows.append(dict(lect_rows[-1]))

            master_rows.append(
                {
                    "experiment_id": "expF3_slide",
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
                    "notes": f"rule={rule} slide_w={slide_w}s F1-best+slide",
                }
            )

        n = len(lect_rows)
        all_summary.append(
            {
                "Setting": setting,
                "Rule": rule,
                "slide_window_sec": slide_w,
                "marker_window_pm_sec": int(args.marker_window_pm),
                "confirm_threshold": args.confirm_threshold,
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

    tables = repo_root / "thesis_project" / "tables"
    f1.write_csv(
        tables / "expF3_slide_summary.csv",
        all_summary,
        [
            "Setting",
            "Rule",
            "slide_window_sec",
            "marker_window_pm_sec",
            "confirm_threshold",
            "min_distance_sec",
            "Precision",
            "Recall",
            "F1",
            "Predicted Boundaries",
        ],
    )

    # Lecture F1 wide: rows = lectures, cols = settings
    settings_list = [s["Setting"] for s in all_summary]
    wide_header = ["Lecture"] + settings_list
    wide_rows: List[Dict[str, object]] = []
    for lec in lecture_ids:
        wr: Dict[str, object] = {"Lecture": lec}
        for row in all_lecture_rows:
            if row["Lecture"] == lec:
                wr[str(row["Setting"])] = row["F1"]
        wide_rows.append(wr)
    f1.write_csv(tables / "expF3_slide_lecture_f1.csv", wide_rows, wide_header)

    f1_best_f1 = 0.1802
    delta_rows: List[Dict[str, object]] = []
    for lec in lecture_ids:
        dr: Dict[str, object] = {"Lecture": lec, "F1_best_F1_ref": f1_best_f1}
        for row in all_lecture_rows:
            if row["Lecture"] != lec:
                continue
            dr[f"deltaF1_vs_F1best_{row['Setting']}"] = round(float(row["F1"]) - f1_best_f1, 4)
        delta_rows.append(dr)
    hdr = ["Lecture", "F1_best_F1_ref"] + [f"deltaF1_vs_F1best_{s}" for s in settings_list]
    f1.write_csv(tables / "expF3_slide_lecture_delta_vs_f1best.csv", delta_rows, hdr)

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
    filtered = [r for r in existing if r.get("experiment_id") != "expF3_slide"]
    filtered.extend(master_rows)
    f1.write_csv(master_csv, filtered, header)

    print("Saved:")
    print(f"  {tables / 'expF3_slide_summary.csv'}")
    print(f"  {tables / 'expF3_slide_lecture_f1.csv'}")
    print(f"  {tables / 'expF3_slide_lecture_delta_vs_f1best.csv'}")
    print(f"  {master_csv}")


if __name__ == "__main__":
    main()
