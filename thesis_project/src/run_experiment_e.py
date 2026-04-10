"""
Experiment E — Prediction count control / boundary pruning (min-distance sweep).

Fixed: sentence windows, window size 3, threshold 0.55, local minima on, min-distance filter on.
Varies: min-distance seconds (20, 30, 45, 60).
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def run_cmd(args: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, object]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_lectures(repo_root: Path, annotations_dir: Path) -> List[str]:
    transcripts_dir = repo_root / "ai_video_rbk" / "transcripts_vtt"
    vtt_ids = {p.stem for p in transcripts_dir.glob("lecture*.vtt")}
    gt_ids = {p.name.replace("_boundaries.txt", "") for p in annotations_dir.glob("lecture*_boundaries.txt")}
    return sorted(vtt_ids.intersection(gt_ids))


def setting_label(min_dist: float) -> str:
    # E1=20s, E2=30s, E3=45s, E4=60s
    mapping = {20.0: "E1", 30.0: "E2", 45.0: "E3", 60.0: "E4"}
    return mapping.get(float(min_dist), f"md{int(min_dist)}")


def interpretation_for(min_dist: float) -> str:
    return {
        20.0: "baseline pruning",
        30.0: "fewer close duplicates",
        45.0: "stronger suppression",
        60.0: "very strict spacing",
    }.get(float(min_dist), "min-distance sweep")


def evaluate_single(
    repo_root: Path,
    lecture_id: str,
    window_size: int,
    threshold: float,
    min_distance_sec: float,
    tolerance_sec: float,
    annotations_dir: Path,
) -> Dict[str, object]:
    src_dir = repo_root / "ai_video_rbk" / "src"
    transcripts_dir = repo_root / "ai_video_rbk" / "transcripts_vtt"

    sid = f"sentence_w{window_size}_t{str(threshold).replace('.', '')}_md{int(min_distance_sec)}"
    out_dir = repo_root / "thesis_project" / "results" / "expE_prediction_pruning" / sid / lecture_id
    out_dir.mkdir(parents=True, exist_ok=True)

    vtt_path = transcripts_dir / f"{lecture_id}.vtt"
    gt_path = annotations_dir / f"{lecture_id}_boundaries.txt"

    windows_path = out_dir / "windows.json"
    embeddings_path = out_dir / "window_embeddings.npy"
    similarities_path = out_dir / "similarities.json"
    boundaries_path = out_dir / f"{lecture_id}_boundaries.json"
    eval_path = out_dir / "evaluation_report.json"

    run_cmd(
        [
            "python3",
            str(src_dir / "semantic_approach.py"),
            "--vtt",
            str(vtt_path),
            "--outdir",
            str(out_dir),
            "--window-size",
            str(window_size),
            "--stride",
            "1",
            "--min-chars",
            "25",
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            "python3",
            str(src_dir / "semantic check.py"),
            "--windows",
            str(windows_path),
            "--embeddings-out",
            str(embeddings_path),
            "--sim-out",
            str(similarities_path),
            "--boundaries-out",
            str(boundaries_path),
            "--threshold",
            str(threshold),
            "--min-distance-seconds",
            str(min_distance_sec),
        ],
        cwd=repo_root,
    )

    run_cmd(
        [
            "python3",
            str(src_dir / "semantic_check_ground.py"),
            "--gt",
            str(gt_path),
            "--pred",
            str(boundaries_path),
            "--tolerance",
            str(tolerance_sec),
            "--report-out",
            str(eval_path),
        ],
        cwd=repo_root,
    )

    m = load_json(eval_path)
    return {
        "min_distance_sec": min_distance_sec,
        "Lecture": lecture_id,
        "Precision": round(float(m["precision"]), 4),
        "Recall": round(float(m["recall"]), 4),
        "F1": round(float(m["f1"]), 4),
        "Predicted Boundaries": int(m["pred_count"]),
        "Ground Truth Boundaries": int(m["gt_count"]),
        "tp": int(m["TP"]),
        "fp": int(m["FP"]),
        "fn": int(m["FN"]),
        "tolerance_sec": float(m["tolerance_sec"]),
    }


def update_master_table(
    repo_root: Path,
    rows: List[Dict[str, object]],
    window_size: int,
    threshold: float,
) -> None:
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

    existing: List[Dict[str, str]] = []
    if master_csv.exists():
        with master_csv.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    filtered = [r for r in existing if r.get("experiment_id") != "expE_prediction_pruning"]
    for r in rows:
        md = float(r["min_distance_sec"])
        setting = f"sentence_w{window_size}_t{str(threshold).replace('.', '')}_md{int(md)}"
        filtered.append(
            {
                "experiment_id": "expE_prediction_pruning",
                "setting": setting,
                "lecture_id": r["Lecture"],
                "pred_count": r["Predicted Boundaries"],
                "gt_count": r["Ground Truth Boundaries"],
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "precision": r["Precision"],
                "recall": r["Recall"],
                "f1": r["F1"],
                "tolerance_sec": r["tolerance_sec"],
                "notes": f"min_distance={md}s; local_minima=yes",
            }
        )
    write_csv(master_csv, filtered, header)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment E: min-distance sweep for boundary pruning (prediction count control)."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--window-size", type=int, default=3, help="Sentence window size (default: 3).")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--min-distances",
        type=float,
        nargs="+",
        default=[20.0, 30.0, 45.0, 60.0],
        help="Min-distance values in seconds to sweep (default: 20 30 45 60).",
    )
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument(
        "--annotations-dir",
        default="ai_video_rbk/annotations",
        help="Annotation directory relative to repo root.",
    )
    parser.add_argument(
        "--lectures",
        nargs="*",
        default=[],
        help="Lecture IDs. Empty means auto-discover from vtt + annotations.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    annotations_dir = (repo_root / args.annotations_dir).resolve()
    lecture_ids = args.lectures if args.lectures else discover_lectures(repo_root, annotations_dir)
    if not lecture_ids:
        raise RuntimeError("No lectures found for Experiment E.")

    min_distances = sorted(set(float(x) for x in args.min_distances))
    all_rows: List[Dict[str, object]] = []

    for md in min_distances:
        sid = f"sentence_w{args.window_size}_t{str(args.threshold).replace('.', '')}_md{int(md)}"
        print(f"\n=== Experiment E | min_distance={md}s | {sid} ===")
        md_rows: List[Dict[str, object]] = []
        for lecture_id in lecture_ids:
            row = evaluate_single(
                repo_root=repo_root,
                lecture_id=lecture_id,
                window_size=args.window_size,
                threshold=args.threshold,
                min_distance_sec=md,
                tolerance_sec=args.tolerance_seconds,
                annotations_dir=annotations_dir,
            )
            md_rows.append(row)
            all_rows.append(row)

        setting_dir = repo_root / "thesis_project" / "results" / "expE_prediction_pruning" / sid
        write_csv(
            setting_dir / "evaluation_table.csv",
            md_rows,
            [
                "min_distance_sec",
                "Lecture",
                "Precision",
                "Recall",
                "F1",
                "Predicted Boundaries",
                "Ground Truth Boundaries",
                "tp",
                "fp",
                "fn",
                "tolerance_sec",
            ],
        )
        n = len(md_rows)
        summary = {
            "experiment": "expE_prediction_pruning",
            "setting": sid,
            "representation": "sentence",
            "window_size": args.window_size,
            "threshold": args.threshold,
            "local_minima": True,
            "min_distance_sec": md,
            "controlled_variables": {
                "threshold": args.threshold,
                "evaluation_tolerance_sec": args.tolerance_seconds,
            },
            "macro_average": {
                "precision": round(sum(float(r["Precision"]) for r in md_rows) / n, 4),
                "recall": round(sum(float(r["Recall"]) for r in md_rows) / n, 4),
                "f1": round(sum(float(r["F1"]) for r in md_rows) / n, 4),
                "predicted_boundaries": round(sum(float(r["Predicted Boundaries"]) for r in md_rows) / n, 2),
            },
            "lectures": md_rows,
        }
        (setting_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Summary table (E1–E4 style)
    by_md: Dict[float, List[Dict[str, object]]] = {}
    for r in all_rows:
        by_md.setdefault(float(r["min_distance_sec"]), []).append(r)

    summary_rows: List[Dict[str, object]] = []
    for md in min_distances:
        rows = by_md[float(md)]
        n = len(rows)
        summary_rows.append(
            {
                "Setting": setting_label(md),
                "Threshold": args.threshold,
                "Local Minima": "Yes",
                "Min-distance": f"{int(md)}s",
                "Prominence": "-",
                "Precision": round(sum(float(r["Precision"]) for r in rows) / n, 4),
                "Recall": round(sum(float(r["Recall"]) for r in rows) / n, 4),
                "F1": round(sum(float(r["F1"]) for r in rows) / n, 4),
                "Predicted Boundaries": round(sum(float(r["Predicted Boundaries"]) for r in rows) / n, 2),
                "Interpretation": interpretation_for(md),
            }
        )

    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expE_pruning_summary.csv",
        summary_rows,
        [
            "Setting",
            "Threshold",
            "Local Minima",
            "Min-distance",
            "Prominence",
            "Precision",
            "Recall",
            "F1",
            "Predicted Boundaries",
            "Interpretation",
        ],
    )

    # Lecture-level F1 matrix
    lecture_ids_sorted = sorted(lecture_ids)
    lecture_f1_rows: List[Dict[str, object]] = []
    for md in min_distances:
        rows = by_md[float(md)]
        by_lecture = {str(r["Lecture"]): float(r["F1"]) for r in rows}
        row: Dict[str, object] = {"Setting": f"{int(md)}s"}
        vals: List[float] = []
        for lecture in lecture_ids_sorted:
            key = lecture.replace("lecture", "L") + " F1"
            val = round(by_lecture.get(lecture, 0.0), 4)
            row[key] = val
            vals.append(val)
        row["Avg F1"] = round(sum(vals) / len(vals), 4) if vals else 0.0
        lecture_f1_rows.append(row)

    lecture_header = ["Setting"] + [lec.replace("lecture", "L") + " F1" for lec in lecture_ids_sorted] + ["Avg F1"]
    write_csv(tables_dir / "expE_pruning_lecture_f1.csv", lecture_f1_rows, lecture_header)

    update_master_table(
        repo_root=repo_root,
        rows=all_rows,
        window_size=args.window_size,
        threshold=args.threshold,
    )

    print("\nSaved:")
    print(f"- {tables_dir / 'expE_pruning_summary.csv'}")
    print(f"- {tables_dir / 'expE_pruning_lecture_f1.csv'}")
    print(f"- {tables_dir / 'experiment_master_table.csv'}")


if __name__ == "__main__":
    main()
