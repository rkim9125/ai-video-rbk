import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def run_cmd(args: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def load_json(path: Path) -> Dict:
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


def evaluate_setting_for_lecture(
    repo_root: Path,
    ws: int,
    threshold: float,
    min_distance_sec: float,
    tolerance_sec: float,
    lecture_id: str,
    annotations_dir: Path,
) -> Dict[str, object]:
    src_dir = repo_root / "ai_video_rbk" / "src"
    transcripts_dir = repo_root / "ai_video_rbk" / "transcripts_vtt"

    setting_id = f"sentence_w{ws}_t{str(threshold).replace('.', '')}_d{int(min_distance_sec)}"
    out_dir = repo_root / "thesis_project" / "results" / "expB_window_size" / setting_id / lecture_id
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
            str(ws),
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

    metrics = load_json(eval_path)
    return {
        "Window Size": ws,
        "Lecture": lecture_id,
        "Precision": round(float(metrics["precision"]), 4),
        "Recall": round(float(metrics["recall"]), 4),
        "F1": round(float(metrics["f1"]), 4),
        "Predicted Boundaries": int(metrics["pred_count"]),
        "Ground Truth Boundaries": int(metrics["gt_count"]),
        "tp": int(metrics["TP"]),
        "fp": int(metrics["FP"]),
        "fn": int(metrics["FN"]),
        "tolerance_sec": float(metrics["tolerance_sec"]),
    }


def interpretation_for_ws(ws: int) -> str:
    mapping = {
        1: "very sensitive / noisy",
        3: "local context",
        5: "balanced baseline",
        7: "smoother context",
        10: "overly broad context",
    }
    return mapping.get(ws, "")


def update_master_table(repo_root: Path, rows: List[Dict[str, object]], threshold: float, min_distance_sec: float) -> None:
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

    filtered = [r for r in existing if r.get("experiment_id") != "expB_window_size"]
    for r in rows:
        setting = f"sentence_w{int(r['Window Size'])}_t{str(threshold).replace('.', '')}_d{int(min_distance_sec)}"
        filtered.append(
            {
                "experiment_id": "expB_window_size",
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
                "notes": "",
            }
        )
    write_csv(master_csv, filtered, header)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment B: sentence-based window size sweep with fixed threshold/min-distance/tolerance."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path.",
    )
    parser.add_argument(
        "--lectures",
        nargs="*",
        default=[],
        help="Lecture IDs, e.g., lecture1 lecture2. Empty means auto-discover.",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="*",
        type=int,
        default=[1, 3, 5, 7, 10],
        help="Sentence window sizes to test (default: 1 3 5 7 10).",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-distance-seconds", type=float, default=20.0)
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument(
        "--annotations-dir",
        default="ai_video_rbk/annotations",
        help="Annotation directory relative to repo root (default: ai_video_rbk/annotations).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    annotations_dir = (repo_root / args.annotations_dir).resolve()
    lecture_ids = args.lectures if args.lectures else discover_lectures(repo_root, annotations_dir)
    if not lecture_ids:
        raise RuntimeError("No lecture files discovered.")

    all_rows: List[Dict[str, object]] = []
    for ws in args.window_sizes:
        setting_rows: List[Dict[str, object]] = []
        print(f"\n=== Experiment B | ws={ws} ===")
        for lecture_id in lecture_ids:
            row = evaluate_setting_for_lecture(
                repo_root=repo_root,
                ws=ws,
                threshold=args.threshold,
                min_distance_sec=args.min_distance_seconds,
                tolerance_sec=args.tolerance_seconds,
                lecture_id=lecture_id,
                annotations_dir=annotations_dir,
            )
            setting_rows.append(row)
            all_rows.append(row)

        setting_id = f"sentence_w{ws}_t{str(args.threshold).replace('.', '')}_d{int(args.min_distance_seconds)}"
        setting_dir = repo_root / "thesis_project" / "results" / "expB_window_size" / setting_id

        # Per-setting lecture table
        write_csv(
            setting_dir / "evaluation_table.csv",
            setting_rows,
            [
                "Window Size",
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

        avg_precision = round(sum(float(r["Precision"]) for r in setting_rows) / len(setting_rows), 4)
        avg_recall = round(sum(float(r["Recall"]) for r in setting_rows) / len(setting_rows), 4)
        avg_f1 = round(sum(float(r["F1"]) for r in setting_rows) / len(setting_rows), 4)
        avg_pred = round(sum(float(r["Predicted Boundaries"]) for r in setting_rows) / len(setting_rows), 2)

        summary = {
            "experiment": "expB_window_size",
            "setting": setting_id,
            "window_size": ws,
            "representation": "sentence",
            "controlled_variables": {
                "threshold": args.threshold,
                "local_minima": True,
                "min_distance_sec": args.min_distance_seconds,
                "evaluation_tolerance_sec": args.tolerance_seconds,
            },
            "lecture_count": len(setting_rows),
            "macro_average": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "predicted_boundaries": avg_pred,
            },
            "lectures": setting_rows,
        }
        (setting_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Table 1: window-size summary
    ws_to_rows: Dict[int, List[Dict[str, object]]] = {}
    for r in all_rows:
        ws_to_rows.setdefault(int(r["Window Size"]), []).append(r)

    summary_rows: List[Dict[str, object]] = []
    for ws in sorted(ws_to_rows):
        rows = ws_to_rows[ws]
        summary_rows.append(
            {
                "Window Size": ws,
                "Precision": round(sum(float(r["Precision"]) for r in rows) / len(rows), 4),
                "Recall": round(sum(float(r["Recall"]) for r in rows) / len(rows), 4),
                "F1": round(sum(float(r["F1"]) for r in rows) / len(rows), 4),
                "Predicted Boundaries": round(sum(float(r["Predicted Boundaries"]) for r in rows) / len(rows), 2),
                "Interpretation": interpretation_for_ws(ws),
            }
        )

    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expB_window_size_summary.csv",
        summary_rows,
        ["Window Size", "Precision", "Recall", "F1", "Predicted Boundaries", "Interpretation"],
    )

    # Table 2: lecture-level F1 table
    lecture_f1_rows: List[Dict[str, object]] = []
    for ws in sorted(ws_to_rows):
        rows = ws_to_rows[ws]
        by_lecture = {r["Lecture"]: float(r["F1"]) for r in rows}
        l1 = by_lecture.get("lecture1", 0.0)
        l2 = by_lecture.get("lecture2", 0.0)
        l3 = by_lecture.get("lecture3", 0.0)
        l4 = by_lecture.get("lecture4", 0.0)
        avg = round((l1 + l2 + l3 + l4) / 4, 4)
        lecture_f1_rows.append(
            {
                "Window Size": ws,
                "L1 F1": round(l1, 4),
                "L2 F1": round(l2, 4),
                "L3 F1": round(l3, 4),
                "L4 F1": round(l4, 4),
                "Avg F1": avg,
            }
        )

    write_csv(
        tables_dir / "expB_window_size_lecture_f1.csv",
        lecture_f1_rows,
        ["Window Size", "L1 F1", "L2 F1", "L3 F1", "L4 F1", "Avg F1"],
    )

    update_master_table(
        repo_root=repo_root,
        rows=all_rows,
        threshold=args.threshold,
        min_distance_sec=args.min_distance_seconds,
    )

    print("\nSaved:")
    print(f"- {tables_dir / 'expB_window_size_summary.csv'}")
    print(f"- {tables_dir / 'expB_window_size_lecture_f1.csv'}")
    print(f"- {tables_dir / 'experiment_master_table.csv'}")


if __name__ == "__main__":
    main()
