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


def evaluate_single(
    repo_root: Path,
    lecture_id: str,
    window_size: int,
    threshold: float,
    min_distance_sec: float,
    tolerance_sec: float,
    annotations_dir: Path,
    rule_id: str,
) -> Dict[str, object]:
    src_dir = repo_root / "ai_video_rbk" / "src"
    transcripts_dir = repo_root / "ai_video_rbk" / "transcripts_vtt"

    setting_id = f"sentence_w{window_size}_t{str(threshold).replace('.', '')}_{rule_id}"
    out_dir = repo_root / "thesis_project" / "results" / "expC_boundary_rule" / setting_id / lecture_id
    out_dir.mkdir(parents=True, exist_ok=True)

    vtt_path = transcripts_dir / f"{lecture_id}.vtt"
    gt_path = annotations_dir / f"{lecture_id}_boundaries.txt"

    windows_path = out_dir / "windows.json"
    embeddings_path = out_dir / "window_embeddings.npy"
    similarities_path = out_dir / "similarities.json"
    boundaries_path = out_dir / f"{lecture_id}_boundaries.json"
    eval_path = out_dir / "evaluation_report.json"

    # 1) Build sentence windows (fixed representation + window size)
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

    # 2) Boundary rule variants
    rule_flags: List[str] = []
    if rule_id == "threshold_only":
        rule_flags = ["--no-local-minima", "--disable-min-distance-filter"]
    elif rule_id == "threshold_localmin":
        rule_flags = ["--disable-min-distance-filter"]
    elif rule_id == "threshold_localmin_mindist":
        rule_flags = []
    else:
        raise ValueError(f"Unknown rule_id: {rule_id}")

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
            *rule_flags,
        ],
        cwd=repo_root,
    )

    # 3) Evaluate
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
        "Rule": rule_id,
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


def rule_interpretation(rule_id: str) -> str:
    mapping = {
        "threshold_only": "most sensitive / highest noise risk",
        "threshold_localmin": "noise-suppressed local dips",
        "threshold_localmin_mindist": "redundancy-reduced practical segmentation",
    }
    return mapping.get(rule_id, "")


def label_rule(rule_id: str) -> str:
    mapping = {
        "threshold_only": "Threshold only",
        "threshold_localmin": "Threshold + Local minima",
        "threshold_localmin_mindist": "Threshold + Local minima + Min-distance",
    }
    return mapping.get(rule_id, rule_id)


def update_master_table(
    repo_root: Path,
    rows: List[Dict[str, object]],
    window_size: int,
    threshold: float,
    min_distance_sec: float,
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

    filtered = [r for r in existing if r.get("experiment_id") != "expC_boundary_rule"]
    for r in rows:
        setting = f"sentence_w{window_size}_t{str(threshold).replace('.', '')}_{r['Rule']}"
        filtered.append(
            {
                "experiment_id": "expC_boundary_rule",
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
                "notes": f"min_distance={min_distance_sec}s",
            }
        )
    write_csv(master_csv, filtered, header)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment C: boundary decision rule comparison.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--window-size", type=int, default=3, help="Fixed sentence window size (default: 3).")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-distance-seconds", type=float, default=20.0)
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
        raise RuntimeError("No lectures found for Experiment C.")

    rules = [
        "threshold_only",
        "threshold_localmin",
        "threshold_localmin_mindist",
    ]

    all_rows: List[Dict[str, object]] = []
    for rule_id in rules:
        print(f"\n=== Experiment C | rule={rule_id} ===")
        rule_rows: List[Dict[str, object]] = []
        for lecture_id in lecture_ids:
            row = evaluate_single(
                repo_root=repo_root,
                lecture_id=lecture_id,
                window_size=args.window_size,
                threshold=args.threshold,
                min_distance_sec=args.min_distance_seconds,
                tolerance_sec=args.tolerance_seconds,
                annotations_dir=annotations_dir,
                rule_id=rule_id,
            )
            rule_rows.append(row)
            all_rows.append(row)

        setting_id = f"sentence_w{args.window_size}_t{str(args.threshold).replace('.', '')}_{rule_id}"
        setting_dir = repo_root / "thesis_project" / "results" / "expC_boundary_rule" / setting_id
        write_csv(
            setting_dir / "evaluation_table.csv",
            rule_rows,
            [
                "Rule",
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

        n = len(rule_rows)
        summary = {
            "experiment": "expC_boundary_rule",
            "setting": setting_id,
            "representation": "sentence",
            "window_size": args.window_size,
            "rule": rule_id,
            "controlled_variables": {
                "threshold": args.threshold,
                "min_distance_sec": args.min_distance_seconds,
                "evaluation_tolerance_sec": args.tolerance_seconds,
            },
            "macro_average": {
                "precision": round(sum(float(r["Precision"]) for r in rule_rows) / n, 4),
                "recall": round(sum(float(r["Recall"]) for r in rule_rows) / n, 4),
                "f1": round(sum(float(r["F1"]) for r in rule_rows) / n, 4),
                "predicted_boundaries": round(sum(float(r["Predicted Boundaries"]) for r in rule_rows) / n, 2),
            },
            "lectures": rule_rows,
        }
        (setting_dir / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Main summary table by rule
    by_rule: Dict[str, List[Dict[str, object]]] = {}
    for r in all_rows:
        by_rule.setdefault(str(r["Rule"]), []).append(r)

    summary_rows: List[Dict[str, object]] = []
    for rule_id in rules:
        rows = by_rule[rule_id]
        n = len(rows)
        summary_rows.append(
            {
                "Rule": label_rule(rule_id),
                "Precision": round(sum(float(r["Precision"]) for r in rows) / n, 4),
                "Recall": round(sum(float(r["Recall"]) for r in rows) / n, 4),
                "F1": round(sum(float(r["F1"]) for r in rows) / n, 4),
                "Predicted Boundaries": round(sum(float(r["Predicted Boundaries"]) for r in rows) / n, 2),
                "Interpretation": rule_interpretation(rule_id),
            }
        )

    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expC_rule_summary.csv",
        summary_rows,
        ["Rule", "Precision", "Recall", "F1", "Predicted Boundaries", "Interpretation"],
    )

    # Lecture-level F1 table (rule x lecture)
    lecture_ids_sorted = sorted(lecture_ids)
    lecture_f1_rows: List[Dict[str, object]] = []
    for rule_id in rules:
        rows = by_rule[rule_id]
        by_lecture = {str(r["Lecture"]): float(r["F1"]) for r in rows}
        row: Dict[str, object] = {"Rule": label_rule(rule_id)}
        vals: List[float] = []
        for lecture in lecture_ids_sorted:
            key = lecture.replace("lecture", "L") + " F1"
            val = round(by_lecture.get(lecture, 0.0), 4)
            row[key] = val
            vals.append(val)
        row["Avg F1"] = round(sum(vals) / len(vals), 4) if vals else 0.0
        lecture_f1_rows.append(row)

    lecture_header = ["Rule"] + [lec.replace("lecture", "L") + " F1" for lec in lecture_ids_sorted] + ["Avg F1"]
    write_csv(tables_dir / "expC_rule_lecture_f1.csv", lecture_f1_rows, lecture_header)

    update_master_table(
        repo_root=repo_root,
        rows=all_rows,
        window_size=args.window_size,
        threshold=args.threshold,
        min_distance_sec=args.min_distance_seconds,
    )

    print("\nSaved:")
    print(f"- {tables_dir / 'expC_rule_summary.csv'}")
    print(f"- {tables_dir / 'expC_rule_lecture_f1.csv'}")
    print(f"- {tables_dir / 'experiment_master_table.csv'}")


if __name__ == "__main__":
    main()
