import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    method_id: str
    representation: str
    window_unit: str
    window_size: str
    threshold: float
    local_minima: bool
    min_distance_sec: float
    tolerance_sec: float


def run_cmd(args: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def parse_vtt(vtt_path: Path) -> List[Dict[str, object]]:
    lines = vtt_path.read_text(encoding="utf-8").splitlines()
    blocks: List[Dict[str, object]] = []
    i = 0

    def ts_to_seconds(ts: str) -> float:
        ts = ts.strip().replace(",", ".")
        parts = ts.split(":")
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + float(ss)
        raise ValueError(f"Invalid timestamp: {ts}")

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper() == "WEBVTT":
            i += 1
            continue

        if "-->" not in line and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        time_line = line
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        start_str, end_str = [part.strip().split(" ")[0] for part in time_line.split("-->")]
        blocks.append(
            {
                "start": ts_to_seconds(start_str),
                "end": ts_to_seconds(end_str),
                "text": " ".join(text_lines).strip(),
            }
        )

    return blocks


def build_time_windows(vtt_path: Path, out_windows_path: Path, window_seconds: float) -> None:
    blocks = parse_vtt(vtt_path)
    if not blocks:
        out_windows_path.write_text("[]", encoding="utf-8")
        return

    t_min = float(blocks[0]["start"])
    t_max = float(blocks[-1]["end"])
    windows = []
    window_id = 0
    start = t_min

    while start < t_max:
        end = start + window_seconds
        texts = []
        for b in blocks:
            b_start = float(b["start"])
            b_end = float(b["end"])
            # overlap between subtitle block and this time window
            if b_end > start and b_start < end:
                txt = str(b["text"]).strip()
                if txt:
                    texts.append(txt)

        merged_text = " ".join(texts).strip()
        if merged_text:
            windows.append(
                {
                    "window_id": window_id,
                    "start": start,
                    "end": end,
                    "text": merged_text,
                }
            )
            window_id += 1
        start = end

    out_windows_path.write_text(json.dumps(windows, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, object]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_single_lecture(
    repo_root: Path,
    lecture_id: str,
    setting_dir: Path,
    config: RunConfig,
    vtt_path: Path,
    gt_path: Path,
    sentence_window_size: int,
    time_window_seconds: float,
) -> Dict[str, object]:
    src_dir = repo_root / "ai_video_rbk" / "src"
    work_dir = setting_dir / lecture_id
    work_dir.mkdir(parents=True, exist_ok=True)

    windows_path = work_dir / "windows.json"
    similarities_path = work_dir / "similarities.json"
    boundaries_path = work_dir / f"{lecture_id}_boundaries.json"
    embeddings_path = work_dir / "window_embeddings.npy"
    eval_path = work_dir / "evaluation_report.json"

    if config.representation == "sentence":
        run_cmd(
            [
                "python3",
                str(src_dir / "semantic_approach.py"),
                "--vtt",
                str(vtt_path),
                "--outdir",
                str(work_dir),
                "--window-size",
                str(sentence_window_size),
                "--stride",
                "1",
                "--min-chars",
                "25",
            ],
            cwd=repo_root,
        )
        # semantic_approach writes windows.json in outdir already.
    else:
        build_time_windows(vtt_path=vtt_path, out_windows_path=windows_path, window_seconds=time_window_seconds)

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
            str(config.threshold),
            "--min-distance-seconds",
            str(config.min_distance_sec),
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
            str(config.tolerance_sec),
            "--report-out",
            str(eval_path),
        ],
        cwd=repo_root,
    )

    metrics = load_json(eval_path)
    pred = load_json(boundaries_path)

    # Save a top-level copy named exactly as requested structure.
    top_level_copy = setting_dir / f"{lecture_id}_boundaries.json"
    top_level_copy.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "Method": config.method_id,
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


def aggregate_rows(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = len(rows)
    return {
        "precision": round(sum(float(r["Precision"]) for r in rows) / n, 4),
        "recall": round(sum(float(r["Recall"]) for r in rows) / n, 4),
        "f1": round(sum(float(r["F1"]) for r in rows) / n, 4),
    }


def update_master_table(master_csv: Path, config: RunConfig, rows: List[Dict[str, object]]) -> None:
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
            reader = csv.DictReader(f)
            existing = list(reader)

    # Remove old rows for this setting then append fresh ones.
    filtered = [r for r in existing if r.get("setting") != config.method_id]
    for r in rows:
        filtered.append(
            {
                "experiment_id": "expA_representation",
                "setting": config.method_id,
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


def run_setting(
    repo_root: Path,
    lecture_ids: List[str],
    config: RunConfig,
    sentence_window_size: int,
    time_window_seconds: float,
    annotations_dir: Path,
) -> None:
    transcripts_dir = repo_root / "ai_video_rbk" / "transcripts_vtt"
    setting_dir = repo_root / "thesis_project" / "results" / "expA_representation" / config.method_id
    setting_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for lecture_id in lecture_ids:
        vtt_path = transcripts_dir / f"{lecture_id}.vtt"
        gt_path = annotations_dir / f"{lecture_id}_boundaries.txt"
        if not vtt_path.exists() or not gt_path.exists():
            print(f"SKIP {lecture_id}: missing input files")
            continue
        print(f"\n=== {config.method_id} | {lecture_id} ===")
        row = evaluate_single_lecture(
            repo_root=repo_root,
            lecture_id=lecture_id,
            setting_dir=setting_dir,
            config=config,
            vtt_path=vtt_path,
            gt_path=gt_path,
            sentence_window_size=sentence_window_size,
            time_window_seconds=time_window_seconds,
        )
        rows.append(row)

    lecture_header = [
        "Method",
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
    ]
    write_csv(setting_dir / "evaluation_table.csv", rows, lecture_header)

    macro = aggregate_rows(rows)
    summary = {
        "experiment": "expA_representation",
        "setting": config.method_id,
        "controlled_variables": {
            "threshold": config.threshold,
            "local_minima": config.local_minima,
            "min_distance_sec": config.min_distance_sec,
            "evaluation_tolerance_sec": config.tolerance_sec,
        },
        "independent_variable": "representation",
        "representation": config.representation,
        "window_unit": config.window_unit,
        "window_size": config.window_size,
        "lecture_count": len(rows),
        "macro_average": macro,
        "lectures": rows,
    }
    (setting_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    master_csv = repo_root / "thesis_project" / "tables" / "experiment_master_table.csv"
    update_master_table(master_csv, config, rows)
    print(f"\nSaved setting outputs to: {setting_dir}")


def discover_lectures(repo_root: Path, annotations_dir: Path) -> List[str]:
    transcripts_dir = repo_root / "ai_video_rbk" / "transcripts_vtt"
    vtt_ids = {p.stem for p in transcripts_dir.glob("lecture*.vtt")}
    gt_ids = {p.name.replace("_boundaries.txt", "") for p in annotations_dir.glob("lecture*_boundaries.txt")}
    ids = sorted(vtt_ids.intersection(gt_ids))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment A (representation) with fixed threshold/min-distance/tolerance and auto summary tables."
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
        help="Lecture IDs (e.g., lecture1 lecture2). Empty means auto-discover.",
    )
    parser.add_argument(
        "--run-sentence",
        action="store_true",
        help="Run sentence-based setting (default runs both if neither flag is set).",
    )
    parser.add_argument(
        "--run-time",
        action="store_true",
        help="Run time-based setting (default runs both if neither flag is set).",
    )
    parser.add_argument(
        "--sentence-window-size",
        type=int,
        default=5,
        help="Sentence-based window size (default: 5).",
    )
    parser.add_argument(
        "--time-window-seconds",
        type=float,
        default=10.0,
        help="Time-based fixed window length in seconds (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Boundary detection threshold (default: 0.55).",
    )
    parser.add_argument(
        "--min-distance-seconds",
        type=float,
        default=20.0,
        help="Minimum distance between boundaries (default: 20).",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=30.0,
        help="Evaluation matching tolerance in seconds (default: 30).",
    )
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
        raise RuntimeError("No lectures found. Check transcripts_vtt and annotations naming.")

    run_sentence = args.run_sentence
    run_time = args.run_time
    if not run_sentence and not run_time:
        run_sentence = True
        run_time = True

    sentence_cfg = RunConfig(
        method_id=f"sentence_w{args.sentence_window_size}_t{str(args.threshold).replace('.', '')}_d{int(args.min_distance_seconds)}",
        representation="sentence",
        window_unit="sentences",
        window_size=str(args.sentence_window_size),
        threshold=args.threshold,
        local_minima=True,
        min_distance_sec=args.min_distance_seconds,
        tolerance_sec=args.tolerance_seconds,
    )
    time_cfg = RunConfig(
        method_id=f"time_{int(args.time_window_seconds)}s_t{str(args.threshold).replace('.', '')}_d{int(args.min_distance_seconds)}",
        representation="time",
        window_unit="seconds",
        window_size=str(int(args.time_window_seconds)),
        threshold=args.threshold,
        local_minima=True,
        min_distance_sec=args.min_distance_seconds,
        tolerance_sec=args.tolerance_seconds,
    )

    if run_sentence:
        run_setting(
            repo_root=repo_root,
            lecture_ids=lecture_ids,
            config=sentence_cfg,
            sentence_window_size=args.sentence_window_size,
            time_window_seconds=args.time_window_seconds,
            annotations_dir=annotations_dir,
        )
    if run_time:
        run_setting(
            repo_root=repo_root,
            lecture_ids=lecture_ids,
            config=time_cfg,
            sentence_window_size=args.sentence_window_size,
            time_window_seconds=args.time_window_seconds,
            annotations_dir=annotations_dir,
        )

    print("\nExperiment A completed.")


if __name__ == "__main__":
    main()
