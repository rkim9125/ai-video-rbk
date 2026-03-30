import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def run_cmd(args: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def hms_to_seconds(hms: str) -> float:
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds_to_hms(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_gt(gt_path: Path) -> List[Dict[str, object]]:
    rows = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$", line)
        if not m:
            continue
        ts, title = m.groups()
        rows.append({"time": hms_to_seconds(ts), "time_str": ts, "title": title})
    return rows


def evaluate(pred_times: List[float], gt: List[Dict[str, object]], tolerance_sec: float) -> Dict[str, float]:
    pred_times = sorted(pred_times)
    gt_used = set()
    tp = 0
    for p in pred_times:
        best = None
        best_diff = None
        for gi, g in enumerate(gt):
            if gi in gt_used:
                continue
            d = abs(float(g["time"]) - p)
            if d <= tolerance_sec and (best_diff is None or d < best_diff):
                best = gi
                best_diff = d
        if best is not None:
            gt_used.add(best)
            tp += 1
    fp = len(pred_times) - tp
    fn = len(gt) - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "pred_count": len(pred_times),
        "gt_count": len(gt),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def dedupe_by_min_distance(times: List[float], min_distance_sec: float) -> List[float]:
    if not times:
        return []
    times = sorted(times)
    selected = [times[0]]
    for t in times[1:]:
        if t - selected[-1] >= min_distance_sec:
            selected.append(t)
    return selected


def parse_vtt_blocks(vtt_path: Path) -> List[Dict[str, object]]:
    lines = vtt_path.read_text(encoding="utf-8").splitlines()
    blocks: List[Dict[str, object]] = []
    i = 0

    def ts_to_sec(ts: str) -> float:
        ts = ts.strip().replace(",", ".")
        p = ts.split(":")
        if len(p) == 3:
            hh, mm, ss = p
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
        if len(p) == 2:
            mm, ss = p
            return int(mm) * 60 + float(ss)
        raise ValueError(ts)

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
        tline = line
        i += 1
        txt = []
        while i < len(lines) and lines[i].strip():
            txt.append(lines[i].strip())
            i += 1
        a, b = [x.strip().split(" ")[0] for x in tline.split("-->")]
        text = " ".join(txt)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        blocks.append({"start": ts_to_sec(a), "end": ts_to_sec(b), "text": text})
    return blocks


def load_semantic_boundaries(
    repo_root: Path,
    lecture_id: str,
    window_size: int,
    threshold: float,
    min_distance_sec: float,
) -> List[float]:
    src_dir = repo_root / "ai_video_rbk" / "src"
    vtt = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lecture_id}.vtt"
    out_dir = repo_root / "thesis_project" / "results" / "expD_structural" / "_intermediate" / lecture_id
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = out_dir / "windows.json"
    boundaries = out_dir / "semantic_boundaries.json"
    embeddings = out_dir / "window_embeddings.npy"
    sims = out_dir / "similarities.json"

    run_cmd(
        [
            "python3",
            str(src_dir / "semantic_approach.py"),
            "--vtt",
            str(vtt),
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
            str(windows),
            "--embeddings-out",
            str(embeddings),
            "--sim-out",
            str(sims),
            "--boundaries-out",
            str(boundaries),
            "--threshold",
            str(threshold),
            "--min-distance-seconds",
            str(min_distance_sec),
        ],
        cwd=repo_root,
    )
    data = json.loads(boundaries.read_text(encoding="utf-8"))
    return [float(x["boundary_time"]) for x in data]


def marker_candidates(blocks: List[Dict[str, object]], marker_patterns: List[str]) -> List[float]:
    pats = [re.compile(p, re.IGNORECASE) for p in marker_patterns]
    out = []
    for b in blocks:
        txt = str(b["text"])
        if any(p.search(txt) for p in pats):
            out.append(float(b["start"]))
    return out


def filler_candidates(
    blocks: List[Dict[str, object]],
    fillers: List[str],
    threshold_count: int = 3,
    local_window_sec: float = 20.0,
) -> List[float]:
    # Count filler occurrences per subtitle block
    fill_re = re.compile(r"\b(" + "|".join(re.escape(x) for x in fillers) + r")\b", re.IGNORECASE)
    points: List[Tuple[float, int]] = []
    for b in blocks:
        c = len(fill_re.findall(str(b["text"])))
        points.append((float(b["start"]), c))

    # Smooth count in local time window and keep local peaks above threshold
    times = [t for t, _ in points]
    vals = []
    for i, (t, _) in enumerate(points):
        s = 0
        for tj, cj in points:
            if abs(tj - t) <= local_window_sec / 2.0:
                s += cj
        vals.append(s)

    out = []
    for i, t in enumerate(times):
        v = vals[i]
        left = vals[i - 1] if i > 0 else v
        right = vals[i + 1] if i < len(vals) - 1 else v
        if v >= threshold_count and v >= left and v >= right:
            out.append(t)
    return out


def slide_candidates(slide_path: Path) -> List[float]:
    if not slide_path.exists():
        return []
    times = []
    for line in slide_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d{2}:\d{2}:\d{2})", line)
        if m:
            times.append(hms_to_seconds(m.group(1)))
    return sorted(times)


def union_times(*groups: List[float], min_distance_sec: float) -> List[float]:
    merged: List[float] = []
    for g in groups:
        merged.extend(g)
    return dedupe_by_min_distance(merged, min_distance_sec=min_distance_sec)


def write_csv(path: Path, rows: List[Dict[str, object]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def update_master_table(repo_root: Path, rows: List[Dict[str, object]]) -> None:
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

    filtered = [r for r in existing if r.get("experiment_id") != "expD_structural"]
    for r in rows:
        filtered.append(
            {
                "experiment_id": "expD_structural",
                "setting": str(r["Model"]),
                "lecture_id": str(r["Lecture"]),
                "pred_count": int(r["Predicted Boundaries"]),
                "gt_count": int(r["Ground Truth Boundaries"]),
                "tp": "",
                "fp": "",
                "fn": "",
                "precision": float(r["Precision"]),
                "recall": float(r["Recall"]),
                "f1": float(r["F1"]),
                "tolerance_sec": "",
                "notes": "semantic+structural fusion",
            }
        )

    write_csv(master_csv, filtered, header)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment D: semantic + structural signal fusion")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--lectures", nargs="*", default=["lecture1", "lecture2", "lecture3", "lecture4"])
    parser.add_argument("--annotations-dir", default="ai_video_rbk/annotations")
    parser.add_argument("--slides-dir", default="thesis_project/data/slide_transitions")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-distance-seconds", type=float, default=20.0)
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ann_dir = (repo_root / args.annotations_dir).resolve()
    slides_dir = (repo_root / args.slides_dir).resolve()

    marker_patterns = [
        r"\bnow\b",
        r"\bnext\b",
        r"\bmove on\b",
        r"\banother important\b",
        r"\btoday we (will|are going to)\b",
        r"\blet'?s move on\b",
    ]
    fillers = ["um", "uh", "okay", "so", "well", "alright"]

    model_defs = [
        ("Baseline", True, False, False, False),
        ("Marker", False, True, False, False),
        ("Filler", False, False, True, False),
        ("Slide", False, False, False, True),
        ("+Marker", True, True, False, False),
        ("+Filler", True, False, True, False),
        ("+Slide", True, False, False, True),
        ("All", True, True, True, True),
    ]

    all_rows: List[Dict[str, object]] = []

    for lecture_id in args.lectures:
        vtt = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lecture_id}.vtt"
        gt = load_gt(ann_dir / f"{lecture_id}_boundaries.txt")
        blocks = parse_vtt_blocks(vtt)

        sem = load_semantic_boundaries(
            repo_root=repo_root,
            lecture_id=lecture_id,
            window_size=args.window_size,
            threshold=args.threshold,
            min_distance_sec=args.min_distance_seconds,
        )
        marker = dedupe_by_min_distance(marker_candidates(blocks, marker_patterns), args.min_distance_seconds)
        filler = dedupe_by_min_distance(filler_candidates(blocks, fillers), args.min_distance_seconds)
        slide = dedupe_by_min_distance(slide_candidates(slides_dir / f"{lecture_id}_slides.txt"), args.min_distance_seconds)

        for name, use_sem, use_marker, use_filler, use_slide in model_defs:
            pred = union_times(
                sem if use_sem else [],
                marker if use_marker else [],
                filler if use_filler else [],
                slide if use_slide else [],
                min_distance_sec=args.min_distance_seconds,
            )
            m = evaluate(pred, gt, args.tolerance_seconds)
            all_rows.append(
                {
                    "Model": name,
                    "Lecture": lecture_id,
                    "Precision": round(m["precision"], 4),
                    "Recall": round(m["recall"], 4),
                    "F1": round(m["f1"], 4),
                    "Predicted Boundaries": m["pred_count"],
                    "Ground Truth Boundaries": m["gt_count"],
                }
            )

    # Table: lecture-level
    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expD_lecture_level_table.csv",
        all_rows,
        ["Model", "Lecture", "Precision", "Recall", "F1", "Predicted Boundaries", "Ground Truth Boundaries"],
    )

    # Table: aggregated model comparison
    by_model: Dict[str, List[Dict[str, object]]] = {}
    for r in all_rows:
        by_model.setdefault(str(r["Model"]), []).append(r)

    summary_rows: List[Dict[str, object]] = []
    for name, use_sem, use_marker, use_filler, use_slide in model_defs:
        rows = by_model.get(name, [])
        if not rows:
            continue
        n = len(rows)
        summary_rows.append(
            {
                "Model": name,
                "Semantic": "✓" if use_sem else "×",
                "Marker": "✓" if use_marker else "×",
                "Filler": "✓" if use_filler else "×",
                "Slide": "✓" if use_slide else "×",
                "Precision": round(sum(float(x["Precision"]) for x in rows) / n, 4),
                "Recall": round(sum(float(x["Recall"]) for x in rows) / n, 4),
                "F1": round(sum(float(x["F1"]) for x in rows) / n, 4),
                "Predicted Boundaries": round(sum(float(x["Predicted Boundaries"]) for x in rows) / n, 2),
            }
        )

    write_csv(
        tables_dir / "expD_model_comparison.csv",
        summary_rows,
        ["Model", "Semantic", "Marker", "Filler", "Slide", "Precision", "Recall", "F1", "Predicted Boundaries"],
    )

    # Notes
    notes = {
        "experiment": "expD_structural",
        "window_size": args.window_size,
        "threshold": args.threshold,
        "min_distance_seconds": args.min_distance_seconds,
        "tolerance_seconds": args.tolerance_seconds,
        "slides_dir": str(slides_dir),
        "slide_files_missing": [
            lecture for lecture in args.lectures if not (slides_dir / f"{lecture}_slides.txt").exists()
        ],
    }
    out_notes = repo_root / "thesis_project" / "results" / "expD_structural" / "run_config.json"
    out_notes.parent.mkdir(parents=True, exist_ok=True)
    out_notes.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    update_master_table(repo_root, all_rows)

    print("Saved:")
    print(f"- {tables_dir / 'expD_model_comparison.csv'}")
    print(f"- {tables_dir / 'expD_lecture_level_table.csv'}")
    print(f"- {tables_dir / 'experiment_master_table.csv'}")
    print(f"- {out_notes}")


if __name__ == "__main__":
    main()
