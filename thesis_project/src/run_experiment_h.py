"""
Experiment H: Hierarchical segmentation + marker fusion (Stage 3).

Stage 1: same as Experiment G (oracle coarse GT vs predicted semantic coarse).
Stage 2: segment-local re-similarity (defaults t=0.60, md=30s).
Stage 3 (marker fusion): reuse Experiment D marker extraction on VTT;
  merge Stage-2 boundaries with marker timestamps:
  - drop a marker if it falls within ±marker_skip_sec of any Stage-2 boundary time;
  - merge lists and apply min-distance pruning (default 30s; prefers lower
    semantic similarity like Experiment G when two candidates conflict).

Outputs:
  - thesis_project/tables/expH_marker_hierarchical_summary.csv
  - thesis_project/tables/expH_marker_hierarchical_lecture_level.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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


def parse_boundary_txt(path: Path) -> List[float]:
    out: List[float] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\d{2}:\d{2}:\d{2})\s+.*$", line)
        if not m:
            continue
        h, mm, s = m.group(1).split(":")
        out.append(int(h) * 3600 + int(mm) * 60 + int(s))
    return sorted(out)


def is_local_minimum(values: List[float], idx: int) -> bool:
    if idx <= 0 or idx >= len(values) - 1:
        return False
    return values[idx] < values[idx - 1] and values[idx] < values[idx + 1]


def semantic_candidates(similarities: List[Dict[str, object]], threshold: float) -> List[Dict[str, object]]:
    vals = [float(x["similarity"]) for x in similarities]
    out: List[Dict[str, object]] = []
    for i, item in enumerate(similarities):
        sim = float(item["similarity"])
        if sim < threshold and is_local_minimum(vals, i):
            out.append(
                {
                    "boundary_time": float(item["right_start"]),
                    "similarity": sim,
                    "left_window_end": float(item["left_end"]),
                    "right_window_start": float(item["right_start"]),
                    "between_windows": [int(item["left_window_id"]), int(item["right_window_id"])],
                    "reason": "low_similarity_and_local_minimum",
                }
            )
    return out


def min_distance_filter(bounds: List[Dict[str, object]], min_distance_sec: float) -> List[Dict[str, object]]:
    if not bounds:
        return []
    sorted_bounds = sorted(bounds, key=lambda b: float(b["boundary_time"]))
    selected: List[Dict[str, object]] = []
    for b in sorted_bounds:
        if not selected:
            selected.append(b)
            continue
        last = selected[-1]
        dt = float(b["boundary_time"]) - float(last["boundary_time"])
        if dt >= min_distance_sec:
            selected.append(b)
            continue
        if float(b.get("similarity", 1.0)) < float(last.get("similarity", 1.0)):
            selected[-1] = b
    return selected


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def stage2_with_segments(
    windows: List[Dict[str, object]],
    embeddings: np.ndarray,
    coarse_bounds: List[float],
    total_end: float,
    threshold: float,
    min_distance_sec: float,
) -> List[Dict[str, object]]:
    segments: List[Tuple[float, float]] = []
    starts = [0.0] + coarse_bounds
    ends = coarse_bounds + [total_end]
    for s, e in zip(starts, ends):
        if e > s:
            segments.append((s, e))

    out: List[Dict[str, object]] = []
    for s, e in segments:
        seg_indices = [
            i
            for i, w in enumerate(windows)
            if float(w["start"]) >= s and float(w["end"]) < e
        ]
        if len(seg_indices) < 2:
            continue
        seg_set = set(seg_indices)

        seg_sims: List[Dict[str, object]] = []
        for i in seg_indices:
            j = i + 1
            if j not in seg_set:
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            seg_sims.append(
                {
                    "left_idx": i,
                    "right_idx": j,
                    "similarity": sim,
                    "left_start": float(windows[i]["start"]),
                    "left_end": float(windows[i]["end"]),
                    "right_start": float(windows[j]["start"]),
                    "right_end": float(windows[j]["end"]),
                }
            )
        if not seg_sims:
            continue

        sim_values = [float(x["similarity"]) for x in seg_sims]
        in_seg: List[Dict[str, object]] = []
        for k, item in enumerate(seg_sims):
            sim = float(item["similarity"])
            if sim < threshold and is_local_minimum(sim_values, k):
                in_seg.append(
                    {
                        "boundary_time": float(item["right_start"]),
                        "similarity": sim,
                        "left_window_end": float(item["left_end"]),
                        "right_window_start": float(item["right_start"]),
                        "between_windows": [int(item["left_idx"]), int(item["right_idx"])],
                        "reason": "segment_local_resimilarity",
                    }
                )

        picked = min_distance_filter(in_seg, min_distance_sec=min_distance_sec)
        out.extend(picked)

    out = sorted(out, key=lambda x: float(x["boundary_time"]))
    for i, b in enumerate(out):
        b["boundary_index"] = i
    return out


def build_similarity_once(repo_root: Path, lecture_id: str, out_dir: Path, window_size: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_sim = out_dir / "similarities.json"
    if local_sim.exists():
        return local_sim

    if int(window_size) == 3:
        e2_sim = (
            repo_root
            / "thesis_project"
            / "results"
            / "expE_prediction_pruning"
            / "sentence_w3_t055_md30"
            / lecture_id
            / "similarities.json"
        )
        if e2_sim.exists():
            return e2_sim

    raise RuntimeError(
        "No similarities cache found. Expected local cache or E2 cache at "
        f"thesis_project/results/expE_prediction_pruning/sentence_w3_t055_md30/{lecture_id}/similarities.json"
    )


def load_e2_cache(repo_root: Path, lecture_id: str, window_size: int) -> Tuple[List[Dict[str, object]], np.ndarray, List[Dict[str, object]]]:
    if int(window_size) != 3:
        raise RuntimeError("This implementation expects window_size=3 to reuse E2 cache.")
    base = (
        repo_root
        / "thesis_project"
        / "results"
        / "expE_prediction_pruning"
        / "sentence_w3_t055_md30"
        / lecture_id
    )
    windows_path = base / "windows.json"
    emb_path = base / "window_embeddings.npy"
    sims_path = base / "similarities.json"
    if not windows_path.exists() or not emb_path.exists() or not sims_path.exists():
        raise RuntimeError(f"Missing E2 cache files under: {base}")
    windows = load_json(windows_path)
    embeddings = np.load(str(emb_path))
    similarities = load_json(sims_path)
    return windows, embeddings, similarities


# --- Experiment D marker extraction (reuse) ---------------------------------

MARKER_PATTERNS = [
    r"\bnow\b",
    r"\bnext\b",
    r"\bmove on\b",
    r"\banother important\b",
    r"\btoday we (will|are going to)\b",
    r"\blet'?s move on\b",
]


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


def marker_candidates(blocks: List[Dict[str, object]], marker_patterns: List[str]) -> List[float]:
    pats = [re.compile(p, re.IGNORECASE) for p in marker_patterns]
    out: List[float] = []
    for b in blocks:
        txt = str(b["text"])
        if any(p.search(txt) for p in pats):
            out.append(float(b["start"]))
    return out


def dedupe_by_min_distance(times: List[float], min_distance_sec: float) -> List[float]:
    if not times:
        return []
    times = sorted(times)
    selected = [times[0]]
    for t in times[1:]:
        if t - selected[-1] >= min_distance_sec:
            selected.append(t)
    return selected


def extract_marker_times_d(
    blocks: List[Dict[str, object]],
    marker_pre_dedupe_min_distance: float,
) -> List[float]:
    """Same pipeline as Experiment D Marker channel before semantic fusion."""
    raw = marker_candidates(blocks, MARKER_PATTERNS)
    return dedupe_by_min_distance(raw, marker_pre_dedupe_min_distance)


def filter_markers_vs_stage2(
    stage2_times: List[float],
    marker_times: List[float],
    proximity_sec: float,
) -> List[float]:
    """Keep marker m only if min_s |m - s| > proximity_sec for stage2_times."""
    if not stage2_times:
        return sorted(marker_times)
    kept: List[float] = []
    for m in sorted(marker_times):
        if all(abs(m - s) > proximity_sec for s in stage2_times):
            kept.append(m)
    return kept


def marker_dict(t: float) -> Dict[str, object]:
    return {
        "boundary_time": float(t),
        "similarity": 1.0,
        "reason": "marker_structural",
        "left_window_end": None,
        "right_window_start": None,
        "between_windows": None,
    }


def merge_stage2_and_markers(
    stage2: List[Dict[str, object]],
    marker_times_after_skip: List[float],
    final_min_distance_sec: float,
) -> Tuple[List[Dict[str, object]], int]:
    """
    Combine Stage-2 dicts with marker-only boundaries; min-distance prune.
    Returns (pred list with boundary_index, count of surviving marker-sourced boundaries).
    """
    combined: List[Dict[str, object]] = []
    combined.extend(stage2)
    for t in marker_times_after_skip:
        combined.append(marker_dict(t))

    pruned = min_distance_filter(combined, min_distance_sec=final_min_distance_sec)
    marker_survived = sum(1 for b in pruned if str(b.get("reason")) == "marker_structural")

    pruned = sorted(pruned, key=lambda x: float(x["boundary_time"]))
    for i, b in enumerate(pruned):
        b["boundary_index"] = i
    return pruned, marker_survived


def evaluate_pred(repo_root: Path, gt_path: Path, pred_path: Path, tolerance_sec: float) -> Dict[str, object]:
    src_dir = repo_root / "ai_video_rbk" / "src"
    eval_path = pred_path.parent / "evaluation_report.json"
    run_cmd(
        [
            "python3",
            str(src_dir / "semantic_check_ground.py"),
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--tolerance",
            str(tolerance_sec),
            "--report-out",
            str(eval_path),
        ],
        cwd=repo_root,
    )
    return load_json(eval_path)


def run_one_lecture(
    *,
    repo_root: Path,
    lecture_id: str,
    coarse_ann_dir: Path,
    fine_ann_dir: Path,
    windows: List[Dict[str, object]],
    embeddings: np.ndarray,
    similarities: List[Dict[str, object]],
    total_end: float,
    stage1_mode: str,
    stage1_threshold: float,
    stage1_min_distance: float,
    stage2_threshold: float,
    stage2_min_distance: float,
    use_marker_stage3: bool,
    marker_skip_sec: float,
    marker_pre_dedupe_sec: float,
    tolerance_sec: float,
    window_size: int,
    out_dir: Path,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = build_similarity_once(repo_root, lecture_id, out_dir / "_sim_cache", window_size)

    if stage1_mode == "none":
        stage1_bounds: List[float] = []
    elif stage1_mode == "oracle":
        stage1_bounds = parse_boundary_txt(coarse_ann_dir / f"{lecture_id}_boundaries.txt")
    else:
        s1_cands = semantic_candidates(similarities, threshold=float(stage1_threshold))
        s1_bounds_json = min_distance_filter(s1_cands, min_distance_sec=float(stage1_min_distance))
        stage1_bounds = sorted(float(x["boundary_time"]) for x in s1_bounds_json)

    if stage1_mode == "none":
        pred = min_distance_filter(
            semantic_candidates(similarities, threshold=stage2_threshold),
            min_distance_sec=stage2_min_distance,
        )
        pred = sorted(pred, key=lambda x: float(x["boundary_time"]))
        for i, b in enumerate(pred):
            b["boundary_index"] = i
        markers_added = 0
    else:
        pred_s2 = stage2_with_segments(
            windows=windows,
            embeddings=embeddings,
            coarse_bounds=stage1_bounds,
            total_end=total_end,
            threshold=stage2_threshold,
            min_distance_sec=stage2_min_distance,
        )
        if use_marker_stage3:
            vtt = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lecture_id}.vtt"
            blocks = parse_vtt_blocks(vtt)
            marker_times = extract_marker_times_d(blocks, marker_pre_dedupe_sec)
            s2_times = [float(x["boundary_time"]) for x in pred_s2]
            markers_filtered = filter_markers_vs_stage2(s2_times, marker_times, marker_skip_sec)
            pred, markers_added = merge_stage2_and_markers(pred_s2, markers_filtered, stage2_min_distance)
        else:
            pred = pred_s2
            markers_added = 0

    pred_path = out_dir / f"{lecture_id}_boundaries.json"
    pred_path.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")

    fine_gt_path = fine_ann_dir / f"{lecture_id}_boundaries.txt"
    m = evaluate_pred(repo_root, fine_gt_path, pred_path, tolerance_sec=float(tolerance_sec))

    return {
        "Stage1 Boundaries": len(stage1_bounds),
        "Markers added (dedup final)": markers_added,
        "metrics": m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment H: hierarchical + marker Stage 3.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--lectures", nargs="*", default=["lecture1", "lecture2", "lecture3", "lecture4"])
    parser.add_argument("--coarse-annotations-dir", default="ai_video_rbk/annotations_corrected")
    parser.add_argument("--fine-annotations-dir", default="ai_video_rbk/annotations_fine")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument("--stage1-threshold", type=float, default=0.35)
    parser.add_argument("--stage1-min-distance", type=float, default=240.0)
    parser.add_argument("--stage2-threshold", type=float, default=0.60)
    parser.add_argument("--stage2-min-distance", type=float, default=30.0)
    parser.add_argument(
        "--marker-skip-sec",
        type=float,
        default=30.0,
        help="Skip marker if within ±this many seconds of any Stage-2 boundary.",
    )
    parser.add_argument(
        "--marker-pre-dedupe-sec",
        type=float,
        default=20.0,
        help="Min-distance dedupe on raw marker hits (Experiment D default).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    coarse_ann_dir = (repo_root / args.coarse_annotations_dir).resolve()
    fine_ann_dir = (repo_root / args.fine_annotations_dir).resolve()
    lecture_ids = args.lectures if args.lectures else discover_lectures(repo_root, fine_ann_dir)
    if not lecture_ids:
        raise RuntimeError("No lectures found for Experiment H.")

    s2_t = float(args.stage2_threshold)
    s2_md = float(args.stage2_min_distance)

    # (setting_id, stage1_mode, use_marker_stage3)
    runs: List[Tuple[str, str, bool]] = [
        ("Baseline", "none", False),
        ("G2_best", "oracle", False),
        ("G4_best", "predicted", False),
        ("H1", "oracle", True),
        ("H2", "predicted", True),
    ]

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for setting_id, stage1_mode, use_marker_stage3 in runs:
        print(f"\n=== {setting_id} | stage1={stage1_mode} | marker3={use_marker_stage3} ===")
        run_rows: List[Dict[str, object]] = []

        for lecture_id in lecture_ids:
            out_dir = repo_root / "thesis_project" / "results" / "expH_marker_hierarchical" / setting_id / lecture_id
            windows, embeddings, similarities = load_e2_cache(
                repo_root=repo_root,
                lecture_id=lecture_id,
                window_size=int(args.window_size),
            )
            total_end = max(float(x["right_end"]) for x in similarities) if similarities else 0.0

            meta = run_one_lecture(
                repo_root=repo_root,
                lecture_id=lecture_id,
                coarse_ann_dir=coarse_ann_dir,
                fine_ann_dir=fine_ann_dir,
                windows=windows,
                embeddings=embeddings,
                similarities=similarities,
                total_end=total_end,
                stage1_mode=stage1_mode,
                stage1_threshold=float(args.stage1_threshold),
                stage1_min_distance=float(args.stage1_min_distance),
                stage2_threshold=s2_t,
                stage2_min_distance=s2_md,
                use_marker_stage3=use_marker_stage3,
                marker_skip_sec=float(args.marker_skip_sec),
                marker_pre_dedupe_sec=float(args.marker_pre_dedupe_sec),
                tolerance_sec=float(args.tolerance_seconds),
                window_size=int(args.window_size),
                out_dir=out_dir,
            )
            m = meta["metrics"]

            row = {
                "Setting": setting_id,
                "Lecture": lecture_id,
                "Stage1": stage1_mode,
                "Stage1 Boundaries": meta["Stage1 Boundaries"],
                "Stage2 Threshold": s2_t,
                "Stage2 Min-distance": f"{int(s2_md)}s",
                "Marker stage3": "yes" if use_marker_stage3 else "no",
                "Markers added (dedup final)": int(meta["Markers added (dedup final)"]),
                "Precision": round(float(m["precision"]), 4),
                "Recall": round(float(m["recall"]), 4),
                "F1": round(float(m["f1"]), 4),
                "Predicted Boundaries": int(m["pred_count"]),
                "Fine GT Boundaries": int(m["gt_count"]),
                "tp": int(m["TP"]),
                "fp": int(m["FP"]),
                "fn": int(m["FN"]),
            }
            run_rows.append(row)
            all_rows.append(row)

        n = len(run_rows)
        summary_rows.append(
            {
                "Setting": setting_id,
                "Stage1": stage1_mode,
                "Marker stage3": "yes" if use_marker_stage3 else "no",
                "Stage2 Threshold": s2_t,
                "Stage2 Min-distance": f"{int(s2_md)}s",
                "Precision": round(sum(float(r["Precision"]) for r in run_rows) / n, 4),
                "Recall": round(sum(float(r["Recall"]) for r in run_rows) / n, 4),
                "F1": round(sum(float(r["F1"]) for r in run_rows) / n, 4),
                "Predicted Boundaries": round(sum(float(r["Predicted Boundaries"]) for r in run_rows) / n, 2),
                "Stage1 Boundaries": round(sum(float(r["Stage1 Boundaries"]) for r in run_rows) / n, 2),
                "Markers added avg": round(sum(float(r["Markers added (dedup final)"]) for r in run_rows) / n, 2),
            }
        )

    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expH_marker_hierarchical_summary.csv",
        summary_rows,
        [
            "Setting",
            "Stage1",
            "Marker stage3",
            "Stage2 Threshold",
            "Stage2 Min-distance",
            "Precision",
            "Recall",
            "F1",
            "Predicted Boundaries",
            "Stage1 Boundaries",
            "Markers added avg",
        ],
    )
    write_csv(
        tables_dir / "expH_marker_hierarchical_lecture_level.csv",
        all_rows,
        [
            "Setting",
            "Lecture",
            "Stage1",
            "Stage1 Boundaries",
            "Stage2 Threshold",
            "Stage2 Min-distance",
            "Marker stage3",
            "Markers added (dedup final)",
            "Precision",
            "Recall",
            "F1",
            "Predicted Boundaries",
            "Fine GT Boundaries",
            "tp",
            "fp",
            "fn",
        ],
    )

    print("\nSaved:")
    print(f"- {tables_dir / 'expH_marker_hierarchical_summary.csv'}")
    print(f"- {tables_dir / 'expH_marker_hierarchical_lecture_level.csv'}")


if __name__ == "__main__":
    main()
