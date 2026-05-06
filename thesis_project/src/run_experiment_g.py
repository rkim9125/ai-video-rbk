"""
Experiment G: Hierarchical 2-stage segmentation.

Stage 1 (coarse):
  - Oracle: coarse GT boundaries
  - Predicted: semantic detector on full lecture (coarse-track params)

Stage 2 (fine, segment-local):
  - Within each coarse segment, detect semantic boundaries with Stage-2 params
  - Merge segment outputs and evaluate on fine GT

Outputs:
  - thesis_project/tables/expG_hierarchical_summary.csv
  - thesis_project/tables/expG_hierarchical_lecture_level.csv
"""

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
    # Segment-local re-similarity:
    # for each segment, recompute adjacent cosine similarities only within that segment.
    segments: List[Tuple[float, float]] = []
    starts = [0.0] + coarse_bounds
    ends = coarse_bounds + [total_end]
    for s, e in zip(starts, ends):
        if e > s:
            segments.append((s, e))

    out: List[Dict[str, object]] = []
    for s, e in segments:
        # Window indices whose time ranges are inside [s, e)
        seg_indices = [
            i
            for i, w in enumerate(windows)
            if float(w["start"]) >= s and float(w["end"]) < e
        ]
        if len(seg_indices) < 2:
            continue
        seg_set = set(seg_indices)

        # Adjacent pairs (i, i+1) only when both belong to this segment
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

    # Intentionally no global min-distance filter after merge.
    out = sorted(out, key=lambda x: float(x["boundary_time"]))
    for i, b in enumerate(out):
        b["boundary_index"] = i
    return out


def build_similarity_once(repo_root: Path, lecture_id: str, out_dir: Path, window_size: int) -> Path:
    """
    Prefer existing E2 caches for stability/reproducibility.
    Falls back to local cache path if already present.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local_sim = out_dir / "similarities.json"
    if local_sim.exists():
        return local_sim

    # Reuse existing Experiment E cache (w=3, t=0.55, md=30) to avoid re-embedding.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment G hierarchical 2-stage protocol.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--lectures", nargs="*", default=[])
    parser.add_argument("--coarse-annotations-dir", default="ai_video_rbk/annotations_corrected")
    parser.add_argument("--fine-annotations-dir", default="ai_video_rbk/annotations_fine")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument("--stage1-threshold", type=float, default=0.35)
    parser.add_argument("--stage1-min-distance", type=float, default=240.0)
    parser.add_argument("--stage2-threshold", type=float, default=0.55)
    parser.add_argument("--stage2-min-distance", type=float, default=30.0)
    parser.add_argument(
        "--run-stage2-grid",
        action="store_true",
        help="Run Stage2 grid for G2/G4: threshold {0.50,0.55,0.60} x min-distance {15,20,30}.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    coarse_ann_dir = (repo_root / args.coarse_annotations_dir).resolve()
    fine_ann_dir = (repo_root / args.fine_annotations_dir).resolve()
    lecture_ids = args.lectures if args.lectures else discover_lectures(repo_root, fine_ann_dir)
    if not lecture_ids:
        raise RuntimeError("No lectures found for Experiment G.")

    # Base run set
    settings: List[Tuple[str, str, float, float]] = [
        ("Baseline", "none", float(args.stage2_threshold), float(args.stage2_min_distance)),
        ("G1", "oracle", float(args.stage2_threshold), float(args.stage2_min_distance)),
        ("G3", "predicted", float(args.stage2_threshold), float(args.stage2_min_distance)),
    ]
    if args.run_stage2_grid:
        for t in [0.50, 0.55, 0.60]:
            for md in [15.0, 20.0, 30.0]:
                settings.append((f"G2_t{str(t).replace('.', '')}_md{int(md)}", "oracle", float(t), float(md)))
                settings.append((f"G4_t{str(t).replace('.', '')}_md{int(md)}", "predicted", float(t), float(md)))

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for setting_id, stage1_mode, s2_thr, s2_md in settings:
        print(f"\n=== {setting_id} | stage1={stage1_mode} | stage2_thr={s2_thr} | stage2_md={s2_md}s ===")
        run_rows: List[Dict[str, object]] = []

        for lecture_id in lecture_ids:
            out_dir = repo_root / "thesis_project" / "results" / "expG_hierarchical" / setting_id / lecture_id
            sim_path = build_similarity_once(
                repo_root=repo_root,
                lecture_id=lecture_id,
                out_dir=out_dir / "_sim_cache",
                window_size=int(args.window_size),
            )
            windows, embeddings, similarities = load_e2_cache(
                repo_root=repo_root,
                lecture_id=lecture_id,
                window_size=int(args.window_size),
            )
            total_end = max(float(x["right_end"]) for x in similarities) if similarities else 0.0

            # Stage 1 coarse boundaries
            if stage1_mode == "none":
                stage1_bounds: List[float] = []
            elif stage1_mode == "oracle":
                stage1_bounds = parse_boundary_txt(coarse_ann_dir / f"{lecture_id}_boundaries.txt")
            else:
                s1_cands = semantic_candidates(similarities, threshold=float(args.stage1_threshold))
                s1_bounds_json = min_distance_filter(s1_cands, min_distance_sec=float(args.stage1_min_distance))
                stage1_bounds = sorted(float(x["boundary_time"]) for x in s1_bounds_json)

            if stage1_mode == "none":
                pred = min_distance_filter(
                    semantic_candidates(similarities, threshold=s2_thr),
                    min_distance_sec=s2_md,
                )
                pred = sorted(pred, key=lambda x: float(x["boundary_time"]))
                for i, b in enumerate(pred):
                    b["boundary_index"] = i
            else:
                pred = stage2_with_segments(
                    windows=windows,
                    embeddings=embeddings,
                    coarse_bounds=stage1_bounds,
                    total_end=total_end,
                    threshold=s2_thr,
                    min_distance_sec=s2_md,
                )

            pred_path = out_dir / f"{lecture_id}_boundaries.json"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")

            fine_gt_path = fine_ann_dir / f"{lecture_id}_boundaries.txt"
            m = evaluate_pred(repo_root, fine_gt_path, pred_path, tolerance_sec=float(args.tolerance_seconds))

            row = {
                "Setting": setting_id,
                "Lecture": lecture_id,
                "Stage1": stage1_mode,
                "Stage1 Boundaries": len(stage1_bounds),
                "Stage2 Threshold": s2_thr,
                "Stage2 Min-distance": f"{int(s2_md)}s",
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
                "Stage2 Threshold": s2_thr,
                "Stage2 Min-distance": f"{int(s2_md)}s",
                "Precision": round(sum(float(r["Precision"]) for r in run_rows) / n, 4),
                "Recall": round(sum(float(r["Recall"]) for r in run_rows) / n, 4),
                "F1": round(sum(float(r["F1"]) for r in run_rows) / n, 4),
                "Predicted Boundaries": round(sum(float(r["Predicted Boundaries"]) for r in run_rows) / n, 2),
                "Stage1 Boundaries": round(sum(float(r["Stage1 Boundaries"]) for r in run_rows) / n, 2),
            }
        )

    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expG_hierarchical_summary.csv",
        summary_rows,
        [
            "Setting",
            "Stage1",
            "Stage2 Threshold",
            "Stage2 Min-distance",
            "Precision",
            "Recall",
            "F1",
            "Predicted Boundaries",
            "Stage1 Boundaries",
        ],
    )
    write_csv(
        tables_dir / "expG_hierarchical_lecture_level.csv",
        all_rows,
        [
            "Setting",
            "Lecture",
            "Stage1",
            "Stage1 Boundaries",
            "Stage2 Threshold",
            "Stage2 Min-distance",
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
    print(f"- {tables_dir / 'expG_hierarchical_summary.csv'}")
    print(f"- {tables_dir / 'expG_hierarchical_lecture_level.csv'}")


if __name__ == "__main__":
    main()
