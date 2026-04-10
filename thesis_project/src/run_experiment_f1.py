"""
Experiment F1 — Marker-gated semantic boundary detection.

Baseline signal pipeline (fixed): sentence windows, w=3, embeddings, adjacent cosine similarities.
Detection change: keep a semantic boundary candidate only if a discourse-marker cue exists
within ±Δ seconds (then apply min-distance 30s).

Semantic confirmation: candidate must be a local minimum and similarity < confirm_threshold
(lower threshold = stricter: fewer dips qualify).

Does not re-embed per sweep: similarities are computed once per lecture and reused.
"""

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def run_cmd(args: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def load_semantic_check_module(repo_root: Path):
    path = repo_root / "ai_video_rbk" / "src" / "semantic check.py"
    spec = importlib.util.spec_from_file_location("semantic_check_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["semantic_check_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_vtt_blocks(vtt_path: Path) -> List[Dict[str, Any]]:
    lines = vtt_path.read_text(encoding="utf-8").splitlines()
    blocks: List[Dict[str, Any]] = []
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


def marker_times_from_vtt(
    vtt_path: Path,
    patterns: List[str],
    dedupe_min_distance: float,
) -> List[float]:
    pats = [re.compile(p, re.IGNORECASE) for p in patterns]
    raw: List[float] = []
    for b in parse_vtt_blocks(vtt_path):
        txt = str(b["text"])
        if any(p.search(txt) for p in pats):
            raw.append(float(b["start"]))
    raw.sort()
    if not raw:
        return []
    selected = [raw[0]]
    for t in raw[1:]:
        if t - selected[-1] >= dedupe_min_distance:
            selected.append(t)
        else:
            pass
    return selected


def gate_semantic_by_markers(
    semantic_raw: List[Dict[str, Any]],
    markers: List[float],
    delta_sec: float,
) -> List[Dict[str, Any]]:
    """Keep boundary b if some marker m satisfies |b_time - m| <= delta_sec."""
    out: List[Dict[str, Any]] = []
    for b in semantic_raw:
        bt = float(b["boundary_time"])
        if any(abs(bt - m) <= delta_sec for m in markers):
            out.append(dict(b))
    for idx, b in enumerate(out):
        b["boundary_index"] = idx
        b["reason"] = "marker_gated_semantic"
    return out


def discover_lectures(repo_root: Path, annotations_dir: Path) -> List[str]:
    transcripts_dir = repo_root / "ai_video_rbk" / "transcripts_vtt"
    vtt_ids = {p.stem for p in transcripts_dir.glob("lecture*.vtt")}
    gt_ids = {p.name.replace("_boundaries.txt", "") for p in annotations_dir.glob("lecture*_boundaries.txt")}
    return sorted(vtt_ids.intersection(gt_ids))


def ensure_similarities_cached(
    repo_root: Path,
    lecture_id: str,
    cache_dir: Path,
    window_size: int,
    sc_mod: Any,
) -> Path:
    """Build windows + embeddings + similarities once per lecture."""
    sim_path = cache_dir / "similarities.json"
    win_path = cache_dir / "windows.json"
    if sim_path.exists() and win_path.exists():
        return sim_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    src_dir = repo_root / "ai_video_rbk" / "src"
    vtt_path = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lecture_id}.vtt"

    run_cmd(
        [
            "python3",
            str(src_dir / "semantic_approach.py"),
            "--vtt",
            str(vtt_path),
            "--outdir",
            str(cache_dir),
            "--window-size",
            str(window_size),
            "--stride",
            "1",
            "--min-chars",
            "25",
        ],
        cwd=repo_root,
    )

    windows = sc_mod.load_windows(str(win_path))
    texts = [w["text"] for w in windows]
    embeddings = sc_mod.compute_embeddings(texts)
    emb_path = cache_dir / "window_embeddings.npy"
    import numpy as np

    np.save(str(emb_path), embeddings)
    similarities = sc_mod.compute_adjacent_similarities(windows, embeddings)
    sc_mod.save_json(similarities, str(sim_path))
    print(f"Cached similarities: {sim_path}")
    return sim_path


def evaluate_pred_json(
    repo_root: Path,
    gt_path: Path,
    pred_path: Path,
    tolerance: float,
    sc_ground: Any,
) -> Dict[str, Any]:
    eval_path = pred_path.parent / "evaluation_report.json"
    run_cmd(
        [
            "python3",
            str(repo_root / "ai_video_rbk" / "src" / "semantic_check_ground.py"),
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--tolerance",
            str(tolerance),
            "--report-out",
            str(eval_path),
        ],
        cwd=repo_root,
    )
    return json.loads(eval_path.read_text(encoding="utf-8"))


def load_ground_module(repo_root: Path):
    sys.path.insert(0, str(repo_root / "ai_video_rbk" / "src"))
    import semantic_check_ground as scg  # type: ignore

    return scg


def write_csv(path: Path, rows: List[Dict[str, object]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def update_master_table(repo_root: Path, rows: List[Dict[str, str]]) -> None:
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
    filtered = [r for r in existing if r.get("experiment_id") != "expF1_marker_gated"]
    filtered.extend(rows)
    write_csv(master_csv, filtered, header)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment F1: marker-gated semantic boundaries.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument(
        "--marker-windows",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0],
        help="Half-width Δ in seconds (±Δ around each marker time).",
    )
    parser.add_argument(
        "--confirm-thresholds",
        type=float,
        nargs="+",
        default=[0.55, 0.60],
        help="Semantic confirmation: keep dips with similarity < this (local minima required).",
    )
    parser.add_argument("--min-distance-final", type=float, default=30.0)
    parser.add_argument("--marker-dedupe-sec", type=float, default=20.0)
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument(
        "--annotations-dir",
        default="ai_video_rbk/annotations",
        help="Relative to repo root.",
    )
    parser.add_argument("--lectures", nargs="*", default=[], help="Empty = auto-discover.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ann_dir = (repo_root / args.annotations_dir).resolve()
    lecture_ids = args.lectures if args.lectures else discover_lectures(repo_root, ann_dir)
    if not lecture_ids:
        raise RuntimeError("No lectures found.")

    sc_mod = load_semantic_check_module(repo_root)

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

    scg = load_ground_module(repo_root)

    all_summary_rows: List[Dict[str, object]] = []
    master_rows: List[Dict[str, str]] = []

    for delta in args.marker_windows:
        for cthr in args.confirm_thresholds:
            setting = f"F1_pm{int(delta)}_c{str(cthr).replace('.', '')}_md{int(args.min_distance_final)}"
            setting_dir = repo_root / "thesis_project" / "results" / "expF1_marker_gated" / setting
            lect_rows: List[Dict[str, object]] = []

            for lecture_id in lecture_ids:
                lec_cache = cache_root / f"sentence_w{args.window_size}" / lecture_id
                sim_path = ensure_similarities_cached(
                    repo_root, lecture_id, lec_cache, args.window_size, sc_mod
                )
                similarities = json.loads(sim_path.read_text(encoding="utf-8"))

                semantic_raw = sc_mod.detect_boundaries(
                    similarities,
                    threshold=cthr,
                    require_local_minimum=True,
                )

                vtt_path = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lecture_id}.vtt"
                markers = marker_times_from_vtt(
                    vtt_path, marker_patterns, dedupe_min_distance=args.marker_dedupe_sec
                )

                gated = gate_semantic_by_markers(semantic_raw, markers, delta_sec=float(delta))
                final_bounds = sc_mod.filter_boundaries_by_min_distance(
                    gated,
                    min_distance_seconds=float(args.min_distance_final),
                )
                for idx, b in enumerate(final_bounds):
                    b["boundary_index"] = idx

                out_dir = setting_dir / lecture_id
                out_dir.mkdir(parents=True, exist_ok=True)
                boundaries_path = out_dir / f"{lecture_id}_boundaries.json"
                boundaries_path.write_text(
                    json.dumps(final_bounds, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                gt_path = ann_dir / f"{lecture_id}_boundaries.txt"
                m = evaluate_pred_json(repo_root, gt_path, boundaries_path, args.tolerance_seconds, scg)

                lect_rows.append(
                    {
                        "Setting": setting,
                        "Lecture": lecture_id,
                        "marker_window_pm_sec": int(delta),
                        "semantic_confirm_threshold": cthr,
                        "min_distance_sec": args.min_distance_final,
                        "Precision": round(float(m["precision"]), 4),
                        "Recall": round(float(m["recall"]), 4),
                        "F1": round(float(m["f1"]), 4),
                        "Predicted Boundaries": int(m["pred_count"]),
                        "GT Boundaries": int(m["gt_count"]),
                        "marker_count_deduped": len(markers),
                        "semantic_raw_before_gate": len(semantic_raw),
                    }
                )

                master_rows.append(
                    {
                        "experiment_id": "expF1_marker_gated",
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
                        "notes": f"marker_pm={delta}s confirm<{cthr} md={args.min_distance_final}",
                    }
                )

            n = len(lect_rows)
            all_summary_rows.append(
                {
                    "Setting": setting,
                    "marker_window_pm_sec": int(delta),
                    "semantic_confirm_threshold": cthr,
                    "min_distance_sec": int(args.min_distance_final),
                    "Precision": round(sum(float(r["Precision"]) for r in lect_rows) / n, 4),
                    "Recall": round(sum(float(r["Recall"]) for r in lect_rows) / n, 4),
                    "F1": round(sum(float(r["F1"]) for r in lect_rows) / n, 4),
                    "Predicted Boundaries": round(sum(float(r["Predicted Boundaries"]) for r in lect_rows) / n, 2),
                }
            )

            eval_tbl = setting_dir / "evaluation_table.csv"
            write_csv(
                eval_tbl,
                lect_rows,
                [
                    "Setting",
                    "Lecture",
                    "marker_window_pm_sec",
                    "semantic_confirm_threshold",
                    "min_distance_sec",
                    "Precision",
                    "Recall",
                    "F1",
                    "Predicted Boundaries",
                    "GT Boundaries",
                    "marker_count_deduped",
                    "semantic_raw_before_gate",
                ],
            )
            summary_json = {
                "experiment": "expF1_marker_gated",
                "setting": setting,
                "macro": all_summary_rows[-1],
                "lectures": lect_rows,
            }
            (setting_dir / "evaluation_summary.json").write_text(
                json.dumps(summary_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expF1_marker_gated_summary.csv",
        all_summary_rows,
        [
            "Setting",
            "marker_window_pm_sec",
            "semantic_confirm_threshold",
            "min_distance_sec",
            "Precision",
            "Recall",
            "F1",
            "Predicted Boundaries",
        ],
    )

    update_master_table(repo_root, master_rows)

    print("\nSaved:")
    print(f"  {tables_dir / 'expF1_marker_gated_summary.csv'}")
    print(f"  {repo_root / 'thesis_project/tables/experiment_master_table.csv'}")
    print("\nCompare with:")
    print("  Semantic baseline (E, md30): results/expE_prediction_pruning/sentence_w3_t055_md30/")
    print("  Marker-only (D): results/expD_structural/ — see expD_model_comparison.csv Marker row")


if __name__ == "__main__":
    main()
