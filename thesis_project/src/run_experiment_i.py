"""
Experiment I — Leave-one-out cross-validation for hyperparameter stability.

Part 1 (coarse GT): Experiment E-style single-stage pruning; grid min-distance ∈ {20,30,45,60};
  fixed w=3, t=0.55, local minima + min-distance (same as E2 aside from md).

Part 2 (fine GT): Experiment G2-style; Stage 1 oracle coarse boundaries; Stage 2 grid
  threshold ∈ {0.50,0.55,0.60} × min-distance ∈ {15,20,30}.

Outputs:
  thesis_project/tables/expI_loocv_E.csv
  thesis_project/tables/expI_loocv_G.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from run_experiment_g import (
    discover_lectures,
    evaluate_pred,
    load_e2_cache,
    min_distance_filter,
    parse_boundary_txt,
    semantic_candidates,
    stage2_with_segments,
    write_csv,
)


def pred_e_from_similarities(similarities: List[Dict[str, object]], threshold: float, md: float) -> List[Dict[str, object]]:
    raw = min_distance_filter(semantic_candidates(similarities, threshold), md)
    out = sorted(raw, key=lambda x: float(x["boundary_time"]))
    for i, b in enumerate(out):
        b["boundary_index"] = i
    return out


def f1_e_lecture(
    repo_root: Path,
    lecture_id: str,
    md: float,
    coarse_ann: Path,
    tolerance: float,
    threshold: float,
    window_size: int,
) -> float:
    _, _, similarities = load_e2_cache(repo_root, lecture_id, window_size)
    pred = pred_e_from_similarities(similarities, threshold, md)
    out_dir = repo_root / "thesis_project" / "results" / "expI_loocv" / "E_preds" / f"md{int(md)}" / lecture_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{lecture_id}_boundaries.json"
    pred_path.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")
    gt_path = coarse_ann / f"{lecture_id}_boundaries.txt"
    m = evaluate_pred(repo_root, gt_path, pred_path, tolerance)
    return float(m["f1"])


def f1_g_lecture(
    repo_root: Path,
    lecture_id: str,
    s2_t: float,
    s2_md: float,
    coarse_ann: Path,
    fine_ann: Path,
    tolerance: float,
    window_size: int,
) -> float:
    windows, embeddings, similarities = load_e2_cache(repo_root, lecture_id, window_size)
    total_end = max(float(x["right_end"]) for x in similarities) if similarities else 0.0
    stage1_bounds = parse_boundary_txt(coarse_ann / f"{lecture_id}_boundaries.txt")
    pred = stage2_with_segments(
        windows=windows,
        embeddings=embeddings,
        coarse_bounds=stage1_bounds,
        total_end=total_end,
        threshold=s2_t,
        min_distance_sec=s2_md,
    )
    t_tag = str(s2_t).replace(".", "")
    out_dir = (
        repo_root
        / "thesis_project"
        / "results"
        / "expI_loocv"
        / "G_preds"
        / f"t{t_tag}_md{int(s2_md)}"
        / lecture_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{lecture_id}_boundaries.json"
    pred_path.write_text(json.dumps(pred, indent=2, ensure_ascii=False), encoding="utf-8")
    gt_path = fine_ann / f"{lecture_id}_boundaries.txt"
    m = evaluate_pred(repo_root, gt_path, pred_path, tolerance)
    return float(m["f1"])


def macro_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pick_best_md_e(
    train_ids: List[str],
    md_grid: List[float],
    f1_fn: Callable[[str, float], float],
) -> Tuple[float, float]:
    """Return (best_md, train_macro_f1_at_best). Tie: smallest md."""
    best_md = md_grid[0]
    best_macro = -1.0
    for md in md_grid:
        macro = macro_mean([f1_fn(lid, md) for lid in train_ids])
        if macro > best_macro + 1e-12:
            best_macro = macro
            best_md = md
        elif abs(macro - best_macro) < 1e-12:
            best_md = min(best_md, md)
    return best_md, best_macro


def pick_best_stage2_g(
    train_ids: List[str],
    thr_grid: List[float],
    md_grid: List[float],
    f1_fn: Callable[[str, float, float], float],
) -> Tuple[float, float, float]:
    """Return (best_t, best_md, train_macro_f1). Tie: lexicographically smallest (t, md)."""
    best_t, best_md = thr_grid[0], md_grid[0]
    best_macro = -1.0
    for t in thr_grid:
        for md in md_grid:
            macro = macro_mean([f1_fn(lid, t, md) for lid in train_ids])
            if macro > best_macro + 1e-12:
                best_macro = macro
                best_t, best_md = t, md
            elif abs(macro - best_macro) < 1e-12:
                if (t, md) < (best_t, best_md):
                    best_t, best_md = t, md
    return best_t, best_md, best_macro


def global_best_e(
    all_ids: List[str],
    md_grid: List[float],
    f1_fn: Callable[[str, float], float],
) -> Tuple[float, float]:
    best_md = md_grid[0]
    best_macro = -1.0
    for md in md_grid:
        macro = macro_mean([f1_fn(lid, md) for lid in all_ids])
        if macro > best_macro + 1e-12:
            best_macro = macro
            best_md = md
        elif abs(macro - best_macro) < 1e-12:
            best_md = min(best_md, md)
    return best_md, best_macro


def global_best_g(
    all_ids: List[str],
    thr_grid: List[float],
    md_grid: List[float],
    f1_fn: Callable[[str, float, float], float],
) -> Tuple[float, float, float]:
    return pick_best_stage2_g(all_ids, thr_grid, md_grid, f1_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment I: LOOCV for E and G hyperparameters.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--lectures", nargs="*", default=["lecture1", "lecture2", "lecture3", "lecture4"])
    parser.add_argument("--coarse-annotations-dir", default="ai_video_rbk/annotations_corrected")
    parser.add_argument("--fine-annotations-dir", default="ai_video_rbk/annotations_fine")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--tolerance-seconds", type=float, default=30.0)
    parser.add_argument("--e-threshold", type=float, default=0.55, help="Experiment E fixed threshold.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    coarse_ann = (repo_root / args.coarse_annotations_dir).resolve()
    fine_ann = (repo_root / args.fine_annotations_dir).resolve()
    lecture_ids = list(args.lectures) if args.lectures else discover_lectures(repo_root, fine_ann)
    lecture_ids = sorted(lecture_ids)
    if len(lecture_ids) != 4:
        raise RuntimeError(f"Experiment I expects 4 lectures; got {len(lecture_ids)}: {lecture_ids}")

    tol = float(args.tolerance_seconds)
    w = int(args.window_size)
    e_thr = float(args.e_threshold)

    md_grid_e = [20.0, 30.0, 45.0, 60.0]
    thr_grid_g = [0.50, 0.55, 0.60]
    md_grid_g = [15.0, 20.0, 30.0]

    def fe(lid: str, md: float) -> float:
        return f1_e_lecture(repo_root, lid, md, coarse_ann, tol, e_thr, w)

    def fg(lid: str, t: float, md: float) -> float:
        return f1_g_lecture(repo_root, lid, t, md, coarse_ann, fine_ann, tol, w)

    # ----- Part E LOOCV -----
    e_rows: List[Dict[str, object]] = []
    e_held_f1s: List[float] = []
    e_best_mds: List[float] = []

    for held in lecture_ids:
        train = [x for x in lecture_ids if x != held]
        best_md, train_macro = pick_best_md_e(train, md_grid_e, fe)
        held_f1 = fe(held, best_md)
        e_best_mds.append(best_md)
        e_held_f1s.append(held_f1)
        e_rows.append(
            {
                "Fold held-out": held,
                "Best min-distance (s)": int(best_md),
                "Train macro F1 (best md)": round(train_macro, 4),
                "Held-out F1": round(held_f1, 4),
            }
        )

    g_md, g_macro_all = global_best_e(lecture_ids, md_grid_e, fe)

    tables_dir = repo_root / "thesis_project" / "tables"
    write_csv(
        tables_dir / "expI_loocv_E.csv",
        e_rows,
        ["Fold held-out", "Best min-distance (s)", "Train macro F1 (best md)", "Held-out F1"],
    )

    # ----- Part G LOOCV -----
    g_rows: List[Dict[str, object]] = []
    g_held_f1s: List[float] = []
    g_best_pairs: List[Tuple[float, float]] = []

    for held in lecture_ids:
        train = [x for x in lecture_ids if x != held]
        bt, bmd, train_macro = pick_best_stage2_g(train, thr_grid_g, md_grid_g, fg)
        held_f1 = fg(held, bt, bmd)
        g_best_pairs.append((bt, bmd))
        g_held_f1s.append(held_f1)
        g_rows.append(
            {
                "Fold held-out": held,
                "Best Stage2 threshold": bt,
                "Best Stage2 min-distance (s)": int(bmd),
                "Train macro F1 (best grid)": round(train_macro, 4),
                "Held-out F1": round(held_f1, 4),
            }
        )

    gt_g, gmd_g, g_macro_all_g = global_best_g(lecture_ids, thr_grid_g, md_grid_g, fg)

    write_csv(
        tables_dir / "expI_loocv_G.csv",
        g_rows,
        [
            "Fold held-out",
            "Best Stage2 threshold",
            "Best Stage2 min-distance (s)",
            "Train macro F1 (best grid)",
            "Held-out F1",
        ],
    )

    mean_loocv_e = macro_mean(e_held_f1s)
    mean_loocv_g = macro_mean(g_held_f1s)

    print("\n=== Experiment I LOOCV summary ===")
    print(f"Saved: {tables_dir / 'expI_loocv_E.csv'}")
    print(f"Saved: {tables_dir / 'expI_loocv_G.csv'}")

    print("\n--- Part E (coarse GT) ---")
    print(f"LOOCV mean held-out F1: {mean_loocv_e:.4f}")
    print(f"Global sweep macro F1 (all 4, best md={int(g_md)}): {g_macro_all:.4f}")
    uniq_e = sorted(set(int(x) for x in e_best_mds))
    if len(uniq_e) == 1:
        print(f"Best min-distance across folds: all folds agree on md={uniq_e[0]}s.")
    else:
        print(f"Best min-distance varies by fold: {dict(zip(lecture_ids, [int(x) for x in e_best_mds]))}")

    print("\n--- Part G (fine GT, oracle stage 1) ---")
    print(f"LOOCV mean held-out F1: {mean_loocv_g:.4f}")
    print(
        f"Global sweep macro F1 (all 4, best t={gt_g}, md={int(gmd_g)}): {g_macro_all_g:.4f}"
    )
    uniq_g = set((t, int(m)) for t, m in g_best_pairs)
    if len(uniq_g) == 1:
        t0, m0 = next(iter(uniq_g))
        print(f"Best (threshold, md) across folds: all folds agree on t={t0}, md={m0}s.")
    else:
        print(f"Best (t, md) varies by fold: {list(zip(lecture_ids, g_best_pairs))}")


if __name__ == "__main__":
    main()
