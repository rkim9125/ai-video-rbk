"""
Export spurious_far false positives with VTT snippets for manual coding.

Uses the same FP buckets as analyze_gt_pred_alignment.py:
  redundant_near_gt: nearest GT <= tolerance (default 30s)
  offset_near_miss: (tolerance, mid_band_max]
  spurious_far: > mid_band_max (default 120s)

Default export: spurious_far only, sorted for review (--sort).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _import_ground(repo_root: Path):
    sys.path.insert(0, str(repo_root / "ai_video_rbk" / "src"))
    import semantic_check_ground as scg  # type: ignore

    return scg


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


def nearest_gt(pred_time: float, gt: List[dict]) -> Tuple[float, dict]:
    best_g = gt[0]
    best_d = abs(pred_time - float(best_g["time"]))
    for g in gt[1:]:
        d = abs(pred_time - float(g["time"]))
        if d < best_d:
            best_d = d
            best_g = g
    return best_d, best_g


def fp_bucket(nearest_dist: float, tol: float, mid_max: float) -> str:
    if nearest_dist <= tol:
        return "redundant_near_gt"
    if nearest_dist <= mid_max:
        return "offset_near_miss"
    return "spurious_far"


def sec_to_hms(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def snippet_for_time(
    blocks: List[Dict[str, object]], center: float, before: float, after: float
) -> Tuple[str, int]:
    t0, t1 = center - before, center + after
    parts: List[str] = []
    n_blocks = 0
    for b in blocks:
        bs, be = float(b["start"]), float(b["end"])
        if be < t0 or bs > t1:
            continue
        n_blocks += 1
        parts.append(f"[{sec_to_hms(bs)}–{sec_to_hms(be)}] {b['text']}")
    return " ".join(parts), n_blocks


def marker_hits_in_text(text: str, tagged: List[Tuple[str, re.Pattern]]) -> List[str]:
    hits = []
    for label, p in tagged:
        if p.search(text):
            hits.append(label)
    return hits


def filler_count(text: str, fillers: List[str]) -> int:
    fill_re = re.compile(r"\b(" + "|".join(re.escape(x) for x in fillers) + r")\b", re.IGNORECASE)
    return len(fill_re.findall(text))


def load_slide_times(slide_path: Path) -> List[float]:
    if not slide_path.exists():
        return []
    times = []
    for line in slide_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d{2}:\d{2}:\d{2})", line)
        if m:
            p = m.group(1).split(":")
            times.append(int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2]))
    return sorted(times)


def nearest_event_delta(times: List[float], t: float) -> Optional[float]:
    if not times:
        return None
    return min(abs(x - t) for x in times)


def collect_rows(
    lecture_id: str,
    repo_root: Path,
    gt_path: Path,
    pred_path: Path,
    vtt_path: Path,
    slide_path: Path,
    tolerance: float,
    mid_band_max: float,
    bucket_filter: str,
    snippet_before: float,
    snippet_after: float,
    slide_hint_window: float,
    marker_window: float,
) -> List[Dict[str, object]]:
    scg = _import_ground(repo_root)
    gt = scg.load_gt_from_txt(str(gt_path))
    pred = scg.load_pred(str(pred_path))
    metrics = scg.compute_metrics(gt, pred, tolerance=tolerance)
    matched_pred: Set[int] = {m["pred_index"] for m in metrics["matches"]}

    blocks = parse_vtt_blocks(vtt_path)
    slide_times = load_slide_times(slide_path)

    marker_tagged: List[Tuple[str, re.Pattern]] = [
        ("now", re.compile(r"\bnow\b", re.IGNORECASE)),
        ("next", re.compile(r"\bnext\b", re.IGNORECASE)),
        ("move_on", re.compile(r"\bmove on\b", re.IGNORECASE)),
        ("another_important", re.compile(r"\banother important\b", re.IGNORECASE)),
        ("today_we", re.compile(r"\btoday we (will|are going to)\b", re.IGNORECASE)),
        ("lets_move_on", re.compile(r"\blet'?s move on\b", re.IGNORECASE)),
    ]
    fillers = ["um", "uh", "okay", "so", "well", "alright"]

    # Marker cue near boundary time (subtitle block start in window)
    marker_starts = []
    for b in blocks:
        if marker_hits_in_text(str(b["text"]), marker_tagged):
            marker_starts.append(float(b["start"]))

    rows_out: List[Dict[str, object]] = []
    for pi, p in enumerate(pred):
        if pi in matched_pred:
            continue
        pt = float(p["boundary_time"])
        d_near, g = nearest_gt(pt, gt)
        bucket = fp_bucket(d_near, tolerance, mid_band_max)
        if bucket_filter != "all" and bucket != bucket_filter:
            continue
        snip, nblk = snippet_for_time(blocks, pt, snippet_before, snippet_after)
        mk_in_snip = marker_hits_in_text(snip, marker_tagged)
        fill_n = filler_count(snip, fillers)
        slide_d = nearest_event_delta(slide_times, pt)
        slide_flag = (
            f"yes Δ={slide_d:.1f}s"
            if slide_d is not None and slide_d <= slide_hint_window
            else ("no" if slide_d is None else f"no (nearest {slide_d:.0f}s)")
        )
        mk_near_t = nearest_event_delta(marker_starts, pt)
        mk_near_flag = (
            f"yes Δ={mk_near_t:.1f}s"
            if mk_near_t is not None and mk_near_t <= marker_window
            else ("no" if mk_near_t is None else f"no (nearest {mk_near_t:.0f}s)")
        )

        rows_out.append(
            {
                "lecture_id": lecture_id,
                "pred_index": pi,
                "pred_time_sec": round(pt, 3),
                "pred_time_hms": sec_to_hms(pt),
                "similarity": round(float(p.get("similarity", 0.0)), 6),
                "fp_bucket": bucket,
                "nearest_gt_dist_sec": round(d_near, 3),
                "nearest_gt_hms": g["time_str"],
                "nearest_gt_title": g["title"],
                "snippet_block_count": nblk,
                "filler_hits_in_snippet": fill_n,
                "marker_patterns_in_snippet": ";".join(mk_in_snip) if mk_in_snip else "",
                "marker_cue_near_pred": mk_near_flag,
                "slide_near_pred": slide_flag,
                "snippet": snip,
            }
        )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export spurious_far (or other) FP rows with VTT snippets.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--lecture", default="lecture1")
    parser.add_argument("--gt", default="", help="Default: ai_video_rbk/annotations/{lecture}_boundaries.txt")
    parser.add_argument(
        "--pred",
        default="",
        help="Default: expE md30 boundaries for lecture.",
    )
    parser.add_argument("--tolerance", type=float, default=30.0)
    parser.add_argument("--mid-band-max", type=float, default=120.0)
    parser.add_argument(
        "--bucket",
        choices=["spurious_far", "offset_near_miss", "redundant_near_gt", "all"],
        default="spurious_far",
    )
    parser.add_argument("--snippet-before", type=float, default=45.0)
    parser.add_argument("--snippet-after", type=float, default=45.0)
    parser.add_argument("--slide-hint-window", type=float, default=30.0)
    parser.add_argument("--marker-hint-window", type=float, default=20.0)
    parser.add_argument(
        "--sort",
        choices=["distance_desc", "similarity_asc"],
        default="distance_desc",
        help="distance_desc: farthest from any GT first; similarity_asc: strongest dip first.",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        default="thesis_project/tables/spurious_far_review",
        help="Relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    lec = args.lecture
    gt_path = Path(args.gt) if args.gt else repo_root / "ai_video_rbk" / "annotations" / f"{lec}_boundaries.txt"
    pred_path = (
        Path(args.pred)
        if args.pred
        else repo_root
        / "thesis_project/results/expE_prediction_pruning/sentence_w3_t055_md30"
        / lec
        / f"{lec}_boundaries.json"
    )
    vtt_path = repo_root / "ai_video_rbk" / "transcripts_vtt" / f"{lec}.vtt"
    slide_path = repo_root / "thesis_project/data/slide_transitions" / f"{lec}_slides.txt"

    rows = collect_rows(
        lecture_id=lec,
        repo_root=repo_root,
        gt_path=gt_path,
        pred_path=pred_path,
        vtt_path=vtt_path,
        slide_path=slide_path,
        tolerance=args.tolerance,
        mid_band_max=args.mid_band_max,
        bucket_filter=args.bucket,
        snippet_before=args.snippet_before,
        snippet_after=args.snippet_after,
        slide_hint_window=args.slide_hint_window,
        marker_window=args.marker_hint_window,
    )

    if args.sort == "distance_desc":
        rows.sort(key=lambda r: float(r["nearest_gt_dist_sec"]), reverse=True)
    else:
        rows.sort(key=lambda r: float(r["similarity"]))

    rows = rows[: max(0, args.limit)]

    out_dir = (repo_root / args.out_dir).resolve() / lec
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "lecture": lec,
        "bucket": args.bucket,
        "sort": args.sort,
        "limit": args.limit,
        "snippet_pm_sec": [args.snippet_before, args.snippet_after],
        "pred_path": str(pred_path.relative_to(repo_root)),
        "vtt_path": str(vtt_path.relative_to(repo_root)),
        "tagging_guide": [
            "1 Transcript noise / ASR artifact",
            "2 Minor discourse shift",
            "3 Example switch / explanation change",
            "4 Potential true subtopic boundary",
            "5 Completely spurious",
        ],
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV without full snippet column for Excel; full text in separate column file optional
    slim_keys = [
        "rank",
        "lecture_id",
        "pred_index",
        "pred_time_sec",
        "pred_time_hms",
        "similarity",
        "fp_bucket",
        "nearest_gt_dist_sec",
        "nearest_gt_hms",
        "nearest_gt_title",
        "snippet_block_count",
        "filler_hits_in_snippet",
        "marker_patterns_in_snippet",
        "marker_cue_near_pred",
        "slide_near_pred",
    ]
    csv_path = out_dir / f"{args.bucket}_top{args.limit}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=slim_keys)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            row = {k: r.get(k, "") for k in slim_keys if k != "rank"}
            row["rank"] = i
            w.writerow(row)

    # Full CSV with snippet
    full_keys = slim_keys + ["snippet"]
    full_csv = out_dir / f"{args.bucket}_top{args.limit}_with_snippets.csv"
    with full_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=full_keys)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            out_row = {k: r.get(k, "") for k in full_keys if k != "rank"}
            out_row["rank"] = i
            w.writerow(out_row)

    # Markdown for reading
    md_lines = [
        f"# {args.bucket} review — `{lec}`",
        "",
        f"- Pred: `{pred_path.relative_to(repo_root)}`",
        f"- Snippet: −{args.snippet_before}s / +{args.snippet_after}s around predicted boundary",
        f"- Sort: `{args.sort}`, limit {args.limit}",
        "",
        "## Tagging (manual)",
        "",
    ]
    for t in meta["tagging_guide"]:
        md_lines.append(f"- {t}")
    md_lines.append("")

    for i, r in enumerate(rows, start=1):
        md_lines.extend(
            [
                f"## {i}. `{r['pred_time_hms']}` (sim={r['similarity']}, nearest GT Δ={r['nearest_gt_dist_sec']}s)",
                "",
                f"- **Nearest GT:** {r['nearest_gt_hms']} — {r['nearest_gt_title']}",
                f"- **Filler hits (snippet):** {r['filler_hits_in_snippet']}",
                f"- **Marker in snippet:** {r['marker_patterns_in_snippet'] or '(none)'}",
                f"- **Marker cue near pred:** {r['marker_cue_near_pred']}",
                f"- **Slide near pred:** {r['slide_near_pred']}",
                "",
                "```text",
                str(r["snippet"])[:12000],
                "```",
                "",
            ]
        )

    md_path = out_dir / f"{args.bucket}_top{args.limit}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {len(rows)} rows to:\n- {csv_path}\n- {full_csv}\n- {md_path}\n- {out_dir / 'export_meta.json'}")


if __name__ == "__main__":
    main()
