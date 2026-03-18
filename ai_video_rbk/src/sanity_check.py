"""
Sanity check: annotations/lecture*_boundaries.txt vs output/silence/lecture*_pred.txt

What this script does:
1. Reads ground-truth annotation boundaries
   - format: HH:MM:SS + label
   - example: 00:05:36 Technical Infrastructure...
2. Reads silence baseline predictions
   - format: MM:SS.xx
   - example: 05:58.97
3. Prints:
   - annotation segments vs predicted silences inside each segment
   - nearest predicted silence for each annotation boundary
   - tolerance-based hit rates (e.g. ±5s, ±10s, ±15s)
"""

import re
from bisect import bisect_left
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
ANNOT_DIR = ROOT / "annotations"
PRED_DIR = ROOT / "output" / "silence"

# You can change these if needed
TOLERANCES = [5.0, 10.0, 15.0]   # seconds
MAX_PREDS_TO_SHOW = 8


def parse_annotation_line(line: str) -> Optional[Tuple[float, str]]:
    """
    Parse annotation line:
    '00:05:36 Technical Infrastructure...' -> (336.0, 'Technical Infrastructure...')
    """
    line = line.strip()
    if not line:
        return None

    m = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\s+(.+)", line)
    if not m:
        return None

    h = int(m.group(1))
    mm = int(m.group(2))
    s = int(m.group(3))
    label = m.group(4).strip()

    sec = h * 3600 + mm * 60 + s
    return sec, label


def parse_pred_line(line: str) -> Optional[float]:
    """
    Parse prediction line:
    '05:58.97' -> 358.97 seconds
    '00:00.00' -> 0.0 seconds
    """
    line = line.strip()
    if not line:
        return None

    m = re.match(r"(\d+):(\d{2})\.(\d{2,})", line)
    if not m:
        return None

    mins = int(m.group(1))
    secs = int(m.group(2))
    frac = m.group(3)

    # keep first 2 decimal digits
    frac = frac.ljust(2, "0")[:2]
    return mins * 60 + secs + int(frac) / 100.0


def sec_to_mmss(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"


def sec_to_hhmmss(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def load_annotations(path: Path) -> List[Tuple[float, str]]:
    annotations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_annotation_line(line)
        if parsed is not None:
            annotations.append(parsed)
    annotations.sort(key=lambda x: x[0])
    return annotations


def load_predictions(path: Path) -> List[float]:
    preds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_pred_line(line)
        if parsed is not None:
            preds.append(parsed)
    preds.sort()
    return preds


def merge_close_predictions(preds: List[float], merge_gap: float = 10.0) -> List[float]:
    """
    Merge prediction timestamps that are very close together.
    Example:
      [100.0, 103.0, 107.0, 150.0] with merge_gap=10
      -> [103.33, 150.0]
    Uses the average timestamp of each group.
    """
    if not preds:
        return []

    groups = [[preds[0]]]

    for p in preds[1:]:
        if p - groups[-1][-1] <= merge_gap:
            groups[-1].append(p)
        else:
            groups.append([p])

    merged = [sum(group) / len(group) for group in groups]
    return merged


def nearest_prediction(target: float, preds: List[float]) -> Optional[Tuple[float, float]]:
    """
    Return (nearest_pred, diff_seconds), where diff = nearest_pred - target
    Uses binary search for efficiency.
    """
    if not preds:
        return None

    idx = bisect_left(preds, target)
    candidates = []

    if idx < len(preds):
        candidates.append(preds[idx])
    if idx > 0:
        candidates.append(preds[idx - 1])

    nearest = min(candidates, key=lambda p: abs(p - target))
    diff = nearest - target
    return nearest, diff


def print_segment_view(annotations: List[Tuple[float, str]], preds: List[float]) -> None:
    """
    Show each annotation boundary and the predicted silence timestamps
    that fall inside the segment before that boundary.
    """
    print("  [Annotation boundaries]                    [Pred silences in segment]")
    print("  " + "-" * 76)

    seg_start = 0.0

    for bound_sec, label in annotations:
        seg_end = bound_sec
        preds_in_seg = [p for p in preds if seg_start <= p < seg_end]

        pred_str = " ".join(sec_to_mmss(p) for p in preds_in_seg[:MAX_PREDS_TO_SHOW])
        if len(preds_in_seg) > MAX_PREDS_TO_SHOW:
            pred_str += f" ... (+{len(preds_in_seg) - MAX_PREDS_TO_SHOW} more)"

        ann_str = f"{sec_to_hhmmss(bound_sec)} {label[:32]}"
        print(f"  {ann_str:<44} | {pred_str}")

        seg_start = bound_sec

    preds_tail = [p for p in preds if p >= seg_start]
    pred_str = " ".join(sec_to_mmss(p) for p in preds_tail[:MAX_PREDS_TO_SHOW])
    if len(preds_tail) > MAX_PREDS_TO_SHOW:
        pred_str += f" ... (+{len(preds_tail) - MAX_PREDS_TO_SHOW} more)"

    print(f"  (after last boundary)                      | {pred_str}")
    print()


def print_nearest_view(annotations: List[Tuple[float, str]], preds: List[float]) -> None:
    """
    For each annotation boundary, print the nearest predicted silence
    and the time difference.
    """
    print("  [Nearest predicted silence for each annotation boundary]")
    print("  " + "-" * 76)

    if not preds:
        print("  No prediction timestamps found.\n")
        return

    for bound_sec, label in annotations:
        result = nearest_prediction(bound_sec, preds)
        if result is None:
            print(f"  GT {sec_to_hhmmss(bound_sec)} | nearest pred: None")
            continue

        nearest_pred, diff = result
        sign = "+" if diff >= 0 else ""
        print(
            f"  GT {sec_to_hhmmss(bound_sec)}"
            f" | pred {sec_to_mmss(nearest_pred):>8}"
            f" | diff {sign}{diff:6.2f}s"
            f" | {label[:40]}"
        )
    print()


def print_tolerance_hits(annotations: List[Tuple[float, str]], preds: List[float]) -> None:
    """
    Simple hit-rate sanity check:
    For each annotation boundary, does there exist a predicted silence within ±tolerance?
    """
    print("  [Tolerance hit rates]")
    print("  " + "-" * 76)

    if not annotations:
        print("  No annotation boundaries found.\n")
        return

    if not preds:
        for tol in TOLERANCES:
            print(f"  ±{tol:.0f}s: 0 / {len(annotations)} hits (0.0%)")
        print()
        return

    for tol in TOLERANCES:
        hits = 0
        for bound_sec, _ in annotations:
            result = nearest_prediction(bound_sec, preds)
            if result is None:
                continue
            _, diff = result
            if abs(diff) <= tol:
                hits += 1

        rate = 100.0 * hits / len(annotations)
        print(f"  ±{tol:.0f}s: {hits} / {len(annotations)} hits ({rate:.1f}%)")

    print()


def run_lecture(lecture_id: str, merge_preds: bool = False, merge_gap: float = 10.0) -> None:
    ann_path = ANNOT_DIR / f"{lecture_id}_boundaries.txt"
    pred_path = PRED_DIR / f"{lecture_id}_pred.txt"

    if not ann_path.exists():
        print(f"[skip] no annotation: {ann_path}")
        return
    if not pred_path.exists():
        print(f"[skip] no prediction: {pred_path}")
        return

    annotations = load_annotations(ann_path)
    preds = load_predictions(pred_path)

    raw_pred_count = len(preds)

    if merge_preds:
        preds = merge_close_predictions(preds, merge_gap=merge_gap)

    print("=" * 90)
    mode = f"{lecture_id} | merged preds={merge_preds}"
    if merge_preds:
        mode += f" (gap={merge_gap:.1f}s)"
    print(f"  {mode}")
    print("=" * 90)
    print()

    print_segment_view(annotations, preds)
    print_nearest_view(annotations, preds)
    print_tolerance_hits(annotations, preds)

    print("  Summary:")
    print(f"    Annotation boundaries: {len(annotations)}")
    print(f"    Pred silence timestamps (raw): {raw_pred_count}")
    print(f"    Pred silence timestamps (used): {len(preds)}")
    print()


def main() -> None:
    ids = set()

    for f in ANNOT_DIR.glob("*_boundaries.txt"):
        ids.add(f.stem.replace("_boundaries", ""))

    for f in PRED_DIR.glob("*_pred.txt"):
        ids.add(f.stem.replace("_pred", ""))

    if not ids:
        print("No lecture files found.")
        return

    # Run both raw and merged views so you can compare easily
    for lecture_id in sorted(ids):
        run_lecture(lecture_id, merge_preds=False)
        run_lecture(lecture_id, merge_preds=True, merge_gap=10.0)


if __name__ == "__main__":
    main()