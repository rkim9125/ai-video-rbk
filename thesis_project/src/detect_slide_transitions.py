import argparse
import re
import subprocess
from pathlib import Path
from typing import List


PTS_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def seconds_to_hms(sec: float) -> str:
    total = int(round(sec))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def detect_scene_times(video_path: Path, scene_threshold: float) -> List[float]:
    """
    Detect scene-change timestamps using ffmpeg's scene filter.
    Higher threshold => fewer transitions.
    Typical useful range for slides: 0.25 ~ 0.45
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-filter:v",
        f"select='gt(scene,{scene_threshold})',metadata=print",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")

    pts = []
    for m in PTS_RE.finditer(text):
        pts.append(float(m.group(1)))
    pts = sorted(set(pts))
    return pts


def dedupe_min_distance(times: List[float], min_distance_sec: float) -> List[float]:
    if not times:
        return []
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] >= min_distance_sec:
            out.append(t)
    return out


def write_slide_txt(times: List[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [seconds_to_hms(t) for t in times]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def lecture_id_from_path(video_path: Path) -> str:
    stem = video_path.stem
    # e.g. lecture1.mp4 -> lecture1
    return stem


def run_one(video_path: Path, output_dir: Path, scene_threshold: float, min_distance_sec: float) -> Path:
    raw = detect_scene_times(video_path, scene_threshold=scene_threshold)
    times = dedupe_min_distance(raw, min_distance_sec=min_distance_sec)
    lecture_id = lecture_id_from_path(video_path)
    out = output_dir / f"{lecture_id}_slides.txt"
    write_slide_txt(times, out)
    print(f"[OK] {video_path.name} -> {out.name} ({len(times)} transitions)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect slide transitions from lecture mp4 using ffmpeg scene detection.")
    parser.add_argument(
        "--video",
        help="Single video path. If omitted, scans --videos-dir for lecture*.mp4.",
    )
    parser.add_argument(
        "--videos-dir",
        default="ai_video_rbk/data",
        help="Directory to scan for lecture*.mp4 when --video is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default="thesis_project/data/slide_transitions",
        help="Output directory for lecture*_slides.txt",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.32,
        help="ffmpeg scene threshold (default: 0.32)",
    )
    parser.add_argument(
        "--min-distance-seconds",
        type=float,
        default=20.0,
        help="Minimum separation between detected transitions (default: 20s)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    if args.video:
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        run_one(
            video_path=video_path,
            output_dir=output_dir,
            scene_threshold=args.scene_threshold,
            min_distance_sec=args.min_distance_seconds,
        )
        return

    videos_dir = (repo_root / args.videos_dir).resolve()
    videos = sorted(videos_dir.glob("lecture*.mp4"))
    if not videos:
        raise RuntimeError(f"No lecture*.mp4 found in: {videos_dir}")

    for video_path in videos:
        run_one(
            video_path=video_path,
            output_dir=output_dir,
            scene_threshold=args.scene_threshold,
            min_distance_sec=args.min_distance_seconds,
        )


if __name__ == "__main__":
    main()
