from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent.parent
import os
os.chdir(ROOT)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ffmpeg_cmd = shutil.which("ffmpeg")
if not ffmpeg_cmd:
    ffmpeg_cmd = "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else "ffmpeg"
    if ffmpeg_cmd == "ffmpeg":
        raise SystemExit("Cannot find ffmpeg. Please install ffmpeg and try again.")

videos = list(DATA_DIR.glob("*.mp4"))
if not videos:
    raise SystemExit(f"No MP4 files found in: {DATA_DIR.absolute()}")

for video_file in videos:
    wav_file = OUTPUT_DIR / f"{video_file.stem}.wav"
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i", str(video_file),
        "-ac", "1",
        "-ar", "16000",
        str(wav_file),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Saved: {wav_file}")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or b"").decode(errors="replace") if (e.stderr or e.stdout) else str(e)
        print(f"Failed: {video_file.name}\n{err}")