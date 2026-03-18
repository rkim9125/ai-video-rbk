from pathlib import Path
import os
from pydub import AudioSegment
from pydub.silence import detect_silence

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

AUDIO_DIR = Path("output/audio")
OUTPUT_DIR = Path("output/silence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SILENCE_LEN = 1000   # ms
SILENCE_THRESH_OFFSET = 16  # dB below average

def ms_to_timestamp(ms: int) -> str:
    total_seconds = ms / 1000
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

wav_files = list(AUDIO_DIR.glob("*.wav"))
if not wav_files:
    raise SystemExit(f"WAV files not found: {AUDIO_DIR.absolute()} (run python3 src/extract_audio.py first)")

for wav_file in wav_files:
    audio = AudioSegment.from_wav(wav_file)

    silence_thresh = audio.dBFS - SILENCE_THRESH_OFFSET
    silent_ranges = detect_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=silence_thresh
    )

    pred_file = OUTPUT_DIR / f"{wav_file.stem}_pred.txt"

    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("00:00.00\n")
        for start_ms, end_ms in silent_ranges:
            midpoint = (start_ms + end_ms) // 2
            f.write(ms_to_timestamp(midpoint) + "\n")

    print(f"Saved: {pred_file}")