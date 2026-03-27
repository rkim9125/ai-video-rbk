import argparse
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


@dataclass
class SubtitleBlock:
    start: float
    end: float
    text: str


@dataclass
class SentenceUnit:
    sentence_id: int
    start: float
    end: float
    text: str


@dataclass
class WindowUnit:
    window_id: int
    start: float
    end: float
    sentence_ids: List[int]
    text: str


def timestamp_to_seconds(ts: str) -> float:
    """
    Convert WebVTT timestamp to seconds.
    Supports formats like:
    00:01:23.456
    01:23.456
    """
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    if len(parts) == 2:
        mm, ss = parts
        return int(mm) * 60 + float(ss)
    raise ValueError(f"Invalid timestamp format: {ts}")


def clean_text(text: str) -> str:
    """
    Minimal cleaning:
    - remove extra spaces
    - remove repeated whitespace/newlines
    - optionally remove bracketed noise labels
    """
    text = text.replace("\n", " ").strip()
    text = re.sub(r"<[^>]+>", "", text)  # remove simple HTML tags
    text = re.sub(r"\[(.*?)\]", "", text)  # remove [Music], [Applause] style
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_vtt(vtt_path: str) -> List[SubtitleBlock]:
    """
    Parse a .vtt file into subtitle blocks.
    """
    content = Path(vtt_path).read_text(encoding="utf-8")
    lines = content.splitlines()

    blocks: List[SubtitleBlock] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip header and blank lines
        if not line or line.upper() == "WEBVTT":
            i += 1
            continue

        # Skip cue identifiers if present
        if "-->" not in line and i + 1 < len(lines) and "-->" in lines[i + 1]:
            i += 1
            line = lines[i].strip()

        if "-->" in line:
            time_line = line
            text_lines = []
            i += 1

            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1

            start_str, end_str = [part.strip().split(" ")[0] for part in time_line.split("-->")]
            start_sec = timestamp_to_seconds(start_str)
            end_sec = timestamp_to_seconds(end_str)
            text = clean_text(" ".join(text_lines))

            if text:
                blocks.append(SubtitleBlock(start=start_sec, end=end_sec, text=text))
        else:
            i += 1

    return blocks


def split_text_into_sentences(text: str) -> List[str]:
    """
    Sentence splitter using punctuation.
    Keeps sentence-ending punctuation attached.
    """
    if not text.strip():
        return []

    # Split after ., ?, ! followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_block_into_sentences(block: SubtitleBlock, sentence_id_start: int) -> List[SentenceUnit]:
    """
    Split one subtitle block into sentence units.
    If multiple sentences are inside one subtitle block, assign timestamps
    proportionally based on character length.
    """
    sentences = split_text_into_sentences(block.text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return [
            SentenceUnit(
                sentence_id=sentence_id_start,
                start=block.start,
                end=block.end,
                text=sentences[0],
            )
        ]

    total_chars = sum(len(s) for s in sentences)
    duration = block.end - block.start
    units: List[SentenceUnit] = []

    current_start = block.start
    next_id = sentence_id_start

    for idx, sent in enumerate(sentences):
        ratio = len(sent) / total_chars if total_chars > 0 else 1 / len(sentences)
        sent_duration = duration * ratio
        sent_end = current_start + sent_duration

        # Ensure last sentence ends exactly at block end
        if idx == len(sentences) - 1:
            sent_end = block.end

        units.append(
            SentenceUnit(
                sentence_id=next_id,
                start=current_start,
                end=sent_end,
                text=sent,
            )
        )

        current_start = sent_end
        next_id += 1

    return units


def build_sentence_units(blocks: List[SubtitleBlock]) -> List[SentenceUnit]:
    """
    Convert subtitle blocks into sentence-level units.
    """
    sentence_units: List[SentenceUnit] = []
    next_id = 0

    for block in blocks:
        split_units = split_block_into_sentences(block, next_id)
        sentence_units.extend(split_units)
        next_id += len(split_units)

    return sentence_units


def merge_short_sentences(
    sentence_units: List[SentenceUnit],
    min_chars: int = 25
) -> List[SentenceUnit]:
    """
    Optional post-processing:
    Merge very short sentence units with the following sentence
    to reduce noise.
    """
    if not sentence_units:
        return []

    merged: List[SentenceUnit] = []
    buffer_unit = sentence_units[0]

    for current in sentence_units[1:]:
        if len(buffer_unit.text) < min_chars:
            buffer_unit = SentenceUnit(
                sentence_id=buffer_unit.sentence_id,
                start=buffer_unit.start,
                end=current.end,
                text=f"{buffer_unit.text} {current.text}".strip(),
            )
        else:
            merged.append(buffer_unit)
            buffer_unit = current

    merged.append(buffer_unit)

    # Reassign sentence IDs cleanly
    for idx, unit in enumerate(merged):
        unit.sentence_id = idx

    return merged


def build_sliding_windows(
    sentence_units: List[SentenceUnit],
    window_size: int = 3,
    stride: int = 1
) -> List[WindowUnit]:
    """
    Build overlapping sentence windows.
    Example:
    window_size=3, stride=1
    [0,1,2], [1,2,3], [2,3,4], ...
    """
    windows: List[WindowUnit] = []

    if len(sentence_units) < window_size:
        return windows

    window_id = 0
    for i in range(0, len(sentence_units) - window_size + 1, stride):
        chunk = sentence_units[i:i + window_size]

        windows.append(
            WindowUnit(
                window_id=window_id,
                start=chunk[0].start,
                end=chunk[-1].end,
                sentence_ids=[s.sentence_id for s in chunk],
                text=" ".join(s.text for s in chunk).strip(),
            )
        )
        window_id += 1

    return windows


def save_json(data, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a WebVTT transcript into sentence units and sliding windows (3 sentences by default)."
    )
    parser.add_argument(
        "--vtt",
        required=True,
        help="Path to the .vtt file (e.g. ai_video_rbk/transcripts_vtt/lecture1.vtt)",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for sentences.json and windows.json (default: current directory)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Number of sentences per sliding window (default: 3)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between windows (default: 1)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=25,
        help="Merge sentence units shorter than this many characters (default: 25)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sentence_output = str(outdir / "sentences.json")
    window_output = str(outdir / "windows.json")

    blocks = parse_vtt(args.vtt)
    print(f"Parsed subtitle blocks: {len(blocks)}")

    sentence_units = build_sentence_units(blocks)
    print(f"Initial sentence units: {len(sentence_units)}")

    # Optional: merge very short sentences to reduce noise
    sentence_units = merge_short_sentences(sentence_units, min_chars=args.min_chars)
    print(f"Sentence units after merging short ones: {len(sentence_units)}")

    windows = build_sliding_windows(
        sentence_units=sentence_units,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"Sliding windows created: {len(windows)}")

    save_json([asdict(s) for s in sentence_units], sentence_output)
    save_json([asdict(w) for w in windows], window_output)

    print(f"Saved sentence units to: {sentence_output}")
    print(f"Saved windows to: {window_output}")

    # Preview a few results
    print("\n--- Preview: sentences ---")
    for s in sentence_units[:5]:
        print(asdict(s))

    print("\n--- Preview: windows ---")
    for w in windows[:5]:
        print(asdict(w))


if __name__ == "__main__":
    main()