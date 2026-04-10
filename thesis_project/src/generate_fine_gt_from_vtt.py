import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ts_to_sec(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    hh, mm, ss = ts.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def sec_to_hms(sec: float) -> str:
    sec_i = int(round(sec))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_vtt_blocks(vtt_path: Path) -> List[Dict[str, object]]:
    lines = vtt_path.read_text(encoding="utf-8").splitlines()
    blocks: List[Dict[str, object]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper() == "WEBVTT" or line.startswith("NOTE"):
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
        txt: List[str] = []
        while i < len(lines) and lines[i].strip():
            txt.append(lines[i].strip())
            i += 1

        a, b = [x.strip().split(" ")[0] for x in tline.split("-->")]
        text = " ".join(txt)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        blocks.append({"start": ts_to_sec(a), "end": ts_to_sec(b), "text": text})
    return blocks


def parse_coarse_boundaries(path: Path) -> List[Tuple[float, str]]:
    """
    Parse coarse section boundaries from lines like:
      00:57:15 Machine Learning Paradigms and Generalization
    """
    if not path.exists():
        return []
    rows: List[Tuple[float, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # tolerate prefixes like "L1:" if present
        line = re.sub(r"^L\d+:", "", line)
        m = re.match(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$", line)
        if not m:
            continue
        ts, title = m.groups()
        rows.append((ts_to_sec(ts), title.strip()))
    rows.sort(key=lambda x: x[0])
    return rows


def section_for_time(t: float, coarse_rows: List[Tuple[float, str]]) -> str:
    if not coarse_rows:
        return "Unassigned Section"
    current = coarse_rows[0][1]
    for st, title in coarse_rows:
        if t >= st:
            current = title
        else:
            break
    return current


def build_patterns() -> Dict[str, List[re.Pattern]]:
    return {
        "A_Concept_Introduction": [
            re.compile(
                r"\b(let'?s define|we define|x is|this is called|the key idea is|in other words|"
                r"what is|definition|means that)\b",
                re.IGNORECASE,
            )
        ],
        "B_Worked_Example": [
            re.compile(
                r"\b(let'?s take (an )?example|for example|for instance|suppose (we )?have|"
                r"consider (the )?case|imagine that|let'?s say)\b",
                re.IGNORECASE,
            )
        ],
        "C_Explanation_Shift_Major": [
            re.compile(
                r"\b(intuitively|formally|mathematically|in practice|practically|from theory to|"
                r"from intuition to|application|real[- ]world|implementation)\b",
                re.IGNORECASE,
            )
        ],
        "D_Practical_Important_Point": [
            re.compile(
                r"\b(this is important|important point|you should remember|keep in mind|"
                r"be careful|note that|in practice|tip|best practice|rule of thumb)\b",
                re.IGNORECASE,
            )
        ],
        "E_High_Value_QA": [
            re.compile(
                r"\b(question|you may ask|you might ask|why (do|does)|how (do|does)|"
                r"common question|good question|q and a|clarify)\b",
                re.IGNORECASE,
            )
        ],
    }


EXCLUDE_PATTERNS = [
    re.compile(r"\b(assignment|deadline|assessment|marks?|attendance|tutorial allocation|email me)\b", re.IGNORECASE),
    re.compile(r"\b(so|okay|ok|um|uh|right)\b[\.,!\?]?$", re.IGNORECASE),
]


def is_excluded(text: str) -> bool:
    t = text.strip()
    if len(t) < 16:
        return True
    return any(p.search(t) for p in EXCLUDE_PATTERNS)


def classify_block(text: str, patterns: Dict[str, List[re.Pattern]]) -> Optional[str]:
    if is_excluded(text):
        return None
    for label, regs in patterns.items():
        if any(r.search(text) for r in regs):
            return label
    return None


def detect_topic(text: str) -> str:
    t = text.lower()
    topic_rules: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\bnumpy\b"), "NumPy Array Operation"),
        (re.compile(r"\bpandas\b|\bdata frame\b|\bdataframe\b"), "DataFrame Processing Example"),
        (re.compile(r"\bartificial intelligence\b|\bai\b"), "AI Definition"),
        (re.compile(r"\bmachine learning\b"), "Machine Learning Concept"),
        (re.compile(r"\bsupervised\b|\bunsupervised\b"), "Supervised vs Unsupervised Learning"),
        (re.compile(r"\bloss\b|\bmse\b|\bcost function\b"), "Loss Function Explanation"),
        (re.compile(r"\bgradient descent\b"), "Gradient Descent"),
        (re.compile(r"\bclassification\b|\bclassifier\b"), "Classification"),
        (re.compile(r"\bregression\b"), "Regression"),
        (re.compile(r"\bneural network\b|\bmlp\b"), "Neural Networks"),
        (re.compile(r"\bcnn\b|\bconvolution\b"), "Convolutional Neural Network"),
        (re.compile(r"\bpooling\b"), "Pooling Layers"),
        (re.compile(r"\brelu\b|\bactivation\b"), "Activation Functions"),
        (re.compile(r"\broc\b|\bauc\b|\bprecision\b|\brecall\b|\bf1\b"), "Evaluation Metrics"),
        (re.compile(r"\bfeature engineering\b|\bfeature\b"), "Feature Engineering"),
        (re.compile(r"\bmodel\b|\btraining\b|\bhyperparameter\b"), "Model Training"),
        (re.compile(r"\bself[- ]driving\b"), "Self-driving Car Example"),
        (re.compile(r"\bstructural engineering\b|\bcivil engineering\b"), "Structural Engineering Example"),
        (re.compile(r"\bsoftware engineering\b"), "Software Engineering Example"),
        (re.compile(r"\bproject\b|\bmvp\b|\bprototype\b"), "Project Design"),
        (re.compile(r"\bcommerciali[sz]ation\b"), "Project Commercialisation"),
        (re.compile(r"\bquestion\b|\bq and a\b|\bclarify\b"), "Student Clarification"),
    ]
    for pat, topic in topic_rules:
        if pat.search(t):
            return topic

    if "example" in t:
        return "Worked Example"
    if "define" in t or "definition" in t or "what is" in t:
        return "Concept Definition"
    return "General Explanation"


def split_qa_type(label: str, text: str, topic: str) -> str:
    if label != "E_High_Value_QA":
        return label
    t = text.lower()
    high_value_terms = [
        "numpy",
        "pandas",
        "machine learning",
        "ai",
        "model",
        "feature",
        "probability",
        "gradient",
        "classification",
        "regression",
        "how do",
        "why",
        "what is",
    ]
    score = 0
    if len(text) >= 45:
        score += 1
    if any(k in t for k in high_value_terms):
        score += 1
    if topic not in {"Student Clarification", "General Explanation"}:
        score += 1
    return "E1_High_Value_QA" if score >= 2 else "E2_Minor_QA"


def normalize_topic(label: str, topic: str, text: str) -> str:
    """
    Force a consistent, analysis-friendly topic schema.
    """
    if topic in {
        "AI Definition",
        "Machine Learning Concept",
        "NumPy Array Operation",
        "DataFrame Processing Example",
        "Software Engineering Example",
        "Structural Engineering Example",
        "Self-driving Car Example",
        "Feature Engineering",
        "Model Training",
        "Evaluation Metrics",
        "Gradient Descent",
        "Loss Function Explanation",
        "Classification",
        "Regression",
        "Neural Networks",
        "Convolutional Neural Network",
        "Pooling Layers",
        "Activation Functions",
        "Project Design",
        "Project Commercialisation",
        "Supervised vs Unsupervised Learning",
    }:
        return topic

    t = text.lower()
    if label == "A_Concept_Introduction":
        if "definition" in t or "what is" in t:
            return "Concept Definition"
        return "Core Concept"
    if label == "B_Worked_Example":
        return "Worked Example"
    if label == "C_Explanation_Shift_Major":
        return "Theory-to-Practice Shift"
    if label == "D_Practical_Important_Point":
        return "Practical Guidance"
    if label == "E1_High_Value_QA":
        if "how do" in t or "how to" in t:
            return "Technical Clarification"
        return "High-value Clarification"
    if label == "E2_Minor_QA":
        return "Minor Clarification"
    return "General Topic"


def post_dedupe(cands: List[Tuple[float, str, str, str]], min_gap_sec: float) -> List[Tuple[float, str, str, str]]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda x: x[0])
    out = [cands[0]]
    for t, label, topic, text in cands[1:]:
        if t - out[-1][0] >= min_gap_sec:
            out.append((t, label, topic, text))
        else:
            # Replace same-zone candidate if new one is stronger cue (A/B/D priority)
            priority = {"A_Concept_Introduction": 5, "B_Worked_Example": 4, "D_Practical_Important_Point": 3, "C_Explanation_Shift_Major": 2, "E_High_Value_QA": 1}
            old_t, old_label, old_topic, old_text = out[-1]
            if priority.get(label, 0) > priority.get(old_label, 0):
                out[-1] = (t, label, topic, text)
    return out


def merge_consecutive_segments(
    cands: List[Tuple[float, str, str, str]],
    merge_gap_sec: float,
) -> List[Tuple[float, str, str, str]]:
    """
    Merge over-segmented consecutive boundaries when type/topic is same
    and time distance is short.
    """
    if not cands:
        return []
    cands = sorted(cands, key=lambda x: x[0])
    merged: List[Tuple[float, str, str, str]] = [cands[0]]
    for t, label, topic, text in cands[1:]:
        last_t, last_label, last_topic, last_text = merged[-1]
        same_cluster = label == last_label and topic == last_topic and (t - last_t) <= merge_gap_sec
        if same_cluster:
            # Keep first timestamp for boundary; enrich note text.
            merged[-1] = (
                last_t,
                last_label,
                last_topic,
                f"{last_text} || {text}",
            )
        else:
            merged.append((t, label, topic, text))
    return merged


def merge_concept_blocks(
    cands: List[Tuple[float, str, str, str]],
    concept_gap_sec: float,
) -> List[Tuple[float, str, str, str]]:
    """
    Additional merge pass for over-segmented concept sequences:
    consecutive A_* entries with same topic within wider gap are merged.
    """
    if not cands:
        return []
    out: List[Tuple[float, str, str, str]] = [cands[0]]
    concept_family = {"AI Definition", "Concept Definition", "Core Concept", "Machine Learning Concept"}
    for t, label, topic, text in cands[1:]:
        lt, ll, ltopic, ltext = out[-1]
        same_family = topic == ltopic or (topic in concept_family and ltopic in concept_family)
        is_concept_chain = (
            label.startswith("A_")
            and ll.startswith("A_")
            and same_family
            and (t - lt) <= concept_gap_sec
        )
        if is_concept_chain:
            merged_topic = ltopic if ltopic in concept_family else topic
            if topic == "AI Definition" or ltopic == "AI Definition":
                merged_topic = "AI Definition"
            out[-1] = (lt, ll, merged_topic, f"{ltext} || {text}")
        else:
            out.append((t, label, topic, text))
    return out


def merge_concept_with_minor_qa_bridge(
    cands: List[Tuple[float, str, str, str]],
    bridge_gap_sec: float,
) -> List[Tuple[float, str, str, str]]:
    """
    Merge A - E2_Minor_QA - A when the two concept items are close and same family.
    This reduces over-segmentation caused by short clarifications inside one concept block.
    """
    if len(cands) < 3:
        return cands
    concept_family = {"AI Definition", "Concept Definition", "Core Concept", "Machine Learning Concept"}
    out: List[Tuple[float, str, str, str]] = []
    i = 0
    while i < len(cands):
        if i + 2 < len(cands):
            a1 = cands[i]
            mid = cands[i + 1]
            a2 = cands[i + 2]
            is_bridge = (
                a1[1].startswith("A_")
                and mid[1] == "E2_Minor_QA"
                and a2[1].startswith("A_")
                and (a2[0] - a1[0]) <= bridge_gap_sec
                and (a1[2] == a2[2] or (a1[2] in concept_family and a2[2] in concept_family))
            )
            if is_bridge:
                merged_topic = "AI Definition" if ("AI Definition" in {a1[2], a2[2]}) else a1[2]
                out.append((a1[0], "A_Concept_Introduction", merged_topic, f"{a1[3]} || {mid[3]} || {a2[3]}"))
                i += 3
                continue
        out.append(cands[i])
        i += 1
    return out


def make_text(text: str) -> str:
    short = re.sub(r"[^A-Za-z0-9 ,.'-]", "", text).strip()
    if len(short) > 80:
        short = short[:77].rstrip() + "..."
    return short


def generate_for_lecture(
    vtt_path: Path,
    coarse_path: Path,
    min_gap_sec: float,
    merge_gap_sec: float,
    concept_gap_sec: float,
    include_text: bool,
) -> List[str]:
    patterns = build_patterns()
    blocks = parse_vtt_blocks(vtt_path)
    coarse_rows = parse_coarse_boundaries(coarse_path)
    cands: List[Tuple[float, str, str, str]] = []
    for b in blocks:
        text = str(b["text"])
        label = classify_block(text, patterns)
        if label is None:
            continue
        topic = detect_topic(text)
        label = split_qa_type(label, text, topic)
        topic = normalize_topic(label, topic, text)
        cands.append((float(b["start"]), label, topic, text))

    selected = post_dedupe(cands, min_gap_sec=min_gap_sec)
    selected = merge_consecutive_segments(selected, merge_gap_sec=merge_gap_sec)
    selected = merge_concept_blocks(selected, concept_gap_sec=concept_gap_sec)
    selected = merge_concept_with_minor_qa_bridge(selected, bridge_gap_sec=180.0)
    lines: List[str] = []
    for i, (t, label, topic, text) in enumerate(selected, start=1):
        seg_id = f"S{i:03d}"
        section = section_for_time(t, coarse_rows)
        hierarchical = f"{section} : {topic}"
        if include_text:
            lines.append(f"{sec_to_hms(t)}\t{seg_id}\t{label}\t{hierarchical}\t{make_text(text)}")
        else:
            lines.append(f"{sec_to_hms(t)}\t{seg_id}\t{label}\t{hierarchical}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate draft fine-grained GT from lecture VTT files.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--transcripts-dir", default="ai_video_rbk/transcripts_vtt")
    parser.add_argument("--coarse-dir", default="ai_video_rbk/annotations_corrected")
    parser.add_argument("--out-dir", default="ai_video_rbk/annotations_fine")
    parser.add_argument("--lectures", nargs="*", default=["lecture1", "lecture2", "lecture3", "lecture4"])
    parser.add_argument("--min-gap-sec", type=float, default=25.0)
    parser.add_argument("--merge-gap-sec", type=float, default=90.0)
    parser.add_argument("--concept-gap-sec", type=float, default=240.0)
    parser.add_argument("--include-text", action="store_true", help="Append raw snippet as last column.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    transcripts_dir = (repo_root / args.transcripts_dir).resolve()
    coarse_dir = (repo_root / args.coarse_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for lec in args.lectures:
        vtt_path = transcripts_dir / f"{lec}.vtt"
        if not vtt_path.exists():
            print(f"SKIP: missing {vtt_path}")
            continue
        coarse_path = coarse_dir / f"{lec}_boundaries.txt"
        if not coarse_path.exists():
            coarse_path = (repo_root / "ai_video_rbk" / "annotations" / f"{lec}_boundaries.txt").resolve()
        lines = generate_for_lecture(
            vtt_path,
            coarse_path,
            min_gap_sec=float(args.min_gap_sec),
            merge_gap_sec=float(args.merge_gap_sec),
            concept_gap_sec=float(args.concept_gap_sec),
            include_text=bool(args.include_text),
        )
        out_path = out_dir / f"{lec}_boundaries.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Saved {out_path} ({len(lines)} boundaries)")


if __name__ == "__main__":
    main()
