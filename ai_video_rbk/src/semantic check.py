import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


def load_windows(json_path: str) -> List[Dict[str, Any]]:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Encode texts into sentence embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def compute_adjacent_similarities(windows: List[Dict[str, Any]], embeddings: np.ndarray) -> List[Dict[str, Any]]:
    """
    Compute cosine similarity between adjacent windows:
    window 0 vs 1, 1 vs 2, 2 vs 3, ...
    """
    similarities = []

    for i in range(len(windows) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])

        similarities.append({
            "left_window_id": windows[i]["window_id"],
            "right_window_id": windows[i + 1]["window_id"],
            "left_start": windows[i]["start"],
            "left_end": windows[i]["end"],
            "right_start": windows[i + 1]["start"],
            "right_end": windows[i + 1]["end"],
            "similarity": sim
        })

    return similarities


def is_local_minimum(values: List[float], idx: int) -> bool:
    """
    Local minimum check for interior points.
    """
    if idx <= 0 or idx >= len(values) - 1:
        return False
    return values[idx] < values[idx - 1] and values[idx] < values[idx + 1]


def detect_boundaries(
    similarities: List[Dict[str, Any]],
    threshold: float = 0.65,
    require_local_minimum: bool = True
) -> List[Dict[str, Any]]:
    """
    Detect topic boundary candidates when similarity is low.
    By default:
    - similarity must be below threshold
    - and be a local minimum
    """
    sim_values = [x["similarity"] for x in similarities]
    boundaries = []

    for i, item in enumerate(similarities):
        low_similarity = item["similarity"] < threshold
        local_min = is_local_minimum(sim_values, i) if require_local_minimum else True

        if low_similarity and local_min:
            # boundary is assumed around the transition between left and right window
            boundary_time = item["right_start"]

            boundaries.append({
                "boundary_index": len(boundaries),
                "between_windows": [item["left_window_id"], item["right_window_id"]],
                "boundary_time": boundary_time,
                "similarity": item["similarity"],
                "left_window_end": item["left_end"],
                "right_window_start": item["right_start"],
                "reason": "low_similarity_and_local_minimum" if require_local_minimum else "low_similarity"
            })

    return boundaries


def save_json(data: Any, output_path: str) -> None:
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def filter_boundaries_by_min_distance(
    boundaries: List[Dict[str, Any]],
    min_distance_seconds: float,
) -> List[Dict[str, Any]]:
    """
    Enforce a minimum time distance between selected boundary candidates.
    If multiple candidates are within `min_distance_seconds`, keep the one
    with the lowest similarity (strongest drop).
    """
    if not boundaries:
        return []

    sorted_bounds = sorted(boundaries, key=lambda b: b["boundary_time"])
    selected: List[Dict[str, Any]] = []

    for b in sorted_bounds:
        if not selected:
            selected.append(b)
            continue

        last = selected[-1]
        dt = float(b["boundary_time"]) - float(last["boundary_time"])
        if dt >= min_distance_seconds:
            selected.append(b)
            continue

        # Too close: replace if this candidate is stronger (lower similarity).
        last_sim = float(last.get("similarity", 1.0))
        b_sim = float(b.get("similarity", 1.0))
        if b_sim < last_sim:
            selected[-1] = b

    return selected


def print_preview(similarities: List[Dict[str, Any]], boundaries: List[Dict[str, Any]], n: int = 10) -> None:
    print("\n=== Similarity preview ===")
    for row in similarities[:n]:
        print(
            f"Window {row['left_window_id']} -> {row['right_window_id']} | "
            f"sim={row['similarity']:.4f} | "
            f"time={row['right_start']:.2f}s"
        )

    print("\n=== Boundary preview ===")
    if not boundaries:
        print("No boundary candidates found.")
    else:
        for b in boundaries[:n]:
            print(
                f"Boundary between windows {b['between_windows'][0]} and {b['between_windows'][1]} | "
                f"time={b['boundary_time']:.2f}s | "
                f"sim={b['similarity']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Embed window texts, compute adjacent cosine similarities, then detect boundary candidates."
    )
    parser.add_argument(
        "--windows",
        default="windows.json",
        help="Path to windows.json (default: windows.json in current directory)",
    )
    parser.add_argument(
        "--embeddings-out",
        default="window_embeddings.npy",
        help="Output path for saved embeddings (default: window_embeddings.npy)",
    )
    parser.add_argument(
        "--sim-out",
        default="similarities.json",
        help="Output path for adjacent similarities (default: similarities.json)",
    )
    parser.add_argument(
        "--boundaries-out",
        default="boundaries.json",
        help="Output path for detected boundaries (default: boundaries.json)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Similarity threshold for boundary candidates (default: 0.55)",
    )
    parser.add_argument(
        "--no-local-minima",
        action="store_true",
        help="Disable local-minimum requirement (use threshold only)",
    )
    parser.add_argument(
        "--min-distance-seconds",
        type=float,
        default=20.0,
        help="Minimum separation between boundary candidates in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--disable-min-distance-filter",
        action="store_true",
        help="Disable min-distance post-filtering",
    )

    args = parser.parse_args()

    windows = load_windows(args.windows)
    print(f"Loaded windows: {len(windows)}")
    texts = [w["text"] for w in windows]

    embeddings = compute_embeddings(texts, model_name=args.model)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(args.embeddings_out, embeddings)
    print(f"Saved embeddings to: {args.embeddings_out}")

    similarities = compute_adjacent_similarities(windows, embeddings)
    save_json(similarities, args.sim_out)
    print(f"Saved similarities to: {args.sim_out}")

    boundaries = detect_boundaries(
        similarities,
        threshold=args.threshold,
        require_local_minimum=not args.no_local_minima,
    )

    if not args.disable_min_distance_filter:
        boundaries = filter_boundaries_by_min_distance(
            boundaries=boundaries,
            min_distance_seconds=args.min_distance_seconds,
        )

    # Keep boundary_index consistent after filtering.
    for idx, b in enumerate(boundaries):
        b["boundary_index"] = idx

    save_json(boundaries, args.boundaries_out)
    print(f"Saved boundaries to: {args.boundaries_out}")

    print_preview(similarities, boundaries, n=10)


if __name__ == "__main__":
    main()