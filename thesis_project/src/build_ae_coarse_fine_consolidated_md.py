"""
Emit a single markdown file with Experiments A–E evaluation under coarse vs fine GT
(full CSV dumps + best-F1 summary). For thesis / external drafting (e.g. §4.4).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_row(rows: List[Dict[str, str]], key: str = "F1") -> Tuple[Optional[Dict[str, str]], Optional[float]]:
    best: Optional[Dict[str, str]] = None
    best_val = float("-inf")
    for r in rows:
        try:
            v = float(r.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if v > best_val:
            best_val = v
            best = r
    if best is None:
        return None, None
    return best, best_val


def row_signature(exp: str, row: Dict[str, str]) -> str:
    # Avoid "|" — breaks markdown tables when pasted.
    if exp == "A":
        return f"{row.get('Method', '')}, {row.get('Window Unit', '')}, w={row.get('Window Size', '')}"
    if exp == "B":
        return f"window_size={row.get('Window Size', '')}"
    if exp == "C":
        return row.get("Rule", "").strip()
    if exp == "D":
        return row.get("Model", "").strip()
    if exp == "E":
        return row.get("Setting", "").strip()
    return str(row)


def csv_to_md_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_No rows._\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")).replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    tables = repo / "thesis_project" / "tables"
    out_path = tables / "experiment_AE_coarse_vs_fine_GT_eval_consolidated.md"

    paths: Dict[str, Tuple[Path, Path]] = {
        "A": (
            tables / "expA" / "expA_summary_table_coarse.csv",
            tables / "expA" / "expA_summary_table_fine.csv",
        ),
        "B": (
            tables / "expB" / "expB_window_size_summary_coarse.csv",
            tables / "expB" / "expB_window_size_summary_fine.csv",
        ),
        "C": (
            tables / "expC" / "expC_rule_summary_coarse.csv",
            tables / "expC" / "expC_rule_summary_fine.csv",
        ),
        "D": (
            tables / "expD" / "expD_model_comparison_coarse.csv",
            tables / "expD" / "expD_model_comparison_fine.csv",
        ),
        "E": (
            tables / "expE" / "expE_pruning_summary_coarse.csv",
            tables / "expE" / "expE_pruning_summary_fine.csv",
        ),
    }

    chunks: List[str] = []
    chunks.append("# Experiments A–E: Coarse vs Fine GT — consolidated evaluation tables\n")
    chunks.append(
        "Generated for drafting (e.g. thesis §4.4 Re-evaluation under Fine GT). "
        "Source snapshots: `thesis_project/tables/expA` … `expE`, files `*_coarse.csv` and `*_fine.csv`.\n"
    )
    chunks.append(
        "**Fine GT** labels: hierarchical `Section : Subtopic` in `ai_video_rbk/annotations_fine/` "
        "(see `thesis_project/src/generate_fine_gt_from_vtt.py`).\n"
    )
    chunks.append("---\n")

    summary_rows: List[str] = []
    summary_rows.append(
        "| Exp | Best F1 (coarse) | Best setting (coarse) | Best F1 (fine) | Best setting (fine) | Same best setting? |"
    )
    summary_rows.append("| --- | --- | --- | --- | --- | --- |")

    for exp, (pc, pf) in paths.items():
        rc = load_csv(pc)
        rf = load_csv(pf)
        bc, f1c = best_row(rc)
        bf, f1f = best_row(rf)
        sig_c = row_signature(exp, bc) if bc else "—"
        sig_f = row_signature(exp, bf) if bf else "—"
        same = "—"
        if bc is not None and bf is not None:
            same = "yes" if sig_c == sig_f else "no"
        summary_rows.append(
            f"| {exp} | {f1c if f1c is not None else '—'} | {sig_c} | "
            f"{f1f if f1f is not None else '—'} | {sig_f} | {same} |"
        )

    chunks.append("## Summary: best macro-F1 row per experiment\n")
    chunks.append("\n".join(summary_rows) + "\n")
    chunks.append(
        "_“Same best setting?” compares a short signature (method/window/rule/model/E-setting); "
        "fine GT can change which row wins even when the grid is unchanged._\n"
    )
    chunks.append("---\n")

    for exp, (pc, pf) in paths.items():
        rc = load_csv(pc)
        rf = load_csv(pf)
        bc, f1c = best_row(rc)
        bf, f1f = best_row(rf)
        chunks.append(f"## Experiment {exp}\n")
        if bc:
            chunks.append(f"- **Best (coarse GT):** F1={f1c} — `{row_signature(exp, bc)}`\n")
        if bf:
            chunks.append(f"- **Best (fine GT):** F1={f1f} — `{row_signature(exp, bf)}`\n")
        chunks.append(f"### {exp} — full table, **coarse** GT\n")
        chunks.append(f"_File:_ `{pc.relative_to(repo)}`\n")
        chunks.append(csv_to_md_table(rc))
        chunks.append(f"### {exp} — full table, **fine** GT\n")
        chunks.append(f"_File:_ `{pf.relative_to(repo)}`\n")
        chunks.append(csv_to_md_table(rf))
        chunks.append("\n---\n")

    out_path.write_text("".join(chunks), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
