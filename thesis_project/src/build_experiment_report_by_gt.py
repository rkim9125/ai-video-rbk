import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def best_row(rows: List[Dict[str, str]], key: str = "F1") -> Dict[str, str]:
    best = None
    best_val = float("-inf")
    for r in rows:
        try:
            v = float(r.get(key, "nan"))
        except Exception:
            continue
        if v > best_val:
            best_val = v
            best = r
    return best or {}


def table_html(rows: List[Dict[str, str]], title: str, highlight_col: str = "F1") -> str:
    if not rows:
        return f"<h3>{html_escape(title)}</h3><p><em>No data</em></p>"
    headers = list(rows[0].keys())
    b = best_row(rows, highlight_col) if highlight_col in headers else {}
    best_val = b.get(highlight_col)
    out = [f"<h3>{html_escape(title)}</h3>", "<table>", "<thead><tr>"]
    out += [f"<th>{html_escape(h)}</th>" for h in headers]
    out += ["</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>")
        for h in headers:
            cls = ""
            if best_val is not None and h == highlight_col and r.get(h) == best_val:
                cls = ' class="best"'
            out.append(f"<td{cls}>{html_escape(r.get(h, ''))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build A-F report for one GT protocol (coarse or fine).")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--gt-protocol", choices=["coarse", "fine"], required=True)
    parser.add_argument("--out", required=True, help="Output HTML path relative to repo root.")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    tables = repo / "thesis_project" / "tables"
    tag = args.gt_protocol

    # Snapshot summaries (already produced during reruns)
    paths: List[Tuple[str, Path]] = [
        ("A", tables / "expA" / f"expA_summary_table_{tag}.csv"),
        ("B", tables / "expB" / f"expB_window_size_summary_{tag}.csv"),
        ("C", tables / "expC" / f"expC_rule_summary_{tag}.csv"),
        ("D", tables / "expD" / f"expD_model_comparison_{tag}.csv"),
        ("E", tables / "expE" / f"expE_pruning_summary_{tag}.csv"),
        ("F1", tables / "expF" / f"expF1_marker_gated_summary_{tag}.csv"),
        ("F2", tables / "expF" / f"expF2_prominence_summary_{tag}.csv"),
        ("F3", tables / "expF" / f"expF3_slide_summary_{tag}.csv"),
    ]
    data = {k: load_csv(p) for k, p in paths}

    best_lines = []
    for k, rows in data.items():
        b = best_row(rows, "F1")
        if b:
            best_lines.append(f"{k} best: {b.get('Setting', b.get('Model', b.get('Method', '-')))} (F1={b.get('F1', '-')})")
        else:
            best_lines.append(f"{k}: no data")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Experiment Report ({html_escape(tag.upper())} GT)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px auto; max-width: 1300px; padding: 0 12px; background: #fafafa; color: #1f2937; }}
    h1, h2, h3 {{ margin: 8px 0; }}
    .sub {{ color: #6b7280; margin-bottom: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; background: white; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 12px; background: white; border: 1px solid #e5e7eb; display: block; overflow-x: auto; white-space: nowrap; }}
    th, td {{ border-bottom: 1px solid #f0f2f5; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    .best {{ background: #e8f7ee; font-weight: 700; color: #166534; }}
    ul {{ margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>Experiment Report ({html_escape(tag.upper())} GT)</h1>
  <div class="sub">Single-protocol view only (no coarse/fine comparison table)</div>
  <div class="card"><strong>Best per experiment</strong><ul>
    {''.join(f"<li>{html_escape(x)}</li>" for x in best_lines)}
  </ul></div>

  <h2>Experiment A</h2><div class="card">{table_html(data['A'], 'A Summary', 'F1')}</div>
  <h2>Experiment B</h2><div class="card">{table_html(data['B'], 'B Summary', 'F1')}</div>
  <h2>Experiment C</h2><div class="card">{table_html(data['C'], 'C Summary', 'F1')}</div>
  <h2>Experiment D</h2><div class="card">{table_html(data['D'], 'D Summary', 'F1')}</div>
  <h2>Experiment E</h2><div class="card">{table_html(data['E'], 'E Summary', 'F1')}</div>
  <h2>Experiment F1</h2><div class="card">{table_html(data['F1'], 'F1 Summary', 'F1')}</div>
  <h2>Experiment F2</h2><div class="card">{table_html(data['F2'], 'F2 Summary', 'F1')}</div>
  <h2>Experiment F3</h2><div class="card">{table_html(data['F3'], 'F3 Summary', 'F1')}</div>
</body>
</html>"""

    out_path = (repo / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
