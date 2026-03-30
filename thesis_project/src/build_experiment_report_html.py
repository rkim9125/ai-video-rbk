import argparse
import csv
from pathlib import Path
from typing import Dict, List


def load_csv(path: Path) -> List[Dict[str, str]]:
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


def table_html(rows: List[Dict[str, str]], title: str, highlight_col: str = "F1") -> str:
    if not rows:
        return f"<h3>{html_escape(title)}</h3><p><em>No data</em></p>"

    headers = list(rows[0].keys())
    best = None
    if highlight_col in headers:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[highlight_col]))
            except Exception:
                vals.append(float("-inf"))
        best = max(vals) if vals else None

    out = [f"<h3>{html_escape(title)}</h3>", "<table>", "<thead><tr>"]
    out += [f"<th>{html_escape(h)}</th>" for h in headers]
    out += ["</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>")
        for h in headers:
            v = r[h]
            cls = ""
            if h == highlight_col and best is not None:
                try:
                    if abs(float(v) - best) < 1e-12:
                        cls = ' class="best"'
                except Exception:
                    pass
            out.append(f"<td{cls}>{html_escape(v)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def best_row(rows: List[Dict[str, str]], key: str = "F1") -> Dict[str, str]:
    if not rows:
        return {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-page HTML report for Experiment A/B/C/D tables.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--out",
        default="thesis_project/tables/experiment_report.html",
        help="Output HTML path relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tables = repo_root / "thesis_project" / "tables"

    a_summary = load_csv(tables / "expA_summary_table.csv")
    a_lecture = load_csv(tables / "expA_lecture_level_table.csv")
    b_summary = load_csv(tables / "expB_window_size_summary.csv")
    b_lecture = load_csv(tables / "expB_window_size_lecture_f1.csv")
    c_summary = load_csv(tables / "expC_rule_summary.csv")
    c_lecture = load_csv(tables / "expC_rule_lecture_f1.csv")
    d_summary = load_csv(tables / "expD_model_comparison.csv")
    d_lecture = load_csv(tables / "expD_lecture_level_table.csv")

    a_best = best_row(a_summary, "F1")
    b_best = best_row(b_summary, "F1")
    c_best = best_row(c_summary, "F1")
    d_best = best_row(d_summary, "F1")

    headline_items = [
        f"A best: {a_best.get('Representation', '-')} (F1={a_best.get('F1', '-')})",
        f"B best: ws={b_best.get('Window Size', '-')} (F1={b_best.get('F1', '-')})",
        f"C best: {c_best.get('Rule', '-')} (F1={c_best.get('F1', '-')})",
        f"D best: {d_best.get('Model', '-')} (F1={d_best.get('F1', '-')})",
        "Overall: structural/marker cues help, but absolute F1 remains low -> hybrid refinement still needed.",
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Experiment Report (A/B/C/D)</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 24px auto;
      max-width: 1300px;
      padding: 0 12px;
      color: #1f2937;
      line-height: 1.45;
      background: #fafafa;
    }}
    h1 {{ margin: 0 0 8px 0; }}
    h2 {{ margin-top: 28px; margin-bottom: 8px; }}
    h3 {{ margin-top: 16px; margin-bottom: 6px; }}
    .sub {{ color: #6b7280; margin-bottom: 16px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 16px;
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
      display: block;
      overflow-x: auto;
      white-space: nowrap;
    }}
    th, td {{
      border-bottom: 1px solid #f0f2f5;
      padding: 10px 12px;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f8fafc;
      font-weight: 600;
    }}
    tr:hover td {{ background: #fcfcfd; }}
    .best {{
      background: #e8f7ee !important;
      font-weight: 700;
      color: #166534;
    }}
    .card {{
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 14px;
      background: white;
      margin-bottom: 14px;
    }}
    .headline {{
      border: 1px solid #dbeafe;
      background: #eff6ff;
      border-radius: 10px;
      padding: 14px 16px;
      margin: 12px 0 20px 0;
    }}
    .headline ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
    .headline li {{
      margin: 3px 0;
    }}
    .bestline {{
      font-size: 14px;
      color: #374151;
      margin: 2px 0 10px 0;
    }}
  </style>
</head>
<body>
  <h1>Experiment Report</h1>
  <div class="sub">A/B/C/D results in readable table format (best F1 highlighted in green)</div>
  <div class="headline">
    <strong>Key Findings</strong>
    <ul>
      <li>{html_escape(headline_items[0])}</li>
      <li>{html_escape(headline_items[1])}</li>
      <li>{html_escape(headline_items[2])}</li>
      <li>{html_escape(headline_items[3])}</li>
      <li>{html_escape(headline_items[4])}</li>
    </ul>
  </div>

  <h2>Experiment A - Representation</h2>
  <div class="bestline">Best setting: {html_escape(a_best.get('Representation', '-'))}, F1={html_escape(a_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(a_summary, "A Summary", highlight_col="F1")}
    {table_html(a_lecture, "A Lecture-level", highlight_col="F1")}
  </div>

  <h2>Experiment B - Window Size Effect</h2>
  <div class="bestline">Best setting: window size {html_escape(b_best.get('Window Size', '-'))}, F1={html_escape(b_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(b_summary, "B Summary", highlight_col="F1")}
    {table_html(b_lecture, "B Lecture-level F1", highlight_col="Avg F1")}
  </div>

  <h2>Experiment C - Boundary Rule</h2>
  <div class="bestline">Best setting: {html_escape(c_best.get('Rule', '-'))}, F1={html_escape(c_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(c_summary, "C Summary", highlight_col="F1")}
    {table_html(c_lecture, "C Lecture-level F1", highlight_col="Avg F1")}
  </div>

  <h2>Experiment D - Structural Signals</h2>
  <div class="bestline">Best setting: {html_escape(d_best.get('Model', '-'))}, F1={html_escape(d_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(d_summary, "D Summary", highlight_col="F1")}
    {table_html(d_lecture, "D Lecture-level", highlight_col="F1")}
  </div>
</body>
</html>"""

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
