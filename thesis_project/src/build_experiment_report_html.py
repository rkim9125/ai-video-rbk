import argparse
import csv
from pathlib import Path
from typing import Dict, List


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_csv_with_fallback(tables_dir: Path, filename: str, folder_hint: str) -> List[Dict[str, str]]:
    primary = tables_dir / filename
    fallback = tables_dir / folder_hint / filename
    if primary.exists():
        return load_csv(primary)
    if fallback.exists():
        return load_csv(fallback)
    return []


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


def best_f1_value(rows: List[Dict[str, str]]) -> float:
    b = best_row(rows, "F1")
    try:
        return float(b.get("F1", "nan"))
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-page HTML report for Experiment A-F tables.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--out",
        default="thesis_project/tables/experiment_report.html",
        help="Output HTML path relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tables = repo_root / "thesis_project" / "tables"

    a_summary = load_csv_with_fallback(tables, "expA_summary_table.csv", "expA")
    a_lecture = load_csv_with_fallback(tables, "expA_lecture_level_table.csv", "expA")
    b_summary = load_csv_with_fallback(tables, "expB_window_size_summary.csv", "expB")
    b_lecture = load_csv_with_fallback(tables, "expB_window_size_lecture_f1.csv", "expB")
    c_summary = load_csv_with_fallback(tables, "expC_rule_summary.csv", "expC")
    c_lecture = load_csv_with_fallback(tables, "expC_rule_lecture_f1.csv", "expC")
    d_summary = load_csv_with_fallback(tables, "expD_model_comparison.csv", "expD")
    d_lecture = load_csv_with_fallback(tables, "expD_lecture_level_table.csv", "expD")
    e_summary = load_csv_with_fallback(tables, "expE_pruning_summary.csv", "expE")
    e_lecture = load_csv_with_fallback(tables, "expE_pruning_lecture_f1.csv", "expE")
    f1_summary = load_csv_with_fallback(tables, "expF1_marker_gated_summary.csv", "expF")
    f2_summary = load_csv_with_fallback(tables, "expF2_prominence_summary.csv", "expF")
    f3_summary = load_csv_with_fallback(tables, "expF3_slide_summary.csv", "expF")
    # snapshot files for explicit coarse/fine comparison
    a_coarse = load_csv_with_fallback(tables, "expA_summary_table_coarse.csv", "expA")
    a_fine = load_csv_with_fallback(tables, "expA_summary_table_fine.csv", "expA")
    b_coarse = load_csv_with_fallback(tables, "expB_window_size_summary_coarse.csv", "expB")
    b_fine = load_csv_with_fallback(tables, "expB_window_size_summary_fine.csv", "expB")
    c_coarse = load_csv_with_fallback(tables, "expC_rule_summary_coarse.csv", "expC")
    c_fine = load_csv_with_fallback(tables, "expC_rule_summary_fine.csv", "expC")
    d_coarse = load_csv_with_fallback(tables, "expD_model_comparison_coarse.csv", "expD")
    d_fine = load_csv_with_fallback(tables, "expD_model_comparison_fine.csv", "expD")
    e_coarse = load_csv_with_fallback(tables, "expE_pruning_summary_coarse.csv", "expE")
    e_fine = load_csv_with_fallback(tables, "expE_pruning_summary_fine.csv", "expE")
    f1_coarse = load_csv_with_fallback(tables, "expF1_marker_gated_summary_coarse.csv", "expF")
    f1_fine = load_csv_with_fallback(tables, "expF1_marker_gated_summary_fine.csv", "expF")
    f2_coarse = load_csv_with_fallback(tables, "expF2_prominence_summary_coarse.csv", "expF")
    f2_fine = load_csv_with_fallback(tables, "expF2_prominence_summary_fine.csv", "expF")
    f3_coarse = load_csv_with_fallback(tables, "expF3_slide_summary_coarse.csv", "expF")
    f3_fine = load_csv_with_fallback(tables, "expF3_slide_summary_fine.csv", "expF")

    a_best = best_row(a_summary, "F1")
    b_best = best_row(b_summary, "F1")
    c_best = best_row(c_summary, "F1")
    d_best = best_row(d_summary, "F1")
    e_best = best_row(e_summary, "F1") if e_summary else {}
    f1_best = best_row(f1_summary, "F1") if f1_summary else {}
    f2_best = best_row(f2_summary, "F1") if f2_summary else {}
    f3_best = best_row(f3_summary, "F1") if f3_summary else {}
    compare_rows = [
        ("A", best_f1_value(a_coarse), best_f1_value(a_fine)),
        ("B", best_f1_value(b_coarse), best_f1_value(b_fine)),
        ("C", best_f1_value(c_coarse), best_f1_value(c_fine)),
        ("D", best_f1_value(d_coarse), best_f1_value(d_fine)),
        ("E", best_f1_value(e_coarse), best_f1_value(e_fine)),
        ("F1", best_f1_value(f1_coarse), best_f1_value(f1_fine)),
        ("F2", best_f1_value(f2_coarse), best_f1_value(f2_fine)),
        ("F3", best_f1_value(f3_coarse), best_f1_value(f3_fine)),
    ]
    compare_table_rows: List[str] = []
    for name, c, f in compare_rows:
        if str(c) == "nan" or str(f) == "nan":
            compare_table_rows.append(f"<tr><td>{html_escape(name)}</td><td>-</td><td>-</td><td>-</td></tr>")
            continue
        delta = f - c
        cls = "best" if delta > 0 else ""
        compare_table_rows.append(
            f"<tr><td>{html_escape(name)}</td><td>{c:.4f}</td><td>{f:.4f}</td><td class=\"{cls}\">{delta:+.4f}</td></tr>"
        )

    headline_items = [
        f"A best: {a_best.get('Representation', '-')} (F1={a_best.get('F1', '-')})",
        f"B best: ws={b_best.get('Window Size', '-')} (F1={b_best.get('F1', '-')})",
        f"C best: {c_best.get('Rule', '-')} (F1={c_best.get('F1', '-')})",
        f"D best: {d_best.get('Model', '-')} (F1={d_best.get('F1', '-')})",
        f"E best: min-distance={e_best.get('Min-distance', '-')} (F1={e_best.get('F1', '-')})"
        if e_best
        else "E: (run thesis_project/src/run_experiment_e.py to populate)",
        f"F1 best: {f1_best.get('Setting', '-')} (F1={f1_best.get('F1', '-')})" if f1_best else "F1: no data",
        f"F2 best: {f2_best.get('Setting', '-')} (F1={f2_best.get('F1', '-')})" if f2_best else "F2: no data",
        f"F3 best: {f3_best.get('Setting', '-')} (F1={f3_best.get('F1', '-')})" if f3_best else "F3: no data",
        "Report generated from latest run outputs (current setup: fine GT).",
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Experiment Report (A/B/C/D/E/F)</title>
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
  <div class="sub">A/B/C/D/E/F results in readable table format (best F1 highlighted in green)</div>
  <div class="headline">
    <strong>Key Findings</strong>
    <ul>
      <li>{html_escape(headline_items[0])}</li>
      <li>{html_escape(headline_items[1])}</li>
      <li>{html_escape(headline_items[2])}</li>
      <li>{html_escape(headline_items[3])}</li>
      <li>{html_escape(headline_items[4])}</li>
      <li>{html_escape(headline_items[5])}</li>
      <li>{html_escape(headline_items[6])}</li>
      <li>{html_escape(headline_items[7])}</li>
      <li>{html_escape(headline_items[8])}</li>
    </ul>
  </div>

  <h2>Coarse vs Fine GT (Best F1)</h2>
  <div class="card">
    <table>
      <thead><tr><th>Experiment</th><th>Coarse Best F1</th><th>Fine Best F1</th><th>Delta (Fine-Coarse)</th></tr></thead>
      <tbody>
        {''.join(compare_table_rows)}
      </tbody>
    </table>
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

  <h2>Experiment E - Prediction count / min-distance sweep</h2>
  <div class="bestline">Best setting: min-distance {html_escape(e_best.get('Min-distance', '-'))}, F1={html_escape(e_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(e_summary, "E Summary (pruning)", highlight_col="F1")}
    {table_html(e_lecture, "E Lecture-level F1", highlight_col="Avg F1")}
  </div>

  <h2>Experiment F1 - Marker-Gated Semantic</h2>
  <div class="bestline">Best setting: {html_escape(f1_best.get('Setting', '-'))}, F1={html_escape(f1_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(f1_summary, "F1 Summary", highlight_col="F1")}
  </div>

  <h2>Experiment F2 - Prominence on F1-best</h2>
  <div class="bestline">Best setting: {html_escape(f2_best.get('Setting', '-'))}, F1={html_escape(f2_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(f2_summary, "F2 Summary", highlight_col="F1")}
  </div>

  <h2>Experiment F3 - Marker + Slide Fusion</h2>
  <div class="bestline">Best setting: {html_escape(f3_best.get('Setting', '-'))}, F1={html_escape(f3_best.get('F1', '-'))}</div>
  <div class="card">
    {table_html(f3_summary, "F3 Summary", highlight_col="F1")}
  </div>
</body>
</html>"""

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
