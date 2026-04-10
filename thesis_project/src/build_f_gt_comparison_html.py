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


def best_row(rows: List[Dict[str, str]]) -> Dict[str, str]:
    best = None
    best_f1 = float("-inf")
    for r in rows:
        try:
            f1 = float(r.get("F1", "nan"))
        except Exception:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best = r
    return best or {}


def table_html(rows: List[Dict[str, str]], title: str) -> str:
    if not rows:
        return f"<h3>{html_escape(title)}</h3><p><em>No data</em></p>"
    headers = list(rows[0].keys())
    best = best_row(rows)
    best_setting = best.get("Setting")
    out = [f"<h3>{html_escape(title)}</h3>", "<table>", "<thead><tr>"]
    out += [f"<th>{html_escape(h)}</th>" for h in headers]
    out += ["</tr></thead><tbody>"]
    for r in rows:
        row_cls = ' class="bestrow"' if r.get("Setting") == best_setting else ""
        out.append(f"<tr{row_cls}>")
        for h in headers:
            out.append(f"<td>{html_escape(r.get(h, ''))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def comparison_row(exp: str, coarse_rows: List[Dict[str, str]], fine_rows: List[Dict[str, str]]) -> Dict[str, str]:
    c = best_row(coarse_rows)
    f = best_row(fine_rows)
    c_f1 = float(c.get("F1", "0") or 0.0)
    f_f1 = float(f.get("F1", "0") or 0.0)
    return {
        "Experiment": exp,
        "Coarse Best Setting": c.get("Setting", "-"),
        "Coarse Best F1": f"{c_f1:.4f}",
        "Fine Best Setting": f.get("Setting", "-"),
        "Fine Best F1": f"{f_f1:.4f}",
        "Delta (Fine - Coarse)": f"{(f_f1 - c_f1):+.4f}",
    }


def summary_table(rows: List[Dict[str, str]]) -> str:
    headers = list(rows[0].keys())
    out = ["<table>", "<thead><tr>"]
    out += [f"<th>{html_escape(h)}</th>" for h in headers]
    out += ["</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>")
        for h in headers:
            v = r[h]
            cls = ' class="pos"' if h == "Delta (Fine - Coarse)" and v.startswith("+") else ""
            out.append(f"<td{cls}>{html_escape(v)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML comparison for F1-F3 across coarse vs fine GT.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--out", default="thesis_project/tables/expF/expF_gt_comparison.html")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tables = repo_root / "thesis_project" / "tables" / "expF"

    f1c = load_csv(tables / "expF1_marker_gated_summary_coarse.csv")
    f1f = load_csv(tables / "expF1_marker_gated_summary_fine.csv")
    f2c = load_csv(tables / "expF2_prominence_summary_coarse.csv")
    f2f = load_csv(tables / "expF2_prominence_summary_fine.csv")
    f3c = load_csv(tables / "expF3_slide_summary_coarse.csv")
    f3f = load_csv(tables / "expF3_slide_summary_fine.csv")

    comp = [
        comparison_row("F1", f1c, f1f),
        comparison_row("F2", f2c, f2f),
        comparison_row("F3", f3c, f3f),
    ]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>F1-F3 GT Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; }}
    h1, h2, h3 {{ margin: 10px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f7f7f7; }}
    .bestrow td {{ background: #eefbea; }}
    .pos {{ color: #1d7a1d; font-weight: bold; }}
    .subtle {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Experiment F1-F3: Coarse vs Fine GT</h1>
  <p class="subtle">Coarse GT: <code>ai_video_rbk/annotations_corrected</code> | Fine GT: <code>ai_video_rbk/annotations_fine</code></p>

  <h2>Best-Setting Comparison</h2>
  {summary_table(comp)}

  <h2>F1 Marker-Gated</h2>
  <div class="grid">
    <div class="card">{table_html(f1c, "Coarse GT")}</div>
    <div class="card">{table_html(f1f, "Fine GT")}</div>
  </div>

  <h2>F2 Prominence</h2>
  <div class="grid">
    <div class="card">{table_html(f2c, "Coarse GT")}</div>
    <div class="card">{table_html(f2f, "Fine GT")}</div>
  </div>

  <h2>F3 Slide-Assisted</h2>
  <div class="grid">
    <div class="card">{table_html(f3c, "Coarse GT")}</div>
    <div class="card">{table_html(f3f, "Fine GT")}</div>
  </div>
</body>
</html>"""

    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
