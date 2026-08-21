"""Multi-Format Executive Report Export Engine.

Generates publication-grade, standalone deliverables:
1. Interactive Standalone HTML Report (Responsive CSS, base64-embedded charts, styled tables)
2. Multi-Sheet Executive Excel Workbook (Executive Summary, Data Profile, Sample Data)
"""

import base64
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.app.config import get_settings


def _image_to_base64(image_path: str) -> Optional[str]:
    """Convert an image file to a base64 data URI string."""
    p = Path(image_path)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = p.suffix.lower().replace(".", "")
        return f"data:image/{ext};base64,{encoded}"
    except Exception:
        return None


def export_to_html(
    report_title: str,
    narrative_text: str,
    dataset_name: str,
    kpis: Optional[Dict[str, Any]] = None,
    chart_paths: Optional[List[str]] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a self-contained, beautifully styled interactive HTML executive report."""
    export_dir = Path(get_settings().data_dir).parent / "outputs" / "exports"
    os.makedirs(export_dir, exist_ok=True)
    filename = f"insightforge_report_{uuid.uuid4().hex[:8]}.html"
    export_path = str((export_dir / filename).resolve())

    # Build KPI cards HTML
    kpi_html = ""
    if kpis:
        cards = []
        for k, v in kpis.items():
            cards.append(f"""
            <div class="kpi-card">
                <div class="kpi-label">{k}</div>
                <div class="kpi-val">{v}</div>
            </div>
            """)
        kpi_html = f'<div class="kpi-grid">{"".join(cards)}</div>'

    # Build Chart embeds HTML
    charts_html = ""
    if chart_paths:
        chart_elements = []
        for cp in chart_paths:
            b64_uri = _image_to_base64(cp)
            if b64_uri:
                chart_elements.append(f"""
                <div class="chart-container">
                    <img src="{b64_uri}" alt="Analytical Visualization" class="chart-img" />
                </div>
                """)
        if chart_elements:
            charts_html = f'<h3>📊 Analytical Visualizations</h3><div class="charts-grid">{"".join(chart_elements)}</div>'

    # Build Data Table HTML
    table_html = ""
    if sample_df is not None and not sample_df.empty:
        df_clean = sample_df.head(15).fillna("")
        table_inner = df_clean.to_html(classes="data-table", index=False)
        table_html = f"<h3>📋 Data Sample Preview (First 15 Rows)</h3>{table_inner}"

    # Simple Markdown to HTML conversion for paragraphs and lists
    formatted_narrative = narrative_text.replace("\n\n", "<br><br>").replace("### ", "<h4>").replace("## ", "<h3>").replace("# ", "<h2>")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title} | InsightForge</title>
    <style>
        :root {{
            --primary: #2563eb;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }}
        .header {{
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #1e293b;
            font-size: 28px;
        }}
        .header .meta {{
            color: var(--text-muted);
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: #f1f5f9;
            padding: 18px;
            border-radius: 8px;
            border-left: 4px solid var(--primary);
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            font-weight: bold;
            color: var(--text-muted);
        }}
        .kpi-val {{
            font-size: 22px;
            font-weight: bold;
            color: #0f172a;
            margin-top: 5px;
        }}
        .narrative-section {{
            background: #ffffff;
            padding: 20px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }}
        .charts-grid {{
            display: flex;
            flex-direction: column;
            gap: 25px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            text-align: center;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            background: #ffffff;
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 13px;
        }}
        .data-table th, .data-table td {{
            border: 1px solid var(--border);
            padding: 8px 12px;
            text-align: left;
        }}
        .data-table th {{
            background: #f8fafc;
            color: #334155;
            font-weight: 600;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ {report_title}</h1>
            <div class="meta">Dataset: <strong>{dataset_name}</strong> | Generated by InsightForge Multi-Agent Engine</div>
        </div>

        {kpi_html}

        <div class="narrative-section">
            <h3>📝 Executive Analytical Narrative</h3>
            <div>{formatted_narrative}</div>
        </div>

        {charts_html}

        {table_html}

        <div class="footer">
            Generated autonomously by <strong>InsightForge</strong> • Multi-Agent Analytical System
        </div>
    </div>
</body>
</html>
"""

    with open(export_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return export_path


def export_to_excel(
    report_title: str,
    dataset_name: str,
    narrative_text: str,
    kpis: Optional[Dict[str, Any]] = None,
    num_stats: Optional[Dict[str, Any]] = None,
    sample_df: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a multi-sheet formatted Excel workbook (.xlsx)."""
    export_dir = Path(get_settings().data_dir).parent / "outputs" / "exports"
    os.makedirs(export_dir, exist_ok=True)
    filename = f"insightforge_workbook_{uuid.uuid4().hex[:8]}.xlsx"
    export_path = str((export_dir / filename).resolve())

    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
        # Sheet 1: Executive Summary
        summary_rows = [
            {"Metric / Property": "Report Title", "Value": report_title},
            {"Metric / Property": "Dataset Analyzed", "Value": dataset_name},
            {"Metric / Property": "Generated By", "Value": "InsightForge Autonomous Analytics"},
        ]
        if kpis:
            for k, v in kpis.items():
                summary_rows.append({"Metric / Property": k, "Value": str(v)})

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

        # Narrative text sub-sheet
        narrative_df = pd.DataFrame([{"Executive Narrative Report": narrative_text}])
        narrative_df.to_excel(writer, sheet_name="Executive_Narrative", index=False)

        # Sheet 2: Statistical Profile
        if num_stats:
            stats_df = pd.DataFrame.from_dict(num_stats, orient="index")
            stats_df.to_excel(writer, sheet_name="Statistical_Profile")

        # Sheet 3: Sample Data
        if sample_df is not None and not sample_df.empty:
            sample_df.head(100).to_excel(writer, sheet_name="Data_Sample", index=False)

    return export_path
