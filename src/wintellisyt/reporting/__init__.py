"""Dependency-free, provenance-labelled SVG and HTML run report."""
import csv
import html
import json
import math
from pathlib import Path


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _svg(title, xlabel, ylabel, series, note):
    points = [(x, y) for _, _, data in series for x, y in data]
    if not points:
        points = [(0, 0), (1, 1)]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    def xmap(x):
        return 80 + (x - xmin) / (xmax - xmin) * 650
    def ymap(y):
        return 350 - (y - ymin) / (ymax - ymin) * 260
    escape = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" '
             f'viewBox="0 0 800 450" role="img" aria-label="{escape(title)}">',
             '<rect width="800" height="450" fill="white"/>',
             f'<text x="80" y="27" font-size="18">{escape(title)}</text>',
             '<path d="M80 90 V350 H730" fill="none" stroke="#333"/>']
    for i in range(5):
        x = xmin + (xmax - xmin) * i / 4
        y = ymin + (ymax - ymin) * i / 4
        parts.extend([f'<text x="{xmap(x):.1f}" y="370" text-anchor="middle" '
                      f'font-size="11">{x:.2g}</text>',
                      f'<text x="72" y="{ymap(y):.1f}" text-anchor="end" '
                      f'font-size="11">{y:.2g}</text>'])
    for index, (name, color, data) in enumerate(series):
        for x, y in data:
            parts.append(f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="2.5" '
                         f'fill="{color}" fill-opacity="0.7"/>')
        parts.append(f'<text x="{80 + (index % 3) * 215}" y="{51 + (index // 3) * 14}" '
                     f'font-size="11" fill="{color}">{escape(name)} (n={len(data)})</text>')
    parts.extend([f'<text x="405" y="397" text-anchor="middle">{escape(xlabel)}</text>',
                  f'<text transform="translate(17 220) rotate(-90)" text-anchor="middle">'
                  f'{escape(ylabel)}</text>',
                  f'<text x="80" y="425" font-size="11">{escape(note)}</text></svg>'])
    return "\n".join(parts)


def write_report(output: Path, *, source, evidence):
    """Write figures from existing run artifacts; manifest need not exist yet."""
    output = Path(output)
    bins, observations, residuals = (_rows(output / name) for name in
                                     ("bins.csv", "observations.csv", "residuals.csv"))
    qa = json.loads((output / "qa.json").read_text(encoding="utf-8"))
    curves = []
    colors = {"raw": "#555555", "screened": "#0072b2", "normal": "#d55e00"}
    for turbine in sorted({row["turbine_id"] for row in bins}):
        for stratum, color in colors.items():
            data = []
            for row in bins:
                if (row["turbine_id"] == turbine and row["stratum"] == stratum
                        and row["supported"].lower() == "true"):
                    low, high, power = (_number(row[k]) for k in
                                        ("lower_mps", "upper_mps", "mean_power_kw"))
                    if None not in (low, high, power):
                        data.append(((low + high) / 2, power))
            if data:
                curves.append((f"{turbine} / {stratum}", color, data))
    def xy(rows, xkey, ykey):
        pairs = [(_number(row.get(xkey)), _number(row.get(ykey))) for row in rows]
        return [(x, y) for x, y in pairs if x is not None and y is not None]
    label = f"Source: {source}; evidence: {evidence}."
    plots = {
        "curves.svg": _svg("Supported descriptive power-curve bins", "Wind speed (m/s)",
                           "Mean electrical power (kW)", curves,
                           label + " Points are bin means; no predictive intervals."),
        "observations.svg": _svg("Raw observations", "Wind speed (m/s)",
                                 "Electrical power (kW)",
                                 [("Raw rows", "#0072b2", xy(observations, "wind_speed_ms",
                                                               "active_power_kw"))], label),
        "residuals.svg": _svg("Held-out residuals", "Wind speed (m/s)", "Residual (kW)",
                              [("Held-out rows", "#d55e00", xy(residuals, "wind_speed_ms",
                                                                "residual_kw"))],
                              label + " Residuals do not establish faults or recoverable loss."),
    }
    for name, content in plots.items():
        (output / name).write_text(content, encoding="utf-8")
    attribution = ""
    if evidence == "verified_real":
        from ..pipeline import source_manifest

        info = source_manifest()[source]
        creators = ", ".join(info["creators"])
        attribution = (
            f'<p>Data by {html.escape(creators)} and {html.escape(info["provider"])}. '
            f'<a href="{html.escape(info["source_url"])}">{html.escape(info["title"])} '
            f'{html.escape(info["version"])}</a>, '
            f'<a href="{html.escape(info["license_url"])}">{html.escape(info["license"])}</a>. '
            'WINTELLISYT maps source fields to a common schema, quarantines duplicate keys '
            'in screened analyses and aggregates observed power into wind speed bins.</p>'
        )
    sections = "\n".join(f'<figure><img src="{name}" alt="{html.escape(name)}"></figure>'
                         for name in plots)
    uncertainty = ('<p><a href="uncertainty.json">Descriptive held-out MAE block bootstrap</a></p>'
                   if (output / "uncertainty.json").is_file() else "")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>SCADA run report</title>
<body><h1>SCADA run report</h1><p>{html.escape(label)}</p>
<p>Descriptive bin means are not a certified power curve. Normal-state results require available
state labels. Residuals are held-out prediction errors, not fault diagnoses or recoverable losses.</p>
{attribution}{sections}<h2>Data quality</h2><pre>{html.escape(json.dumps(qa, indent=2))}</pre>
{uncertainty}<p>No predictive intervals are estimated. See the run manifest for input hashes and provenance
when available.</p></body></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")
