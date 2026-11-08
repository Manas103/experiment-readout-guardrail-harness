"""Renders the 144-slice release-gate family as a single static Plotly
dashboard: one point per (metric, geofence) slice, x-axis the measured log
rate ratio, y-axis -log10(p-value), colored by whether BH flagged it and
shaped by whether it was a seeded shift. This is the "Plotly" entry on
the stack line; it is a saved static HTML artifact, not a running
dashboard server.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from erg.autonomy_metrics import simulate_release_corpus
from erg.autonomy_release_gate import run_release_gate


def main() -> None:
    rng = np.random.default_rng(2026090103)
    corpus = simulate_release_corpus(
        n_seeded_shifts=30,
        shift_multiplier=1.60,
        baseline_exposure_miles=500_000.0,
        candidate_exposure_miles=500_000.0,
        rng=rng,
    )
    result = run_release_gate(corpus)

    is_seeded = np.array([o.is_seeded_shift for o in corpus.observations])
    families = np.array([o.family for o in corpus.observations])
    labels = [f"{o.metric_name} / {o.geofence}" for o in corpus.observations]
    neg_log10_p = -np.log10(np.clip(result.p_values, 1e-300, None))

    fig = go.Figure()
    group_specs = [
        ("null, not flagged", ~is_seeded & ~result.bh_flagged, "lightgray", "circle"),
        ("null, false alarm (BH)", ~is_seeded & result.bh_flagged, "orange", "circle"),
        ("seeded shift, missed", is_seeded & ~result.bh_flagged, "crimson", "x"),
        ("seeded shift, caught", is_seeded & result.bh_flagged, "seagreen", "star"),
    ]
    for name, mask, color, symbol in group_specs:
        fig.add_trace(
            go.Scatter(
                x=result.log_rate_ratios[mask],
                y=neg_log10_p[mask],
                mode="markers",
                name=f"{name} (n={int(mask.sum())})",
                marker=dict(color=color, symbol=symbol, size=9, line=dict(width=1, color="black")),
                text=[f"{lbl} [{fam}]" for lbl, fam, m in zip(labels, families, mask) if m],
                hoverinfo="text+x+y",
            )
        )
    fig.add_hline(y=-np.log10(0.05), line_dash="dot", line_color="gray", annotation_text="raw p=0.05")
    fig.update_layout(
        title="Fleet autonomy release gate: 144 (metric x geofence) slices, BH fdr_q=0.05",
        xaxis_title="log rate ratio (candidate / baseline)",
        yaxis_title="-log10(p-value)",
        template="plotly_white",
        width=1000,
        height=650,
    )

    out_path = Path(__file__).resolve().parent.parent / "docs" / "plots" / "autonomy_release_dashboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Wrote dashboard to {out_path}")


if __name__ == "__main__":
    main()
