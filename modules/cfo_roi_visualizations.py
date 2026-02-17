"""
CFO ROI Visualizations
Charts for the CFO ROI Model tab in the Flow Metrics Dashboard.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional


COLORS = {
    "primary": "#3498DB",
    "secondary": "#2ECC71",
    "accent": "#9B59B6",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "neutral": "#95A5A6",
    "dark": "#2C3E50",
    "flow": "#3498DB",
    "ai": "#9B59B6",
    "business": "#2ECC71",
    "overlap": "#E74C3C",
    "other": "#95A5A6",
    "low": "#F39C12",
    "high": "#3498DB",
}

GROUP_COLORS = {
    "Flow": COLORS["flow"],
    "AI": COLORS["ai"],
    "Business Outcomes": COLORS["business"],
    "Overlap": COLORS["overlap"],
    "Other": COLORS["other"],
}


def create_expenses_breakdown(engine) -> go.Figure:
    """Horizontal bar chart of expenses by region."""
    if not engine.expenses:
        return _empty_figure("No expense data loaded")

    regions = [e.region for e in engine.expenses]
    totals = [e.total for e in engine.expenses]
    ftes = [e.fte_count for e in engine.expenses]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=regions,
            x=totals,
            orientation="h",
            marker_color=COLORS["primary"],
            text=[f"${v:,.0f} ({f:.0f} FTEs)" for v, f in zip(totals, ftes)],
            textposition="auto",
            hovertemplate="%{y}<br>Total: $%{x:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text="Engineering Cost by Region", font=dict(size=16)),
        xaxis_title="Total Cost ($)",
        yaxis=dict(autorange="reversed"),
        height=max(250, len(regions) * 50 + 100),
        margin=dict(l=160, r=20, t=40, b=40),
    )
    return fig


def create_investment_breakdown(engine) -> go.Figure:
    """Grouped bar chart: annual recurring vs one-time investment."""
    if not engine.investments:
        return _empty_figure("No investment data loaded")

    cats = [i.category for i in engine.investments]
    annual = [i.annual_recurring for i in engine.investments]
    one_time = [i.one_time for i in engine.investments]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Annual Recurring",
            x=cats,
            y=annual,
            marker_color=COLORS["primary"],
            text=[f"${v:,.0f}" for v in annual],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="One-time",
            x=cats,
            y=one_time,
            marker_color=COLORS["warning"],
            text=[f"${v:,.0f}" for v in one_time],
            textposition="outside",
        )
    )

    max_val = max(max(annual, default=0), max(one_time, default=0))
    fig.update_layout(
        title=dict(text="Investment Breakdown", font=dict(size=16)),
        barmode="group",
        yaxis_title="Amount ($)",
        yaxis=dict(range=[0, max_val * 1.3]),
        xaxis=dict(tickangle=-30),
        height=400,
        margin=dict(l=60, r=20, t=40, b=120),
    )
    return fig


def create_roi_by_group_chart(engine, scenario: str = "low") -> go.Figure:
    """Stacked/grouped bar showing ROI contributions by group for a scenario."""
    groups = engine.roi_rows_by_group()
    if not groups:
        return _empty_figure("No ROI data loaded")

    group_names = []
    group_totals = []
    colors = []

    for gname in ["Flow", "AI", "Business Outcomes", "Overlap", "Other"]:
        if gname not in groups:
            continue
        total = sum(r.low if scenario == "low" else r.high for r in groups[gname])
        group_names.append(gname)
        group_totals.append(total)
        colors.append(GROUP_COLORS.get(gname, COLORS["neutral"]))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=group_names,
            y=group_totals,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in group_totals],
            textposition="outside",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    min_val = min(group_totals, default=0)
    max_val = max(group_totals, default=0)
    padding = (max_val - min_val) * 0.2 if max_val != min_val else abs(max_val) * 0.3

    fig.update_layout(
        title=dict(
            text=f"ROI by Category ({scenario.title()} Scenario)",
            font=dict(size=16),
        ),
        yaxis_title="Annual Impact ($)",
        yaxis=dict(range=[min_val - padding, max_val + padding]),
        height=400,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def create_roi_waterfall(engine, scenario: str = "low") -> go.Figure:
    """Waterfall from Gross ROI → Net Annual → Net Y1 for both scenarios."""
    s = engine.compute_summary(scenario)

    labels = [
        "Flow ROI",
        "AI ROI",
        "Business Outcomes",
        "Overlap Adj.",
        "Gross ROI",
        "Annual Investment",
        "Net Annual ROI",
        f"{engine.scenario_a_name} One-time",
        f"Net Y1 ({engine.scenario_a_name})",
        f"{engine.scenario_b_name} Bonus",
        f"Net Y1 ({engine.scenario_b_name})",
    ]
    values = [
        s["flow_total"],
        s["ai_total"],
        s["business_total"],
        s["overlap_total"],
        s["gross_roi"],
        -s["annual_investment"],
        s["net_annual_roi"],
        -s["scenario_a_one_time"],
        s["net_roi_y1_a"],
        -s["scenario_b_bonus"],
        s["net_roi_y1_b"],
    ]
    measures = [
        "relative", "relative", "relative", "relative",
        "total",
        "relative",
        "total",
        "relative",
        "total",
        "relative",
        "total",
    ]

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=[f"${abs(v):,.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=9),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": COLORS["secondary"]}},
            decreasing={"marker": {"color": COLORS["danger"]}},
            totals={"marker": {"color": COLORS["primary"]}},
        )
    )

    max_val = max(values, default=0)
    min_val = min(values, default=0)
    padding = (max_val - min_val) * 0.25

    fig.update_layout(
        title=dict(
            text=f"ROI Waterfall ({scenario.title()} Scenario)",
            font=dict(size=16),
        ),
        yaxis_title="Amount ($)",
        yaxis=dict(range=[min_val - padding, max_val + padding]),
        xaxis=dict(tickangle=-45),
        height=550,
        margin=dict(l=60, r=20, t=60, b=160),
    )
    return fig


def create_scenario_comparison(engine) -> go.Figure:
    """Side-by-side comparison of Low vs High scenarios for key metrics."""
    both = engine.compute_both_scenarios()
    lo = both["low"]
    hi = both["high"]

    metrics = [
        "Flow ROI",
        "AI ROI",
        "Gross ROI",
        "Net Annual ROI",
        f"Net Y1 ({engine.scenario_a_name})",
        f"Net Y1 ({engine.scenario_b_name})",
    ]
    low_vals = [
        lo["flow_total"], lo["ai_total"], lo["gross_roi"],
        lo["net_annual_roi"], lo["net_roi_y1_a"], lo["net_roi_y1_b"],
    ]
    high_vals = [
        hi["flow_total"], hi["ai_total"], hi["gross_roi"],
        hi["net_annual_roi"], hi["net_roi_y1_a"], hi["net_roi_y1_b"],
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Conservative (Low)",
            x=metrics,
            y=low_vals,
            marker_color=COLORS["low"],
            text=[f"${v:,.0f}" for v in low_vals],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Optimistic (High)",
            x=metrics,
            y=high_vals,
            marker_color=COLORS["high"],
            text=[f"${v:,.0f}" for v in high_vals],
            textposition="outside",
        )
    )

    max_val = max(max(high_vals, default=0), max(low_vals, default=0))
    fig.update_layout(
        title=dict(text="Conservative vs Optimistic Scenario", font=dict(size=16)),
        barmode="group",
        yaxis_title="Amount ($)",
        yaxis=dict(range=[0, max_val * 1.25]),
        xaxis=dict(tickangle=-30),
        height=450,
        margin=dict(l=60, r=20, t=60, b=120),
    )
    return fig


def create_multi_year_projection(engine) -> go.Figure:
    """Line + bar chart showing Net ROI over 3 years for both scenarios."""
    lo = engine.compute_summary("low")
    hi = engine.compute_summary("high")

    years = ["Year 1", "Year 2", "Year 3"]

    # Scenario A
    a_lo = [lo["net_roi_y1_a"], lo["net_roi_y2"], lo["net_roi_y3"]]
    a_hi = [hi["net_roi_y1_a"], hi["net_roi_y2"], hi["net_roi_y3"]]

    # Scenario B
    b_lo = [lo["net_roi_y1_b"], lo["net_roi_y2"], lo["net_roi_y3"]]
    b_hi = [hi["net_roi_y1_b"], hi["net_roi_y2"], hi["net_roi_y3"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[engine.scenario_a_name, engine.scenario_b_name],
        shared_yaxes=True,
    )

    fig.add_trace(
        go.Bar(name="Conservative", x=years, y=a_lo, marker_color=COLORS["low"],
               text=[f"${v:,.0f}" for v in a_lo], textposition="outside",
               showlegend=True),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(name="Optimistic", x=years, y=a_hi, marker_color=COLORS["high"],
               text=[f"${v:,.0f}" for v in a_hi], textposition="outside",
               showlegend=True),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(name="Conservative", x=years, y=b_lo, marker_color=COLORS["low"],
               text=[f"${v:,.0f}" for v in b_lo], textposition="outside",
               showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(name="Optimistic", x=years, y=b_hi, marker_color=COLORS["high"],
               text=[f"${v:,.0f}" for v in b_hi], textposition="outside",
               showlegend=False),
        row=1, col=2,
    )

    all_vals = a_lo + a_hi + b_lo + b_hi
    max_val = max(all_vals, default=0)
    min_val = min(all_vals, default=0)
    padding = (max_val - min_val) * 0.2

    fig.update_layout(
        title=dict(text="3-Year Net ROI Projection", font=dict(size=16)),
        barmode="group",
        height=450,
        margin=dict(l=60, r=20, t=60, b=40),
        yaxis=dict(range=[min_val - padding, max_val + padding]),
    )
    fig.update_yaxes(title_text="Net ROI ($)", row=1, col=1)

    return fig


def create_payback_gauge(engine) -> go.Figure:
    """Gauge charts showing payback period in months for both scenarios."""
    lo = engine.compute_summary("low")
    hi = engine.compute_summary("high")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=[engine.scenario_a_name, engine.scenario_b_name],
    )

    for col, (name, lo_val, hi_val) in enumerate(
        [
            (engine.scenario_a_name, lo["payback_months_a"], hi["payback_months_a"]),
            (engine.scenario_b_name, lo["payback_months_b"], hi["payback_months_b"]),
        ],
        start=1,
    ):
        avg_payback = (lo_val + hi_val) / 2
        display_val = min(avg_payback, 24)

        color = COLORS["secondary"] if avg_payback <= 6 else (
            COLORS["warning"] if avg_payback <= 12 else COLORS["danger"]
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=display_val,
                number=dict(suffix=" mo", font=dict(size=24)),
                gauge=dict(
                    axis=dict(range=[0, 24], ticksuffix=" mo"),
                    bar=dict(color=color, thickness=0.7),
                    steps=[
                        {"range": [0, 6], "color": "rgba(46,204,113,0.2)"},
                        {"range": [6, 12], "color": "rgba(243,156,18,0.2)"},
                        {"range": [12, 24], "color": "rgba(231,76,60,0.2)"},
                    ],
                ),
            ),
            row=1, col=col,
        )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[
            dict(
                text=f"Range: {lo['payback_months_a']:.1f}–{hi['payback_months_a']:.1f} months",
                x=0.22, y=-0.05, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="gray"),
            ),
            dict(
                text=f"Range: {lo['payback_months_b']:.1f}–{hi['payback_months_b']:.1f} months",
                x=0.78, y=-0.05, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color="gray"),
            ),
        ],
    )
    return fig


def create_roi_item_detail_chart(engine, scenario: str = "low") -> go.Figure:
    """Horizontal bar chart of each individual ROI line item, color-coded by group."""
    if not engine.roi_rows:
        return _empty_figure("No ROI data loaded")

    # Sort by absolute value
    items = sorted(
        engine.roi_rows,
        key=lambda r: abs(r.low if scenario == "low" else r.high),
        reverse=True,
    )

    cats = [r.category for r in items]
    vals = [r.low if scenario == "low" else r.high for r in items]
    colors = [GROUP_COLORS.get(r.group, COLORS["neutral"]) for r in items]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=cats,
            x=vals,
            orientation="h",
            marker_color=colors,
            text=[f"${v:,.0f}" for v in vals],
            textposition="auto",
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"ROI Line Items ({scenario.title()} Scenario)",
            font=dict(size=16),
        ),
        xaxis_title="Amount ($)",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(cats) * 35 + 100),
        margin=dict(l=280, r=20, t=40, b=40),
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, showarrow=False, font_size=14,
        xref="paper", yref="paper",
    )
    fig.update_layout(height=250, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig
