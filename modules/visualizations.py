"""
Visualization Engine
Creates interactive Plotly charts for the Flow Metrics Dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class VisualizationEngine:
    """
    Engine for creating interactive visualizations of flow and financial metrics.
    """

    # Color scheme for consistent styling
    COLORS = {
        "primary": "#3498DB",
        "secondary": "#2ECC71",
        "accent": "#9B59B6",
        "warning": "#F39C12",
        "danger": "#E74C3C",
        "neutral": "#95A5A6",
        "dark": "#2C3E50",
        "light": "#ECF0F1",
        "throughput": "#3498DB",
        "cycle_time": "#E74C3C",
        "flow_efficiency": "#2ECC71",
        "wip": "#F39C12",
        "flow_predictability": "#9B59B6",
    }

    def __init__(self, flow_engine, financial_engine):
        """
        Initialize visualization engine.

        Args:
            flow_engine: FlowMetricsEngine instance
            financial_engine: FinancialMetricsEngine instance
        """
        self.flow = flow_engine
        self.financial = financial_engine

    def create_littles_law_visualization(self) -> go.Figure:
        """
        Create visualization showing Little's Law relationship.
        WIP = Throughput × Cycle Time
        """
        metrics = self.flow.get_all_metrics()
        validation = self.flow.validate_littles_law()

        # Create 3D surface showing the relationship
        throughput_range = np.linspace(10, 100, 30)
        cycle_time_range = np.linspace(5, 30, 30)
        T, C = np.meshgrid(throughput_range, cycle_time_range)
        WIP = (T / 7) * C  # Convert throughput to daily

        fig = go.Figure()

        # Add surface
        fig.add_trace(
            go.Surface(
                x=T,
                y=C,
                z=WIP,
                colorscale="Viridis",
                opacity=0.7,
                name="Little's Law Surface",
                showscale=True,
                colorbar=dict(title="WIP"),
            )
        )

        # Add current position marker
        current_throughput = metrics["throughput"]
        current_cycle = metrics["cycle_time"]
        current_wip = metrics["wip"]

        fig.add_trace(
            go.Scatter3d(
                x=[current_throughput],
                y=[current_cycle],
                z=[current_wip],
                mode="markers",
                marker=dict(size=10, color="red", symbol="diamond"),
                name=f"Current Position (WIP={current_wip:.0f})",
            )
        )

        # Add expected WIP marker
        fig.add_trace(
            go.Scatter3d(
                x=[current_throughput],
                y=[current_cycle],
                z=[validation["expected_wip"]],
                mode="markers",
                marker=dict(size=8, color="green", symbol="circle"),
                name=f"Expected WIP ({validation['expected_wip']:.0f})",
            )
        )

        fig.update_layout(
            title=dict(
                text="Little's Law: WIP = Throughput × Cycle Time", font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="Throughput (items/week)",
                yaxis_title="Cycle Time (days)",
                zaxis_title="WIP (items)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            height=500,
            margin=dict(l=0, r=0, t=40, b=120),
            annotations=[
                dict(
                    text="<b>How to read:</b> Red diamond = your position. Green dot = expected by Little's Law. If red is above surface, you have excess WIP.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.22,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    align="center",
                )
            ],
        )

        return fig

    def create_dependency_network(self) -> go.Figure:
        """
        Create a network diagram showing metric dependencies.
        """
        # Node positions (arranged in a circle)
        metrics = [
            "Throughput",
            "Cycle Time",
            "Flow Efficiency",
            "WIP",
            "Predictability",
        ]
        n = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_pos = np.cos(angles) * 2
        y_pos = np.sin(angles) * 2

        # Define edges with strengths
        edges = [
            (0, 3, 0.8, "direct"),  # Throughput -> WIP
            (1, 3, 0.9, "direct"),  # Cycle Time -> WIP
            (2, 1, 0.8, "inverse"),  # Flow Efficiency -> Cycle Time
            (2, 0, 0.7, "direct"),  # Flow Efficiency -> Throughput
            (3, 2, 0.6, "inverse"),  # WIP -> Flow Efficiency
            (3, 4, 0.5, "inverse"),  # WIP -> Predictability
            (4, 0, 0.5, "direct"),  # Predictability -> Throughput
        ]

        fig = go.Figure()

        # Draw edges
        for source, target, strength, rel_type in edges:
            color = (
                self.COLORS["secondary"]
                if rel_type == "direct"
                else self.COLORS["danger"]
            )
            width = strength * 4

            # Create curved edge
            mid_x = (x_pos[source] + x_pos[target]) / 2 + 0.3
            mid_y = (y_pos[source] + y_pos[target]) / 2 + 0.3

            fig.add_trace(
                go.Scatter(
                    x=[x_pos[source], mid_x, x_pos[target]],
                    y=[y_pos[source], mid_y, y_pos[target]],
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="text",
                    hovertext=f"{metrics[source]} → {metrics[target]}<br>Strength: {strength:.1f}<br>Type: {rel_type}",
                    showlegend=False,
                )
            )

        # Draw nodes
        colors = [
            self.COLORS["throughput"],
            self.COLORS["cycle_time"],
            self.COLORS["flow_efficiency"],
            self.COLORS["wip"],
            self.COLORS["flow_predictability"],
        ]

        current_values = self.flow.get_all_metrics()
        value_list = [
            current_values["throughput"],
            current_values["cycle_time"],
            current_values["flow_efficiency"] * 100,
            current_values["wip"],
            current_values["flow_predictability"] * 100,
        ]

        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_pos,
                mode="markers+text",
                marker=dict(size=50, color=colors, line=dict(width=2, color="white")),
                text=metrics,
                textposition="bottom center",
                hoverinfo="text",
                hovertext=[
                    f"{m}<br>Value: {v:.1f}" for m, v in zip(metrics, value_list)
                ],
                showlegend=False,
            )
        )

        # Add legend
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(width=3, color=self.COLORS["secondary"]),
                name="Direct Relationship",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(width=3, color=self.COLORS["danger"]),
                name="Inverse Relationship",
            )
        )

        fig.update_layout(
            title=dict(text="Metric Dependency Network", font=dict(size=16)),
            showlegend=True,
            legend=dict(x=0, y=1),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=450,
            margin=dict(l=20, r=20, t=40, b=80),
            annotations=[
                dict(
                    text="<b>Legend:</b> Green = direct relationship (↑↑), Red = inverse (↑↓). Thicker = stronger.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.18,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    align="center",
                )
            ],
        )

        return fig

    def create_correlation_heatmap(self) -> go.Figure:
        """
        Create a heatmap showing correlations between metrics.
        """
        matrix = self.flow.get_relationship_matrix()
        metrics = [
            "Throughput",
            "Cycle Time",
            "Flow Efficiency",
            "WIP",
            "Predictability",
        ]

        # Create custom colorscale (blue for negative, white for zero, red for positive)
        colorscale = [
            [0, "#3498DB"],  # Strong negative
            [0.5, "#FFFFFF"],  # Zero
            [1, "#E74C3C"],  # Strong positive
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=metrics,
                y=metrics,
                colorscale=colorscale,
                zmid=0,
                text=np.round(matrix, 2),
                texttemplate="%{text}",
                textfont=dict(size=12),
                hovertemplate="%{y} → %{x}<br>Correlation: %{z:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(text="Metric Correlation Matrix", font=dict(size=16)),
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
            height=400,
            margin=dict(l=100, r=20, t=40, b=120),
            annotations=[
                dict(
                    text="<b>Reading:</b> Red (+1) = move together, Blue (-1) = move opposite, White (0) = independent",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.28,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )
            ],
        )

        return fig

    def create_radar_chart(self, benchmarks: Dict[str, float]) -> go.Figure:
        """
        Create radar chart comparing current values to benchmarks.
        """
        metrics = self.flow.get_all_metrics()

        categories = [
            "Throughput",
            "Cycle Time<br>(inverse)",
            "Flow<br>Efficiency",
            "WIP<br>(inverse)",
            "Predictability",
        ]

        # Normalize values to 0-100 scale for radar chart
        current_normalized = [
            min(100, (metrics["throughput"] / benchmarks["throughput"]) * 100),
            min(
                100, (benchmarks["cycle_time"] / metrics["cycle_time"]) * 100
            ),  # Inverse
            min(
                100, (metrics["flow_efficiency"] / benchmarks["flow_efficiency"]) * 100
            ),
            min(100, (benchmarks["wip"] / metrics["wip"]) * 100),  # Inverse
            min(
                100,
                (metrics["flow_predictability"] / benchmarks["flow_predictability"])
                * 100,
            ),
        ]

        benchmark_normalized = [100, 100, 100, 100, 100]  # Benchmark is always 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=current_normalized + [current_normalized[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(52, 152, 219, 0.4)",
                line=dict(color=self.COLORS["primary"], width=3),
                name="Current",
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=benchmark_normalized + [benchmark_normalized[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(46, 204, 113, 0.1)",
                line=dict(color=self.COLORS["secondary"], width=3, dash="dash"),
                name="Benchmark",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 120],
                    ticksuffix="%",
                    tickfont=dict(size=12, color="#AAAAAA"),
                    gridcolor="rgba(255,255,255,0.2)",
                    linecolor="rgba(255,255,255,0.3)",
                ),
                angularaxis=dict(
                    tickfont=dict(size=11, color="#FFFFFF"),
                    gridcolor="rgba(255,255,255,0.15)",
                ),
                bgcolor="rgba(30,30,30,0.6)",
            ),
            showlegend=True,
            legend=dict(x=0.85, y=1.15, font=dict(color="#FFFFFF")),
            title=dict(text="Performance vs Benchmark", font=dict(size=16)),
            height=450,
            margin=dict(l=80, r=80, t=80, b=100),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(
                    text="<b>Reading:</b> Blue area = current performance. Dashed = benchmark (100%). Larger blue area = better.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.18,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )
            ],
        )

        return fig

    def create_weighted_score_gauge(
        self, weights: Dict[str, float], benchmarks: Dict[str, float]
    ) -> go.Figure:
        """
        Create a gauge chart showing weighted performance score.
        """
        score = self.flow.calculate_weighted_score(weights, benchmarks)

        # Determine color based on score
        if score >= 80:
            color = self.COLORS["secondary"]
        elif score >= 60:
            color = self.COLORS["warning"]
        else:
            color = self.COLORS["danger"]

        fig = go.Figure(
            go.Indicator(
                mode="gauge",
                value=score,
                domain={"x": [0, 1], "y": [0.1, 0.9]},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickfont": {"size": 10},
                    },
                    "bar": {"color": color, "thickness": 0.75},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(231, 76, 60, 0.3)"},
                        {"range": [40, 70], "color": "rgba(243, 156, 18, 0.3)"},
                        {"range": [70, 100], "color": "rgba(46, 204, 113, 0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 80,
                    },
                },
            )
        )

        # Calculate delta from target
        delta = score - 80
        delta_color = self.COLORS["secondary"] if delta >= 0 else self.COLORS["danger"]
        delta_symbol = "▲" if delta >= 0 else "▼"

        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=60),
            annotations=[
                # Main score - centered in the gauge
                dict(
                    text=f"<b>{score:.1f}</b>",
                    x=0.5,
                    y=0.35,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=42, color=color),
                    align="center",
                ),
                # Label below score
                dict(
                    text="Weighted Score",
                    x=0.5,
                    y=0.22,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="gray"),
                    align="center",
                ),
                # Delta from target
                dict(
                    text=f"{delta_symbol} {abs(delta):.1f} vs target (80)",
                    x=0.5,
                    y=0.12,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=11, color=delta_color),
                    align="center",
                ),
                # Legend at bottom
                dict(
                    text="<b>Score:</b> 0-40 Poor | 40-70 Fair | 70-100 Good",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.02,
                    showarrow=False,
                    font=dict(size=9, color="gray"),
                ),
            ],
        )

        return fig

    def create_metric_breakdown_chart(self) -> go.Figure:
        """
        Create bar chart showing each metric's contribution.
        """
        metrics = self.flow.get_all_metrics()

        # Normalize to percentage of max for visualization
        data = {
            "Metric": [
                "Throughput",
                "Cycle Time",
                "Flow Efficiency",
                "WIP",
                "Predictability",
            ],
            "Value": [
                metrics["throughput"],
                metrics["cycle_time"],
                metrics["flow_efficiency"] * 100,
                metrics["wip"],
                metrics["flow_predictability"] * 100,
            ],
            "Unit": ["items/wk", "days", "%", "items", "%"],
            "Color": [
                self.COLORS["throughput"],
                self.COLORS["cycle_time"],
                self.COLORS["flow_efficiency"],
                self.COLORS["wip"],
                self.COLORS["flow_predictability"],
            ],
        }

        fig = go.Figure()

        for i, (metric, value, unit, color) in enumerate(
            zip(data["Metric"], data["Value"], data["Unit"], data["Color"])
        ):
            fig.add_trace(
                go.Bar(
                    x=[metric],
                    y=[value],
                    name=metric,
                    marker_color=color,
                    text=[f"{value:.1f} {unit}"],
                    textposition="outside",
                    hovertemplate=f"{metric}<br>Value: {value:.1f} {unit}<extra></extra>",
                )
            )

        fig.update_layout(
            title=dict(text="Current Metric Values", font=dict(size=16)),
            showlegend=False,
            xaxis_title="Metric",
            yaxis_title="Value",
            height=400,
            margin=dict(l=60, r=20, t=40, b=60),
        )

        return fig

    def create_financial_waterfall(self) -> go.Figure:
        """
        Create waterfall chart showing revenue to profit breakdown.
        """
        breakdown = self.financial.generate_financial_breakdown()

        # Prepare data for waterfall
        labels = [item["subcategory"] for item in breakdown]
        values = [item["value"] for item in breakdown]

        # Add net profit at the end
        net_profit = sum(values)
        labels.append("Net Profit")
        values.append(net_profit)

        # Determine measure type for waterfall
        measures = ["absolute"] + ["relative"] * (len(values) - 2) + ["total"]

        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=measures,
                x=labels,
                y=values,
                text=[f"${abs(v):,.0f}" for v in values],
                textposition="outside",
                textfont=dict(size=10),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": self.COLORS["secondary"]}},
                decreasing={"marker": {"color": self.COLORS["danger"]}},
                totals={"marker": {"color": self.COLORS["primary"]}},
            )
        )

        # Calculate y-axis range to fit all labels
        max_val = max(values) if values else 0
        min_val = min(values) if values else 0
        y_padding = (max_val - min_val) * 0.2

        fig.update_layout(
            title=dict(text="Weekly Financial Breakdown", font=dict(size=16)),
            showlegend=False,
            xaxis_title="Category",
            yaxis_title="Amount ($)",
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[min_val - y_padding, max_val + y_padding]),
            height=550,
            margin=dict(l=60, r=20, t=60, b=180),
            annotations=[
                dict(
                    text="<b>Reading:</b> Green = income, Red = costs. Final bar = net profit. Hidden costs reduce true profitability.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.38,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )
            ],
        )

        return fig

    def create_delay_cost_chart(self) -> go.Figure:
        """
        Create chart showing cost of delay breakdown.
        """
        delay_costs = self.financial.calculate_cost_of_delay()

        labels = ["Delay per Item", "Total WIP Delay", "Opportunity Cost"]
        values = [
            delay_costs["delay_per_item"],
            delay_costs["total_wip_delay"],
            delay_costs["opportunity_cost"],
        ]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker=dict(
                        colors=[
                            self.COLORS["warning"],
                            self.COLORS["danger"],
                            self.COLORS["accent"],
                        ]
                    ),
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>",
                )
            ]
        )

        # Add center annotation
        fig.add_annotation(
            text=f"Total<br>${delay_costs['total_delay_cost']:,.0f}",
            x=0.5,
            y=0.5,
            font_size=14,
            showarrow=False,
        )

        fig.update_layout(
            title=dict(text="Cost of Delay Breakdown", font=dict(size=16)),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    def create_roi_comparison_chart(self, improvement_pct: float = 10.0) -> go.Figure:
        """
        Create chart comparing ROI of improving each metric.
        """
        sensitivities = self.financial.calculate_metric_financial_sensitivity(
            improvement_pct
        )

        metrics = list(sensitivities.keys())
        annual_impacts = [sensitivities[m]["total_annual_impact"] for m in metrics]

        # Format metric names
        metric_labels = [m.replace("_", " ").title() for m in metrics]

        # Sort by impact
        sorted_data = sorted(
            zip(metric_labels, annual_impacts), key=lambda x: x[1], reverse=True
        )
        metric_labels, annual_impacts = zip(*sorted_data)

        colors = [
            self.COLORS["secondary"] if v > 0 else self.COLORS["danger"]
            for v in annual_impacts
        ]

        fig = go.Figure(
            go.Bar(
                x=list(metric_labels),
                y=list(annual_impacts),
                marker_color=colors,
                text=[f"${v:,.0f}" for v in annual_impacts],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate="%{x}<br>Annual Impact: $%{y:,.0f}<extra></extra>",
            )
        )

        # Calculate y-axis range to fit all labels
        max_val = max(annual_impacts) if annual_impacts else 0
        min_val = min(annual_impacts) if annual_impacts else 0
        y_range_padding = (
            (max_val - min_val) * 0.25 if max_val != min_val else abs(max_val) * 0.25
        )

        fig.update_layout(
            title=dict(
                text=f"Annual Financial Impact of {improvement_pct:.1f}% Improvement",
                font=dict(size=16),
            ),
            xaxis_title="Metric",
            yaxis_title="Annual Impact ($)",
            yaxis=dict(range=[min_val - y_range_padding, max_val + y_range_padding]),
            height=500,
            margin=dict(l=60, r=20, t=60, b=120),
            annotations=[
                dict(
                    text=f"<b>Reading:</b> Taller bars = higher financial impact from {improvement_pct:.1f}% improvement. Focus efforts on tallest bars first.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.22,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )
            ],
        )

        return fig

    def create_scenario_analysis(
        self, metric_name: str, improvement_pct: float
    ) -> go.Figure:
        """
        Create chart showing impact of scenario change.
        """
        metric_key = metric_name.lower().replace(" ", "_")
        projected = self.flow.simulate_metric_change(metric_key, improvement_pct)
        current = self.flow.get_all_metrics()

        metrics = [
            "Throughput",
            "Cycle Time",
            "Flow Efficiency",
            "WIP",
            "Predictability",
        ]
        keys = [
            "throughput",
            "cycle_time",
            "flow_efficiency",
            "wip",
            "flow_predictability",
        ]

        current_vals = [
            (
                current[k]
                if k not in ["flow_efficiency", "flow_predictability"]
                else current[k] * 100
            )
            for k in keys
        ]
        projected_vals = [
            (
                projected[k]
                if k not in ["flow_efficiency", "flow_predictability"]
                else projected[k] * 100
            )
            for k in keys
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Current",
                x=metrics,
                y=current_vals,
                marker_color=self.COLORS["neutral"],
                text=[f"{v:.1f}" for v in current_vals],
                textposition="outside",
            )
        )

        fig.add_trace(
            go.Bar(
                name="Projected",
                x=metrics,
                y=projected_vals,
                marker_color=self.COLORS["primary"],
                text=[f"{v:.1f}" for v in projected_vals],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Impact of {improvement_pct:+}% {metric_name} Change",
                font=dict(size=16),
            ),
            barmode="group",
            xaxis_title="Metric",
            yaxis_title="Value",
            legend=dict(x=0.8, y=1.1),
            height=400,
            margin=dict(l=60, r=20, t=60, b=60),
        )

        return fig

    def create_sensitivity_heatmap(self, improvement_pct: float = 10.0) -> go.Figure:
        """
        Create heatmap showing sensitivity of all metrics to changes.
        """
        sensitivity_data = self.flow.get_sensitivity_data(improvement_pct)

        metrics = [
            "Throughput",
            "Cycle Time",
            "Flow Efficiency",
            "WIP",
            "Predictability",
        ]

        # Convert to matrix
        matrix = []
        for metric_key in [
            "throughput",
            "cycle_time",
            "flow_efficiency",
            "wip",
            "flow_predictability",
        ]:
            matrix.append(sensitivity_data[metric_key])

        matrix = np.array(matrix)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=metrics,
                y=[f"{improvement_pct}% ↑ {m}" for m in metrics],
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(matrix, 1),
                texttemplate="%{text}%",
                textfont=dict(size=11),
                hovertemplate="Changing %{y}<br>Impact on %{x}: %{z:.1f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Sensitivity Analysis: Impact of {improvement_pct}% Improvement",
                font=dict(size=16),
            ),
            xaxis_title="Affected Metric",
            yaxis_title="Changed Metric",
            height=400,
            margin=dict(l=140, r=20, t=40, b=120),
            annotations=[
                dict(
                    text=f"<b>Reading:</b> Each row shows effect of {improvement_pct}% improvement in that metric on all others. Green = positive, Red = negative.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.28,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )
            ],
        )

        return fig

    def create_priority_matrix(
        self, weights: Dict[str, float], benchmarks: Dict[str, float]
    ) -> go.Figure:
        """
        Create a priority matrix (effort vs impact).
        """
        opportunities = self.flow.identify_improvement_opportunities(benchmarks)

        if not opportunities:
            fig = go.Figure()
            fig.add_annotation(
                text="All metrics at or above benchmark!",
                x=0.5,
                y=0.5,
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(
                title=dict(text="Improvement Priority Matrix", font=dict(size=16)),
                xaxis_title="Relative Effort",
                yaxis_title="Financial Impact ($)",
                height=450,
                margin=dict(l=60, r=20, t=40, b=100),
                annotations=[
                    dict(
                        text="<b>Quadrants:</b> Top-left = Quick Wins (do first), Top-right = Major Projects, Bottom-left = Fill-ins, Bottom-right = Reconsider",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.2,
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                    )
                ],
            )
            return fig

        # Calculate effort (inverse of weight - harder to improve high-weight items)
        efforts = []
        impacts = []
        labels = []
        colors = []

        for opp in opportunities:
            metric_key = opp["metric"].lower().replace(" ", "_")
            effort = (1 - weights.get(metric_key, 0.2)) * 100 + opp["gap"] * 0.5
            impact = opp["potential_savings"]

            efforts.append(effort)
            impacts.append(impact)
            labels.append(opp["metric"])

            if opp["impact"] == "high":
                colors.append(self.COLORS["secondary"])
            elif opp["impact"] == "medium":
                colors.append(self.COLORS["warning"])
            else:
                colors.append(self.COLORS["danger"])

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=efforts,
                y=impacts,
                mode="markers+text",
                marker=dict(size=20, color=colors, line=dict(width=2, color="white")),
                text=labels,
                textposition="top center",
                hovertemplate="%{text}<br>Effort: %{x:.0f}<br>Impact: $%{y:,.0f}<extra></extra>",
            )
        )

        # Add quadrant lines
        max_effort = max(efforts) if efforts else 100
        max_impact = max(impacts) if impacts else 10000

        fig.add_hline(
            y=max_impact / 2, line_dash="dash", line_color="gray", opacity=0.5
        )
        fig.add_vline(
            x=max_effort / 2, line_dash="dash", line_color="gray", opacity=0.5
        )

        # Add quadrant labels
        fig.add_annotation(
            x=max_effort * 0.25,
            y=max_impact * 0.85,
            text="Quick Wins",
            showarrow=False,
            font=dict(color="green", size=12),
        )
        fig.add_annotation(
            x=max_effort * 0.75,
            y=max_impact * 0.85,
            text="Major Projects",
            showarrow=False,
            font=dict(color="blue", size=12),
        )
        fig.add_annotation(
            x=max_effort * 0.25,
            y=max_impact * 0.15,
            text="Fill-ins",
            showarrow=False,
            font=dict(color="gray", size=12),
        )
        fig.add_annotation(
            x=max_effort * 0.75,
            y=max_impact * 0.15,
            text="Reconsider",
            showarrow=False,
            font=dict(color="red", size=12),
        )

        fig.update_layout(
            title=dict(text="Improvement Priority Matrix", font=dict(size=16)),
            xaxis_title="Relative Effort",
            yaxis_title="Financial Impact ($)",
            height=450,
            margin=dict(l=60, r=20, t=40, b=60),
        )

        return fig

    def create_improvement_roadmap(
        self, opportunities: List[Dict], weeks: int = 24
    ) -> go.Figure:
        """
        Create a timeline/roadmap showing projected improvements.
        """
        if not opportunities:
            fig = go.Figure()
            fig.add_annotation(
                text="No improvement opportunities identified",
                x=0.5,
                y=0.5,
                showarrow=False,
                font_size=14,
            )
            fig.update_layout(height=300)
            return fig

        # Simulate cumulative improvements over time
        week_list = list(range(1, weeks + 1))
        cumulative_savings = [0]

        for i, week in enumerate(week_list[1:], 1):
            # Assume we tackle opportunities in order of priority
            # Scale milestone frequency based on total weeks
            milestone_interval = max(2, weeks // 6)
            opp_idx = min(i // milestone_interval, len(opportunities) - 1)
            weekly_savings = sum(
                opp["potential_savings"] / 52 for opp in opportunities[: opp_idx + 1]
            )
            cumulative_savings.append(cumulative_savings[-1] + weekly_savings)

        fig = go.Figure()

        # Add cumulative savings line
        fig.add_trace(
            go.Scatter(
                x=week_list,
                y=cumulative_savings,
                mode="lines+markers",
                name="Cumulative Savings",
                line=dict(color=self.COLORS["secondary"], width=3),
                fill="tozeroy",
                fillcolor="rgba(46, 204, 113, 0.2)",
                marker=dict(size=4),
            )
        )

        # Add milestone markers
        milestone_interval = max(2, weeks // 6)
        for i, opp in enumerate(opportunities[: min(5, weeks // milestone_interval)]):
            milestone_week = (i + 1) * milestone_interval
            if milestone_week <= weeks:
                milestone_value = (
                    cumulative_savings[milestone_week - 1]
                    if milestone_week <= len(cumulative_savings)
                    else cumulative_savings[-1]
                )
                fig.add_trace(
                    go.Scatter(
                        x=[milestone_week],
                        y=[milestone_value],
                        mode="markers+text",
                        marker=dict(
                            size=15, color=self.COLORS["primary"], symbol="star"
                        ),
                        text=[opp["metric"]],
                        textposition="top center",
                        showlegend=False,
                        hovertemplate=f"Week {milestone_week}<br>{opp['metric']}<br>Savings: ${milestone_value:,.0f}<extra></extra>",
                    )
                )

        fig.update_layout(
            title=dict(
                text=f"Projected Improvement Roadmap ({weeks} Weeks)",
                font=dict(size=16),
            ),
            xaxis_title="Week",
            yaxis_title="Cumulative Savings ($)",
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            height=400,
            margin=dict(l=60, r=20, t=40, b=60),
        )

        return fig
