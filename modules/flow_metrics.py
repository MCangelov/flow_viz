"""
Flow Metrics Engine
Handles calculations and relationships between the 5 core flow metrics:
- Throughput
- Cycle Time
- Flow Efficiency
- Work in Progress (WIP)
- Flow Predictability
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class MetricRelationship:
    """Represents a relationship between two metrics"""

    source: str
    target: str
    relationship_type: str  # 'direct', 'inverse', 'complex'
    strength: float  # 0.0 to 1.0
    formula: str


class FlowMetricsEngine:
    """
    Engine for calculating flow metrics and their interdependencies.
    Implements Little's Law and other flow relationships.
    """

    def __init__(
        self,
        throughput: float,
        cycle_time: float,
        flow_efficiency: float,
        wip: float,
        flow_predictability: float,
    ):
        """
        Initialize the flow metrics engine with current values.

        Args:
            throughput: Items completed per week
            cycle_time: Average days from start to completion
            flow_efficiency: Ratio of active work time to total time (0-1)
            wip: Number of items currently in progress
            flow_predictability: Measure of consistency (0-1, higher = more predictable)
        """
        self.throughput = throughput
        self.cycle_time = cycle_time
        self.flow_efficiency = flow_efficiency
        self.wip = wip
        self.flow_predictability = flow_predictability

        # Define metric relationships
        self._relationships = self._define_relationships()

    def _define_relationships(self) -> List[MetricRelationship]:
        """Define the mathematical relationships between metrics"""
        return [
            # Little's Law relationships
            MetricRelationship(
                source="throughput",
                target="wip",
                relationship_type="direct",
                strength=1.0,
                formula="WIP = Throughput × Cycle Time",
            ),
            MetricRelationship(
                source="cycle_time",
                target="wip",
                relationship_type="direct",
                strength=1.0,
                formula="WIP = Throughput × Cycle Time",
            ),
            MetricRelationship(
                source="wip",
                target="cycle_time",
                relationship_type="direct",
                strength=0.9,
                formula="Cycle Time = WIP / Throughput",
            ),
            # Flow Efficiency relationships
            MetricRelationship(
                source="flow_efficiency",
                target="cycle_time",
                relationship_type="inverse",
                strength=0.8,
                formula="Higher efficiency → Lower cycle time",
            ),
            MetricRelationship(
                source="flow_efficiency",
                target="throughput",
                relationship_type="direct",
                strength=0.7,
                formula="Higher efficiency → Higher throughput",
            ),
            # WIP relationships
            MetricRelationship(
                source="wip",
                target="flow_efficiency",
                relationship_type="inverse",
                strength=0.6,
                formula="Higher WIP → Lower efficiency (context switching)",
            ),
            MetricRelationship(
                source="wip",
                target="flow_predictability",
                relationship_type="inverse",
                strength=0.5,
                formula="Higher WIP → More variability",
            ),
            # Predictability relationships
            MetricRelationship(
                source="flow_predictability",
                target="cycle_time",
                relationship_type="complex",
                strength=0.6,
                formula="Higher predictability → More consistent cycle times",
            ),
            MetricRelationship(
                source="flow_predictability",
                target="throughput",
                relationship_type="direct",
                strength=0.5,
                formula="Higher predictability → Steadier throughput",
            ),
        ]

    def get_all_metrics(self) -> Dict[str, float]:
        """Return all current metric values"""
        return {
            "throughput": self.throughput,
            "cycle_time": self.cycle_time,
            "flow_efficiency": self.flow_efficiency,
            "wip": self.wip,
            "flow_predictability": self.flow_predictability,
        }

    def validate_littles_law(self) -> Dict[str, float]:
        """
        Check if current values satisfy Little's Law.
        WIP = Throughput × Cycle Time (with time unit conversion)

        Returns:
            Dictionary with calculated WIP and deviation
        """
        # Convert throughput from items/week to items/day
        throughput_per_day = self.throughput / 7.0

        # Calculate expected WIP according to Little's Law
        expected_wip = throughput_per_day * self.cycle_time

        # Calculate deviation
        deviation = self.wip - expected_wip
        deviation_pct = (deviation / expected_wip * 100) if expected_wip > 0 else 0

        return {
            "expected_wip": expected_wip,
            "actual_wip": self.wip,
            "deviation": deviation,
            "deviation_pct": deviation_pct,
            "is_balanced": abs(deviation_pct) < 10,  # Within 10% is considered balanced
        }

    def calculate_theoretical_throughput(self) -> float:
        """Calculate theoretical throughput based on WIP and cycle time"""
        if self.cycle_time > 0:
            # Throughput = WIP / Cycle Time (convert to weekly)
            return (self.wip / self.cycle_time) * 7.0
        return 0.0

    def calculate_theoretical_cycle_time(self) -> float:
        """Calculate theoretical cycle time based on WIP and throughput"""
        throughput_per_day = self.throughput / 7.0
        if throughput_per_day > 0:
            return self.wip / throughput_per_day
        return 0.0

    def calculate_active_time(self) -> float:
        """Calculate active work time based on flow efficiency"""
        return self.cycle_time * self.flow_efficiency

    def calculate_wait_time(self) -> float:
        """Calculate wait/blocked time"""
        return self.cycle_time * (1 - self.flow_efficiency)

    def get_metric_relationships(self) -> List[MetricRelationship]:
        """Return all defined metric relationships"""
        return self._relationships

    def get_relationship_matrix(self) -> np.ndarray:
        """
        Generate a correlation-style matrix showing relationships.
        Positive values = direct relationship
        Negative values = inverse relationship
        """
        metrics = [
            "throughput",
            "cycle_time",
            "flow_efficiency",
            "wip",
            "flow_predictability",
        ]
        n = len(metrics)
        matrix = np.eye(n)  # Start with identity matrix

        # Fill in relationships
        relationship_map = {
            ("throughput", "wip"): 0.8,
            ("cycle_time", "wip"): 0.9,
            ("wip", "cycle_time"): 0.9,
            ("flow_efficiency", "cycle_time"): -0.8,
            ("flow_efficiency", "throughput"): 0.7,
            ("wip", "flow_efficiency"): -0.6,
            ("wip", "flow_predictability"): -0.5,
            ("flow_predictability", "cycle_time"): -0.3,
            ("flow_predictability", "throughput"): 0.5,
            ("throughput", "cycle_time"): -0.4,
        }

        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if (m1, m2) in relationship_map:
                    matrix[i, j] = relationship_map[(m1, m2)]
                elif (m2, m1) in relationship_map:
                    matrix[i, j] = relationship_map[(m2, m1)]

        return matrix

    def simulate_metric_change(
        self, metric_name: str, change_pct: float
    ) -> Dict[str, float]:
        """
        Simulate the impact of changing one metric on others.

        Args:
            metric_name: Name of metric to change
            change_pct: Percentage change (positive or negative)

        Returns:
            Dictionary of projected new values for all metrics
        """
        current = self.get_all_metrics()
        projected = current.copy()

        # Apply the direct change
        multiplier = 1 + (change_pct / 100)

        if metric_name == "throughput":
            projected["throughput"] = current["throughput"] * multiplier
            # Higher throughput with same WIP → lower cycle time
            if current["wip"] > 0:
                projected["cycle_time"] = current["wip"] / (
                    projected["throughput"] / 7.0
                )
            # Slight improvement in efficiency due to momentum
            projected["flow_efficiency"] = min(
                1.0, current["flow_efficiency"] * (1 + change_pct * 0.002)
            )

        elif metric_name == "cycle_time":
            # For cycle time, improvement means REDUCTION
            projected["cycle_time"] = current["cycle_time"] / multiplier
            # Lower cycle time with same WIP → higher throughput
            projected["throughput"] = (current["wip"] / projected["cycle_time"]) * 7.0
            # Faster cycles often mean better efficiency
            projected["flow_efficiency"] = min(
                1.0, current["flow_efficiency"] * (1 + change_pct * 0.003)
            )

        elif metric_name == "flow_efficiency":
            projected["flow_efficiency"] = min(
                1.0, current["flow_efficiency"] * multiplier
            )
            # Higher efficiency → lower cycle time
            efficiency_ratio = projected["flow_efficiency"] / current["flow_efficiency"]
            projected["cycle_time"] = current["cycle_time"] / efficiency_ratio
            # Recalculate throughput via Little's Law
            projected["throughput"] = (current["wip"] / projected["cycle_time"]) * 7.0

        elif metric_name == "wip":
            # For WIP, improvement means REDUCTION
            projected["wip"] = current["wip"] / multiplier
            # Lower WIP improves efficiency (less context switching)
            projected["flow_efficiency"] = min(
                1.0, current["flow_efficiency"] * (1 + change_pct * 0.004)
            )
            # Maintain throughput, reduce cycle time
            projected["cycle_time"] = projected["wip"] / (current["throughput"] / 7.0)
            # Lower WIP improves predictability
            projected["flow_predictability"] = min(
                1.0, current["flow_predictability"] * (1 + change_pct * 0.003)
            )

        elif metric_name == "flow_predictability":
            projected["flow_predictability"] = min(
                1.0, current["flow_predictability"] * multiplier
            )
            # Higher predictability has indirect positive effects
            projected["flow_efficiency"] = min(
                1.0, current["flow_efficiency"] * (1 + change_pct * 0.001)
            )
            projected["throughput"] = current["throughput"] * (1 + change_pct * 0.002)

        return projected

    def identify_improvement_opportunities(
        self, benchmarks: Dict[str, float]
    ) -> List[Dict]:
        """
        Identify which metrics have the most room for improvement.

        Args:
            benchmarks: Target values for each metric

        Returns:
            List of improvement opportunities sorted by potential impact
        """
        current = self.get_all_metrics()
        opportunities = []

        # Calculate gaps for each metric
        metric_configs = {
            "throughput": {"higher_is_better": True, "financial_impact_factor": 500},
            "cycle_time": {"higher_is_better": False, "financial_impact_factor": 300},
            "flow_efficiency": {
                "higher_is_better": True,
                "financial_impact_factor": 400,
            },
            "wip": {"higher_is_better": False, "financial_impact_factor": 200},
            "flow_predictability": {
                "higher_is_better": True,
                "financial_impact_factor": 250,
            },
        }

        for metric, config in metric_configs.items():
            current_val = current[metric]
            benchmark_val = benchmarks[metric]

            if config["higher_is_better"]:
                gap = ((benchmark_val - current_val) / benchmark_val) * 100
                has_gap = current_val < benchmark_val
            else:
                gap = ((current_val - benchmark_val) / current_val) * 100
                has_gap = current_val > benchmark_val

            if has_gap:
                impact = (
                    "high" if abs(gap) > 30 else "medium" if abs(gap) > 15 else "low"
                )
                potential_savings = abs(gap) * config["financial_impact_factor"]

                opportunities.append(
                    {
                        "metric": metric.replace("_", " ").title(),
                        "current": current_val,
                        "benchmark": benchmark_val,
                        "gap": abs(gap),
                        "impact": impact,
                        "potential_savings": potential_savings,
                        "direction": (
                            "increase" if config["higher_is_better"] else "decrease"
                        ),
                    }
                )

        # Sort by potential impact
        opportunities.sort(key=lambda x: x["potential_savings"], reverse=True)

        return opportunities

    def calculate_weighted_score(
        self, weights: Dict[str, float], benchmarks: Dict[str, float]
    ) -> float:
        """
        Calculate a weighted performance score (0-100).

        Args:
            weights: Importance weight for each metric (should sum to 1)
            benchmarks: Target values for comparison

        Returns:
            Weighted score from 0-100
        """
        current = self.get_all_metrics()
        total_score = 0.0

        for metric, weight in weights.items():
            current_val = current[metric]
            benchmark_val = benchmarks[metric]

            # Normalize score (100 = at or above benchmark)
            if metric in ["cycle_time", "wip"]:
                # Lower is better
                if current_val <= benchmark_val:
                    score = 100
                else:
                    score = max(0, (benchmark_val / current_val) * 100)
            else:
                # Higher is better
                if current_val >= benchmark_val:
                    score = 100
                else:
                    score = max(0, (current_val / benchmark_val) * 100)

            total_score += score * weight

        return total_score

    def generate_scenario_data(
        self,
        metric_name: str,
        range_pct: Tuple[float, float] = (-50, 100),
        steps: int = 20,
    ) -> List[Dict]:
        """
        Generate data for scenario analysis across a range of changes.

        Args:
            metric_name: Metric to vary
            range_pct: (min_change, max_change) as percentages
            steps: Number of data points

        Returns:
            List of dictionaries with scenario results
        """
        scenarios = []
        change_values = np.linspace(range_pct[0], range_pct[1], steps)

        for change in change_values:
            projected = self.simulate_metric_change(metric_name, change)
            scenarios.append({"change_pct": change, **projected})

        return scenarios

    def get_sensitivity_data(
        self, improvement_pct: float = 10.0
    ) -> Dict[str, List[float]]:
        """
        Generate sensitivity analysis data for all metrics.
        Shows how improvement_pct% change in each metric affects others.
        """
        metrics = [
            "throughput",
            "cycle_time",
            "flow_efficiency",
            "wip",
            "flow_predictability",
        ]
        sensitivities = {}

        for metric in metrics:
            # Calculate improvement impact on all metrics
            projected = self.simulate_metric_change(metric, improvement_pct)
            current = self.get_all_metrics()

            changes = []
            for m in metrics:
                if current[m] > 0:
                    pct_change = ((projected[m] - current[m]) / current[m]) * 100
                else:
                    pct_change = 0
                changes.append(pct_change)

            sensitivities[metric] = changes

        return sensitivities
