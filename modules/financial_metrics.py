"""
Financial Metrics Engine
Calculates financial impacts based on flow metrics including:
- Revenue projections
- Cost analysis
- Cost of delay
- WIP carrying costs
- ROI calculations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FinancialImpact:
    """Represents the financial impact of a metric change"""

    metric: str
    current_cost: float
    projected_cost: float
    savings: float
    roi_pct: float


class FinancialMetricsEngine:
    """
    Engine for calculating financial impacts of flow metrics.
    Links operational performance to business outcomes.
    """

    def __init__(
        self,
        flow_engine,
        revenue_per_item: float,
        cost_per_item: float,
        hourly_labor_cost: float,
        delay_cost_per_day: float,
        wip_carrying_cost_pct: float,
        team_size: int,
    ):
        """
        Initialize financial metrics engine.

        Args:
            flow_engine: FlowMetricsEngine instance with current metrics
            revenue_per_item: Revenue generated per completed item
            cost_per_item: Direct cost per item
            hourly_labor_cost: Average hourly labor cost
            delay_cost_per_day: Cost of delay per item per day
            wip_carrying_cost_pct: Monthly carrying cost as percentage of item value
            team_size: Number of team members
        """
        self.flow = flow_engine
        self.revenue_per_item = revenue_per_item
        self.cost_per_item = cost_per_item
        self.hourly_labor_cost = hourly_labor_cost
        self.delay_cost_per_day = delay_cost_per_day
        self.wip_carrying_cost_pct = wip_carrying_cost_pct
        self.team_size = team_size

    def calculate_weekly_revenue(self) -> float:
        """Calculate weekly revenue based on throughput"""
        return self.flow.throughput * self.revenue_per_item

    def calculate_weekly_cost(self) -> float:
        """Calculate weekly direct costs"""
        direct_cost = self.flow.throughput * self.cost_per_item
        labor_cost = self.team_size * self.hourly_labor_cost * 40  # 40 hours/week
        return direct_cost + labor_cost

    def calculate_weekly_profit(self) -> float:
        """Calculate weekly profit"""
        return self.calculate_weekly_revenue() - self.calculate_weekly_cost()

    def calculate_profit_margin(self) -> float:
        """Calculate profit margin percentage"""
        revenue = self.calculate_weekly_revenue()
        if revenue > 0:
            return self.calculate_weekly_profit() / revenue
        return 0.0

    def calculate_cost_of_delay(self) -> Dict[str, float]:
        """
        Calculate cost of delay based on cycle time and WIP.

        Returns:
            Dictionary with various delay cost components
        """
        flow_metrics = self.flow.get_all_metrics()

        # Wait time per item (non-value-add time)
        wait_time_days = flow_metrics["cycle_time"] * (
            1 - flow_metrics["flow_efficiency"]
        )

        # Total delay cost for items in progress
        delay_per_item = wait_time_days * self.delay_cost_per_day
        total_wip_delay = delay_per_item * flow_metrics["wip"]

        # Opportunity cost - revenue delayed
        avg_delay_weeks = flow_metrics["cycle_time"] / 7.0
        opportunity_cost = (
            avg_delay_weeks * self.revenue_per_item * (flow_metrics["wip"] / 10)
        )

        return {
            "delay_per_item": delay_per_item,
            "total_wip_delay": total_wip_delay,
            "opportunity_cost": opportunity_cost,
            "total_delay_cost": total_wip_delay + opportunity_cost,
            "wait_time_days": wait_time_days,
        }

    def calculate_wip_carrying_cost(self) -> float:
        """
        Calculate the monthly cost of carrying WIP inventory.
        Based on the value of items in progress and carrying cost percentage.
        """
        flow_metrics = self.flow.get_all_metrics()

        # Value of WIP (using cost per item as proxy)
        wip_value = flow_metrics["wip"] * self.cost_per_item

        # Monthly carrying cost
        carrying_cost = wip_value * self.wip_carrying_cost_pct

        return carrying_cost

    def calculate_efficiency_impact(self) -> Dict[str, float]:
        """
        Calculate financial impact of current flow efficiency.
        """
        flow_metrics = self.flow.get_all_metrics()

        # Hours per week the team could save with better efficiency
        total_team_hours = self.team_size * 40
        wasted_hours = total_team_hours * (1 - flow_metrics["flow_efficiency"])

        # Cost of wasted time
        waste_cost = wasted_hours * self.hourly_labor_cost

        # Potential additional throughput with 100% efficiency
        potential_additional_items = self.flow.throughput * (
            1 / flow_metrics["flow_efficiency"] - 1
        )
        potential_revenue = potential_additional_items * self.revenue_per_item

        return {
            "wasted_hours_per_week": wasted_hours,
            "waste_cost_per_week": waste_cost,
            "potential_additional_items": potential_additional_items,
            "potential_additional_revenue": potential_revenue,
        }

    def calculate_predictability_impact(self) -> Dict[str, float]:
        """
        Calculate financial impact of flow predictability.
        Lower predictability = higher risk and buffer costs.
        """
        flow_metrics = self.flow.get_all_metrics()

        # Unpredictability factor (0 = perfectly predictable)
        unpredictability = 1 - flow_metrics["flow_predictability"]

        # Buffer inventory needed due to unpredictability
        safety_stock_items = flow_metrics["wip"] * unpredictability * 0.3
        safety_stock_cost = (
            safety_stock_items * self.cost_per_item * self.wip_carrying_cost_pct
        )

        # Risk premium (expediting, overtime, etc.)
        risk_premium = self.calculate_weekly_revenue() * unpredictability * 0.05

        # Planning overhead
        planning_overhead = (
            self.team_size * self.hourly_labor_cost * unpredictability * 5
        )

        return {
            "safety_stock_items": safety_stock_items,
            "safety_stock_cost": safety_stock_cost,
            "risk_premium": risk_premium,
            "planning_overhead": planning_overhead,
            "total_unpredictability_cost": safety_stock_cost
            + risk_premium
            + planning_overhead,
        }

    def get_all_metrics(self) -> Dict[str, float]:
        """Return all calculated financial metrics"""
        delay_costs = self.calculate_cost_of_delay()
        efficiency_impact = self.calculate_efficiency_impact()
        predictability_impact = self.calculate_predictability_impact()

        return {
            "weekly_revenue": self.calculate_weekly_revenue(),
            "weekly_cost": self.calculate_weekly_cost(),
            "weekly_profit": self.calculate_weekly_profit(),
            "profit_margin": self.calculate_profit_margin(),
            "total_delay_cost": delay_costs["total_delay_cost"],
            "delay_per_item": delay_costs["delay_per_item"],
            "wip_carrying_cost": self.calculate_wip_carrying_cost(),
            "waste_cost": efficiency_impact["waste_cost_per_week"],
            "unpredictability_cost": predictability_impact[
                "total_unpredictability_cost"
            ],
            "opportunity_cost": delay_costs["opportunity_cost"],
        }

    def calculate_improvement_roi(
        self, metric_name: str, improvement_pct: float, investment_cost: float = 0
    ) -> Dict[str, float]:
        """
        Calculate ROI of improving a specific metric.

        Args:
            metric_name: Name of the metric to improve
            improvement_pct: Percentage improvement
            investment_cost: Cost to achieve the improvement

        Returns:
            Dictionary with ROI calculations
        """
        # Get current financial state
        current_financials = self.get_all_metrics()

        # Simulate the improvement
        projected_flow = self.flow.simulate_metric_change(metric_name, improvement_pct)

        # Create temporary engine with projected values
        from modules.flow_metrics import FlowMetricsEngine

        projected_flow_engine = FlowMetricsEngine(
            throughput=projected_flow["throughput"],
            cycle_time=projected_flow["cycle_time"],
            flow_efficiency=projected_flow["flow_efficiency"],
            wip=projected_flow["wip"],
            flow_predictability=projected_flow["flow_predictability"],
        )

        projected_financial_engine = FinancialMetricsEngine(
            flow_engine=projected_flow_engine,
            revenue_per_item=self.revenue_per_item,
            cost_per_item=self.cost_per_item,
            hourly_labor_cost=self.hourly_labor_cost,
            delay_cost_per_day=self.delay_cost_per_day,
            wip_carrying_cost_pct=self.wip_carrying_cost_pct,
            team_size=self.team_size,
        )

        projected_financials = projected_financial_engine.get_all_metrics()

        # Calculate benefits
        weekly_profit_increase = (
            projected_financials["weekly_profit"] - current_financials["weekly_profit"]
        )
        annual_profit_increase = weekly_profit_increase * 52

        delay_cost_reduction = (
            current_financials["total_delay_cost"]
            - projected_financials["total_delay_cost"]
        )
        annual_delay_savings = delay_cost_reduction * 52

        total_annual_benefit = annual_profit_increase + annual_delay_savings

        # Calculate ROI
        if investment_cost > 0:
            roi = ((total_annual_benefit - investment_cost) / investment_cost) * 100
            payback_weeks = (
                investment_cost / weekly_profit_increase
                if weekly_profit_increase > 0
                else float("inf")
            )
        else:
            roi = float("inf") if total_annual_benefit > 0 else 0
            payback_weeks = 0

        return {
            "current_weekly_profit": current_financials["weekly_profit"],
            "projected_weekly_profit": projected_financials["weekly_profit"],
            "weekly_profit_increase": weekly_profit_increase,
            "annual_profit_increase": annual_profit_increase,
            "delay_cost_reduction": delay_cost_reduction,
            "annual_delay_savings": annual_delay_savings,
            "total_annual_benefit": total_annual_benefit,
            "investment_cost": investment_cost,
            "roi_pct": roi,
            "payback_weeks": payback_weeks,
        }

    def generate_financial_breakdown(self) -> List[Dict]:
        """
        Generate a detailed financial breakdown for visualization.
        """
        metrics = self.get_all_metrics()

        breakdown = [
            {
                "category": "Revenue",
                "subcategory": "Weekly Revenue",
                "value": metrics["weekly_revenue"],
                "type": "income",
            },
            {
                "category": "Costs",
                "subcategory": "Direct Costs",
                "value": -self.flow.throughput * self.cost_per_item,
                "type": "cost",
            },
            {
                "category": "Costs",
                "subcategory": "Labor Costs",
                "value": -self.team_size * self.hourly_labor_cost * 40,
                "type": "cost",
            },
            {
                "category": "Hidden Costs",
                "subcategory": "Cost of Delay",
                "value": -metrics["total_delay_cost"],
                "type": "hidden",
            },
            {
                "category": "Hidden Costs",
                "subcategory": "WIP Carrying Cost",
                "value": -metrics["wip_carrying_cost"] / 4,
                "type": "hidden",
            },
            {
                "category": "Hidden Costs",
                "subcategory": "Inefficiency Cost",
                "value": -metrics["waste_cost"],
                "type": "hidden",
            },
            {
                "category": "Hidden Costs",
                "subcategory": "Unpredictability Cost",
                "value": -metrics["unpredictability_cost"],
                "type": "hidden",
            },
        ]

        return breakdown

    def calculate_metric_financial_sensitivity(
        self, improvement_pct: float = 10.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate how sensitive financial outcomes are to each metric.
        Shows dollar impact of improvement_pct% improvement in each metric.
        """
        metrics = [
            "throughput",
            "cycle_time",
            "flow_efficiency",
            "wip",
            "flow_predictability",
        ]
        sensitivities = {}

        current = self.get_all_metrics()

        for metric in metrics:
            roi_data = self.calculate_improvement_roi(metric, improvement_pct, 0)

            sensitivities[metric] = {
                "weekly_profit_change": roi_data["weekly_profit_increase"],
                "annual_profit_change": roi_data["annual_profit_increase"],
                "delay_cost_change": roi_data["delay_cost_reduction"],
                "total_annual_impact": roi_data["total_annual_benefit"],
            }

        return sensitivities

    def project_financials_over_time(
        self, weeks: int = 52, improvement_rate_per_week: float = 0.5
    ) -> List[Dict]:
        """
        Project financial metrics over time with gradual improvement.

        Args:
            weeks: Number of weeks to project
            improvement_rate_per_week: Percentage improvement per week

        Returns:
            List of weekly projections
        """
        projections = []
        current_metrics = self.flow.get_all_metrics()

        cumulative_profit = 0

        for week in range(weeks):
            # Calculate improvement factor
            improvement_factor = 1 + (improvement_rate_per_week / 100) * week

            # Project flow metrics
            projected_throughput = current_metrics["throughput"] * min(
                improvement_factor, 2.0
            )
            projected_cycle_time = current_metrics["cycle_time"] / min(
                improvement_factor, 2.0
            )
            projected_efficiency = min(
                1.0, current_metrics["flow_efficiency"] * improvement_factor
            )

            # Calculate financials
            weekly_revenue = projected_throughput * self.revenue_per_item
            weekly_cost = projected_throughput * self.cost_per_item + (
                self.team_size * self.hourly_labor_cost * 40
            )
            weekly_profit = weekly_revenue - weekly_cost
            cumulative_profit += weekly_profit

            projections.append(
                {
                    "week": week + 1,
                    "throughput": projected_throughput,
                    "cycle_time": projected_cycle_time,
                    "flow_efficiency": projected_efficiency,
                    "weekly_revenue": weekly_revenue,
                    "weekly_cost": weekly_cost,
                    "weekly_profit": weekly_profit,
                    "cumulative_profit": cumulative_profit,
                }
            )

        return projections

    def calculate_break_even_analysis(self) -> Dict[str, float]:
        """
        Calculate break-even point and related metrics.
        """
        # Fixed costs (labor)
        fixed_costs = self.team_size * self.hourly_labor_cost * 40

        # Variable cost per item
        variable_cost = self.cost_per_item

        # Contribution margin per item
        contribution_margin = self.revenue_per_item - variable_cost

        # Break-even point in items
        if contribution_margin > 0:
            break_even_items = fixed_costs / contribution_margin
        else:
            break_even_items = float("inf")

        # Current position relative to break-even
        current_throughput = self.flow.throughput
        margin_of_safety = current_throughput - break_even_items
        margin_of_safety_pct = (
            (margin_of_safety / current_throughput * 100)
            if current_throughput > 0
            else 0
        )

        return {
            "fixed_costs_weekly": fixed_costs,
            "variable_cost_per_item": variable_cost,
            "contribution_margin": contribution_margin,
            "break_even_items_per_week": break_even_items,
            "current_throughput": current_throughput,
            "margin_of_safety_items": margin_of_safety,
            "margin_of_safety_pct": margin_of_safety_pct,
        }
