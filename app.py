"""
Flow Metrics Dashboard - Main Application
Interactive visualization of flow metrics and their financial impact
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from modules.flow_metrics import FlowMetricsEngine
from modules.financial_metrics import FinancialMetricsEngine
from modules.visualizations import VisualizationEngine
from modules.cfo_roi_tab import render_cfo_roi_tab
from utils.helpers import format_currency, format_percentage, load_excel_data

# Page configuration
st.set_page_config(
    page_title="Flow Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2C3E50;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #1E1E1E;
        border-left: 4px solid #3498DB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
        color: #FFFFFF;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #D4EDDA;
        border-left: 4px solid #28A745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables for persistence"""
    defaults = {
        # Flow Metric Parameters
        "throughput": 40.0,
        "cycle_time": 14.0,
        "flow_efficiency": 0.15,
        "wip": 100.0,
        "flow_predictability": 0.75,
        # Financial Parameters
        "revenue_per_item": 10000.0,
        "cost_per_item": 6000.0,
        "hourly_labor_cost": 75.0,
        "delay_cost_per_day": 500.0,
        "wip_carrying_cost_pct": 0.02,
        "team_size": 10,
        # Subjective Weights (user-adjustable importance)
        "weight_throughput": 0.20,
        "weight_cycle_time": 0.20,
        "weight_flow_efficiency": 0.20,
        "weight_wip": 0.20,
        "weight_predictability": 0.20,
        # Industry benchmarks
        "benchmark_throughput": 40.0,
        "benchmark_cycle_time": 14.0,
        "benchmark_flow_efficiency": 0.15,
        "benchmark_wip": 100.0,
        "benchmark_predictability": 0.75,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_saveable_state():
    """Get all saveable session state values"""
    keys = [
        "throughput",
        "cycle_time",
        "flow_efficiency",
        "wip",
        "flow_predictability",
        "revenue_per_item",
        "cost_per_item",
        "hourly_labor_cost",
        "delay_cost_per_day",
        "wip_carrying_cost_pct",
        "team_size",
        "weight_throughput",
        "weight_cycle_time",
        "weight_flow_efficiency",
        "weight_wip",
        "weight_predictability",
        "benchmark_throughput",
        "benchmark_cycle_time",
        "benchmark_flow_efficiency",
        "benchmark_wip",
        "benchmark_predictability",
    ]
    return {k: st.session_state[k] for k in keys}


def load_state_from_dict(data):
    """Load state from dictionary"""
    for key, value in data.items():
        if key in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Render the sidebar with input controls"""
    st.sidebar.markdown("## ⚙️ Configuration Panel")

    # Tabs for different parameter categories
    tab1, tab2, tab3, tab4 = st.sidebar.tabs(
        ["📈 Flow", "💰 Financial", "⚖️ Weights", "🎯 Benchmarks"]
    )

    with tab1:
        st.markdown("### Flow Metric Inputs")

        # Initialize widget keys if not present
        if "throughput_slider_key" not in st.session_state:
            st.session_state.throughput_slider_key = st.session_state.throughput
        if "throughput_input_key" not in st.session_state:
            st.session_state.throughput_input_key = st.session_state.throughput
        if "cycle_time_slider_key" not in st.session_state:
            st.session_state.cycle_time_slider_key = st.session_state.cycle_time
        if "cycle_time_input_key" not in st.session_state:
            st.session_state.cycle_time_input_key = st.session_state.cycle_time
        if "flow_eff_slider_key" not in st.session_state:
            st.session_state.flow_eff_slider_key = int(
                st.session_state.flow_efficiency * 100
            )
        if "flow_eff_input_key" not in st.session_state:
            st.session_state.flow_eff_input_key = int(
                st.session_state.flow_efficiency * 100
            )
        if "wip_slider_key" not in st.session_state:
            st.session_state.wip_slider_key = st.session_state.wip
        if "wip_input_key" not in st.session_state:
            st.session_state.wip_input_key = st.session_state.wip
        if "pred_slider_key" not in st.session_state:
            st.session_state.pred_slider_key = st.session_state.flow_predictability
        if "pred_input_key" not in st.session_state:
            st.session_state.pred_input_key = st.session_state.flow_predictability

        # Throughput
        def sync_throughput_slider():
            st.session_state.throughput = st.session_state.throughput_slider_key
            st.session_state.throughput_input_key = (
                st.session_state.throughput_slider_key
            )

        def sync_throughput_input():
            st.session_state.throughput = st.session_state.throughput_input_key
            st.session_state.throughput_slider_key = (
                st.session_state.throughput_input_key
            )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Throughput (items/week)",
                min_value=1.0,
                max_value=200.0,
                step=1.0,
                help="Number of items completed per week",
                key="throughput_slider_key",
                on_change=sync_throughput_slider,
            )
        with col2:
            st.number_input(
                "Value",
                min_value=1.0,
                max_value=200.0,
                step=1.0,
                key="throughput_input_key",
                on_change=sync_throughput_input,
                label_visibility="collapsed",
            )

        # Cycle Time
        def sync_cycle_time_slider():
            st.session_state.cycle_time = st.session_state.cycle_time_slider_key
            st.session_state.cycle_time_input_key = (
                st.session_state.cycle_time_slider_key
            )

        def sync_cycle_time_input():
            st.session_state.cycle_time = st.session_state.cycle_time_input_key
            st.session_state.cycle_time_slider_key = (
                st.session_state.cycle_time_input_key
            )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Cycle Time (days)",
                min_value=1.0,
                max_value=90.0,
                step=0.5,
                help="Average time from start to completion",
                key="cycle_time_slider_key",
                on_change=sync_cycle_time_slider,
            )
        with col2:
            st.number_input(
                "Value",
                min_value=1.0,
                max_value=90.0,
                step=0.5,
                key="cycle_time_input_key",
                on_change=sync_cycle_time_input,
                label_visibility="collapsed",
            )

        # Flow Efficiency
        def sync_flow_eff_slider():
            st.session_state.flow_efficiency = (
                st.session_state.flow_eff_slider_key / 100.0
            )
            st.session_state.flow_eff_input_key = st.session_state.flow_eff_slider_key

        def sync_flow_eff_input():
            st.session_state.flow_efficiency = (
                st.session_state.flow_eff_input_key / 100.0
            )
            st.session_state.flow_eff_slider_key = st.session_state.flow_eff_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Flow Efficiency (%)",
                min_value=1,
                max_value=100,
                step=1,
                help="Percentage of time actively working vs waiting",
                key="flow_eff_slider_key",
                on_change=sync_flow_eff_slider,
            )
        with col2:
            st.number_input(
                "Value",
                min_value=1,
                max_value=100,
                step=1,
                key="flow_eff_input_key",
                on_change=sync_flow_eff_input,
                label_visibility="collapsed",
            )

        # WIP
        def sync_wip_slider():
            st.session_state.wip = st.session_state.wip_slider_key
            st.session_state.wip_input_key = st.session_state.wip_slider_key

        def sync_wip_input():
            st.session_state.wip = st.session_state.wip_input_key
            st.session_state.wip_slider_key = st.session_state.wip_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Work in Progress (items)",
                min_value=1.0,
                max_value=500.0,
                step=1.0,
                help="Current number of items in progress",
                key="wip_slider_key",
                on_change=sync_wip_slider,
            )
        with col2:
            st.number_input(
                "Value",
                min_value=1.0,
                max_value=500.0,
                step=1.0,
                key="wip_input_key",
                on_change=sync_wip_input,
                label_visibility="collapsed",
            )

        # Flow Predictability
        def sync_pred_slider():
            st.session_state.flow_predictability = st.session_state.pred_slider_key
            st.session_state.pred_input_key = st.session_state.pred_slider_key

        def sync_pred_input():
            st.session_state.flow_predictability = st.session_state.pred_input_key
            st.session_state.pred_slider_key = st.session_state.pred_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Flow Predictability",
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                help="1.0 = highly predictable, 0.1 = highly variable",
                key="pred_slider_key",
                on_change=sync_pred_slider,
            )
        with col2:
            st.number_input(
                "Value",
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                key="pred_input_key",
                on_change=sync_pred_input,
                label_visibility="collapsed",
            )

    with tab2:
        st.markdown("### Financial Parameters")

        # Initialize financial widget keys
        if "rev_slider_key" not in st.session_state:
            st.session_state.rev_slider_key = st.session_state.revenue_per_item
        if "rev_input_key" not in st.session_state:
            st.session_state.rev_input_key = st.session_state.revenue_per_item
        if "cost_slider_key" not in st.session_state:
            st.session_state.cost_slider_key = st.session_state.cost_per_item
        if "cost_input_key" not in st.session_state:
            st.session_state.cost_input_key = st.session_state.cost_per_item
        if "labor_slider_key" not in st.session_state:
            st.session_state.labor_slider_key = st.session_state.hourly_labor_cost
        if "labor_input_key" not in st.session_state:
            st.session_state.labor_input_key = st.session_state.hourly_labor_cost
        if "delay_slider_key" not in st.session_state:
            st.session_state.delay_slider_key = st.session_state.delay_cost_per_day
        if "delay_input_key" not in st.session_state:
            st.session_state.delay_input_key = st.session_state.delay_cost_per_day
        if "carry_slider_key" not in st.session_state:
            st.session_state.carry_slider_key = (
                st.session_state.wip_carrying_cost_pct * 100
            )
        if "carry_input_key" not in st.session_state:
            st.session_state.carry_input_key = (
                st.session_state.wip_carrying_cost_pct * 100
            )
        if "team_slider_key" not in st.session_state:
            st.session_state.team_slider_key = st.session_state.team_size
        if "team_input_key" not in st.session_state:
            st.session_state.team_input_key = st.session_state.team_size

        # Revenue per Item
        def sync_rev_slider():
            st.session_state.revenue_per_item = st.session_state.rev_slider_key
            st.session_state.rev_input_key = st.session_state.rev_slider_key

        def sync_rev_input():
            st.session_state.revenue_per_item = st.session_state.rev_input_key
            st.session_state.rev_slider_key = st.session_state.rev_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Revenue per Item ($)",
                min_value=100.0,
                max_value=100000.0,
                step=500.0,
                help="Revenue generated per completed item",
                key="rev_slider_key",
                on_change=sync_rev_slider,
            )
        with col2:
            st.number_input(
                "Revenue Value",
                min_value=0.0,
                max_value=1000000.0,
                step=500.0,
                key="rev_input_key",
                on_change=sync_rev_input,
                label_visibility="collapsed",
            )

        # Cost per Item
        def sync_cost_slider():
            st.session_state.cost_per_item = st.session_state.cost_slider_key
            st.session_state.cost_input_key = st.session_state.cost_slider_key

        def sync_cost_input():
            st.session_state.cost_per_item = st.session_state.cost_input_key
            st.session_state.cost_slider_key = st.session_state.cost_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Cost per Item ($)",
                min_value=100.0,
                max_value=100000.0,
                step=500.0,
                help="Direct cost per item",
                key="cost_slider_key",
                on_change=sync_cost_slider,
            )
        with col2:
            st.number_input(
                "Cost Value",
                min_value=0.0,
                max_value=1000000.0,
                step=500.0,
                key="cost_input_key",
                on_change=sync_cost_input,
                label_visibility="collapsed",
            )

        # Hourly Labor Cost
        def sync_labor_slider():
            st.session_state.hourly_labor_cost = st.session_state.labor_slider_key
            st.session_state.labor_input_key = st.session_state.labor_slider_key

        def sync_labor_input():
            st.session_state.hourly_labor_cost = st.session_state.labor_input_key
            st.session_state.labor_slider_key = st.session_state.labor_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Hourly Labor Cost ($)",
                min_value=10.0,
                max_value=500.0,
                step=5.0,
                help="Average hourly labor cost",
                key="labor_slider_key",
                on_change=sync_labor_slider,
            )
        with col2:
            st.number_input(
                "Labor Value",
                min_value=0.0,
                max_value=1000.0,
                step=5.0,
                key="labor_input_key",
                on_change=sync_labor_input,
                label_visibility="collapsed",
            )

        # Delay Cost per Day
        def sync_delay_slider():
            st.session_state.delay_cost_per_day = st.session_state.delay_slider_key
            st.session_state.delay_input_key = st.session_state.delay_slider_key

        def sync_delay_input():
            st.session_state.delay_cost_per_day = st.session_state.delay_input_key
            st.session_state.delay_slider_key = st.session_state.delay_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Cost of Delay ($/day)",
                min_value=10.0,
                max_value=10000.0,
                step=50.0,
                help="Cost of delay per item per day",
                key="delay_slider_key",
                on_change=sync_delay_slider,
            )
        with col2:
            st.number_input(
                "Delay Value",
                min_value=0.0,
                max_value=50000.0,
                step=50.0,
                key="delay_input_key",
                on_change=sync_delay_input,
                label_visibility="collapsed",
            )

        # WIP Carrying Cost
        def sync_carry_slider():
            st.session_state.wip_carrying_cost_pct = (
                st.session_state.carry_slider_key / 100.0
            )
            st.session_state.carry_input_key = st.session_state.carry_slider_key

        def sync_carry_input():
            st.session_state.wip_carrying_cost_pct = (
                st.session_state.carry_input_key / 100.0
            )
            st.session_state.carry_slider_key = st.session_state.carry_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "WIP Carrying Cost (%/month)",
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                help="Monthly carrying cost as percentage of item value",
                key="carry_slider_key",
                on_change=sync_carry_slider,
            )
        with col2:
            st.number_input(
                "Carrying Cost Value",
                min_value=0.0,
                max_value=20.0,
                step=0.5,
                key="carry_input_key",
                on_change=sync_carry_input,
                label_visibility="collapsed",
            )

        # Team Size
        def sync_team_slider():
            st.session_state.team_size = st.session_state.team_slider_key
            st.session_state.team_input_key = st.session_state.team_slider_key

        def sync_team_input():
            st.session_state.team_size = st.session_state.team_input_key
            st.session_state.team_slider_key = st.session_state.team_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Team Size",
                min_value=1,
                max_value=100,
                step=1,
                help="Number of team members",
                key="team_slider_key",
                on_change=sync_team_slider,
            )
        with col2:
            st.number_input(
                "Team Size Value",
                min_value=1,
                max_value=500,
                step=1,
                key="team_input_key",
                on_change=sync_team_input,
                label_visibility="collapsed",
            )

    with tab3:
        st.markdown("### Importance Weights")
        st.markdown("*Adjust how much each metric matters to your organization*")

        # Initialize widget keys if not present
        if "wt_slider_key" not in st.session_state:
            st.session_state.wt_slider_key = st.session_state.weight_throughput
        if "wt_input_key" not in st.session_state:
            st.session_state.wt_input_key = st.session_state.weight_throughput
        if "wc_slider_key" not in st.session_state:
            st.session_state.wc_slider_key = st.session_state.weight_cycle_time
        if "wc_input_key" not in st.session_state:
            st.session_state.wc_input_key = st.session_state.weight_cycle_time
        if "wf_slider_key" not in st.session_state:
            st.session_state.wf_slider_key = st.session_state.weight_flow_efficiency
        if "wf_input_key" not in st.session_state:
            st.session_state.wf_input_key = st.session_state.weight_flow_efficiency
        if "ww_slider_key" not in st.session_state:
            st.session_state.ww_slider_key = st.session_state.weight_wip
        if "ww_input_key" not in st.session_state:
            st.session_state.ww_input_key = st.session_state.weight_wip
        if "wp_slider_key" not in st.session_state:
            st.session_state.wp_slider_key = st.session_state.weight_predictability
        if "wp_input_key" not in st.session_state:
            st.session_state.wp_input_key = st.session_state.weight_predictability

        # Throughput Weight
        def sync_wt_slider():
            st.session_state.weight_throughput = st.session_state.wt_slider_key
            st.session_state.wt_input_key = st.session_state.wt_slider_key

        def sync_wt_input():
            st.session_state.weight_throughput = st.session_state.wt_input_key
            st.session_state.wt_slider_key = st.session_state.wt_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Throughput Importance",
                0.0,
                1.0,
                step=0.05,
                key="wt_slider_key",
                on_change=sync_wt_slider,
            )
        with col2:
            st.number_input(
                "Throughput Weight",
                0.0,
                1.0,
                step=0.05,
                key="wt_input_key",
                on_change=sync_wt_input,
                label_visibility="collapsed",
            )

        # Cycle Time Weight
        def sync_wc_slider():
            st.session_state.weight_cycle_time = st.session_state.wc_slider_key
            st.session_state.wc_input_key = st.session_state.wc_slider_key

        def sync_wc_input():
            st.session_state.weight_cycle_time = st.session_state.wc_input_key
            st.session_state.wc_slider_key = st.session_state.wc_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Cycle Time Importance",
                0.0,
                1.0,
                step=0.05,
                key="wc_slider_key",
                on_change=sync_wc_slider,
            )
        with col2:
            st.number_input(
                "Cycle Time Weight",
                0.0,
                1.0,
                step=0.05,
                key="wc_input_key",
                on_change=sync_wc_input,
                label_visibility="collapsed",
            )

        # Flow Efficiency Weight
        def sync_wf_slider():
            st.session_state.weight_flow_efficiency = st.session_state.wf_slider_key
            st.session_state.wf_input_key = st.session_state.wf_slider_key

        def sync_wf_input():
            st.session_state.weight_flow_efficiency = st.session_state.wf_input_key
            st.session_state.wf_slider_key = st.session_state.wf_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Flow Efficiency Importance",
                0.0,
                1.0,
                step=0.05,
                key="wf_slider_key",
                on_change=sync_wf_slider,
            )
        with col2:
            st.number_input(
                "Flow Efficiency Weight",
                0.0,
                1.0,
                step=0.05,
                key="wf_input_key",
                on_change=sync_wf_input,
                label_visibility="collapsed",
            )

        # WIP Weight
        def sync_ww_slider():
            st.session_state.weight_wip = st.session_state.ww_slider_key
            st.session_state.ww_input_key = st.session_state.ww_slider_key

        def sync_ww_input():
            st.session_state.weight_wip = st.session_state.ww_input_key
            st.session_state.ww_slider_key = st.session_state.ww_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "WIP Importance",
                0.0,
                1.0,
                step=0.05,
                key="ww_slider_key",
                on_change=sync_ww_slider,
            )
        with col2:
            st.number_input(
                "WIP Weight",
                0.0,
                1.0,
                step=0.05,
                key="ww_input_key",
                on_change=sync_ww_input,
                label_visibility="collapsed",
            )

        # Predictability Weight
        def sync_wp_slider():
            st.session_state.weight_predictability = st.session_state.wp_slider_key
            st.session_state.wp_input_key = st.session_state.wp_slider_key

        def sync_wp_input():
            st.session_state.weight_predictability = st.session_state.wp_input_key
            st.session_state.wp_slider_key = st.session_state.wp_input_key

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Predictability Importance",
                0.0,
                1.0,
                step=0.05,
                key="wp_slider_key",
                on_change=sync_wp_slider,
            )
        with col2:
            st.number_input(
                "Predictability Weight",
                0.0,
                1.0,
                step=0.05,
                key="wp_input_key",
                on_change=sync_wp_input,
                label_visibility="collapsed",
            )

        total_weight = (
            st.session_state.weight_throughput
            + st.session_state.weight_cycle_time
            + st.session_state.weight_flow_efficiency
            + st.session_state.weight_wip
            + st.session_state.weight_predictability
        )
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Total Weight: {total_weight:.2f} (should equal 1.0)")
        else:
            st.success(f"Total Weight: {total_weight:.2f} ✓")

    with tab4:
        st.markdown("### Industry Benchmarks")
        st.markdown("*Set comparison targets*")

        # Initialize sidebar benchmark widget keys
        if "sb_bench_tp_slider_key" not in st.session_state:
            st.session_state.sb_bench_tp_slider_key = (
                st.session_state.benchmark_throughput
            )
        if "sb_bench_tp_input_key" not in st.session_state:
            st.session_state.sb_bench_tp_input_key = (
                st.session_state.benchmark_throughput
            )
        if "sb_bench_ct_slider_key" not in st.session_state:
            st.session_state.sb_bench_ct_slider_key = (
                st.session_state.benchmark_cycle_time
            )
        if "sb_bench_ct_input_key" not in st.session_state:
            st.session_state.sb_bench_ct_input_key = (
                st.session_state.benchmark_cycle_time
            )
        if "sb_bench_fe_slider_key" not in st.session_state:
            st.session_state.sb_bench_fe_slider_key = int(
                st.session_state.benchmark_flow_efficiency * 100
            )
        if "sb_bench_fe_input_key" not in st.session_state:
            st.session_state.sb_bench_fe_input_key = int(
                st.session_state.benchmark_flow_efficiency * 100
            )
        if "sb_bench_wip_slider_key" not in st.session_state:
            st.session_state.sb_bench_wip_slider_key = st.session_state.benchmark_wip
        if "sb_bench_wip_input_key" not in st.session_state:
            st.session_state.sb_bench_wip_input_key = st.session_state.benchmark_wip
        if "sb_bench_pred_slider_key" not in st.session_state:
            st.session_state.sb_bench_pred_slider_key = (
                st.session_state.benchmark_predictability
            )
        if "sb_bench_pred_input_key" not in st.session_state:
            st.session_state.sb_bench_pred_input_key = (
                st.session_state.benchmark_predictability
            )

        # Benchmark Throughput
        def sync_sb_bench_tp_slider():
            st.session_state.benchmark_throughput = (
                st.session_state.sb_bench_tp_slider_key
            )
            st.session_state.sb_bench_tp_input_key = (
                st.session_state.sb_bench_tp_slider_key
            )
            # Sync with main tab6 keys if they exist
            if "bench_tp_slider_key" in st.session_state:
                del st.session_state["bench_tp_slider_key"]
            if "bench_tp_input_key" in st.session_state:
                del st.session_state["bench_tp_input_key"]

        def sync_sb_bench_tp_input():
            st.session_state.benchmark_throughput = (
                st.session_state.sb_bench_tp_input_key
            )
            st.session_state.sb_bench_tp_slider_key = (
                st.session_state.sb_bench_tp_input_key
            )
            if "bench_tp_slider_key" in st.session_state:
                del st.session_state["bench_tp_slider_key"]
            if "bench_tp_input_key" in st.session_state:
                del st.session_state["bench_tp_input_key"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Benchmark Throughput",
                min_value=1.0,
                max_value=200.0,
                step=5.0,
                help="Target throughput to compare against",
                key="sb_bench_tp_slider_key",
                on_change=sync_sb_bench_tp_slider,
            )
        with col2:
            st.number_input(
                "Throughput Value",
                min_value=1.0,
                max_value=500.0,
                step=5.0,
                key="sb_bench_tp_input_key",
                on_change=sync_sb_bench_tp_input,
                label_visibility="collapsed",
            )

        # Benchmark Cycle Time
        def sync_sb_bench_ct_slider():
            st.session_state.benchmark_cycle_time = (
                st.session_state.sb_bench_ct_slider_key
            )
            st.session_state.sb_bench_ct_input_key = (
                st.session_state.sb_bench_ct_slider_key
            )
            if "bench_ct_slider_key" in st.session_state:
                del st.session_state["bench_ct_slider_key"]
            if "bench_ct_input_key" in st.session_state:
                del st.session_state["bench_ct_input_key"]

        def sync_sb_bench_ct_input():
            st.session_state.benchmark_cycle_time = (
                st.session_state.sb_bench_ct_input_key
            )
            st.session_state.sb_bench_ct_slider_key = (
                st.session_state.sb_bench_ct_input_key
            )
            if "bench_ct_slider_key" in st.session_state:
                del st.session_state["bench_ct_slider_key"]
            if "bench_ct_input_key" in st.session_state:
                del st.session_state["bench_ct_input_key"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Benchmark Cycle Time (days)",
                min_value=1.0,
                max_value=90.0,
                step=1.0,
                help="Target cycle time (lower is better)",
                key="sb_bench_ct_slider_key",
                on_change=sync_sb_bench_ct_slider,
            )
        with col2:
            st.number_input(
                "Cycle Time Value",
                min_value=1.0,
                max_value=180.0,
                step=1.0,
                key="sb_bench_ct_input_key",
                on_change=sync_sb_bench_ct_input,
                label_visibility="collapsed",
            )

        # Benchmark Flow Efficiency
        def sync_sb_bench_fe_slider():
            st.session_state.benchmark_flow_efficiency = (
                st.session_state.sb_bench_fe_slider_key / 100.0
            )
            st.session_state.sb_bench_fe_input_key = (
                st.session_state.sb_bench_fe_slider_key
            )
            if "bench_fe_slider_key" in st.session_state:
                del st.session_state["bench_fe_slider_key"]
            if "bench_fe_input_key" in st.session_state:
                del st.session_state["bench_fe_input_key"]

        def sync_sb_bench_fe_input():
            st.session_state.benchmark_flow_efficiency = (
                st.session_state.sb_bench_fe_input_key / 100.0
            )
            st.session_state.sb_bench_fe_slider_key = (
                st.session_state.sb_bench_fe_input_key
            )
            if "bench_fe_slider_key" in st.session_state:
                del st.session_state["bench_fe_slider_key"]
            if "bench_fe_input_key" in st.session_state:
                del st.session_state["bench_fe_input_key"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Benchmark Flow Efficiency (%)",
                min_value=1,
                max_value=100,
                step=5,
                help="Target flow efficiency percentage",
                key="sb_bench_fe_slider_key",
                on_change=sync_sb_bench_fe_slider,
            )
        with col2:
            st.number_input(
                "Flow Efficiency Value",
                min_value=1,
                max_value=100,
                step=1,
                key="sb_bench_fe_input_key",
                on_change=sync_sb_bench_fe_input,
                label_visibility="collapsed",
            )

        # Benchmark WIP
        def sync_sb_bench_wip_slider():
            st.session_state.benchmark_wip = st.session_state.sb_bench_wip_slider_key
            st.session_state.sb_bench_wip_input_key = (
                st.session_state.sb_bench_wip_slider_key
            )
            if "bench_wip_slider_key" in st.session_state:
                del st.session_state["bench_wip_slider_key"]
            if "bench_wip_input_key" in st.session_state:
                del st.session_state["bench_wip_input_key"]

        def sync_sb_bench_wip_input():
            st.session_state.benchmark_wip = st.session_state.sb_bench_wip_input_key
            st.session_state.sb_bench_wip_slider_key = (
                st.session_state.sb_bench_wip_input_key
            )
            if "bench_wip_slider_key" in st.session_state:
                del st.session_state["bench_wip_slider_key"]
            if "bench_wip_input_key" in st.session_state:
                del st.session_state["bench_wip_input_key"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Benchmark WIP",
                min_value=1.0,
                max_value=500.0,
                step=5.0,
                help="Target WIP level (lower is typically better)",
                key="sb_bench_wip_slider_key",
                on_change=sync_sb_bench_wip_slider,
            )
        with col2:
            st.number_input(
                "WIP Value",
                min_value=1.0,
                max_value=1000.0,
                step=5.0,
                key="sb_bench_wip_input_key",
                on_change=sync_sb_bench_wip_input,
                label_visibility="collapsed",
            )

        # Benchmark Predictability
        def sync_sb_bench_pred_slider():
            st.session_state.benchmark_predictability = (
                st.session_state.sb_bench_pred_slider_key
            )
            st.session_state.sb_bench_pred_input_key = (
                st.session_state.sb_bench_pred_slider_key
            )
            if "bench_pred_slider_key" in st.session_state:
                del st.session_state["bench_pred_slider_key"]
            if "bench_pred_input_key" in st.session_state:
                del st.session_state["bench_pred_input_key"]

        def sync_sb_bench_pred_input():
            st.session_state.benchmark_predictability = (
                st.session_state.sb_bench_pred_input_key
            )
            st.session_state.sb_bench_pred_slider_key = (
                st.session_state.sb_bench_pred_input_key
            )
            if "bench_pred_slider_key" in st.session_state:
                del st.session_state["bench_pred_slider_key"]
            if "bench_pred_input_key" in st.session_state:
                del st.session_state["bench_pred_input_key"]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.slider(
                "Benchmark Predictability",
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                help="Target predictability (higher is better)",
                key="sb_bench_pred_slider_key",
                on_change=sync_sb_bench_pred_slider,
            )
        with col2:
            st.number_input(
                "Predictability Value",
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                key="sb_bench_pred_input_key",
                on_change=sync_sb_bench_pred_input,
                label_visibility="collapsed",
            )


def render_metric_cards(flow_engine, financial_engine):
    """Render the top metric summary cards"""
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = flow_engine.get_all_metrics()

    with col1:
        delta = metrics["throughput"] - st.session_state.benchmark_throughput
        st.metric(
            label="📦 Throughput",
            value=f"{metrics['throughput']:.1f}/wk",
            delta=f"{delta:+.1f} vs benchmark",
        )

    with col2:
        delta = st.session_state.benchmark_cycle_time - metrics["cycle_time"]
        st.metric(
            label="⏱️ Cycle Time",
            value=f"{metrics['cycle_time']:.1f} days",
            delta=f"{delta:+.1f} vs benchmark",
        )

    with col3:
        delta_pct = (
            metrics["flow_efficiency"] - st.session_state.benchmark_flow_efficiency
        ) * 100
        st.metric(
            label="⚡ Flow Efficiency",
            value=f"{metrics['flow_efficiency']*100:.1f}%",
            delta=f"{delta_pct:+.1f}%",
        )

    with col4:
        delta = st.session_state.benchmark_wip - metrics["wip"]
        st.metric(
            label="📋 WIP",
            value=f"{metrics['wip']:.0f} items",
            delta=f"{delta:+.0f} vs benchmark",
            delta_color="normal",
        )

    with col5:
        delta_pct = (
            metrics["flow_predictability"] - st.session_state.benchmark_predictability
        ) * 100
        st.metric(
            label="🎯 Predictability",
            value=f"{metrics['flow_predictability']*100:.0f}%",
            delta=f"{delta_pct:+.1f}%",
        )


def render_financial_summary(financial_engine):
    """Render financial summary section"""
    st.markdown(
        '<p class="section-header">💰 Financial Impact Summary</p>',
        unsafe_allow_html=True,
    )

    metrics = financial_engine.get_all_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Weekly Revenue", value=format_currency(metrics["weekly_revenue"])
        )

    with col2:
        st.metric(
            label="Weekly Profit",
            value=format_currency(metrics["weekly_profit"]),
            delta=f"{metrics['profit_margin']*100:.1f}% margin",
        )

    with col3:
        st.metric(
            label="Cost of Delay",
            value=format_currency(metrics["total_delay_cost"]),
            delta="per week",
            delta_color="inverse",
        )

    with col4:
        st.metric(
            label="WIP Carrying Cost",
            value=format_currency(metrics["wip_carrying_cost"]),
            delta="per month",
            delta_color="inverse",
        )


def main():
    """Main application entry point"""
    # Initialize state
    initialize_session_state()

    # Render sidebar controls
    render_sidebar()

    # Header
    st.markdown(
        '<h1 class="main-header">📊 Flow Metrics & Financial Dashboard</h1>',
        unsafe_allow_html=True,
    )

    # Initialize calculation engines
    flow_engine = FlowMetricsEngine(
        throughput=st.session_state.throughput,
        cycle_time=st.session_state.cycle_time,
        flow_efficiency=st.session_state.flow_efficiency,
        wip=st.session_state.wip,
        flow_predictability=st.session_state.flow_predictability,
    )

    financial_engine = FinancialMetricsEngine(
        flow_engine=flow_engine,
        revenue_per_item=st.session_state.revenue_per_item,
        cost_per_item=st.session_state.cost_per_item,
        hourly_labor_cost=st.session_state.hourly_labor_cost,
        delay_cost_per_day=st.session_state.delay_cost_per_day,
        wip_carrying_cost_pct=st.session_state.wip_carrying_cost_pct,
        team_size=st.session_state.team_size,
    )

    viz_engine = VisualizationEngine(flow_engine, financial_engine)

    # Get weights for weighted calculations
    weights = {
        "throughput": st.session_state.weight_throughput,
        "cycle_time": st.session_state.weight_cycle_time,
        "flow_efficiency": st.session_state.weight_flow_efficiency,
        "wip": st.session_state.weight_wip,
        "flow_predictability": st.session_state.weight_predictability,
    }

    benchmarks = {
        "throughput": st.session_state.benchmark_throughput,
        "cycle_time": st.session_state.benchmark_cycle_time,
        "flow_efficiency": st.session_state.benchmark_flow_efficiency,
        "wip": st.session_state.benchmark_wip,
        "flow_predictability": st.session_state.benchmark_predictability,
    }

    # Top metric cards
    render_metric_cards(flow_engine, financial_engine)

    st.markdown("---")

    # Financial summary
    render_financial_summary(financial_engine)

    st.markdown("---")

    # Main visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "🔗 Metric Dependencies",
            "📊 Flow Analysis",
            "💵 Financial Impact",
            "🎯 What-If Scenarios",
            "📈 Optimization",
            "🎯 Benchmarks",
            "💼 CFO ROI Model",
        ]
    )

    with tab1:
        st.markdown(
            '<p class="section-header"></p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class="info-box">
            <strong>Little's Law:</strong> WIP = Throughput × Cycle Time<br>
            This fundamental relationship shows how these three metrics are mathematically linked.
            </div>
            """,
                unsafe_allow_html=True,
            )

            fig_littles = viz_engine.create_littles_law_visualization()
            st.plotly_chart(fig_littles, use_container_width=True)

        with col2:
            fig_network = viz_engine.create_dependency_network()
            st.plotly_chart(fig_network, use_container_width=True)

        st.markdown("#### Correlation Heatmap")
        fig_correlation = viz_engine.create_correlation_heatmap()
        st.plotly_chart(fig_correlation, use_container_width=True)

    with tab2:
        st.markdown(
            '<p class="section-header"></p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1.2, 0.8])

        with col1:
            fig_radar = viz_engine.create_radar_chart(benchmarks)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            with st.expander("ℹ️ How is this calculated?"):
                st.markdown(
                    """
                    **Weighted Performance Score (0-100):**
                    1. For each metric, calculate: `(Current / Benchmark) × 100`
                      - For Cycle Time & WIP: `(Benchmark / Current) × 100` (lower is better)
                    2. Multiply each score by its **importance weight** from the Weights tab
                    3. Sum all weighted scores
                    
                    **Example:** If Throughput is 50 (benchmark 60) with weight 0.25:
                    - Score = (50/60) × 100 = 83.3%
                    - Weighted contribution = 83.3 × 0.25 = 20.8 points
                    """
                )
            fig_gauge = viz_engine.create_weighted_score_gauge(weights, benchmarks)
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab3:
        st.markdown(
            '<p class="section-header"></p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Revenue vs Costs Breakdown")
            with st.expander("ℹ️ How is this calculated?"):
                st.markdown(
                    f"""
                    **Weekly Financial Breakdown:**
                    - **Revenue** = Throughput × Revenue per Item
                      - `{st.session_state.throughput:.1f} × ${st.session_state.revenue_per_item:,.0f} = ${st.session_state.throughput * st.session_state.revenue_per_item:,.0f}`
                    - **Direct Costs** = Throughput × Cost per Item
                      - `{st.session_state.throughput:.1f} × ${st.session_state.cost_per_item:,.0f} = ${st.session_state.throughput * st.session_state.cost_per_item:,.0f}`
                    - **Labor Costs** = Team Size × Hourly Rate × 40 hrs
                      - `{st.session_state.team_size} × ${st.session_state.hourly_labor_cost:.0f} × 40 = ${st.session_state.team_size * st.session_state.hourly_labor_cost * 40:,.0f}`
                    - **Hidden Costs** = Delay + Carrying + Inefficiency costs
                    - **Net Profit** = Revenue - All Costs
                    """
                )
            fig_waterfall = viz_engine.create_financial_waterfall()
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with col2:
            with st.expander("ℹ️ How is this calculated?"):
                wait_time = st.session_state.cycle_time * (
                    1 - st.session_state.flow_efficiency
                )
                delay_per_item = wait_time * st.session_state.delay_cost_per_day
                st.markdown(
                    f"""
                    **Cost of Delay Components:**
                    - **Wait Time** = Cycle Time × (1 - Flow Efficiency)
                      - `{st.session_state.cycle_time:.1f} × (1 - {st.session_state.flow_efficiency:.2f}) = {wait_time:.1f} days`
                    - **Delay per Item** = Wait Time × Cost of Delay/Day
                      - `{wait_time:.1f} × ${st.session_state.delay_cost_per_day:.0f} = ${delay_per_item:,.0f}`
                    - **Total WIP Delay** = Delay per Item × WIP
                      - `${delay_per_item:,.0f} × {st.session_state.wip:.0f} = ${delay_per_item * st.session_state.wip:,.0f}`
                    - **Opportunity Cost** = Revenue delayed while items sit in WIP
                    """
                )
            fig_delay = viz_engine.create_delay_cost_chart()
            st.plotly_chart(fig_delay, use_container_width=True)

        st.markdown("#### ROI by Metric Improvement")
        roi_improvement_pct = st.slider(
            "Improvement percentage to analyze",
            min_value=0.5,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Adjust to see financial impact of different improvement levels",
            key="roi_improvement_slider",
        )
        fig_roi = viz_engine.create_roi_comparison_chart(roi_improvement_pct)
        st.plotly_chart(fig_roi, use_container_width=True)

    with tab4:
        st.markdown(
            '<p class="section-header"></p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-box">
            Explore how changes to individual metrics would impact your overall performance and financials.
            Use the sliders below to model different scenarios.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            scenario_metric = st.selectbox(
                "Select metric to analyze",
                [
                    "Throughput",
                    "Cycle Time",
                    "Flow Efficiency",
                    "WIP",
                    "Flow Predictability",
                ],
            )

            improvement_pct = st.slider(
                "Improvement percentage",
                min_value=-50.0,
                max_value=100.0,
                value=20.0,
                step=0.5,
                help="Positive = improvement, Negative = decline",
            )

        with col2:
            fig_scenario = viz_engine.create_scenario_analysis(
                scenario_metric, improvement_pct
            )
            st.plotly_chart(fig_scenario, use_container_width=True)

        st.markdown("#### Multi-Metric Sensitivity Analysis")
        sensitivity_pct = st.slider(
            "Improvement percentage for sensitivity analysis",
            min_value=0.5,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Adjust to see how different improvement levels ripple through all metrics",
            key="sensitivity_improvement_slider",
        )
        fig_sensitivity = viz_engine.create_sensitivity_heatmap(sensitivity_pct)
        st.plotly_chart(fig_sensitivity, use_container_width=True)

    with tab5:
        st.markdown(
            '<p class="section-header">Optimization Recommendations</p>',
            unsafe_allow_html=True,
        )

        # Calculate optimization opportunities
        opportunities = flow_engine.identify_improvement_opportunities(benchmarks)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Improvement Priority Matrix")
            fig_priority = viz_engine.create_priority_matrix(weights, benchmarks)
            st.plotly_chart(fig_priority, use_container_width=True)

        with col2:
            st.markdown("#### Top Recommendations")
            if opportunities:
                for i, opp in enumerate(opportunities[:5], 1):
                    impact_color = (
                        "🟢"
                        if opp["impact"] == "high"
                        else "🟡" if opp["impact"] == "medium" else "🔴"
                    )
                    st.markdown(
                        f"""
                        **{i}. {opp['metric']}** {impact_color}
                        - Gap: {opp['gap']:.1f}%
                        - Potential savings: {format_currency(opp['potential_savings'])}
                        """
                    )
            else:
                st.success("🎉 All metrics are at or above benchmark!")

        st.markdown("#### Projected Improvement Roadmap")
        with st.expander("ℹ️ How is this calculated?"):
            st.markdown(
                """
                **Projected Improvement Roadmap Logic:**
                
                This chart projects cumulative financial savings over time based on systematically addressing improvement opportunities.
                
                **Calculation Method:**
                1. **Identify Opportunities:** Metrics where current performance is below benchmark are ranked by potential financial impact
                2. **Phased Implementation:** The model assumes you tackle opportunities sequentially, starting with highest-impact items
                3. **Milestone Intervals:** The roadmap divides the timeline into phases (roughly 6 phases total), with one major improvement initiative per phase
                4. **Cumulative Savings:** Each week, savings accumulate based on:
                  - `Weekly Savings = Σ (Potential Annual Savings / 52)` for all initiatives started
                5. **Star Markers:** Show when each initiative is expected to be completed and its cumulative impact
                
                **Key Assumptions:**
                - Improvements are implemented in priority order (highest ROI first)
                - Each initiative takes approximately `Total Weeks / 6` weeks to implement
                - Benefits begin immediately after implementation
                - Savings are linear and sustained throughout the period
                
                **How to Use:**
                - **Shorter timeframes (12-24 weeks):** Aggressive improvement sprints
                - **Medium timeframes (24-52 weeks):** Balanced annual planning
                - **Longer timeframes (52-104 weeks):** Strategic multi-year roadmaps
                
                **For Stakeholders:**
                - The final cumulative savings value represents the total projected financial benefit
                - Star markers indicate key milestones for tracking progress
                - Adjust the duration to match your organization's improvement capacity
                """
            )

        roadmap_weeks = st.slider(
            "Roadmap duration (weeks)",
            min_value=4,
            max_value=104,
            value=24,
            step=4,
            help="Adjust the time horizon for the improvement projection",
            key="roadmap_weeks_slider",
        )
        fig_roadmap = viz_engine.create_improvement_roadmap(
            opportunities, roadmap_weeks
        )
        st.plotly_chart(fig_roadmap, use_container_width=True)

    with tab6:
        st.markdown(
            '<p class="section-header">Benchmark Configuration</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-box">
            <strong>Setting Benchmarks:</strong> Benchmarks are target values for comparison. You can set them based on:
            <ul>
            <li><strong>Historical Best:</strong> Your team's best-performing period</li>
            <li><strong>Industry Standards:</strong> Typical values for your industry</li>
            <li><strong>Business Requirements:</strong> What's needed to meet customer/revenue goals</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initialize benchmark widget keys
        if "bench_tp_slider_key" not in st.session_state:
            st.session_state.bench_tp_slider_key = st.session_state.benchmark_throughput
        if "bench_tp_input_key" not in st.session_state:
            st.session_state.bench_tp_input_key = st.session_state.benchmark_throughput
        if "bench_ct_slider_key" not in st.session_state:
            st.session_state.bench_ct_slider_key = st.session_state.benchmark_cycle_time
        if "bench_ct_input_key" not in st.session_state:
            st.session_state.bench_ct_input_key = st.session_state.benchmark_cycle_time
        if "bench_fe_slider_key" not in st.session_state:
            st.session_state.bench_fe_slider_key = int(
                st.session_state.benchmark_flow_efficiency * 100
            )
        if "bench_fe_input_key" not in st.session_state:
            st.session_state.bench_fe_input_key = int(
                st.session_state.benchmark_flow_efficiency * 100
            )
        if "bench_wip_slider_key" not in st.session_state:
            st.session_state.bench_wip_slider_key = st.session_state.benchmark_wip
        if "bench_wip_input_key" not in st.session_state:
            st.session_state.bench_wip_input_key = st.session_state.benchmark_wip
        if "bench_pred_slider_key" not in st.session_state:
            st.session_state.bench_pred_slider_key = (
                st.session_state.benchmark_predictability
            )
        if "bench_pred_input_key" not in st.session_state:
            st.session_state.bench_pred_input_key = (
                st.session_state.benchmark_predictability
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Flow Metric Benchmarks")

            # Benchmark Throughput
            def sync_bench_tp_slider():
                st.session_state.benchmark_throughput = (
                    st.session_state.bench_tp_slider_key
                )
                st.session_state.bench_tp_input_key = (
                    st.session_state.bench_tp_slider_key
                )
                # Sync with sidebar keys
                if "sb_bench_tp_slider_key" in st.session_state:
                    del st.session_state["sb_bench_tp_slider_key"]
                if "sb_bench_tp_input_key" in st.session_state:
                    del st.session_state["sb_bench_tp_input_key"]

            def sync_bench_tp_input():
                st.session_state.benchmark_throughput = (
                    st.session_state.bench_tp_input_key
                )
                st.session_state.bench_tp_slider_key = (
                    st.session_state.bench_tp_input_key
                )
                if "sb_bench_tp_slider_key" in st.session_state:
                    del st.session_state["sb_bench_tp_slider_key"]
                if "sb_bench_tp_input_key" in st.session_state:
                    del st.session_state["sb_bench_tp_input_key"]

            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "Benchmark Throughput (items/week)",
                    min_value=1.0,
                    max_value=200.0,
                    step=1.0,
                    help="Target throughput to compare against",
                    key="bench_tp_slider_key",
                    on_change=sync_bench_tp_slider,
                )
            with c2:
                st.number_input(
                    "Throughput Value",
                    min_value=1.0,
                    max_value=500.0,
                    step=1.0,
                    key="bench_tp_input_key",
                    on_change=sync_bench_tp_input,
                    label_visibility="collapsed",
                )

            # Benchmark Cycle Time
            def sync_bench_ct_slider():
                st.session_state.benchmark_cycle_time = (
                    st.session_state.bench_ct_slider_key
                )
                st.session_state.bench_ct_input_key = (
                    st.session_state.bench_ct_slider_key
                )
                if "sb_bench_ct_slider_key" in st.session_state:
                    del st.session_state["sb_bench_ct_slider_key"]
                if "sb_bench_ct_input_key" in st.session_state:
                    del st.session_state["sb_bench_ct_input_key"]

            def sync_bench_ct_input():
                st.session_state.benchmark_cycle_time = (
                    st.session_state.bench_ct_input_key
                )
                st.session_state.bench_ct_slider_key = (
                    st.session_state.bench_ct_input_key
                )
                if "sb_bench_ct_slider_key" in st.session_state:
                    del st.session_state["sb_bench_ct_slider_key"]
                if "sb_bench_ct_input_key" in st.session_state:
                    del st.session_state["sb_bench_ct_input_key"]

            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "Benchmark Cycle Time (days)",
                    min_value=1.0,
                    max_value=90.0,
                    step=0.5,
                    help="Target cycle time (lower is better)",
                    key="bench_ct_slider_key",
                    on_change=sync_bench_ct_slider,
                )
            with c2:
                st.number_input(
                    "Cycle Time Value",
                    min_value=1.0,
                    max_value=180.0,
                    step=0.5,
                    key="bench_ct_input_key",
                    on_change=sync_bench_ct_input,
                    label_visibility="collapsed",
                )

            # Benchmark Flow Efficiency
            def sync_bench_fe_slider():
                st.session_state.benchmark_flow_efficiency = (
                    st.session_state.bench_fe_slider_key / 100.0
                )
                st.session_state.bench_fe_input_key = (
                    st.session_state.bench_fe_slider_key
                )
                if "sb_bench_fe_slider_key" in st.session_state:
                    del st.session_state["sb_bench_fe_slider_key"]
                if "sb_bench_fe_input_key" in st.session_state:
                    del st.session_state["sb_bench_fe_input_key"]

            def sync_bench_fe_input():
                st.session_state.benchmark_flow_efficiency = (
                    st.session_state.bench_fe_input_key / 100.0
                )
                st.session_state.bench_fe_slider_key = (
                    st.session_state.bench_fe_input_key
                )
                if "sb_bench_fe_slider_key" in st.session_state:
                    del st.session_state["sb_bench_fe_slider_key"]
                if "sb_bench_fe_input_key" in st.session_state:
                    del st.session_state["sb_bench_fe_input_key"]

            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "Benchmark Flow Efficiency (%)",
                    min_value=1,
                    max_value=100,
                    step=1,
                    help="Target flow efficiency percentage",
                    key="bench_fe_slider_key",
                    on_change=sync_bench_fe_slider,
                )
            with c2:
                st.number_input(
                    "Flow Efficiency Value",
                    min_value=1,
                    max_value=100,
                    step=1,
                    key="bench_fe_input_key",
                    on_change=sync_bench_fe_input,
                    label_visibility="collapsed",
                )

            # Benchmark WIP
            def sync_bench_wip_slider():
                st.session_state.benchmark_wip = st.session_state.bench_wip_slider_key
                st.session_state.bench_wip_input_key = (
                    st.session_state.bench_wip_slider_key
                )
                if "sb_bench_wip_slider_key" in st.session_state:
                    del st.session_state["sb_bench_wip_slider_key"]
                if "sb_bench_wip_input_key" in st.session_state:
                    del st.session_state["sb_bench_wip_input_key"]

            def sync_bench_wip_input():
                st.session_state.benchmark_wip = st.session_state.bench_wip_input_key
                st.session_state.bench_wip_slider_key = (
                    st.session_state.bench_wip_input_key
                )
                if "sb_bench_wip_slider_key" in st.session_state:
                    del st.session_state["sb_bench_wip_slider_key"]
                if "sb_bench_wip_input_key" in st.session_state:
                    del st.session_state["sb_bench_wip_input_key"]

            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "Benchmark WIP (items)",
                    min_value=1.0,
                    max_value=500.0,
                    step=1.0,
                    help="Target WIP level (lower is typically better)",
                    key="bench_wip_slider_key",
                    on_change=sync_bench_wip_slider,
                )
            with c2:
                st.number_input(
                    "WIP Value",
                    min_value=1.0,
                    max_value=1000.0,
                    step=1.0,
                    key="bench_wip_input_key",
                    on_change=sync_bench_wip_input,
                    label_visibility="collapsed",
                )

            # Benchmark Predictability
            def sync_bench_pred_slider():
                st.session_state.benchmark_predictability = (
                    st.session_state.bench_pred_slider_key
                )
                st.session_state.bench_pred_input_key = (
                    st.session_state.bench_pred_slider_key
                )
                if "sb_bench_pred_slider_key" in st.session_state:
                    del st.session_state["sb_bench_pred_slider_key"]
                if "sb_bench_pred_input_key" in st.session_state:
                    del st.session_state["sb_bench_pred_input_key"]

            def sync_bench_pred_input():
                st.session_state.benchmark_predictability = (
                    st.session_state.bench_pred_input_key
                )
                st.session_state.bench_pred_slider_key = (
                    st.session_state.bench_pred_input_key
                )
                if "sb_bench_pred_slider_key" in st.session_state:
                    del st.session_state["sb_bench_pred_slider_key"]
                if "sb_bench_pred_input_key" in st.session_state:
                    del st.session_state["sb_bench_pred_input_key"]

            c1, c2 = st.columns([3, 1])
            with c1:
                st.slider(
                    "Benchmark Predictability",
                    min_value=0.1,
                    max_value=1.0,
                    step=0.05,
                    help="Target predictability (higher is better)",
                    key="bench_pred_slider_key",
                    on_change=sync_bench_pred_slider,
                )
            with c2:
                st.number_input(
                    "Predictability Value",
                    min_value=0.1,
                    max_value=1.0,
                    step=0.05,
                    key="bench_pred_input_key",
                    on_change=sync_bench_pred_input,
                    label_visibility="collapsed",
                )

        with col2:
            st.markdown("#### Current vs Benchmark Summary")

            comparison_data = {
                "Metric": [
                    "Throughput",
                    "Cycle Time",
                    "Flow Efficiency",
                    "WIP",
                    "Predictability",
                ],
                "Current": [
                    f"{st.session_state.throughput:.1f}",
                    f"{st.session_state.cycle_time:.1f}",
                    f"{st.session_state.flow_efficiency*100:.1f}%",
                    f"{st.session_state.wip:.0f}",
                    f"{st.session_state.flow_predictability*100:.0f}%",
                ],
                "Benchmark": [
                    f"{st.session_state.benchmark_throughput:.1f}",
                    f"{st.session_state.benchmark_cycle_time:.1f}",
                    f"{st.session_state.benchmark_flow_efficiency*100:.1f}%",
                    f"{st.session_state.benchmark_wip:.0f}",
                    f"{st.session_state.benchmark_predictability*100:.0f}%",
                ],
                "Status": [
                    (
                        "✅"
                        if st.session_state.throughput
                        >= st.session_state.benchmark_throughput
                        else "❌"
                    ),
                    (
                        "✅"
                        if st.session_state.cycle_time
                        <= st.session_state.benchmark_cycle_time
                        else "❌"
                    ),
                    (
                        "✅"
                        if st.session_state.flow_efficiency
                        >= st.session_state.benchmark_flow_efficiency
                        else "❌"
                    ),
                    (
                        "✅"
                        if st.session_state.wip <= st.session_state.benchmark_wip
                        else "❌"
                    ),
                    (
                        "✅"
                        if st.session_state.flow_predictability
                        >= st.session_state.benchmark_predictability
                        else "❌"
                    ),
                ],
            }

            st.dataframe(
                pd.DataFrame(comparison_data), use_container_width=True, hide_index=True
            )

            st.markdown("#### Benchmark Presets")
            preset = st.selectbox(
                "Load a preset",
                ["Custom", "Conservative", "Industry Average", "World-Class"],
                help="Load predefined benchmark values",
            )

            all_benchmark_keys = [
                "bench_tp_slider_key",
                "bench_tp_input_key",
                "bench_ct_slider_key",
                "bench_ct_input_key",
                "bench_fe_slider_key",
                "bench_fe_input_key",
                "bench_wip_slider_key",
                "bench_wip_input_key",
                "bench_pred_slider_key",
                "bench_pred_input_key",
                "sb_bench_tp_slider_key",
                "sb_bench_tp_input_key",
                "sb_bench_ct_slider_key",
                "sb_bench_ct_input_key",
                "sb_bench_fe_slider_key",
                "sb_bench_fe_input_key",
                "sb_bench_wip_slider_key",
                "sb_bench_wip_input_key",
                "sb_bench_pred_slider_key",
                "sb_bench_pred_input_key",
            ]

            if preset == "Conservative":
                if st.button("Apply Conservative Preset"):
                    st.session_state.benchmark_throughput = 40.0
                    st.session_state.benchmark_cycle_time = 21.0
                    st.session_state.benchmark_flow_efficiency = 0.15
                    st.session_state.benchmark_wip = 120.0
                    st.session_state.benchmark_predictability = 0.65
                    for key in all_benchmark_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            elif preset == "Industry Average":
                if st.button("Apply Industry Average Preset"):
                    st.session_state.benchmark_throughput = 50.0
                    st.session_state.benchmark_cycle_time = 14.0
                    st.session_state.benchmark_flow_efficiency = 0.25
                    st.session_state.benchmark_wip = 90.0
                    st.session_state.benchmark_predictability = 0.75
                    for key in all_benchmark_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            elif preset == "World-Class":
                if st.button("Apply World-Class Preset"):
                    st.session_state.benchmark_throughput = 80.0
                    st.session_state.benchmark_cycle_time = 7.0
                    st.session_state.benchmark_flow_efficiency = 0.40
                    st.session_state.benchmark_wip = 50.0
                    st.session_state.benchmark_predictability = 0.90
                    for key in all_benchmark_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

    with tab7:
        render_cfo_roi_tab()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    Flow Metrics Dashboard v1.0 | Built with Streamlit<br>
    Data updates in real-time as you adjust parameters
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
