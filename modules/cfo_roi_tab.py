"""
CFO ROI Tab - Streamlit Integration
Renders the CFO ROI Model section with file upload, editable inputs,
and dynamic visualizations.
"""

import streamlit as st
import pandas as pd
from modules.cfo_roi_engine import (
    CFOROIEngine,
    ExpenseRow,
    InvestmentRow,
    ROIModelRow,
    auto_detect_group,
    load_cfo_data_from_excel,
    load_cfo_data_from_csv,
    parse_expenses_sheet,
    parse_investment_sheet,
    parse_roi_model_sheet,
)
from modules.cfo_roi_visualizations import (
    create_expenses_breakdown,
    create_investment_breakdown,
    create_roi_by_group_chart,
    create_roi_waterfall,
    create_scenario_comparison,
    create_multi_year_projection,
    create_payback_gauge,
    create_roi_item_detail_chart,
)
from utils.helpers import format_currency


def _init_cfo_session_state():
    """Initialize CFO-specific session state."""
    defaults = {
        "cfo_expenses": [],
        "cfo_investments": [],
        "cfo_roi_rows": [],
        "cfo_data_loaded": False,
        "cfo_scenario_a_name": "Consultants",
        "cfo_scenario_b_name": "Tryzens",
        "cfo_scenario_a_one_time": 250000.0,
        "cfo_scenario_b_bonus_pct": 10.0,
        "cfo_y2_growth_pct": 5.0,
        "cfo_y3_growth_pct": 3.0,
        "cfo_overlap_pct": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _handle_file_upload():
    """Handle file uploads for CFO data."""
    st.markdown("#### 📁 Load Data")
    st.markdown(
        """
        Upload an **Excel file** with sheets named *Expenses*, *Investment*, and *ROI Model*,
        or upload individual **CSV files** for each section.
        """
    )

    upload_mode = st.radio(
        "Upload mode",
        ["Single Excel file (multi-sheet)", "Individual CSV files"],
        horizontal=True,
        key="cfo_upload_mode",
    )

    if upload_mode == "Single Excel file (multi-sheet)":
        uploaded = st.file_uploader(
            "Upload Excel (.xlsx)",
            type=["xlsx", "xls"],
            key="cfo_excel_upload",
        )
        if uploaded is not None:
            import tempfile, os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                data = load_cfo_data_from_excel(tmp_path)
                st.session_state.cfo_expenses = data["expenses"]
                st.session_state.cfo_investments = data["investments"]
                st.session_state.cfo_roi_rows = data["roi_rows"]
                st.session_state.cfo_data_loaded = True

                counts = (
                    f"{len(data['expenses'])} expense rows, "
                    f"{len(data['investments'])} investment rows, "
                    f"{len(data['roi_rows'])} ROI line items"
                )
                st.success(f"Loaded: {counts}")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
            finally:
                os.unlink(tmp_path)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            exp_file = st.file_uploader("Expenses CSV", type=["csv"], key="cfo_csv_exp")
            if exp_file:
                df = pd.read_csv(exp_file).dropna(how="all")
                st.session_state.cfo_expenses = parse_expenses_sheet(df)
                st.success(f"{len(st.session_state.cfo_expenses)} expense rows")

        with col2:
            inv_file = st.file_uploader("Investment CSV", type=["csv"], key="cfo_csv_inv")
            if inv_file:
                df = pd.read_csv(inv_file).dropna(how="all")
                st.session_state.cfo_investments = parse_investment_sheet(df)
                st.success(f"{len(st.session_state.cfo_investments)} investment rows")

        with col3:
            roi_file = st.file_uploader("ROI Model CSV", type=["csv"], key="cfo_csv_roi")
            if roi_file:
                df = pd.read_csv(roi_file).dropna(how="all")
                st.session_state.cfo_roi_rows = parse_roi_model_sheet(df)
                st.success(f"{len(st.session_state.cfo_roi_rows)} ROI rows")

        if (
            st.session_state.cfo_expenses
            or st.session_state.cfo_investments
            or st.session_state.cfo_roi_rows
        ):
            st.session_state.cfo_data_loaded = True


def _render_editable_expenses():
    """Render editable expense table."""
    st.markdown("#### 💼 Expenses")
    expenses = st.session_state.cfo_expenses

    if not expenses:
        st.info("No expense data. Upload a file or add rows manually.")
        if st.button("➕ Add expense row", key="add_exp"):
            st.session_state.cfo_expenses.append(
                ExpenseRow(region="New Region", fte_count=1, cost_per_fte=50000, total=50000)
            )
            st.rerun()
        return

    # Show as editable dataframe
    df = pd.DataFrame([
        {
            "Region": e.region,
            "FTE Count": e.fte_count,
            "Cost per FTE ($)": e.cost_per_fte,
            "Total ($)": e.total,
            "Notes": e.notes,
        }
        for e in expenses
    ])

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="cfo_expense_editor",
        column_config={
            "FTE Count": st.column_config.NumberColumn(min_value=0, step=1),
            "Cost per FTE ($)": st.column_config.NumberColumn(min_value=0, step=1000, format="$%d"),
            "Total ($)": st.column_config.NumberColumn(min_value=0, format="$%d"),
        },
    )

    # Sync back
    new_expenses = []
    for _, row in edited.iterrows():
        region = str(row.get("Region", "") or "")
        if not region:
            continue
        fte = float(row.get("FTE Count", 0) or 0)
        cost = float(row.get("Cost per FTE ($)", 0) or 0)
        total_val = float(row.get("Total ($)", 0) or 0)
        # Recalculate total if it looks stale
        if total_val == 0 and fte > 0 and cost > 0:
            total_val = fte * cost
        notes = str(row.get("Notes", "") or "")
        new_expenses.append(ExpenseRow(region=region, fte_count=fte, cost_per_fte=cost, total=total_val, notes=notes))

    st.session_state.cfo_expenses = new_expenses
    total = sum(e.total for e in new_expenses)
    st.markdown(f"**Total Engineering Cost: {format_currency(total)}**")


def _render_editable_investments():
    """Render editable investment table."""
    st.markdown("#### 🏗️ Investments")
    investments = st.session_state.cfo_investments

    if not investments:
        st.info("No investment data. Upload a file or add rows manually.")
        if st.button("➕ Add investment row", key="add_inv"):
            st.session_state.cfo_investments.append(
                InvestmentRow(category="New Item", annual_recurring=0, one_time=0)
            )
            st.rerun()
        return

    df = pd.DataFrame([
        {
            "Category": i.category,
            "Annual Recurring ($)": i.annual_recurring,
            "One-time ($)": i.one_time,
            "Notes": i.notes,
        }
        for i in investments
    ])

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="cfo_invest_editor",
        column_config={
            "Annual Recurring ($)": st.column_config.NumberColumn(min_value=0, format="$%d"),
            "One-time ($)": st.column_config.NumberColumn(min_value=0, format="$%d"),
        },
    )

    new_inv = []
    for _, row in edited.iterrows():
        cat = str(row.get("Category", "") or "")
        if not cat:
            continue
        annual = float(row.get("Annual Recurring ($)", 0) or 0)
        one_time = float(row.get("One-time ($)", 0) or 0)
        notes = str(row.get("Notes", "") or "")
        new_inv.append(InvestmentRow(category=cat, annual_recurring=annual, one_time=one_time, notes=notes))

    st.session_state.cfo_investments = new_inv

    total_annual = sum(i.annual_recurring for i in new_inv)
    total_once = sum(i.one_time for i in new_inv)
    st.markdown(f"**Annual: {format_currency(total_annual)} | One-time: {format_currency(total_once)}**")


def _render_editable_roi_model():
    """Render editable ROI model table with group assignment."""
    st.markdown("#### 📊 ROI Model Line Items")
    roi_rows = st.session_state.cfo_roi_rows

    if not roi_rows:
        st.info("No ROI data. Upload a file or add rows manually.")
        if st.button("➕ Add ROI row", key="add_roi"):
            st.session_state.cfo_roi_rows.append(
                ROIModelRow(category="New Item", low=0, high=0, group="Other")
            )
            st.rerun()
        return

    df = pd.DataFrame([
        {
            "Category": r.category,
            "Group": r.group,
            "Low ($)": r.low,
            "High ($)": r.high,
            "Notes": r.notes,
        }
        for r in roi_rows
    ])

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="cfo_roi_editor",
        column_config={
            "Group": st.column_config.SelectboxColumn(
                options=["Flow", "AI", "Business Outcomes", "Overlap", "Other"],
                required=True,
            ),
            "Low ($)": st.column_config.NumberColumn(format="$%d"),
            "High ($)": st.column_config.NumberColumn(format="$%d"),
        },
    )

    new_rows = []
    for _, row in edited.iterrows():
        cat = str(row.get("Category", "") or "")
        if not cat:
            continue
        group = str(row.get("Group", "") or "")
        if not group or group == "nan":
            group = auto_detect_group(cat)
        low = float(row.get("Low ($)", 0) or 0)
        high = float(row.get("High ($)", 0) or 0)
        notes = str(row.get("Notes", "") or "")
        new_rows.append(ROIModelRow(category=cat, low=low, high=high, notes=notes, group=group))

    st.session_state.cfo_roi_rows = new_rows


def _render_scenario_config():
    """Render scenario configuration controls."""
    st.markdown("#### ⚙️ Scenario & Projection Settings")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.cfo_scenario_a_name = st.text_input(
            "Scenario A Name",
            value=st.session_state.cfo_scenario_a_name,
            key="cfo_sa_name_input",
        )
        st.session_state.cfo_scenario_a_one_time = st.number_input(
            f"{st.session_state.cfo_scenario_a_name} One-time Cost ($)",
            min_value=0.0,
            value=st.session_state.cfo_scenario_a_one_time,
            step=10000.0,
            format="%.0f",
            key="cfo_sa_cost_input",
        )

    with col2:
        st.session_state.cfo_scenario_b_name = st.text_input(
            "Scenario B Name",
            value=st.session_state.cfo_scenario_b_name,
            key="cfo_sb_name_input",
        )
        st.session_state.cfo_scenario_b_bonus_pct = st.number_input(
            f"{st.session_state.cfo_scenario_b_name} Bonus (% of Net Annual ROI)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.cfo_scenario_b_bonus_pct,
            step=1.0,
            key="cfo_sb_pct_input",
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        st.session_state.cfo_y2_growth_pct = st.number_input(
            "Year 2 Growth (%)",
            min_value=-50.0,
            max_value=100.0,
            value=st.session_state.cfo_y2_growth_pct,
            step=0.5,
            key="cfo_y2_input",
        )
    with col4:
        st.session_state.cfo_y3_growth_pct = st.number_input(
            "Year 3 Growth (%)",
            min_value=-50.0,
            max_value=100.0,
            value=st.session_state.cfo_y3_growth_pct,
            step=0.5,
            key="cfo_y3_input",
        )
    with col5:
        st.session_state.cfo_overlap_pct = st.number_input(
            "Overlap Adj. (%) — if no explicit row",
            min_value=0.0,
            max_value=50.0,
            value=st.session_state.cfo_overlap_pct,
            step=1.0,
            key="cfo_overlap_input",
            help="Applied only if no 'Overlap' rows exist in the ROI data",
        )


def _render_summary_cards(engine):
    """Render KPI cards for both scenarios."""
    lo = engine.compute_summary("low")
    hi = engine.compute_summary("high")

    st.markdown("#### 📋 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Eng. Cost",
            format_currency(lo["total_engineering_cost"]),
        )
    with col2:
        st.metric(
            "Gross ROI (range)",
            f"{format_currency(lo['gross_roi'])} – {format_currency(hi['gross_roi'])}",
        )
    with col3:
        st.metric(
            "Net Annual ROI",
            f"{format_currency(lo['net_annual_roi'])} – {format_currency(hi['net_annual_roi'])}",
        )
    with col4:
        st.metric(
            f"Payback ({engine.scenario_a_name})",
            f"{lo['payback_months_a']:.1f} – {hi['payback_months_a']:.1f} mo",
        )
    with col5:
        st.metric(
            f"Payback ({engine.scenario_b_name})",
            f"{lo['payback_months_b']:.1f} – {hi['payback_months_b']:.1f} mo",
        )


def _render_charts(engine):
    """Render all CFO ROI charts."""
    scenario_toggle = st.radio(
        "Chart scenario",
        ["Conservative (Low)", "Optimistic (High)"],
        horizontal=True,
        key="cfo_chart_scenario",
    )
    scenario = "low" if "Low" in scenario_toggle else "high"

    # Row 1: Expenses + Investment
    col1, col2 = st.columns(2)
    with col1:
        fig = create_expenses_breakdown(engine)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_investment_breakdown(engine)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: ROI by group + item detail
    col1, col2 = st.columns(2)
    with col1:
        fig = create_roi_by_group_chart(engine, scenario)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_roi_item_detail_chart(engine, scenario)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Waterfall
    fig = create_roi_waterfall(engine, scenario)
    st.plotly_chart(fig, use_container_width=True)

    # Row 4: Scenario comparison + Multi-year
    col1, col2 = st.columns(2)
    with col1:
        fig = create_scenario_comparison(engine)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = create_multi_year_projection(engine)
        st.plotly_chart(fig, use_container_width=True)

    # Row 5: Payback gauges
    fig = create_payback_gauge(engine)
    st.plotly_chart(fig, use_container_width=True)


def render_cfo_roi_tab():
    """Main entry point — call this from your app's tab."""
    _init_cfo_session_state()

    st.markdown(
        '<p class="section-header">💼 CFO ROI Model</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        <strong>CFO ROI Model:</strong> Upload your Expenses, Investment, and ROI Model data
        (Excel or CSV), edit values inline, and see dynamic ROI projections with two configurable scenarios.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File upload
    with st.expander("📁 Upload Data", expanded=not st.session_state.cfo_data_loaded):
        _handle_file_upload()

    # Scenario settings
    with st.expander("⚙️ Scenario Settings", expanded=False):
        _render_scenario_config()

    # Editable data tables
    data_tab1, data_tab2, data_tab3 = st.tabs(
        ["💼 Expenses", "🏗️ Investments", "📊 ROI Line Items"]
    )
    with data_tab1:
        _render_editable_expenses()
    with data_tab2:
        _render_editable_investments()
    with data_tab3:
        _render_editable_roi_model()

    # Build engine and render
    if st.session_state.cfo_roi_rows or st.session_state.cfo_expenses:
        engine = CFOROIEngine(
            expenses=st.session_state.cfo_expenses,
            investments=st.session_state.cfo_investments,
            roi_rows=st.session_state.cfo_roi_rows,
            scenario_a_name=st.session_state.cfo_scenario_a_name,
            scenario_b_name=st.session_state.cfo_scenario_b_name,
            scenario_a_one_time=st.session_state.cfo_scenario_a_one_time,
            scenario_b_bonus_pct=st.session_state.cfo_scenario_b_bonus_pct,
            y2_growth_pct=st.session_state.cfo_y2_growth_pct,
            y3_growth_pct=st.session_state.cfo_y3_growth_pct,
            overlap_pct=st.session_state.cfo_overlap_pct,
        )

        st.markdown("---")
        _render_summary_cards(engine)
        st.markdown("---")
        _render_charts(engine)
    else:
        st.info("Upload data or add rows manually to see ROI analysis.")
