"""
CFO ROI Model Engine
Handles parsing of uploaded Expenses/Investment/ROI Model data,
auto-categorization, and all ROI calculations including multi-year
projections and multi-scenario support.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ExpenseRow:
    region: str
    fte_count: float
    cost_per_fte: float
    total: float
    notes: str = ""


@dataclass
class InvestmentRow:
    category: str
    annual_recurring: float
    one_time: float
    notes: str = ""


@dataclass
class ROIModelRow:
    category: str
    low: float
    high: float
    notes: str = ""
    group: str = "Other"  # Flow, AI, Business Outcomes, Overlap, Other
    resources: List[str] = field(default_factory=list)


CATEGORY_PREFIXES = {
    "Flow": ["flow:", "flow "],
    "AI": ["ai:", "ai "],
    "Business Outcomes": ["business"],
    "Overlap": ["overlap"],
}


def auto_detect_group(category_name: str) -> str:
    """Auto-detect ROI row group from category name prefix."""
    lower = category_name.strip().lower()
    for group, prefixes in CATEGORY_PREFIXES.items():
        for prefix in prefixes:
            if lower.startswith(prefix):
                return group
    return "Other"


class CFOROIEngine:
    """
    Engine for the CFO ROI Model.
    Parses uploaded data, computes totals, ROI, payback, and multi-year projections.
    """

    def __init__(
        self,
        expenses: List[ExpenseRow],
        investments: List[InvestmentRow],
        roi_rows: List[ROIModelRow],
        scenario_a_name: str = "Consultants",
        scenario_b_name: str = "Tryzens",
        scenario_a_one_time: float = 250_000,
        scenario_b_bonus_pct: float = 10.0,
        y2_growth_pct: float = 5.0,
        y3_growth_pct: float = 3.0,
        overlap_pct: float = 0.0,  # only used if no explicit Overlap row
    ):
        self.expenses = expenses
        self.investments = investments
        self.roi_rows = roi_rows

        self.scenario_a_name = scenario_a_name
        self.scenario_b_name = scenario_b_name
        self.scenario_a_one_time = scenario_a_one_time
        self.scenario_b_bonus_pct = scenario_b_bonus_pct

        self.y2_growth_pct = y2_growth_pct
        self.y3_growth_pct = y3_growth_pct
        self.overlap_pct = overlap_pct

    # ── Expense helpers ──────────────────────────────────────────────

    def total_engineering_cost(self) -> float:
        return sum(e.total for e in self.expenses)

    def expenses_dataframe(self) -> pd.DataFrame:
        if not self.expenses:
            return pd.DataFrame(columns=["Region", "FTE Count", "Cost per FTE ($)", "Total ($)", "Notes"])
        rows = [
            {
                "Region": e.region,
                "FTE Count": e.fte_count,
                "Cost per FTE ($)": e.cost_per_fte,
                "Total ($)": e.total,
                "Notes": e.notes,
            }
            for e in self.expenses
        ]
        return pd.DataFrame(rows)

    # ── Investment helpers ────────────────────────────────────────────

    def total_annual_investment(self) -> float:
        return sum(i.annual_recurring for i in self.investments)

    def total_one_time_investment(self) -> float:
        return sum(i.one_time for i in self.investments)

    def investments_dataframe(self) -> pd.DataFrame:
        if not self.investments:
            return pd.DataFrame(columns=["Category", "Annual Recurring ($)", "One-time ($)", "Notes"])
        rows = [
            {
                "Category": i.category,
                "Annual Recurring ($)": i.annual_recurring,
                "One-time ($)": i.one_time,
                "Notes": i.notes,
            }
            for i in self.investments
        ]
        return pd.DataFrame(rows)

    # ── ROI Model helpers ────────────────────────────────────────────

    def roi_rows_by_group(self) -> Dict[str, List[ROIModelRow]]:
        groups: Dict[str, List[ROIModelRow]] = {}
        for r in self.roi_rows:
            groups.setdefault(r.group, []).append(r)
        return groups

    def _sum_group(self, group_name: str, scenario: str) -> float:
        """Sum low or high values for a group. scenario = 'low' or 'high'."""
        return sum(
            (r.low if scenario == "low" else r.high)
            for r in self.roi_rows
            if r.group == group_name
        )

    def roi_model_dataframe(self) -> pd.DataFrame:
        if not self.roi_rows:
            return pd.DataFrame(columns=["Category", "Group", "Low ($)", "High ($)", "Notes"])
        rows = [
            {
                "Category": r.category,
                "Group": r.group,
                "Low ($)": r.low,
                "High ($)": r.high,
                "Notes": r.notes,
            }
            for r in self.roi_rows
        ]
        return pd.DataFrame(rows)

    # ── Core ROI calculations ────────────────────────────────────────

    def compute_summary(self, scenario: str = "low") -> Dict[str, float]:
        """
        Compute full ROI summary for a given scenario ('low' or 'high').
        Returns dict with all the key financial figures.
        """
        flow_total = self._sum_group("Flow", scenario)
        ai_total = self._sum_group("AI", scenario)
        business_total = self._sum_group("Business Outcomes", scenario)
        overlap_total = self._sum_group("Overlap", scenario)

        # If no explicit overlap rows, apply overlap_pct to (flow + ai)
        if not any(r.group == "Overlap" for r in self.roi_rows) and self.overlap_pct > 0:
            overlap_total = -(flow_total + ai_total) * (self.overlap_pct / 100)

        gross_roi = flow_total + ai_total + business_total + overlap_total
        annual_investment = self.total_annual_investment()
        net_annual_roi = gross_roi - annual_investment

        # Scenario A: fixed one-time cost
        scenario_a_one_time = self.scenario_a_one_time
        net_roi_y1_a = net_annual_roi - scenario_a_one_time

        # Scenario B: percentage bonus of net annual ROI
        scenario_b_bonus = net_annual_roi * (self.scenario_b_bonus_pct / 100)
        net_roi_y1_b = net_annual_roi - scenario_b_bonus

        # Year 2 & 3 projections (compounding growth on net annual ROI)
        net_roi_y2 = net_annual_roi * (1 + self.y2_growth_pct / 100)
        net_roi_y3 = net_roi_y2 * (1 + self.y3_growth_pct / 100)

        # Payback months
        monthly_roi = net_annual_roi / 12 if net_annual_roi > 0 else 0
        payback_a = (
            (scenario_a_one_time + annual_investment) / monthly_roi
            if monthly_roi > 0
            else float("inf")
        )
        payback_b = (
            (scenario_b_bonus + annual_investment) / monthly_roi
            if monthly_roi > 0
            else float("inf")
        )

        # ROI ratios
        total_cost_a = annual_investment + scenario_a_one_time
        total_cost_b = annual_investment + scenario_b_bonus
        roi_ratio_y1_a = net_roi_y1_a / total_cost_a if total_cost_a > 0 else 0
        roi_ratio_y1_b = net_roi_y1_b / total_cost_b if total_cost_b > 0 else 0
        roi_ratio_y2 = net_roi_y2 / annual_investment if annual_investment > 0 else 0
        roi_ratio_y3 = net_roi_y3 / annual_investment if annual_investment > 0 else 0

        return {
            "total_engineering_cost": self.total_engineering_cost(),
            "flow_total": flow_total,
            "ai_total": ai_total,
            "business_total": business_total,
            "overlap_total": overlap_total,
            "gross_roi": gross_roi,
            "annual_investment": annual_investment,
            "one_time_investment": self.total_one_time_investment(),
            "net_annual_roi": net_annual_roi,
            # Scenario A
            "scenario_a_one_time": scenario_a_one_time,
            "net_roi_y1_a": net_roi_y1_a,
            "payback_months_a": payback_a,
            "roi_ratio_y1_a": roi_ratio_y1_a,
            # Scenario B
            "scenario_b_bonus": scenario_b_bonus,
            "net_roi_y1_b": net_roi_y1_b,
            "payback_months_b": payback_b,
            "roi_ratio_y1_b": roi_ratio_y1_b,
            # Multi-year
            "net_roi_y2": net_roi_y2,
            "net_roi_y3": net_roi_y3,
            "roi_ratio_y2": roi_ratio_y2,
            "roi_ratio_y3": roi_ratio_y3,
        }

    def compute_both_scenarios(self) -> Dict[str, Dict[str, float]]:
        return {
            "low": self.compute_summary("low"),
            "high": self.compute_summary("high"),
        }


# ── File parsing ────────────────────────────────────────────────────


def parse_expenses_sheet(df: pd.DataFrame) -> List[ExpenseRow]:
    """Parse an Expenses DataFrame with flexible column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Try to find the right columns by partial match
    col_map = _find_columns(
        df.columns,
        {
            "region": ["region", "name", "vendor", "team", "location"],
            "fte_count": ["fte", "count", "headcount", "people", "staff"],
            "cost_per_fte": ["cost per", "rate", "salary", "cost/fte"],
            "total": ["total"],
            "notes": ["notes", "note", "comment"],
        },
    )

    rows = []
    for _, row in df.iterrows():
        region = str(row.get(col_map.get("region", ""), "") or "")
        if not region or region.lower().startswith("total"):
            continue

        fte = _to_float(row.get(col_map.get("fte_count", ""), 0))
        cost = _to_float(row.get(col_map.get("cost_per_fte", ""), 0))

        total_col = col_map.get("total", "")
        if total_col and pd.notna(row.get(total_col)):
            total = _to_float(row[total_col])
        else:
            total = fte * cost

        notes = str(row.get(col_map.get("notes", ""), "") or "")

        rows.append(ExpenseRow(region=region, fte_count=fte, cost_per_fte=cost, total=total, notes=notes))

    return rows


def parse_investment_sheet(df: pd.DataFrame) -> List[InvestmentRow]:
    """Parse an Investment DataFrame with flexible column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_map = _find_columns(
        df.columns,
        {
            "category": ["category", "item", "name", "tool", "description"],
            "annual": ["annual", "recurring", "yearly"],
            "one_time": ["one-time", "one time", "onetime", "setup"],
            "notes": ["notes", "note", "comment"],
        },
    )

    rows = []
    for _, row in df.iterrows():
        cat = str(row.get(col_map.get("category", ""), "") or "")
        if not cat or cat.lower().startswith("total"):
            continue

        annual = _to_float(row.get(col_map.get("annual", ""), 0))
        one_time = _to_float(row.get(col_map.get("one_time", ""), 0))
        notes = str(row.get(col_map.get("notes", ""), "") or "")

        rows.append(InvestmentRow(category=cat, annual_recurring=annual, one_time=one_time, notes=notes))

    return rows


def parse_roi_model_sheet(df: pd.DataFrame) -> List[ROIModelRow]:
    """Parse an ROI Model DataFrame with flexible column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_map = _find_columns(
        df.columns,
        {
            "category": ["category", "item", "name", "description"],
            "low": ["low", "conservative", "min"],
            "high": ["high", "optimistic", "max"],
            "notes": ["notes", "note", "comment"],
        },
    )

    # Check for resource columns (any column with "resource" or URLs)
    resource_cols = [i for i, c in enumerate(df.columns) if "resource" in str(c).lower()]

    rows = []
    for row_idx, row in df.iterrows():
        cat = str(row.iloc[df.columns.get_loc(col_map["category"])] if "category" in col_map else "")
        if not cat or cat == "nan":
            continue
        # Skip summary/total rows
        cat_lower = cat.lower()
        skip_prefixes = ["total", "gross", "net ", "payback", "flow roi", "ai roi"]
        if any(cat_lower.startswith(p) for p in skip_prefixes):
            continue

        low = _to_float(row.get(col_map.get("low", ""), 0))
        high = _to_float(row.get(col_map.get("high", ""), 0))
        notes = str(row.get(col_map.get("notes", ""), "") or "")
        if notes == "nan":
            notes = ""
        group = auto_detect_group(cat)

        resources = []
        for rc_idx in resource_cols:
            val = row.iloc[rc_idx]
            if isinstance(val, str) and val.startswith("http"):
                resources.append(val)

        rows.append(
            ROIModelRow(category=cat, low=low, high=high, notes=notes, group=group, resources=resources)
        )

    return rows


def _clean_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame from Excel that may have:
    - Data starting at column B (first col all NaN)
    - Headers in the first data row instead of the header row
    """
    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    # If all column names are 'Unnamed: N', the real headers are in the first row
    if all(str(c).startswith("Unnamed") for c in df.columns):
        # Find the first row that looks like a header (contains string values)
        for idx in range(min(5, len(df))):
            row = df.iloc[idx]
            non_null = row.dropna()
            if len(non_null) > 0 and all(isinstance(v, str) for v in non_null):
                df.columns = [str(v) if pd.notna(v) else f"col_{i}" for i, v in enumerate(row)]
                df = df.iloc[idx + 1:].reset_index(drop=True)
                break

    # Drop fully empty rows
    df = df.dropna(how="all")
    return df


def load_cfo_data_from_excel(filepath: str) -> Dict[str, Any]:
    """
    Load all three sheets from an Excel file.
    Returns dict with 'expenses', 'investments', 'roi_rows' lists.
    """
    xls = pd.ExcelFile(filepath)
    sheet_names_lower = {s.lower(): s for s in xls.sheet_names}

    result: Dict[str, Any] = {"expenses": [], "investments": [], "roi_rows": []}

    # Find sheets by partial name match
    for sn_lower, sn in sheet_names_lower.items():
        df = pd.read_excel(xls, sheet_name=sn)
        df = _clean_excel_df(df)

        if "expense" in sn_lower:
            result["expenses"] = parse_expenses_sheet(df)
        elif "invest" in sn_lower:
            result["investments"] = parse_investment_sheet(df)
        elif "roi" in sn_lower and "overview" not in sn_lower:
            result["roi_rows"] = parse_roi_model_sheet(df)

    return result


def load_cfo_data_from_csv(filepath: str, data_type: str) -> Any:
    """
    Load a single CSV file for one of the three data types.
    data_type: 'expenses', 'investments', 'roi_model'
    """
    df = pd.read_csv(filepath)
    df = df.dropna(how="all")

    if data_type == "expenses":
        return parse_expenses_sheet(df)
    elif data_type == "investments":
        return parse_investment_sheet(df)
    elif data_type == "roi_model":
        return parse_roi_model_sheet(df)
    return []


# ── Internal helpers ─────────────────────────────────────────────────


def _find_columns(columns: pd.Index, search_map: Dict[str, List[str]]) -> Dict[str, str]:
    """Fuzzy-match column names. Returns {logical_name: actual_column_name}."""
    result = {}
    cols_lower = {str(c).lower(): str(c) for c in columns}

    for logical, patterns in search_map.items():
        for pattern in patterns:
            for cl, co in cols_lower.items():
                if pattern in cl and logical not in result:
                    result[logical] = co
                    break
            if logical in result:
                break

    return result


def _to_float(val: Any) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        # Try stripping currency symbols
        cleaned = str(val).replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0
