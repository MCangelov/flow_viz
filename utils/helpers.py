"""
Helper Functions
Utility functions for the Flow Metrics Dashboard
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any
from pathlib import Path


def format_currency(value: float, decimals: int = 0, prefix: str = "$") -> str:
    """
    Format a number as currency.

    Args:
        value: The numeric value to format
        decimals: Number of decimal places
        prefix: Currency prefix (default: $)

    Returns:
        Formatted currency string
    """
    if value >= 1_000_000:
        return f"{prefix}{value/1_000_000:,.{decimals}f}M"
    elif value >= 1_000:
        return f"{prefix}{value/1_000:,.{decimals}f}K"
    else:
        return f"{prefix}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as percentage.

    Args:
        value: Decimal value (e.g., 0.15 for 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def load_excel_data(
    filepath: str, sheet_name: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Load data from an Excel file.

    Args:
        filepath: Path to the Excel file
        sheet_name: Specific sheet to load (default: first sheet)

    Returns:
        DataFrame or None if file not found
    """
    try:
        path = Path(filepath)
        if not path.exists():
            print(f"Warning: File not found: {filepath}")
            return None

        if sheet_name:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        else:
            df = pd.read_excel(filepath)

        return df

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None


def normalize_value(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = 0,
    target_max: float = 100,
) -> float:
    """
    Normalize a value from one range to another.

    Args:
        value: The value to normalize
        min_val: Minimum of original range
        max_val: Maximum of original range
        target_min: Minimum of target range
        target_max: Maximum of target range

    Returns:
        Normalized value
    """
    if max_val == min_val:
        return (target_min + target_max) / 2

    normalized = (value - min_val) / (max_val - min_val)
    return target_min + normalized * (target_max - target_min)


def calculate_percentile(value: float, data: List[float]) -> float:
    """
    Calculate the percentile of a value within a dataset.

    Args:
        value: The value to evaluate
        data: List of comparison values

    Returns:
        Percentile (0-100)
    """
    if not data:
        return 50.0

    sorted_data = sorted(data)
    count_below = sum(1 for x in sorted_data if x < value)
    return (count_below / len(sorted_data)) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division by zero

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max bounds.

    Args:
        value: The value to clamp
        min_val: Minimum bound
        max_val: Maximum bound

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def moving_average(data: List[float], window: int = 3) -> List[float]:
    """
    Calculate moving average of a data series.

    Args:
        data: List of numeric values
        window: Window size for averaging

    Returns:
        List of moving averages
    """
    if len(data) < window:
        return data

    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start : i + 1]) / (i - start + 1))

    return result


def calculate_trend(data: List[float]) -> Dict[str, float]:
    """
    Calculate trend statistics for a data series.

    Args:
        data: List of numeric values

    Returns:
        Dictionary with trend statistics
    """
    if len(data) < 2:
        return {"slope": 0, "direction": "stable", "change_pct": 0}

    # Simple linear regression
    x = np.arange(len(data))
    slope = np.polyfit(x, data, 1)[0]

    # Determine direction
    if slope > 0.01 * np.mean(data):
        direction = "increasing"
    elif slope < -0.01 * np.mean(data):
        direction = "decreasing"
    else:
        direction = "stable"

    # Calculate percentage change
    change_pct = ((data[-1] - data[0]) / data[0] * 100) if data[0] != 0 else 0

    return {"slope": slope, "direction": direction, "change_pct": change_pct}


def format_delta(
    current: float, previous: float, is_inverse: bool = False
) -> Dict[str, Any]:
    """
    Format the delta between two values for display.

    Args:
        current: Current value
        previous: Previous value
        is_inverse: If True, negative delta is good

    Returns:
        Dictionary with delta info
    """
    delta = current - previous
    delta_pct = (delta / previous * 100) if previous != 0 else 0

    # Determine if change is positive
    is_positive = delta > 0 if not is_inverse else delta < 0

    return {
        "delta": delta,
        "delta_pct": delta_pct,
        "is_positive": is_positive,
        "display": f"{delta:+.1f} ({delta_pct:+.1f}%)",
    }


def generate_sample_data(
    n_weeks: int = 52, base_throughput: float = 50, variability: float = 0.15
) -> pd.DataFrame:
    """
    Generate sample historical data for testing.

    Args:
        n_weeks: Number of weeks of data
        base_throughput: Average throughput
        variability: Coefficient of variation

    Returns:
        DataFrame with sample metrics
    """
    np.random.seed(42)

    weeks = list(range(1, n_weeks + 1))

    # Generate correlated metrics
    throughput = np.random.normal(
        base_throughput, base_throughput * variability, n_weeks
    )
    throughput = np.clip(throughput, base_throughput * 0.5, base_throughput * 1.5)

    # WIP follows throughput with lag
    wip = throughput * 14 / 7 + np.random.normal(0, 10, n_weeks)
    wip = np.clip(wip, 20, 200)

    # Cycle time from Little's Law with noise
    cycle_time = wip / (throughput / 7) + np.random.normal(0, 2, n_weeks)
    cycle_time = np.clip(cycle_time, 5, 30)

    # Flow efficiency
    flow_efficiency = np.random.normal(0.2, 0.05, n_weeks)
    flow_efficiency = np.clip(flow_efficiency, 0.05, 0.5)

    # Predictability inversely related to variability
    predictability = np.random.normal(0.75, 0.1, n_weeks)
    predictability = np.clip(predictability, 0.4, 0.95)

    return pd.DataFrame(
        {
            "week": weeks,
            "throughput": throughput,
            "cycle_time": cycle_time,
            "flow_efficiency": flow_efficiency,
            "wip": wip,
            "flow_predictability": predictability,
        }
    )


def validate_metric_value(metric_name: str, value: float) -> bool:
    """
    Validate that a metric value is within reasonable bounds.

    Args:
        metric_name: Name of the metric
        value: Value to validate

    Returns:
        True if valid, False otherwise
    """
    bounds = {
        "throughput": (0, 1000),
        "cycle_time": (0.1, 365),
        "flow_efficiency": (0, 1),
        "wip": (0, 10000),
        "flow_predictability": (0, 1),
    }

    if metric_name not in bounds:
        return True

    min_val, max_val = bounds[metric_name]
    return min_val <= value <= max_val


def calculate_weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted average.

    Args:
        values: List of values
        weights: List of weights (same length as values)

    Returns:
        Weighted average
    """
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0

    return sum(v * w for v, w in zip(values, weights)) / total_weight
