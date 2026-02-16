"""
Business layer: inventory reorder points, safety stock, and simulation.
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def safety_stock(
    lead_time_days: float,
    demand_std: float,
    service_level: float = 0.95,
) -> float:
    """
    Safety stock = z * sqrt(lead_time) * demand_std
    Assumes demand is normally distributed.
    """
    z = scipy_stats.norm.ppf(service_level)
    return z * np.sqrt(lead_time_days) * demand_std


def compute_reorder_point(
    forecast_demand: float,
    lead_time_days: float,
    demand_std: float,
    service_level: float = 0.95,
) -> float:
    """
    Reorder point = (forecast * lead_time) + safety_stock
    """
    ss = safety_stock(lead_time_days, demand_std, service_level)
    return forecast_demand * lead_time_days + ss


def simulate_inventory_policy(
    actuals: np.ndarray,
    initial_stock: float,
    lead_time_days: int,
    reorder_qty: float,
    reorder_point: float,
) -> dict:
    """
    Simulate inventory levels day-by-day. Reorder when stock <= reorder_point.
    Demand comes from actuals. Returns stock_levels, stockouts, orders_placed.
    """
    n = len(actuals)
    stock = initial_stock
    stock_levels = []
    stockouts = 0
    orders_in_transit = []  # (arrival_day, qty)
    orders_placed = 0

    for t in range(n):
        # Receive orders that arrive today
        while orders_in_transit and orders_in_transit[0][0] <= t:
            _, qty = orders_in_transit.pop(0)
            stock += qty
        # Demand
        demand = actuals[t]
        stock -= demand
        if stock < 0:
            stockouts += 1
            stock = 0
        stock_levels.append(stock)
        # Reorder if below ROP (one order at a time - no new order while one is in transit)
        if stock <= reorder_point and not orders_in_transit:
            orders_in_transit.append((t + lead_time_days, reorder_qty))
            orders_placed += 1

    return {
        "stock_levels": np.array(stock_levels),
        "stockout_days": stockouts,
        "stockout_rate": stockouts / n if n > 0 else 0,
        "orders_placed": orders_placed,
    }


def estimate_stockout_reduction(
    baseline_result: dict,
    optimized_result: dict,
) -> dict:
    """Compare baseline vs optimized inventory policy."""
    return {
        "stockout_reduction_pct": (baseline_result["stockout_rate"] - optimized_result["stockout_rate"])
        / max(baseline_result["stockout_rate"], 1e-8)
        * 100,
        "baseline_stockouts": baseline_result["stockout_days"],
        "optimized_stockouts": optimized_result["stockout_days"],
    }


def revenue_impact_estimation(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    margin_pct: float = 0.3,
) -> dict:
    """
    Estimate revenue impact of forecast accuracy.
    Simplified: assumes better forecasts reduce both overstock and stockout costs.
    """
    mae = np.mean(np.abs(actuals - forecasts))
    # Rough revenue-at-risk from forecast error (MAE as proxy for lost/misallocated sales)
    revenue_at_risk = mae * margin_pct
    return {
        "mae": mae,
        "revenue_at_risk_per_day": revenue_at_risk,
        "note": "Lower MAE implies better allocation and less revenue loss",
    }
