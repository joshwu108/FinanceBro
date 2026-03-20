"""Serializers — Convert pandas objects to JSON-serializable dicts.

All functions handle NaN, Inf, and Timestamp objects, rounding floats
to 6 decimal places for JSON safety.
"""

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def _clean_value(v: Any) -> Any:
    """Sanitize a single value for JSON serialization.

    - NaN / Inf → None
    - numpy scalars → Python scalars
    - Timestamps → ISO strings
    - Floats → rounded to 6 decimals
    """
    if v is None:
        return None

    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None

    if isinstance(v, (np.floating, np.complexfloating)):
        fval = float(v)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return round(fval, 6)

    if isinstance(v, np.integer):
        return int(v)

    if isinstance(v, np.bool_):
        return bool(v)

    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()

    if isinstance(v, date):
        return v.isoformat()

    if isinstance(v, np.ndarray):
        return [_clean_value(x) for x in v.tolist()]

    if isinstance(v, float):
        return round(v, 6)

    return v


def serialize_series(s: pd.Series) -> Dict[str, Any]:
    """Convert a pandas Series to {index: [...], values: [...]}.

    NaN and Inf values are converted to null (None).
    Float values are rounded to 6 decimal places.
    """
    if s is None or s.empty:
        return {"index": [], "values": []}

    return {
        "index": [_clean_value(idx) for idx in s.index],
        "values": [_clean_value(val) for val in s.values],
    }


def serialize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert a DataFrame to {columns: [...], index: [...], data: [[...]]}.

    NaN and Inf values are converted to null (None).
    Float values are rounded to 6 decimal places.
    """
    if df is None or df.empty:
        return {"columns": [], "index": [], "data": []}

    return {
        "columns": list(df.columns),
        "index": [_clean_value(idx) for idx in df.index],
        "data": [
            [_clean_value(val) for val in row]
            for row in df.values.tolist()
        ],
    }


def serialize_equity_curve(equity_curve: pd.Series) -> List[Dict[str, Any]]:
    """Convert an equity Series to [{date: "2020-01-02", value: 100123.45}, ...].

    Dates are formatted as ISO strings.  NaN/Inf values are converted to null.
    """
    if equity_curve is None or equity_curve.empty:
        return []

    result: List[Dict[str, Any]] = []
    for idx, val in equity_curve.items():
        date_str = _clean_value(idx)
        clean_val = _clean_value(val)
        result.append({"date": date_str, "value": clean_val})

    return result


def serialize_trade_log(trade_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean trade log dicts.

    - Timestamp values → ISO strings
    - Float values → rounded to 6 decimal places
    - NaN/Inf → null
    """
    if not trade_log:
        return []

    cleaned: List[Dict[str, Any]] = []
    for trade in trade_log:
        cleaned_trade = {k: _clean_value(v) for k, v in trade.items()}
        cleaned.append(cleaned_trade)

    return cleaned


def serialize_ohlcv(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert an OHLCV DataFrame to a list of row dicts.

    Returns:
        [{date: "2020-01-02", open: 74.06, high: 75.15, low: 73.80,
          close: 75.09, volume: 135480400}, ...]
    """
    if df is None or df.empty:
        return []

    result: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        entry: Dict[str, Any] = {"date": _clean_value(idx)}
        for col in ("open", "high", "low", "close", "volume"):
            if col in row.index:
                entry[col] = _clean_value(row[col])
        result.append(entry)

    return result


def serialize_weights(weights: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a portfolio weights DataFrame to a list of row dicts.

    Each row is a rebalance date with asset weights:
        [{date: "2020-01-02", AAPL: 0.5, MSFT: 0.5}, ...]
    """
    if weights is None or weights.empty:
        return []

    result: List[Dict[str, Any]] = []
    for idx, row in weights.iterrows():
        entry: Dict[str, Any] = {"date": _clean_value(idx)}
        for col in row.index:
            entry[str(col)] = _clean_value(row[col])
        result.append(entry)

    return result
