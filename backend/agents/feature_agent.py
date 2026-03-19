"""FeatureAgent — Constructs predictive features from raw market data.
Wraps and hardens backend/app/services/feature_engineering.py.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ta

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FeatureAgent(BaseAgent):
    """Generates predictive features and targets from OHLCV data.

    All features use ONLY past data at each point in time.
    Target y(t+k) is strictly separated from feature matrix X(t).
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "target_horizons": [1, 5],
        "default_target": "fwd_direction_5d",
        "include_trend": True,
        "include_momentum": True,
        "include_volatility": True,
        "include_volume": True,
        "include_price_features": True,
        "include_time_features": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}
        self._feature_metadata: List[Dict[str, Any]] = []

    # ── BaseAgent contract ───────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "cleaned_data": (
                "pd.DataFrame with columns: open, high, low, close, volume; "
                "DatetimeIndex or 'date' column"
            ),
            "feature_config": "(optional) dict overriding DEFAULT_CONFIG keys",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "feature_matrix": (
                "pd.DataFrame — features only, no target columns, no raw OHLCV"
            ),
            "target": (
                "pd.Series — forward-looking target aligned to feature_matrix index"
            ),
            "feature_metadata": (
                "list[dict] — name, description, lookback_window per feature"
            ),
            "all_targets": (
                "pd.DataFrame — all computed target columns, aligned to feature_matrix"
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the feature engineering pipeline.

        Args:
            inputs: dict with 'cleaned_data' (DataFrame) and optional
                    'feature_config' (dict).

        Returns:
            dict with 'feature_matrix', 'target', 'feature_metadata',
            'all_targets'.
        """
        # Input validation
        if "cleaned_data" not in inputs:
            raise ValueError("inputs must contain 'cleaned_data'")
        cleaned_data = inputs["cleaned_data"]
        if not isinstance(cleaned_data, pd.DataFrame):
            raise TypeError(
                f"'cleaned_data' must be a pd.DataFrame, got {type(cleaned_data)}"
            )
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(cleaned_data.columns)
        if missing:
            raise ValueError(f"'cleaned_data' missing required columns: {missing}")

        # Resolve effective config locally (don't mutate self._config)
        effective_config = {
            **self._config,
            **(inputs.get("feature_config") or {}),
        }

        df = cleaned_data.copy()
        df = self._ensure_datetime_index(df)

        # Phase 1: Compute features (past-only rolling windows)
        self._feature_metadata = []
        df = self._add_features(df, effective_config)

        # Phase 2: Compute targets (forward-looking, kept separate)
        targets = self._compute_targets(df, effective_config)

        # Phase 3: Build feature matrix (exclude raw OHLCV and targets)
        feature_cols = [m["name"] for m in self._feature_metadata]
        feature_matrix = df[feature_cols].copy()

        # Phase 4: Replace inf with NaN
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

        # Phase 5: Align and drop NaN rows (NO ffill/bfill)
        target_col = effective_config["default_target"]
        target = targets[target_col]

        common_index = feature_matrix.index.intersection(target.index)
        feature_matrix = feature_matrix.loc[common_index]
        target = target.loc[common_index]

        valid_mask = feature_matrix.notna().all(axis=1) & target.notna()
        rows_before = len(feature_matrix)
        feature_matrix = feature_matrix.loc[valid_mask]
        target = target.loc[valid_mask]
        rows_dropped = rows_before - len(feature_matrix)

        # Align all targets to the same clean index
        all_targets = targets.loc[feature_matrix.index]

        outputs: Dict[str, Any] = {
            "feature_matrix": feature_matrix,
            "target": target,
            "feature_metadata": self._feature_metadata,
            "all_targets": all_targets,
        }

        # Phase 6: Validate
        self.validate(inputs, outputs)

        # Phase 7: Record metrics
        self._metrics = {
            "feature_count": len(feature_cols),
            "row_count": len(feature_matrix),
            "rows_dropped_nan": rows_dropped,
            "date_range_start": str(feature_matrix.index.min()),
            "date_range_end": str(feature_matrix.index.max()),
            "target_column": target_col,
            "target_horizons": effective_config["target_horizons"],
            "all_target_columns": list(all_targets.columns),
        }

        logger.info(
            "FeatureAgent complete: %d features, %d rows, %d dropped",
            len(feature_cols),
            len(feature_matrix),
            rows_dropped,
        )

        return outputs

    def latest_inference_features(
        self,
        cleaned_data: pd.DataFrame,
        feature_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.Timestamp, pd.Series]:
        """Return the latest timestamp and feature vector usable without a target.

        Uses the same feature definitions as ``run`` but does not require
        forward-looking labels, so the most recent bar with complete rolling
        windows can be scored for live inference.
        """
        if not isinstance(cleaned_data, pd.DataFrame):
            raise TypeError(
                f"'cleaned_data' must be a pd.DataFrame, got {type(cleaned_data)}"
            )
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(cleaned_data.columns)
        if missing:
            raise ValueError(f"'cleaned_data' missing required columns: {missing}")

        effective_config = {
            **self._config,
            **(feature_config or {}),
        }
        self._feature_metadata = []
        df = cleaned_data.copy()
        df = self._ensure_datetime_index(df)
        df = self._add_features(df, effective_config)

        feature_cols = [m["name"] for m in self._feature_metadata]
        feature_matrix = df[feature_cols].copy()
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

        valid = feature_matrix.dropna(how="any")
        if valid.empty:
            raise ValueError(
                "No row with complete features — need more history for indicators"
            )

        last_ts = pd.Timestamp(valid.index[-1])
        row = valid.iloc[-1]
        return last_ts, row

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Paranoid validation for leakage and data integrity."""
        X = outputs["feature_matrix"]
        y = outputs["target"]

        # 1. No NaN or inf in outputs
        if X.isna().any().any():
            raise ValueError("Feature matrix contains NaN")
        if not np.isfinite(X.values).all():
            raise ValueError("Feature matrix contains inf")
        if y.isna().any():
            raise ValueError("Target contains NaN")

        # 2. No target columns leaked into feature matrix
        target_keywords = ("target", "fwd_return", "fwd_direction")
        for col in X.columns:
            for kw in target_keywords:
                if kw in col:
                    raise ValueError(
                        f"Target column '{col}' found in feature matrix"
                    )

        # 3. No raw OHLCV in feature matrix
        raw_cols = {"open", "high", "low", "close", "volume"}
        leaked = raw_cols.intersection(set(X.columns))
        if leaked:
            raise ValueError(f"Raw OHLCV columns in features: {leaked}")

        # 4. Temporal ordering
        if not X.index.is_monotonic_increasing:
            raise ValueError("Feature index is not temporally ordered")

        # 5. High-correlation warning (potential leakage)
        correlations = X.corrwith(y).abs()
        suspicious = correlations[correlations > 0.95]
        if not suspicious.empty:
            logger.warning(
                "Features with >0.95 |correlation| to target (possible leakage): %s",
                suspicious.to_dict(),
            )

        # 6. Index alignment
        if not X.index.equals(y.index):
            raise ValueError("Feature matrix and target indices don't match")

        return True

    def log_metrics(self) -> None:
        """Persist metrics from the most recent run to experiments/."""
        if not self._metrics:
            logger.warning("No metrics to log — run() has not been called")
            return

        # Anchor to repo root, not cwd
        experiments_dir = Path(__file__).parent.parent / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc)
        log_entry = {
            "agent": "FeatureAgent",
            "timestamp": now.isoformat(),
            **self._metrics,
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"feature_agent_{ts}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    # ── Internal: ensure DatetimeIndex ───────────────────────────

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a sorted DatetimeIndex."""
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # ── Internal: feature computation (past-only) ────────────────

    def _add_features(
        self, df: pd.DataFrame, cfg: Dict[str, Any]
    ) -> pd.DataFrame:
        """Dispatch to all enabled feature categories."""
        if cfg["include_trend"]:
            df = self._add_trend_indicators(df)
        if cfg["include_momentum"]:
            df = self._add_momentum_indicators(df)
        if cfg["include_volatility"]:
            df = self._add_volatility_indicators(df)
        if cfg["include_volume"]:
            df = self._add_volume_indicators(df)
        if cfg["include_price_features"]:
            df = self._add_price_features(df)
        if cfg["include_time_features"]:
            df = self._add_time_features(df)
        return df

    def _register(self, name: str, description: str, lookback: int) -> None:
        """Register a feature in metadata."""
        self._feature_metadata.append({
            "name": name,
            "description": description,
            "lookback_window": lookback,
        })

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-following indicators (past-only rolling windows).

        Raw SMA/EMA price levels are computed internally but NOT registered
        as features — only their normalized ratios (close/sma, close/ema)
        are used as features (see _add_price_features). This avoids
        non-stationary absolute price levels in the feature matrix.
        """
        # Compute raw SMAs for internal use (ratios computed in _add_price_features)
        for window in (5, 10, 20, 50, 200):
            df[f"sma_{window}"] = ta.trend.sma_indicator(
                df["close"], window=window
            )

        # Register normalized SMA ratios as features
        for window in (5, 10, 20, 50, 200):
            col = f"close_sma{window}_ratio"
            df[col] = df["close"] / df[f"sma_{window}"].replace(0, np.nan)
            self._register(col, f"Close / SMA-{window} ratio", window)

        # Compute raw EMAs for internal use
        df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
        df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)

        # MACD is already a difference of EMAs — stationary
        df["macd"] = ta.trend.macd(df["close"])
        self._register("macd", "MACD line", 26)

        df["macd_signal"] = ta.trend.macd_signal(df["close"])
        self._register("macd_signal", "MACD signal line", 35)

        df["macd_diff"] = ta.trend.macd_diff(df["close"])
        self._register("macd_diff", "MACD histogram", 35)

        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
        self._register("adx", "Average Directional Index", 14)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators (past-only)."""
        df["rsi"] = ta.momentum.rsi(df["close"])
        self._register("rsi", "Relative Strength Index (14d)", 14)

        df["stoch"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
        self._register("stoch", "Stochastic Oscillator %K", 14)

        df["stoch_signal"] = ta.momentum.stoch_signal(
            df["high"], df["low"], df["close"]
        )
        self._register("stoch_signal", "Stochastic Oscillator %D", 17)

        df["williams_r"] = ta.momentum.williams_r(
            df["high"], df["low"], df["close"]
        )
        self._register("williams_r", "Williams %R", 14)

        df["roc"] = ta.momentum.roc(df["close"])
        self._register("roc", "Rate of Change", 12)

        df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])
        self._register("cci", "Commodity Channel Index", 20)

        df["mfi"] = ta.volume.money_flow_index(
            df["high"], df["low"], df["close"], df["volume"]
        )
        self._register("mfi", "Money Flow Index", 14)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators (past-only rolling windows).

        Bollinger Band and Keltner Channel absolute levels are NOT
        registered as features — only normalized/relative versions are
        used to avoid non-stationary price-level features.
        """
        # Compute raw bands internally
        bb_upper = ta.volatility.bollinger_hband(df["close"])
        bb_lower = ta.volatility.bollinger_lband(df["close"])

        # Normalized Bollinger Band features
        df["bb_width"] = (bb_upper - bb_lower) / df["close"]
        self._register("bb_width", "Bollinger Band width / close", 20)

        bb_range = bb_upper - bb_lower
        df["bb_position"] = (
            (df["close"] - bb_lower) / bb_range.replace(0, np.nan)
        )
        self._register("bb_position", "Price position within Bollinger Bands", 20)

        df["close_to_bb_upper"] = (bb_upper - df["close"]) / df["close"]
        self._register("close_to_bb_upper", "Distance to BB upper / close", 20)

        df["close_to_bb_lower"] = (df["close"] - bb_lower) / df["close"]
        self._register("close_to_bb_lower", "Distance to BB lower / close", 20)

        # ATR normalized by close
        raw_atr = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"]
        )
        df["atr_pct"] = raw_atr / df["close"]
        self._register("atr_pct", "ATR as percentage of close", 14)

        # Normalized Keltner Channel features
        kc_upper = ta.volatility.keltner_channel_hband(
            df["high"], df["low"], df["close"]
        )
        kc_lower = ta.volatility.keltner_channel_lband(
            df["high"], df["low"], df["close"]
        )
        df["kc_width"] = (kc_upper - kc_lower) / df["close"]
        self._register("kc_width", "Keltner Channel width / close", 20)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators (past-only).

        OBV and ADL are cumulative since inception — non-stationary and
        leak dataset-boundary information.  We register their rate-of-change
        instead (stationary).
        """
        # Volume relative to its own moving average (normalized)
        vol_sma_5 = df["volume"].rolling(window=5).mean()
        vol_sma_20 = df["volume"].rolling(window=20).mean()
        df["volume_ratio_5"] = df["volume"] / vol_sma_5.replace(0, np.nan)
        self._register("volume_ratio_5", "Volume / 5d volume SMA", 5)

        df["volume_ratio_20"] = df["volume"] / vol_sma_20.replace(0, np.nan)
        self._register("volume_ratio_20", "Volume / 20d volume SMA", 20)

        # OBV rate-of-change (stationary) instead of raw cumulative OBV
        obv = ta.volume.on_balance_volume(df["close"], df["volume"])
        df["obv_roc"] = obv.pct_change()
        self._register("obv_roc", "OBV rate of change (1d)", 2)

        df["volume_roc"] = df["volume"].pct_change()
        self._register("volume_roc", "Volume Rate of Change (1d)", 1)

        # ADL rate-of-change (stationary) instead of raw cumulative ADL
        adl = ta.volume.acc_dist_index(
            df["high"], df["low"], df["close"], df["volume"]
        )
        df["adl_roc"] = adl.pct_change()
        self._register("adl_roc", "A/D Line rate of change (1d)", 2)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-derived features (past-only).

        Note: rolling std is computed on *returns* not price levels,
        because returns are closer to stationary.
        """
        # Past returns — positive period = backward-looking
        df["return_1d"] = df["close"].pct_change(periods=1)
        self._register("return_1d", "1-day past return", 1)

        df["return_2d"] = df["close"].pct_change(periods=2)
        self._register("return_2d", "2-day past return", 2)

        df["return_5d"] = df["close"].pct_change(periods=5)
        self._register("return_5d", "5-day past return", 5)

        df["return_20d"] = df["close"].pct_change(periods=20)
        self._register("return_20d", "20-day past return", 20)

        # Intraday spreads
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
        self._register("hl_spread", "High-Low spread / close", 0)

        df["oc_spread"] = (
            (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        )
        self._register("oc_spread", "Open-Close spread / open", 0)

        hl_range = df["high"] - df["low"]
        df["price_position"] = (
            (df["close"] - df["low"]) / hl_range.replace(0, np.nan)
        )
        self._register("price_position", "Close position within day range", 0)

        # Rolling volatility of RETURNS (reuse return_1d, not a fresh pct_change)
        df["return_std_5"] = df["return_1d"].rolling(window=5).std()
        self._register("return_std_5", "5-day return volatility", 5)

        df["return_std_20"] = df["return_1d"].rolling(window=20).std()
        self._register("return_std_20", "20-day return volatility", 20)

        # EMA ratios (only if trend indicators were computed)
        if "ema_12" in df.columns:
            df["close_ema12_ratio"] = (
                df["close"] / df["ema_12"].replace(0, np.nan)
            )
            self._register("close_ema12_ratio", "Close / EMA-12 ratio", 12)

        if "ema_26" in df.columns:
            df["close_ema26_ratio"] = (
                df["close"] / df["ema_26"].replace(0, np.nan)
            )
            self._register("close_ema26_ratio", "Close / EMA-26 ratio", 26)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar features (no leakage risk).

        Removed days_since_start — it leaks dataset boundary information.
        Removed is_weekend — markets are closed on weekends.
        """
        df["day_of_week"] = df.index.dayofweek
        self._register("day_of_week", "Day of week (0=Mon, 4=Fri)", 0)

        df["month"] = df.index.month
        self._register("month", "Month of year", 0)

        df["quarter"] = df.index.quarter
        self._register("quarter", "Quarter of year", 0)

        return df

    # ── Internal: target construction (forward-looking) ──────────

    def _compute_targets(
        self, df: pd.DataFrame, cfg: Dict[str, Any]
    ) -> pd.DataFrame:
        """Compute forward-looking targets.

        Uses shift(-k) to look ahead — this is correct ONLY for target
        construction.  These columns must NEVER appear in the feature
        matrix.

        Forward return formula:
            fwd_return_kd(t) = (price(t+k) - price(t)) / price(t)

        IMPORTANT: fwd_direction preserves NaN where fwd_return is NaN
        (the last k rows have no future data).  A naive ``(series > 0).astype(int)``
        would silently convert NaN → False → 0, corrupting labels.
        """
        targets: Dict[str, pd.Series] = {}
        for k in cfg["target_horizons"]:
            future_price = df["close"].shift(-k)
            fwd_return = (future_price - df["close"]) / df["close"]
            targets[f"fwd_return_{k}d"] = fwd_return
            # Preserve NaN at tail boundary — do NOT cast NaN to 0
            direction = pd.Series(np.nan, index=df.index)
            valid = fwd_return.notna()
            direction.loc[valid] = (fwd_return.loc[valid] > 0).astype(int)
            targets[f"fwd_direction_{k}d"] = direction

        return pd.DataFrame(targets, index=df.index)
