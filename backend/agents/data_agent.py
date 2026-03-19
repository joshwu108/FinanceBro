"""DataAgent — Fetches, validates, and persists historical OHLCV data.
Works with backend/app/services/data_collector.py.

constraints:
  - No forward-filling (ffill) of prices — drop NaN rows instead
  - No survivorship bias — track and report failed/delisted symbols
  - Flag daily returns >5% as potential anomalies
  - All symbols aligned to common date index (intersection, not union)
  - Sorted DatetimeIndex, no missing values in final output
"""

import json
import logging
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DataAgent(BaseAgent):
    """Fetches and cleans historical OHLCV data for the quantitative finance pipeline.
    All output DataFrames have:
      - Columns: open, high, low, close, volume
      - DatetimeIndex sorted in ascending order, unique timestamps
      - No NaN or inf values
      - No forward-filled gaps
      - Valid OHLC bar invariants (high >= low, etc.)
    """
    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
    DEFAULT_CONFIG: Dict[str, Any] = {
        "data_dir": "data",
        "max_retries": 3,
        "retry_backoff_base": 2.0,
        "period": "2y",
        "interval": "1d",
        "anomaly_threshold_pct": 5.0,
        "min_rows": 30,
        "gap_calendar_days_threshold": 5,
        "alignment_drop_warn_pct": 20.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config: Dict[str, Any] = {**self.DEFAULT_CONFIG, **(config or {})}
        self._metrics: Dict[str, Any] = {}

    # ── BaseAgent contract ───────────────────────────────────────

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "symbols": "list[str] — ticker symbols to fetch",
            "start_date": "(optional) str — ISO date, e.g. '2022-01-01'",
            "end_date": "(optional) str — ISO date, e.g. '2023-12-31'",
            "config": "(optional) dict overriding DEFAULT_CONFIG keys",
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "cleaned_data": (
                "dict[str, pd.DataFrame] — symbol -> OHLCV DataFrame "
                "with DatetimeIndex, columns: open, high, low, close, volume"
            ),
            "data_quality_report": (
                "dict — symbols_requested, symbols_fetched, symbols_failed, "
                "per_symbol quality metrics"
            ),
        }

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch, validate, align, and persist OHLCV data.

        Args:
            inputs: dict with 'symbols' (required), optional 'start_date',
                    'end_date', 'config'.

        Returns:
            dict with 'cleaned_data' and 'data_quality_report'.
        """
        run_id = uuid.uuid4().hex[:12]

        # ── Input validation ──
        symbols = inputs.get("symbols")
        if not symbols:
            raise ValueError("inputs must contain non-empty 'symbols' list")

        effective_config = {**self._config, **(inputs.get("config") or {})}
        start_date = inputs.get("start_date")
        end_date = inputs.get("end_date")

        # Validate date format if provided
        parsed_start = None
        parsed_end = None
        if start_date:
            try:
                parsed_start = date.fromisoformat(start_date)
            except ValueError:
                raise ValueError(
                    f"start_date must be ISO format (YYYY-MM-DD), got: {start_date!r}"
                )
        if end_date:
            try:
                parsed_end = date.fromisoformat(end_date)
            except ValueError:
                raise ValueError(
                    f"end_date must be ISO format (YYYY-MM-DD), got: {end_date!r}"
                )
        if parsed_start and parsed_end and parsed_start >= parsed_end:
            raise ValueError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )

        # ── Phase 1: Fetch raw data per symbol ──
        # NOTE: Fetching is sequential. For large symbol lists, consider
        # wrapping with concurrent.futures.ThreadPoolExecutor.
        raw_data: Dict[str, pd.DataFrame] = {}
        failed_symbols: Set[str] = set()

        for symbol in symbols:
            df = self._fetch_symbol(
                symbol,
                start_date=start_date,
                end_date=end_date,
                period=effective_config["period"],
                interval=effective_config["interval"],
                max_retries=effective_config["max_retries"],
                backoff_base=effective_config["retry_backoff_base"],
            )
            if df is not None and not df.empty:
                raw_data[symbol] = df
            else:
                failed_symbols.add(symbol)
                logger.warning(
                    "Symbol %s returned no data — tracked as survivorship bias risk",
                    symbol,
                )

        # ── Phase 2: Normalize and validate each symbol ──
        cleaned: Dict[str, pd.DataFrame] = {}
        per_symbol_reports: Dict[str, Dict[str, Any]] = {}

        for symbol, df in raw_data.items():
            # Check for stock splits before discarding extra columns
            split_warning = self._check_splits(df, symbol)

            # Record source timezone before stripping
            source_tz = self._get_source_tz(df)

            try:
                normalized = self._normalize(df)
                self._check_positive_prices(normalized, symbol)
            except ValueError as exc:
                logger.warning(
                    "%s: skipped during normalization — %s", symbol, exc
                )
                failed_symbols.add(symbol)
                continue

            anomalies = self._detect_anomalies(
                normalized, effective_config["anomaly_threshold_pct"]
            )
            gaps = self._detect_gaps(
                normalized, effective_config["gap_calendar_days_threshold"]
            )
            missing_bdays = self._detect_missing_business_days(normalized)

            # Drop NaN rows — NO ffill/bfill
            rows_before = len(normalized)
            normalized = normalized.dropna()
            rows_dropped = rows_before - len(normalized)

            if rows_dropped > 0:
                logger.info(
                    "%s: dropped %d rows with NaN (no forward-fill)",
                    symbol,
                    rows_dropped,
                )

            if len(normalized) < effective_config["min_rows"]:
                logger.warning(
                    "%s: only %d rows after cleaning (min %d) — skipping",
                    symbol,
                    len(normalized),
                    effective_config["min_rows"],
                )
                failed_symbols.add(symbol)
                continue

            cleaned[symbol] = normalized
            per_symbol_reports[symbol] = {
                "rows": len(normalized),
                "rows_dropped_nan": rows_dropped,
                "date_range_start": str(normalized.index.min().date()),
                "date_range_end": str(normalized.index.max().date()),
                "source_timezone": source_tz,
                "missing_day_gaps": gaps,
                "missing_business_days": missing_bdays,
                "anomalies": anomalies,
                "anomaly_count": len(anomalies),
                "split_warning": split_warning,
            }

        # ── Phase 3: Align multi-symbol date indices ──
        survivorship_warnings: List[str] = []

        if len(cleaned) > 1:
            cleaned, alignment_warnings = self._align_indices(
                cleaned, effective_config["alignment_drop_warn_pct"]
            )
            survivorship_warnings.extend(alignment_warnings)
            for symbol in cleaned:
                per_symbol_reports[symbol]["rows_after_alignment"] = len(
                    cleaned[symbol]
                )

        # Check for empty dataset after all filtering
        if not cleaned:
            raise ValueError(
                f"All symbols failed — no data available for pipeline. "
                f"Failed: {sorted(failed_symbols)}"
            )

        # ── Phase 4: Save to parquet ──
        data_dir = Path(effective_config["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        for symbol, df in cleaned.items():
            path = data_dir / f"{symbol}.parquet"
            if path.exists():
                logger.warning("Overwriting existing parquet: %s", path)
            df.to_parquet(path)
            logger.info("Saved %s to %s (%d rows)", symbol, path, len(df))

        # ── Phase 5: Build quality report ──
        failed_list = sorted(failed_symbols)

        # NOTE: Survivorship bias from the input symbol list itself is NOT
        # checked. The caller is responsible for providing a point-in-time
        # constituent list (e.g., historical S&P 500 members).
        if survivorship_warnings:
            survivorship_warnings.append(
                "NOTE: Input symbol list bias is not checked — ensure "
                "point-in-time constituent lists are used for backtesting."
            )

        quality_report: Dict[str, Any] = {
            "run_id": run_id,
            "symbols_requested": list(symbols),
            "symbols_fetched": list(cleaned.keys()),
            "symbols_failed": failed_list,
            "survivorship_bias_warnings": survivorship_warnings,
            "per_symbol": per_symbol_reports,
        }

        outputs: Dict[str, Any] = {
            "cleaned_data": cleaned,
            "data_quality_report": quality_report,
        }

        # ── Phase 6: Validate ──
        self.validate(inputs, outputs)

        # ── Phase 7: Record metrics and log experiment ──
        self._metrics = {
            "run_id": run_id,
            "symbols_requested": len(symbols),
            "symbols_fetched": len(cleaned),
            "symbols_failed": len(failed_list),
            "failed_list": failed_list,
            "total_rows": sum(len(df) for df in cleaned.values()),
            "total_anomalies": sum(
                r["anomaly_count"] for r in per_symbol_reports.values()
            ),
            "symbols": list(symbols),
        }

        logger.info(
            "DataAgent complete: %d/%d symbols fetched, %d total rows",
            len(cleaned),
            len(symbols),
            self._metrics["total_rows"],
        )

        self.log_metrics()

        return outputs

    def validate(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Validate data integrity of outputs."""
        cleaned = outputs["cleaned_data"]
        report = outputs["data_quality_report"]

        required_report_keys = {
            "symbols_requested", "symbols_fetched", "symbols_failed", "per_symbol",
        }
        missing_report_keys = required_report_keys - set(report.keys())
        if missing_report_keys:
            raise ValueError(
                f"data_quality_report missing required keys: {missing_report_keys}"
            )

        for symbol, df in cleaned.items():
            # 1. Correct columns
            missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"{symbol}: missing columns {missing}")

            extra = set(df.columns) - set(self.REQUIRED_COLUMNS)
            if extra:
                raise ValueError(f"{symbol}: unexpected columns {extra}")

            # 2. DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{symbol}: index is not DatetimeIndex")

            # 3. Sorted
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"{symbol}: index not sorted ascending")

            # 4. No duplicate timestamps
            if not df.index.is_unique:
                raise ValueError(f"{symbol}: duplicate timestamps detected")

            # 5. No NaN
            if df.isna().any().any():
                raise ValueError(f"{symbol}: contains NaN values")

            # 6. No inf
            if not np.isfinite(df.values).all():
                raise ValueError(f"{symbol}: contains inf values")

            # 7. Positive prices
            for col in ("open", "high", "low", "close"):
                if (df[col] <= 0).any():
                    raise ValueError(f"{symbol}: non-positive values in {col}")

            # 8. Non-negative volume
            if (df["volume"] < 0).any():
                raise ValueError(f"{symbol}: negative volume detected")

            # 9. OHLC bar invariants
            if not (df["high"] >= df["low"]).all():
                raise ValueError(f"{symbol}: high < low detected")
            if not (df["high"] >= df["open"]).all():
                raise ValueError(f"{symbol}: high < open detected")
            if not (df["high"] >= df["close"]).all():
                raise ValueError(f"{symbol}: high < close detected")
            if not (df["low"] <= df["open"]).all():
                raise ValueError(f"{symbol}: low > open detected")
            if not (df["low"] <= df["close"]).all():
                raise ValueError(f"{symbol}: low > close detected")

        # 10. If multiple symbols, indices must match
        if len(cleaned) > 1:
            reference_idx = next(iter(cleaned.values())).index
            for symbol, df in cleaned.items():
                if not df.index.equals(reference_idx):
                    raise ValueError(
                        f"{symbol}: index does not match reference after alignment"
                    )

        return True

    def log_metrics(self) -> None:
        """Persist metrics from the most recent run to experiments/.

        Conforms to the experiment template structure with DataAgent-specific
        metrics under the 'metrics' key.
        """
        if not self._metrics:
            logger.warning("No metrics to log — run() has not been called")
            return

        experiments_dir = Path(__file__).parent.parent / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        now = datetime.now(timezone.utc)
        run_id = self._metrics.get("run_id", uuid.uuid4().hex[:12])

        log_entry = {
            "experiment_id": f"data_{run_id}",
            "date": now.strftime("%Y-%m-%d"),
            "agent": "DataAgent",
            "stage": "data",
            "timestamp": now.isoformat(),
            "symbols": self._metrics.get("symbols", []),
            "model": None,
            "features": [],
            "parameters": {
                "transaction_costs_bps": None,
                "slippage_bps": None,
                "max_position_size": None,
                "benchmark": None,
            },
            "out_of_sample": None,
            "walk_forward_folds": None,
            "metrics": {
                "symbols_requested": self._metrics.get("symbols_requested", 0),
                "symbols_fetched": self._metrics.get("symbols_fetched", 0),
                "symbols_failed": self._metrics.get("symbols_failed", 0),
                "total_rows": self._metrics.get("total_rows", 0),
                "total_anomalies": self._metrics.get("total_anomalies", 0),
            },
            "benchmark_comparison": None,
            "overfitting_score": None,
            "statistical_significance": None,
            "failed_list": self._metrics.get("failed_list", []),
            "notes": "DataAgent data fetch and clean run",
        }

        ts = now.strftime("%Y%m%d_%H%M%S")
        log_path = experiments_dir / f"data_agent_{ts}.json"
        log_path.write_text(json.dumps(log_entry, indent=2, default=str))
        logger.info("Metrics logged to %s", log_path)

    # ── Internal: fetch a single symbol ──────────────────────────

    def _fetch_symbol(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str,
        interval: str,
        max_retries: int,
        backoff_base: float,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from yfinance with retry and exponential backoff."""
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Fetching %s (attempt %d/%d)", symbol, attempt + 1, max_retries
                )
                ticker = yf.Ticker(symbol)

                if start_date and end_date:
                    data = ticker.history(
                        start=start_date, end=end_date, interval=interval
                    )
                else:
                    data = ticker.history(period=period, interval=interval)

                if data is not None and not data.empty:
                    logger.info("Fetched %d rows for %s", len(data), symbol)
                    return data

                logger.warning(
                    "No data returned for %s (attempt %d)", symbol, attempt + 1
                )
            except Exception as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Error fetching %s (attempt %d/%d), retrying: %s: %s",
                        symbol, attempt + 1, max_retries,
                        type(exc).__name__, exc,
                    )
                else:
                    logger.error(
                        "Final retry exhausted for %s: %s: %s",
                        symbol, type(exc).__name__, exc,
                    )

            if attempt < max_retries - 1:
                sleep_time = min(backoff_base ** attempt, 30.0)
                time.sleep(sleep_time)

        return None

    # ── Internal: check for unadjusted stock splits ──────────────

    @staticmethod
    def _check_splits(df: pd.DataFrame, symbol: str) -> Optional[str]:
        """Check if raw data contains stock splits that may not be adjusted.

        Returns a warning string if splits are detected, None otherwise.
        """
        cols_lower = {c.lower(): c for c in df.columns}
        splits_col = cols_lower.get("stock splits")
        if splits_col is None:
            return None

        splits = df[splits_col]
        nonzero_splits = splits[splits != 0]
        if nonzero_splits.empty:
            return None

        split_dates = [
            str(d.date()) if hasattr(d, "date") else str(d)
            for d in nonzero_splits.index[:5]
        ]
        warning = (
            f"{symbol}: {len(nonzero_splits)} stock split(s) detected on "
            f"{split_dates}. Verify data is split-adjusted."
        )
        logger.warning(warning)
        return warning

    # ── Internal: extract source timezone ────────────────────────

    @staticmethod
    def _get_source_tz(df: pd.DataFrame) -> Optional[str]:
        """Extract timezone info from a DataFrame's index before stripping."""
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            return str(df.index.tz)
        return None

    # ── Internal: normalize raw yfinance output ──────────────────

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize yfinance output to standard OHLCV schema.

        - Lowercase column names
        - Keep only OHLCV columns
        - Deduplicate timestamps (keep first)
        - Ensure DatetimeIndex sorted ascending
        - Replace inf with NaN (will be dropped later)
        """
        result = df.copy()

        # Lowercase columns
        result.columns = [c.lower() for c in result.columns]

        # Keep only required columns
        available = [c for c in self.REQUIRED_COLUMNS if c in result.columns]
        if set(available) != set(self.REQUIRED_COLUMNS):
            missing = set(self.REQUIRED_COLUMNS) - set(available)
            raise ValueError(f"Raw data missing required columns: {missing}")

        result = result[list(self.REQUIRED_COLUMNS)]

        # Ensure DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)

        # Remove timezone info for consistency (convert to UTC first)
        if result.index.tz is not None:
            result.index = result.index.tz_convert("UTC").tz_localize(None)

        # Deduplicate timestamps (keep first occurrence)
        if not result.index.is_unique:
            dupes = result.index[result.index.duplicated()].tolist()
            logger.warning(
                "Dropping %d duplicate timestamp(s): %s",
                len(dupes),
                dupes[:5],
            )
            result = result[~result.index.duplicated(keep="first")]

        # Sort ascending
        result = result.sort_index()

        # Replace inf with NaN (will be dropped, not filled)
        result = result.replace([np.inf, -np.inf], np.nan)

        return result

    # ── Internal: positive price check ───────────────────────────

    def _check_positive_prices(self, df: pd.DataFrame, symbol: str) -> None:
        """Raise if any OHLC price is non-positive (before NaN drop)."""
        for col in ("open", "high", "low", "close"):
            series = df[col].dropna()
            if (series <= 0).any():
                bad_dates = series[series <= 0].index.tolist()
                raise ValueError(
                    f"{symbol}: non-positive prices in '{col}' on {bad_dates[:5]}"
                )

    # ── Internal: anomaly detection ──────────────────────────────

    def _detect_anomalies(
        self, df: pd.DataFrame, threshold_pct: float
    ) -> List[Dict[str, Any]]:
        """Flag daily close-to-close returns exceeding threshold_pct."""
        returns = df["close"].pct_change().abs()
        threshold = threshold_pct / 100.0
        anomalous = returns[returns > threshold].dropna()

        anomalies = []
        for date_val, ret in anomalous.items():
            anomalies.append({
                "date": str(date_val.date()) if hasattr(date_val, "date") else str(date_val),
                "return_pct": round(float(ret) * 100, 2),
            })

        if anomalies:
            logger.warning(
                "Detected %d anomalous daily returns (>%.1f%%): %s",
                len(anomalies),
                threshold_pct,
                [a["date"] for a in anomalies[:5]],
            )

        return anomalies

    # ── Internal: gap detection ──────────────────────────────────

    def _detect_gaps(
        self, df: pd.DataFrame, calendar_days_threshold: int
    ) -> List[Dict[str, Any]]:
        """Detect gaps in trading dates exceeding threshold calendar days.

        Normal weekends (2 calendar days) and typical holidays (3 days)
        are expected.  Gaps beyond the threshold are flagged.

        NOTE: This uses a calendar-day heuristic. For production-grade gap
        detection, integrate an exchange calendar library (e.g.,
        ``exchange_calendars`` or ``pandas_market_calendars``).
        """
        if len(df) < 2:
            return []

        date_diffs = df.index[1:] - df.index[:-1]
        gaps = []

        for i, diff in enumerate(date_diffs):
            days = diff.days
            if days >= calendar_days_threshold:
                gaps.append({
                    "from_date": str(df.index[i].date()),
                    "to_date": str(df.index[i + 1].date()),
                    "calendar_days": days,
                })

        if gaps:
            logger.info("Detected %d date gaps exceeding %d days", len(gaps), calendar_days_threshold)

        return gaps

    def _detect_missing_business_days(
        self, df: pd.DataFrame
    ) -> List[str]:
        """Detect weekdays missing from the index (excludes weekends).

        Uses pandas business day range as an approximation of trading days.
        Does not account for exchange holidays — use ``exchange_calendars``
        for production-grade detection.
        """
        if len(df) < 2:
            return []

        # Normalize to midnight for business day comparison
        actual_dates = df.index.normalize()
        expected = pd.bdate_range(start=actual_dates.min(), end=actual_dates.max())
        missing = expected.difference(actual_dates)

        missing_dates = [str(d.date()) for d in missing]

        if missing_dates:
            logger.info(
                "Detected %d missing business days (may include exchange holidays): %s...",
                len(missing_dates),
                missing_dates[:5],
            )

        return missing_dates

    # ── Internal: multi-symbol alignment ─────────────────────────

    def _align_indices(
        self,
        data: Dict[str, pd.DataFrame],
        drop_warn_pct: float = 20.0,
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Align all symbols to their common date intersection.

        Does NOT forward-fill — dates without data for all symbols are
        simply excluded.

        Returns:
            Tuple of (aligned_data, survivorship_warnings).
        """
        indices = [df.index for df in data.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)

        if len(common_index) == 0:
            raise ValueError(
                "No common trading dates across symbols — cannot align"
            )

        aligned = {}
        warnings: List[str] = []

        # Check for symbols with significantly shorter histories
        max_len = max(len(df) for df in data.values())
        for symbol, df in data.items():
            if len(df) < max_len * 0.5:
                warn_msg = (
                    f"{symbol}: only {len(df)} rows vs max {max_len} — "
                    f"possible partial delisting or late listing"
                )
                warnings.append(warn_msg)
                logger.warning(warn_msg)

        for symbol, df in data.items():
            original_len = len(df)
            aligned[symbol] = df.loc[common_index].copy()
            dropped = original_len - len(aligned[symbol])

            if dropped > 0:
                drop_pct = (dropped / original_len) * 100
                logger.info(
                    "%s: dropped %d rows during alignment (%d -> %d, %.1f%%)",
                    symbol,
                    dropped,
                    original_len,
                    len(aligned[symbol]),
                    drop_pct,
                )

                if drop_pct >= drop_warn_pct:
                    warn_msg = (
                        f"{symbol}: alignment dropped {drop_pct:.1f}% of data "
                        f"({dropped}/{original_len} rows) — possible partial "
                        f"delisting or data gap. Review for survivorship bias."
                    )
                    warnings.append(warn_msg)
                    logger.warning(warn_msg)

        return aligned, warnings
