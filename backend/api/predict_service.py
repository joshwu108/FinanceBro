"""Online prediction: load cached OHLCV, build latest features, score with saved model."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_MODELS_DIR = BACKEND_ROOT / "models"

_bundle_cache: Dict[str, Dict[str, Any]] = {}


def clear_inference_bundle_cache(symbol: str | None = None) -> None:
    """Drop cached joblib bundles so the next request reloads from disk."""
    if symbol is None:
        _bundle_cache.clear()
    else:
        _bundle_cache.pop(symbol.upper(), None)


def inference_artifact_path(symbol: str) -> Path:
    return DEFAULT_MODELS_DIR / f"{symbol.upper()}_inference.joblib"


def get_inference_bundle(symbol: str, *, use_cache: bool = True) -> Dict[str, Any]:
    sym = symbol.upper()
    if use_cache and sym in _bundle_cache:
        return _bundle_cache[sym]
    path = inference_artifact_path(sym)
    bundle = ModelAgent.load_inference_bundle(path)
    if use_cache:
        _bundle_cache[sym] = bundle
    return bundle


def load_ohlcv(
    symbol: str,
    *,
    data_dir: Path | None = None,
    max_stale_hours: float = 24.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return OHLCV for ``symbol``, using parquet when fresh to avoid network I/O."""
    sym = symbol.upper()
    root = data_dir or DEFAULT_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sym}.parquet"

    if path.is_file() and not force_refresh:
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        if age_h < max_stale_hours:
            logger.debug("Using cached parquet for %s (age %.2fh)", sym, age_h)
            return pd.read_parquet(path)

    agent = DataAgent({**DataAgent.DEFAULT_CONFIG, "data_dir": str(root)})
    out = agent.run({"symbols": [sym]})
    if sym not in out["cleaned_data"]:
        raise ValueError(f"No OHLCV available for {sym}")
    return out["cleaned_data"][sym]


def predict_for_symbol(
    symbol: str,
    *,
    force_refresh_data: bool = False,
    max_stale_hours: float = 24.0,
) -> Dict[str, Any]:
    """Load data, compute the latest feature row, run the frozen classifier."""
    sym = symbol.upper()
    ohlcv = load_ohlcv(
        sym,
        force_refresh=force_refresh_data,
        max_stale_hours=max_stale_hours,
    )
    bundle = get_inference_bundle(sym)

    feature_agent = FeatureAgent()
    ts, feat_row = feature_agent.latest_inference_features(ohlcv)

    names: list[str] = bundle["feature_names"]
    missing = set(names) - set(feat_row.index)
    if missing:
        raise ValueError(
            f"Feature mismatch for {sym}: bundle expects columns not present: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )

    x = feat_row[names].to_numpy(dtype=float, copy=False).reshape(1, -1)
    scaler = bundle["scaler"]
    model = bundle["model"]
    x_scaled = scaler.transform(x)
    proba = float(model.predict_proba(x_scaled)[0, 1])
    pred_class = int(proba >= 0.5)
    signal = "long" if pred_class == 1 else "short"

    return {
        "symbol": sym,
        "prediction": round(proba, 6),
        "signal": signal,
        "timestamp": ts.isoformat(),
        "model_type": bundle.get("model_type"),
    }
