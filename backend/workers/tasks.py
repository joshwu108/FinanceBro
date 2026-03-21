import json
import logging
import os

import redis

from workers.celery_app import app
redis_host = os.getenv("REDIS_HOST", "localhost")
r = redis.Redis(host=redis_host, port=6379, db=1)

logger = logging.getLogger(__name__)


def _task_redis_url() -> str:
    task_redis_url = (os.getenv("TASK_REDIS_URL") or "").strip()
    if task_redis_url:
        return task_redis_url

    base_redis_url = (os.getenv("REDIS_URL") or "").strip()
    if base_redis_url:
        if "/" in base_redis_url.rsplit(":", 1)[-1]:
            return base_redis_url
        return f"{base_redis_url}/1"

    return "redis://localhost:6379/1"


SIGNAL_CHANNEL = "financebro:signals"


@app.task(name="tasks.evaluate_tick")
def evaluate_tick(symbol, price, features):
    raw_cached = r.get(f"features:{symbol}")
    if raw_cached:
        try:
            features = json.loads(raw_cached)
            prediction = 1.0 if price > features.get("open", 0) else 0.0
        except json.JSONDecodeError:
            result = {"symbol": symbol, "signal": "ERROR", "message": "Corrupted feature vector"}
            r.publish(SIGNAL_CHANNEL, json.dumps(result))
            return result
    else:
        result = {"symbol": symbol, "signal": "NEUTRAL"}
        r.publish(SIGNAL_CHANNEL, json.dumps(result))
        return result

    result = {
        "symbol": symbol,
        "signal": "LONG" if prediction > 0.5 else "SHORT",
        "confidence": prediction,
    }
    r.publish(SIGNAL_CHANNEL, json.dumps(result))
    return result
