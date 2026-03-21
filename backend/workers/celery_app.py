import os
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv

ROOT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ROOT_ENV_PATH, override=False)

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

app = Celery(
    "financebro",
    broker=broker_url,
    backend=result_backend,
    include=["workers.tasks"],
)

app.conf.update(
    result_expires=3600,
    task_serializer="json",
    broker_connection_retry_on_startup=True,
    task_default_queue='financebro_queue',
)

if __name__ == "__main__":
    app.start()