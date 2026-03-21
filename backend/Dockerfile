# Stage 1: Build & Base
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies (needed for confluent-kafka and scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    librdkafka-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production Image
FROM python:3.11-slim

WORKDIR /app

# Copy system libraries for confluent-kafka
RUN apt-get update && apt-get install -y \
    librdkafka1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Ensure the app can find our modules
ENV PYTHONPATH=/app

# Default port for FastAPI
EXPOSE 8000

# We don't define a CMD here because we'll override it in docker-compose 
# to run either the API or the Worker.
