# Multi-stage Dockerfile for AI Research Assistant
# CPU-only default, slim image without docling

# ============================================================
# Builder stage: install dependencies
# ============================================================
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt


# ============================================================
# Runtime stage: minimal runtime image
# ============================================================
FROM python:3.11-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./
COPY --chown=appuser:appuser README.md ./
COPY --chown=appuser:appuser LICENSE ./

# Create data directories with correct permissions
RUN mkdir -p /app/data/papers /app/data/processed /app/data/cache /app/data/eval/reports /app/data/graph && \
    chown -R appuser:appuser /app/data

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Healthcheck - uses the healthcheck module
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -m src.healthcheck || exit 1

# Default command shows help; use `docker compose run app ...` for commands
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]