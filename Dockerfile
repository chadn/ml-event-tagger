# Multi-stage build for ML Event Tagger API
# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml README.md ./

# Copy source code for editable install
COPY ml_event_tagger/ /app/ml_event_tagger/

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Install dependencies to /app/.venv
RUN uv venv /app/.venv && \
    uv pip install --no-cache -e .

# Stage 2: Runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY ml_event_tagger/ /app/ml_event_tagger/
COPY models/ /app/models/

# Copy metadata files
COPY pyproject.toml README.md /app/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run the application
CMD ["uvicorn", "ml_event_tagger.serve:app", "--host", "0.0.0.0", "--port", "8000"]

