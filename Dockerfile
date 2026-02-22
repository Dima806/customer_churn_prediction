# ===========================================================================
# Customer Churn Dashboard – Dockerfile
# Compatible with HuggingFace Spaces (port 7860, user 1000).
# ===========================================================================

FROM python:3.12-slim

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# ---------------------------------------------------------------------------
# Working directory and user
# ---------------------------------------------------------------------------
WORKDIR /app

# HuggingFace Spaces runs containers as a non-root user (uid 1000)
RUN useradd -m -u 1000 appuser

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
COPY pyproject.toml ./
# Install only runtime dependencies (skip dev group)
RUN uv sync --no-dev

# ---------------------------------------------------------------------------
# Application code and artefacts
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# Predictions and model artefacts are expected to have been pre-built
# (run `make all` locally before building the image)
COPY predictions/ ./predictions/
COPY artifacts/ ./artifacts/

# Ensure the non-root user owns the files
RUN chown -R appuser:appuser /app
USER appuser

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 7860
ENV PORT=7860

CMD ["uv", "run", "python", "dashboard/app.py"]
