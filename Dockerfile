# ==========================================
# Stage 1: Builder Layer (Discarded after build)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Create and activate a Virtual Environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements separate to leverage Docker layer caching efficiently
COPY requirements.txt .

# Upgrade compiler tools and install Python dependencies natively inside the venv
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Final Production Image (Ultra-lean & Hardened)
# ==========================================
FROM python:3.11-slim

# Create a system user without an interactive shell (-s /bin/false) for system hardening
RUN useradd -m -u 1000 -s /bin/false appuser

WORKDIR /app

# Pull the clean, pre-compiled virtual environment from the builder stage
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Explicitly copy source folders. Prevents exposure of raw host files like .git or Dockerfiles
COPY --chown=appuser:appuser ./automation ./automation
COPY --chown=appuser:appuser ./dashboard ./dashboard
COPY --chown=appuser:appuser ./pipeline ./pipeline
COPY --chown=appuser:appuser ./utils ./utils
COPY --chown=appuser:appuser ./collectors ./collectors
COPY --chown=appuser:appuser ./config ./config

# Explicitly initialize the shared volume mount point and apply owner permissions before attachment
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Hardening environment flags for Python runtime stability
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose Streamlit's structural port (or 8000 later for FastAPI)
EXPOSE 8501

# Drop privileges down to the non-root system user
USER appuser

# Safe default entrypoint fallback executed in executive form (JSON Array format)
CMD ["python", "automation/flow.py"]