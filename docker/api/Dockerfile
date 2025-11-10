# Stage 1: Build the application
FROM python:3.9-slim as builder

# Install poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy only the necessary files to install dependencies
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Stage 2: Create the final image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create a non-root user
RUN useradd --create-home appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the installed dependencies from the builder stage
COPY --from=builder /app/.venv /.venv

# Set the PATH to include the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/

# Create necessary directories and set permissions
RUN mkdir -p logs \
    && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app/src:/app
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV LOG_LEVEL=INFO
ENV POSTGRES_HOST=localhost
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB_NAME=churn_db
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]