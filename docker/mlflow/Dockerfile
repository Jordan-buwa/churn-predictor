# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install MLflow and Azure ML SDK
RUN pip install --no-cache-dir \
    mlflow \
    azure-ai-ml \
    azure-identity \
    psycopg2-binary \
    boto3

# Create non-root user
RUN useradd -m -s /bin/bash mlflow && \
    mkdir -p /mlflow/artifacts && \
    chown -R mlflow:mlflow /mlflow /app

# Switch to non-root user
USER mlflow

# Environment variables (will be set at runtime)
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
ENV AZURE_TENANT_ID=${AZURE_TENANT_ID}
ENV AZURE_CLIENT_SECRET=${AZURE_CLIENT_SECRET}

# Expose MLflow UI port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start MLflow server (proxies to Azure ML)
CMD ["sh", "-c", "\
    mlflow server \
      --backend-store-uri \"$MLFLOW_TRACKING_URI\" \
      --default-artifact-root \"$MLFLOW_TRACKING_URI\" \
      --host 0.0.0.0 \
      --port 5000 \
      --serve-artifacts \
      --gunicorn-opts '--timeout 120 --workers 2' \
"]