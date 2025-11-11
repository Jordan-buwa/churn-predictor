FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Azure ML dependencies
RUN pip install azure-ai-ml azure-identity azure-storage-blob

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create directories
RUN mkdir -p /app/models /app/artifacts /app/logs /app/mlruns

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV ARTIFACT_DIR=/app/artifacts
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=production

# Run training with Azure ML integration
CMD ["python", "src/models/train_xgb.py"]