# Base Image
FROM python:3.11-slim

# Set environment variables for path and logging
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# Install critical system dependencies:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and configuration
COPY src/ ./src/
COPY config/ ./config/

# Create application data and log directories (consistent with docker-compose volumes)
RUN mkdir -p /app/data/raw /app/data/processed /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV LOG_DIR=/app/logs

# Entry point for the data pipeline worker
ENTRYPOINT ["python", "src/data_pipeline/pipeline_data.py"]
CMD ["--help"]