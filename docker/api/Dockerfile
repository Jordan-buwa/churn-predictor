# Base Image
FROM python:3.11-slim

# Set environment variables for path
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# Install critical system dependencies:
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, configuration, and necessary files
COPY src/ ./src/
COPY models ./models
COPY retrain.py .

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Default command for the API service (using secure runtime variables)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]