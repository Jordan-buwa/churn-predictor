FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models ./models
COPY data/ ./data

COPY retrain.py .

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Use CMD with runtime environment variables (do NOT hardcode secrets)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]