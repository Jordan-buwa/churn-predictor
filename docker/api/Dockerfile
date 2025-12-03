FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
COPY requirements-core.txt .
COPY requirements-azure.txt .
COPY requirements-ml.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models ./models
COPY data/ ./data
COPY test_input.json .
COPY retrain.py .

# Set Python path
ENV PYTHONPATH=/app
ENV POSTGRES_HOST=${POSTGRES_HOST}
ENV POSTGRES_PORT=5432
ENV POSTGRES_DB_NAME=${POSTGRES_DB_NAME}
ENV POSTGRES_DB_USER=${POSTGRES_DB_USER}
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
ENV API_KEY_SECRET=${API_KEY_SECRET}
# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
