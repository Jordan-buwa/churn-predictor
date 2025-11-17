# Build stage
FROM python:3.11-slim AS build
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential gcc g++ gfortran cmake \
    libpq-dev libssl-dev libffi-dev libatlas3-base liblapack-dev git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models ./models
COPY retrain.py .

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /app /app

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
