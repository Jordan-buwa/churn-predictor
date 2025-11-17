# ---------- BUILD STAGE ----------
    FROM python:3.11-slim AS build
    WORKDIR /app
    
    # Install required system deps
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        cmake \
        libpq-dev \
        libssl-dev \
        libffi-dev \
        libatlas3-base \
        liblapack-dev \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy requirement files split into layers
    COPY requirements-core.txt .
    COPY requirements-ml.txt .
    COPY requirements-azure.txt .
    
    RUN pip install --upgrade pip
    
    # Install smallest → largest
    RUN pip install --no-cache-dir -r requirements-core.txt
    RUN pip install --no-cache-dir -r requirements-ml.txt
    RUN pip install --no-cache-dir -r requirements-azure.txt
    
    # Copy only runtime code
    COPY src/ ./src/
    COPY models ./models
    COPY retrain.py .
    
    # ---------- RUNTIME STAGE ----------
    FROM python:3.11-slim
    WORKDIR /app
    
    # Copy installed packages
    COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
    
    # Copy app
    COPY --from=build /app/src ./src
    COPY --from=build /app/models ./models
    COPY --from=build /app/retrain.py ./retrain.py
    
    ENV PYTHONPATH=/app
    EXPOSE 8000
    CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    