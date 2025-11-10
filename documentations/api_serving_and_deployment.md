# API Serving and Deployment Guide

This guide covers running the FastAPI service locally, via Docker Compose, and deploying to production. It also explains the test setup used in CI.

## Local Development (Uvicorn)
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Set environment variables (optional):
  - `ENVIRONMENT=development`
  - `DATABASE_URL=postgresql://<user>:<password>@<host>/<db>`
  - For quick local testing without Postgres: `DATABASE_URL=sqlite:///./dev.db`
- Start the server:
  - `uvicorn src.api.main:app --reload --port 8000`
- Check health endpoints:
  - `GET http://localhost:8000/health` (machine-readable)
  - `GET http://localhost:8000/health-ui` (HTML)

## Docker Compose
- Build and run the API:
  - `docker compose -f docker-compose.api.yml up --build`
- The compose file maps volumes for models, data, logs, and config, and reads env vars from `.env`.
- Health check is configured on port `8000`; access `http://localhost:8000/health`.

## Environment Configuration
- `ENVIRONMENT`: `development`, `test`, or `production`.
  - In `test`, the API skips heavy router imports and DB pool initialization to keep tests fast and isolated.
- `DATABASE_URL`: overrides DB connection.
  - Use `sqlite:///./<file>.db` for local/test runs.
  - Defaults to Postgres connection if not provided.
- Other variables consumed by the app are defined in `docker/api/Dockerfile` and `docker-compose.api.yml` (e.g., Postgres and MLflow settings).

## CI Test Setup (Pytest)
- CI runs `pytest` scoped to API tests only: `pytest -q tests/api`.
- Tests use a local SQLite DB:
  - `ENVIRONMENT=test`
  - `DATABASE_URL=sqlite:///./test_api.db`
- Import-time safeguards:
  - DB pool initialization is skipped in test environment.
  - Heavy routers are conditionally omitted in tests to avoid external dependencies.

## Production Deployment
1. Build the Docker image:
   - `docker build -f docker/api/Dockerfile -t churn-api:latest .`
2. Push to your registry (example):
   - `docker tag churn-api:latest <registry>/churn-api:latest`
   - `docker push <registry>/churn-api:latest`
3. Run container with environment:
   - `docker run -d -p 8000:8000 --env-file .env --name churn-api <registry>/churn-api:latest`
4. Verify health:
   - `curl http://localhost:8000/health`

## Notes
- Pydantic v2 deprecation warnings are present; migration is non-blocking for serving and testing.
- DVC-managed model artifacts are not required to start the API; endpoints may report models as not loaded if artifacts are unavailable.
- For secure deployments, provide DB credentials and any secrets via your orchestrator’s secret management.