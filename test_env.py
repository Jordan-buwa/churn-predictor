# test_env.py
from dotenv import load_dotenv
import os

load_dotenv()

print("Postgres user:", os.getenv("POSTGRES_DB_USER"))
print("MLflow URI:", os.getenv("MLFLOW_TRACKING_URI"))
