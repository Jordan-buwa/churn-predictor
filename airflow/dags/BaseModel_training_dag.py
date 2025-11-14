from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import yaml
from pathlib import Path
from datetime import timedelta
import sys
import os

# Helper: Load YAML config
def load_model_config(model_yaml_path: str):
    with open(model_yaml_path, "r") as f:
        return yaml.safe_load(f)

def get_schedule_from_yaml(model_yaml_path: str, default_days=7):
    cfg = load_model_config(model_yaml_path)
    days = cfg.get("training_schedule", {}).get("days", default_days)
    # Convert days to cron string (run every N days at midnight)
    return f"0 0 */{days} * *"

# Default DAG args from config_airflow.yaml
CONFIG_PATH = Path(__file__).parents[1] / "config" / "config_airflow.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

default_cfg = cfg.get("dags", {}).get("default", {})
default_args = {
    "owner": default_cfg.get("owner", "airflow"),
    "depends_on_past": default_cfg.get("depends_on_past", False),
    "email_on_failure": default_cfg.get("email_on_failure", False),
    "email_on_retry": default_cfg.get("email_on_retry", False),
    "retries": default_cfg.get("retries", 0),
    "retry_delay": timedelta(minutes=default_cfg.get("retry_delay_minutes", 5)),
}

# -----------------------------
# Training functions
# -----------------------------
def train_rf(**kwargs):
    sys.path.append(str(Path(__file__).parents[1]))
    from src.models.train_rf import main as train_rf_main
    train_rf_main()

def train_xgb(**kwargs):
    sys.path.append(str(Path(__file__).parents[1]))
    from src.models.train_xgb import main as train_xgb_main
    train_xgb_main()

def train_nn(**kwargs):
    sys.path.append(str(Path(__file__).parents[1]))
    from src.models.train_nn import main as train_nn_main
    train_nn_main()

# -----------------------------
# Load schedules dynamically from YAMLs
# -----------------------------
BASE_DIR = Path(__file__).parents[1]
model_yamls = {
    "rf": BASE_DIR / "config" / "config_train_rf.yaml",
    "xgb": BASE_DIR / "config" / "config_train_xgb.yaml",
    "nn": BASE_DIR / "config" / "config_train_nn.yaml",
}

# Choose the smallest days among models to trigger combined DAG
days_list = [
    yaml_cfg.get("training_schedule", {}).get("days", 7)
    for yaml_cfg in (load_model_config(p) for p in model_yamls.values())
]
min_days = min(days_list)
schedule_interval = f"0 0 */{min_days} * *"  # run every min_days at 00:00

# -----------------------------
# DAG definition
# -----------------------------
with DAG(
    dag_id="train_all_models_combined",
    default_args=default_args,
    description="Train RF, XGB, and NN models in parallel",
    schedule_interval=schedule_interval,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=2,  # keep lightweight
    tags=["training", "models"]
) as dag:

    task_rf = PythonOperator(
        task_id="train_rf",
        python_callable=train_rf,
        provide_context=True,
    )

    task_xgb = PythonOperator(
        task_id="train_xgb",
        python_callable=train_xgb,
        provide_context=True,
    )

    task_nn = PythonOperator(
        task_id="train_nn",
        python_callable=train_nn,
        provide_context=True,
    )

    # Parallel execution
    [task_rf, task_xgb, task_nn]
