from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import json

# Default args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

# DAG definition
with DAG(
    'drift_detection_dag',
    default_args=default_args,
    description='Detect feature and target drift against reference data',
    schedule_interval='@daily',  # run daily, adjust as needed
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['churn', 'drift']
) as dag:

    # Path to reference processed data
    REF_PATH = "data/processed/processed_data.csv"

    # Drift thresholds (example)
    FEATURE_DRIFT_THRESHOLD = 0.1
    TARGET_DRIFT_THRESHOLD = 0.05

    # Placeholder: replace with your data extraction from Postgres/Azure
    def extract_new_data(**kwargs):
        # Example: read from grouped tables in airflow/seed
        # Here you would query Postgres or Azure blob
        # For demo, load from a JSON or CSV
        new_data_path = "data/raw/simulated_realistic_sample.csv"
        df_new = pd.read_csv(new_data_path)
        return df_new.to_dict(orient='records')

    def detect_drift(**kwargs):
        ti = kwargs['ti']

        # Load reference data
        if not os.path.exists(REF_PATH):
            raise FileNotFoundError(f"Reference file not found: {REF_PATH}")
        df_ref = pd.read_csv(REF_PATH)

        # Load new data
        new_records = ti.xcom_pull(task_ids='extract_new_data')
        df_new = pd.DataFrame(new_records)

        # Simple drift calculations (Kolmogorov-Smirnov or population ratios)
        feature_drift_ratios = {}
        for col in df_ref.columns:
            if col == 'churn':
                continue
            ref_vals = df_ref[col].fillna(0)
            new_vals = df_new[col].fillna(0)
            drift_ratio = abs(new_vals.mean() - ref_vals.mean()) / (ref_vals.std() + 1e-6)
            feature_drift_ratios[col] = drift_ratio

        # Target drift (proportion difference)
        target_ref = df_ref['churn'].value_counts(normalize=True).get(1, 0)
        target_new = df_new['churn'].value_counts(normalize=True).get(1, 0)
        target_drift = abs(target_new - target_ref)

        # Determine if retraining needed
        retrain_needed = (
            max(feature_drift_ratios.values()) > FEATURE_DRIFT_THRESHOLD
            or target_drift > TARGET_DRIFT_THRESHOLD
        )

        # Push XCom
        ti.xcom_push(key='retrain_needed', value=retrain_needed)
        ti.xcom_push(key='feature_drift', value=json.dumps(feature_drift_ratios))
        ti.xcom_push(key='target_drift', value=target_drift)

        return retrain_needed

    extract_task = PythonOperator(
        task_id='extract_new_data',
        python_callable=extract_new_data
    )

    drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift
    )

    extract_task >> drift_task
