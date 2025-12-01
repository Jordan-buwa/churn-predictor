# dags/drift_detection_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import json
import psycopg2

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
    description='Detect feature and target drift using combined_features table',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['churn', 'drift']
) as dag:

    # Postgres connection details (adjust as needed)
    PG_CONN_INFO = {
        'host': 'localhost',
        'dbname': 'your_db',
        'user': 'your_user',
        'password': 'your_password'
    }

    REF_PATH = "data/processed/processed_data.csv"
    FEATURE_DRIFT_THRESHOLD = 0.1
    TARGET_DRIFT_THRESHOLD = 0.05

    def extract_new_data(**kwargs):
        """Pull new production data from combined_features table or fallback to CSV"""
        try:
            conn = psycopg2.connect(**PG_CONN_INFO)
            query = "SELECT * FROM combined_features;"
            df_new = pd.read_sql(query, conn)
            conn.close()
            print("Loaded new data from combined_features table")
        except Exception as e:
            print(f"Postgres fetch failed ({e}), falling back to CSV")
            df_new = pd.read_csv("data/processed/processed_data.csv")
        return df_new.to_dict(orient='records')

    def detect_drift(**kwargs):
        ti = kwargs['ti']

        # Load reference data
        if os.path.exists(REF_PATH):
            df_ref = pd.read_csv(REF_PATH)
        else:
            df_ref = pd.DataFrame()  # empty fallback

        new_records = ti.xcom_pull(task_ids='extract_new_data')
        df_new = pd.DataFrame(new_records)

        # Skip if reference or new data is empty
        if df_ref.empty or df_new.empty:
            print("Reference or new data empty. Skipping drift detection.")
            retrain_needed = False
            feature_drift_ratios = {}
            target_drift = 0.0
        else:
            # Feature drift calculation (mean shift / std)
            feature_drift_ratios = {}
            for col in df_ref.columns:
                if col == 'churn' or col not in df_new.columns:
                    continue
                ref_vals = df_ref[col].fillna(0)
                new_vals = df_new[col].fillna(0)
                drift_ratio = abs(new_vals.mean() - ref_vals.mean()) / (ref_vals.std() + 1e-6)
                feature_drift_ratios[col] = drift_ratio

            # Target drift
            target_ref = df_ref['churn'].value_counts(normalize=True).get(1, 0)
            target_new = df_new['churn'].value_counts(normalize=True).get(1, 0)
            target_drift = abs(target_new - target_ref)

            retrain_needed = (
                max(feature_drift_ratios.values(), default=0) > FEATURE_DRIFT_THRESHOLD
                or target_drift > TARGET_DRIFT_THRESHOLD
            )

        # Push XComs
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
