# dags/retrain_trigger_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
import pandas as pd
import os
import sys

# Add project root to import retrain.py
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from retrain import ModelRetrainer  # DO NOT MODIFY

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

with DAG(
    'retrain_trigger_dag',
    default_args=default_args,
    description='Trigger retraining if drift detected',
    schedule_interval=None,  # manually triggered by drift DAG
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['churn', 'retrain']
) as dag:

    def check_drift(**kwargs):
        ti = kwargs['ti']
        retrain_needed = ti.xcom_pull(
            dag_id='drift_detection_dag',
            task_ids='detect_drift',
            key='retrain_needed'
        )
        return 'run_retraining' if retrain_needed else 'skip_retraining'

    branch_task = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=check_drift
    )

    def run_retraining(**kwargs):
        retrainer = ModelRetrainer(config_path="config/config_retrain.yaml")

        # Load data from combined_features table
        try:
            import psycopg2
            conn = psycopg2.connect(host='localhost', dbname='your_db',
                                    user='your_user', password='your_password')
            df_new = pd.read_sql("SELECT * FROM combined_features;", conn)
            conn.close()
            print("Loaded new data from combined_features")
        except Exception as e:
            print(f"Postgres fetch failed ({e}), fallback to CSV")
            df_new = pd.read_csv("data/processed/processed_data.csv")

        # Load reference training data
        df_ref = pd.read_csv("data/processed/processed_data.csv")

        # Determine which data to use (example: combined training + new data)
        X = pd.concat([df_ref.drop(columns=['churn']), df_new.drop(columns=['churn'])], axis=0)
        y = pd.concat([df_ref['churn'], df_new['churn']], axis=0)

        retrainer.models_to_retrain = ["xgboost", "random_forest", "neural_net"]
        report = retrainer.run_retraining_pipeline()
        print("Retraining report:", report)

    run_task = PythonOperator(
        task_id='run_retraining',
        python_callable=run_retraining
    )

    skip_task = DummyOperator(task_id='skip_retraining')

    branch_task >> run_task
    branch_task >> skip_task
