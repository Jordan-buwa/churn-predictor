
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime
import sys
import os

# Add project root to path so retrain.py can be imported
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

from retrain import ModelRetrainer
import pandas as pd

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
    schedule_interval=None,  # triggered manually by drift DAG or sensor
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['churn', 'retrain']
) as dag:

    # Branching to decide if retraining is needed
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

    # Retraining task
    def run_retraining(**kwargs):
        retrainer = ModelRetrainer(config_path="config/config_retrain.yaml")

        # Determine which data to use
        # Load reference training data
        df_ref = pd.read_csv("data/processed/processed_data.csv")

        # Load new production data
        df_new = pd.read_csv("data/production/new_data.csv")

        # Example: simple threshold usage
        # Here you could pull actual feature drift from XCom and decide
        # For demonstration, retrain with training + new data
        X = pd.concat([df_ref.drop(columns=['churn']), df_new.drop(columns=['churn'])], axis=0)
        y = pd.concat([df_ref['churn'], df_new['churn']], axis=0)

        # Overwrite the retrainer data to include combined dataset
        retrainer.models_to_retrain = ["xgboost", "random_forest", "neural_net"]
        report = retrainer.run_retraining_pipeline()

        print("Retraining report:", report)

    run_task = PythonOperator(
        task_id='run_retraining',
        python_callable=run_retraining
    )

    # Dummy task if retraining skipped
    from airflow.operators.dummy import DummyOperator
    skip_task = DummyOperator(task_id='skip_retraining')

    branch_task >> run_task
    branch_task >> skip_task
