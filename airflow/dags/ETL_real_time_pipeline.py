# dags/real_time_data_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.api.utils.customer_data import CustomerData
from src.api.utils.database import save_customer_data, generate_batch_id
import json

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

with DAG(
    dag_id='real_time_data_pipeline',
    default_args=default_args,
    description='ETL pipeline for incoming production customer data',
    schedule_interval=None,  # triggered manually
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['churn', 'realtime']
) as dag:

    def fetch_real_time_data(**kwargs):
        """
        Pull new real-time data.
        In practice: staging table, API, or Kafka.
        For demo: JSON file in data/production
        """
        file_path = "data/production/customers.json"
        with open(file_path, "r") as f:
            records = json.load(f)
        return records

    def save_real_time_data(**kwargs):
        """
        Save new real-time records into Postgres tables.
        """
        ti = kwargs['ti']
        records = ti.xcom_pull(task_ids='fetch_real_time_data')
        batch_id = generate_batch_id()

        for record in records:
            try:
                customer = CustomerData(**record)
                save_customer_data(customer, batch_id)
            except Exception as e:
                print(f"Skipping record due to validation error: {e}")

    fetch_task = PythonOperator(
        task_id='fetch_real_time_data',
        python_callable=fetch_real_time_data
    )

    save_task = PythonOperator(
        task_id='save_real_time_data',
        python_callable=save_real_time_data
    )

    fetch_task >> save_task
