from airflow import DAG
from airflow.operators.python import PythonOperator
# from airflow.python import PythonOperator
from datetime import datetime, timedelta
from src.api.utils.customer_data import CustomerData
from src.api.utils.database import save_customer_data, generate_batch_id

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

with DAG(
    'real_time_data_pipeline',
    default_args=default_args,
    description='ETL pipeline for incoming production customer data',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['churn', 'realtime']
) as dag:

    def save_real_time_data(**kwargs):
        """
        Pull data from your FastAPI endpoint or temporary staging table.
        Here we assume you pass a list of dicts via XCom or external trigger.
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

    def fetch_real_time_data(**kwargs):
        """
        This can pull from a staging table, Kafka, or API
        Simulated here as reading from a JSON file for example purposes
        """
        import json
        file_path = "data/production/customers.json"
        with open(file_path, "r") as f:
            records = json.load(f)
        return records

    fetch_task = PythonOperator(
        task_id='fetch_real_time_data',
        python_callable=fetch_real_time_data
    )

    save_task = PythonOperator(
        task_id='save_real_time_data',
        python_callable=save_real_time_data
    )

    fetch_task >> save_task
