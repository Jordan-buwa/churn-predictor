# dags/data_pipeline_dag.py
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from src.data_pipeline import preprocess_data, validate_after_preprocess 

# folder to save processed data
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def run_preprocessing(**kwargs):
    df_processed = preprocess_data()
    df_validated = validate_after_preprocess(df_processed)
    # store in XCom for downstream tasks
    kwargs['ti'].xcom_push(key='validated_df', value=df_validated.to_dict())
    return "Preprocessing done"

def store_processed_data(**kwargs):
    # pull validated data from XCom
    ti = kwargs['ti']
    validated_df_dict = ti.xcom_pull(key='validated_df', task_ids='preprocess_and_validate')
    df_validated = pd.DataFrame(validated_df_dict)
    
    # save locally
    filename = os.path.join(PROCESSED_DATA_DIR, f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_validated.to_csv(filename, index=False)
    print(f"Processed data saved at {filename}")

# Define DAG
with DAG(
    dag_id="data_pipeline_dag",
    start_date=datetime(2025, 11, 14),
    schedule_interval="@daily",
    catchup=False,
    tags=["data_pipeline"],
) as dag:

    preprocess_and_validate = PythonOperator(
        task_id="preprocess_and_validate",
        python_callable=run_preprocessing,
        provide_context=True,
    )

    store_data = PythonOperator(
        task_id="store_data",
        python_callable=store_processed_data,
        provide_context=True,
    )

    # DAG flow
    preprocess_and_validate >> store_data

