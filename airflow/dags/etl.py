from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from src.data_pipeline.ingest import DataIngestion
from src.data_pipeline.preprocess import DataPreprocessor
from src.data_pipeline.validate_after_preprocess import validate_after_preprocess  

def ingest_task():
    ingestion = DataIngestion("config/config_ingest.yaml")
    df = ingestion.load_data()
    df.to_csv("/tmp/ingested_data.csv", index=False)

def preprocess_task():
    df = pd.read_csv("/tmp/ingested_data.csv")
    preprocessor = DataPreprocessor(data_raw=df)
    df_processed = preprocessor.run_preprocessing_pipeline()
    df_processed.to_csv("/tmp/processed_data.csv", index=False)

def validate_task():
    df = pd.read_csv("/tmp/processed_data.csv")
    validate_after_preprocess(df)  # your validation logic

with DAG(
    "customer_data_pipeline",
    start_date=datetime(2025, 11, 14),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_task
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_task
    )

    task_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_task
    )

    # Define dependencies
    task_ingest >> task_preprocess >> task_validate
