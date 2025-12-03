import time
import os
from prometheus_client import start_http_server, Gauge
from mlflow.tracking import MlflowClient
from datetime import datetime
import numpy as np

#  Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8080")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", 9100))
SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL", 15))

# Prometheus Metrics
# Gauge for the latest primary metric of a specific run (e.g., AUC, F1)
MLFLOW_RUN_METRIC = Gauge(
    'mlflow_run_metric_value',
    'Latest value of a tracked MLflow metric for a run.',
    ['experiment_name', 'run_id', 'metric_name']
)

# Gauge for a run's status (1=RUNNING, 2=SCHEDULED, 3=FINISHED, 4=FAILED, 5=KILLED)
MLFLOW_RUN_STATUS = Gauge(
    'mlflow_run_status',
    'Status of an MLflow run (1=Active, 3=Finished, 4=Failed, etc.).',
    ['experiment_name', 'run_id', 'status_name']
)


def get_mlflow_client():
    """Initializes and returns the MLflowClient."""
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client


def scrape_metrics():
    """Fetches MLflow run metrics and updates Prometheus gauges."""
    client = get_mlflow_client()
    try:

        experiments = client.search_experiments()
        experiment_names = [exp.name for exp in experiments]

        quoted_names = [f"'{name}'" for name in experiment_names]
        filter_string = f"tags.mlflow.experimentName IN ({', '.join(quoted_names)})"

        # Use filter_string and experiment_ids argument
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id for exp in experiments],
            filter_string=filter_string,
            order_by=["start_time DESC"],
            max_results=100  # Limit results to avoid excessive load
        )

        # Clear old metrics before adding new ones
        MLFLOW_RUN_METRIC.clear()
        MLFLOW_RUN_STATUS.clear()

        for run in runs:
            try:
                exp_name = client.get_experiment(run.info.experiment_id).name
            except Exception:
                exp_name = f"Unknown_Experiment_{run.info.experiment_id}"

            run_id = run.info.run_id
            status_name = run.info.status

            # Update Run Status Gauge
            status_map = {'RUNNING': 1, 'SCHEDULED': 2,
                          'FINISHED': 3, 'FAILED': 4, 'KILLED': 5}
            status_value = status_map.get(status_name, 0)
            MLFLOW_RUN_STATUS.labels(
                exp_name, run_id, status_name).set(status_value)

            # Update Metric Gauges
            for metric_key, metric_value in run.data.metrics.items():
                # Get the latest value for the metric
                latest_metric = client.get_metric_history(run_id, metric_key)
                if latest_metric:
                    latest_value = latest_metric[-1].value
                    MLFLOW_RUN_METRIC.labels(
                        exp_name, run_id, metric_key).set(latest_value)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully scraped {len(runs)} MLflow runs.")

    except Exception as e:
        # We need a robust check for client compatibility before proceeding
        if "unexpected keyword argument 'experiment_names'" in str(e):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FATAL ERROR: MLflow Client version mismatch (used deprecated filter). Ensure the fix has been saved and container restarted.")
        else:
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error during MLflow scraping: {e}")


def main():
    """Starts the HTTP server and metric collection loop."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting MLflow Metrics Exporter on port {EXPORTER_PORT}...")
    start_http_server(EXPORTER_PORT)

    while True:
        scrape_metrics()
        time.sleep(SCRAPE_INTERVAL)


if __name__ == '__main__':
    main()
