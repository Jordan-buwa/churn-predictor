import time
import os
from prometheus_client import start_http_server, Gauge
from mlflow.tracking import MlflowClient
from datetime import datetime
import numpy as np

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8080")
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", 9100))
SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL", 15))

# Prometheus Metrics
MLFLOW_RUN_METRIC = Gauge(
    'mlflow_run_metric_value',
    'Latest value of a tracked MLflow metric for a run.',
    ['experiment_name', 'run_id', 'metric_name']
)

MLFLOW_RUN_STATUS = Gauge(
    'mlflow_run_status',
    'Status of an MLflow run (1=Active, 3=Finished, 4=Failed, etc.).',
    ['experiment_name', 'run_id', 'status_name']
)


def get_mlflow_client():
    """Initializes and returns the MLflowClient."""
    # The client automatically picks up the environment variable MLFLOW_TRACKING_URI
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client


def scrape_metrics():
    """Fetches MLflow run metrics and updates Prometheus gauges."""
    client = get_mlflow_client()
    try:
        # Search all experiments
        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]

        # Search runs across all experiments using only the IDs (The fix)
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            order_by=["start_time DESC"],
            max_results=100  # Limit results to avoid excessive load
        )

        # Clear old metrics before adding new ones
        MLFLOW_RUN_METRIC.clear()
        MLFLOW_RUN_STATUS.clear()

        for run in runs:
            try:
                # Fetch experiment name from ID
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
            for metric_key in run.data.metrics.keys():
                # Get the latest value for the metric history
                latest_metric_history = client.get_metric_history(
                    run_id, metric_key)
                if latest_metric_history:
                    # The latest entry is the last in the list
                    latest_value = latest_metric_history[-1].value

                    MLFLOW_RUN_METRIC.labels(
                        exp_name, run_id, metric_key).set(latest_value)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully scraped {len(runs)} MLflow runs and updated Prometheus metrics.")

    except Exception as e:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CRITICAL Error during MLflow scraping: {e}")


def main():
    """Starts the HTTP server and metric collection loop."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting MLflow Metrics Exporter on port {EXPORTER_PORT}...")
    start_http_server(EXPORTER_PORT)

    while True:
        scrape_metrics()
        time.sleep(SCRAPE_INTERVAL)


if __name__ == '__main__':
    main()
