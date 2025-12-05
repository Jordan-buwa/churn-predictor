# grafana_dashboard.py

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

GRAFANA_URL = os.getenv("GRAFANA_URL")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY")
FOLDER_ID = int(os.getenv("GRAFANA_FOLDER_ID", 0))  # default to 0 if empty
OVERWRITE = True

if not GRAFANA_URL or not GRAFANA_API_KEY:
    raise ValueError("Please set GRAFANA_URL and GRAFANA_API_KEY in .env")

# Define the full dashboard
dashboard = {
    "dashboard": {
        "id": None,
        "uid": "ml-drift-monitoring",
        "title": "ML Drift Monitoring Dashboard",
        "tags": ["mlops", "drift", "monitoring"],
        "timezone": "browser",
        "schemaVersion": 35,
        "version": 0,
        "refresh": "5s",
        "time": {"from": "now-1h", "to": "now"},
        "templating": {"list": []},
        "panels": [
            # Panel 1
            {
                "id": 1,
                "title": "Overall Drift Status",
                "type": "stat",
                "targets": [{"expr": "drift_status", "legendFormat": "{{ instance }}"}],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "mappings": [
                            {"type": "value", "options": {"0": {"text": "STABLE"}, "1": {"text": "WARNING"}, "2": {"text": "CRITICAL"}}}
                        ],
                        "thresholds": {"steps": [{"value": None, "color": "green"}, {"value": 1, "color": "yellow"}, {"value": 2, "color": "red"}]},
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            },
            # Panel 2
            {
                "id": 2,
                "title": "Drift Severity Over Time",
                "type": "timeseries",
                "targets": [{"expr": "drift_severity", "legendFormat": "{{severity_level}}"}],
                "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}},
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            },
            # Panel 3
            {
                "id": 3,
                "title": "Feature Drift Metrics",
                "type": "gauge",
                "targets": [{"expr": "feature_drift_ratio", "legendFormat": "Drift Ratio"}],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {"steps": [{"value": None, "color": "green"}, {"value": 0.1, "color": "yellow"}, {"value": 0.3, "color": "red"}]},
                        "min": 0,
                        "max": 1,
                    }
                },
                "gridPos": {"h": 8, "w": 6, "x": 0, "y": 8},
            },
            # Panel 4
            {
                "id": 4,
                "title": "Target Drift - PSI Score",
                "type": "gauge",
                "targets": [{"expr": "target_psi_score", "legendFormat": "PSI Score"}],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {"steps": [{"value": None, "color": "green"}, {"value": 0.1, "color": "yellow"}, {"value": 0.2, "color": "red"}]},
                        "min": 0,
                        "max": 1,
                    }
                },
                "gridPos": {"h": 8, "w": 6, "x": 6, "y": 8},
            },
            # Panel 5
            {
                "id": 5,
                "title": "Alerts by Type",
                "type": "bargauge",
                "targets": [{"expr": "rate(alerts_triggered_total[5m])", "legendFormat": "{{alert_type}}"}],
                "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}},
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            },
            # Panel 6
            {
                "id": 6,
                "title": "Feature-Level Drift Scores",
                "type": "heatmap",
                "targets": [{"expr": "psi_score", "legendFormat": "{{feature}}"}],
                "fieldConfig": {"defaults": {"color": {"mode": "scheme", "scheme": "OrRd"}}},
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
            },
            # Panel 7
            {
                "id": 7,
                "title": "Monitoring Performance",
                "type": "timeseries",
                "targets": [
                    {"expr": "rate(data_points_processed_total[5m])", "legendFormat": "Data Points/Min"},
                    {"expr": "monitoring_cycle_duration_sum / monitoring_cycle_duration_count", "legendFormat": "Avg Cycle Duration"},
                ],
                "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}},
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
            },
        ],
    },
    "folderId": FOLDER_ID,
    "overwrite": OVERWRITE,
}

# Push dashboard to Grafana
headers = {"Authorization": f"Bearer {GRAFANA_API_KEY}", "Content-Type": "application/json"}

response = requests.post(
    f"{GRAFANA_URL}/api/dashboards/db", headers=headers, data=json.dumps(dashboard)
)

if response.status_code in [200, 201]:
    print("Dashboard created/updated successfully!")
    print("Response:", response.json())
else:
    print("Error creating dashboard!")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
