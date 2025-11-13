{
  "dashboard": {
    "id": null,
    "title": "ML Drift Monitoring Dashboard",
    "tags": ["mlops", "drift", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Overall Drift Status",
        "type": "stat",
        "targets": [
          {
            "expr": "drift_status",
            "legendFormat": "{{ instance }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": { "mode": "thresholds" },
            "mappings": [
              {
                "type": "value",
                "options": {
                  "0": { "text": "STABLE" },
                  "1": { "text": "WARNING" },
                  "2": { "text": "CRITICAL" }
                }
              }
            ],
            "thresholds": {
              "steps": [
                { "value": null, "color": "green" },
                { "value": 1, "color": "yellow" },
                { "value": 2, "color": "red" }
              ]
            }
          }
        },
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 }
      },
      {
        "id": 2,
        "title": "Drift Severity Over Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "drift_severity",
            "legendFormat": "{{severity_level}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": { "mode": "palette-classic" },
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "linear",
              "barAlignment": 0,
              "lineWidth": 2,
              "fillOpacity": 10,
              "gradientMode": "none",
              "spanNulls": false,
              "showPoints": "auto",
              "pointSize": 5,
              "stacking": { "mode": "none", "group": "A" },
              "axisPlacement": "auto",
              "axisLabel": "",
              "scaleDistribution": { "type": "linear" },
              "axisCenteredZero": false,
              "hideFrom": { "tooltip": false, "viz": false, "legend": false }
            }
          }
        },
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 }
      },
      {
        "id": 3,
        "title": "Feature Drift Metrics",
        "type": "gauge",
        "targets": [
          { "expr": "feature_drift_ratio", "legendFormat": "Drift Ratio" }
        ],
        "fieldConfig": {
          "defaults": {
            "color": { "mode": "thresholds" },
            "mappings": [],
            "thresholds": {
              "steps": [
                { "value": null, "color": "green" },
                { "value": 0.1, "color": "yellow" },
                { "value": 0.3, "color": "red" }
              ]
            },
            "min": 0,
            "max": 1
          }
        },
        "gridPos": { "h": 8, "w": 6, "x": 0, "y": 8 }
      },
      {
        "id": 4,
        "title": "Target Drift - PSI Score",
        "type": "gauge",
        "targets": [
          { "expr": "target_psi_score", "legendFormat": "PSI Score" }
        ],
        "fieldConfig": {
          "defaults": {
            "color": { "mode": "thresholds" },
            "mappings": [],
            "thresholds": {
              "steps": [
                { "value": null, "color": "green" },
                { "value": 0.1, "color": "yellow" },
                { "value": 0.2, "color": "red" }
              ]
            },
            "min": 0,
            "max": 1
          }
        },
        "gridPos": { "h": 8, "w": 6, "x": 6, "y": 8 }
      },
      {
        "id": 5,
        "title": "Alerts by Type",
        "type": "bargauge",
        "targets": [
          { "expr": "rate(alerts_triggered_total[5m])", "legendFormat": "{{alert_type}}" }
        ],
        "fieldConfig": {
          "defaults": {
            "color": { "mode": "palette-classic" },
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "linear",
              "barAlignment": 0,
              "lineWidth": 1,
              "fillOpacity": 80,
              "gradientMode": "none",
              "spanNulls": false,
              "showPoints": "never",
              "pointSize": 5,
              "stacking": { "mode": "normal", "group": "A" },
              "axisPlacement": "auto",
              "axisLabel": "",
              "scaleDistribution": { "type": "linear" },
              "axisCenteredZero": false,
              "hideFrom": { "tooltip": false, "viz": false, "legend": false }
            }
          }
        },
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 8 }
      },
      {
        "id": 6,
        "title": "Feature-Level Drift Scores",
        "type": "heatmap",
        "targets": [{ "expr": "psi_score", "legendFormat": "{{feature}}" }],
        "fieldConfig": {
          "defaults": { "color": { "mode": "scheme", "scheme": "OrRd" } }
        },
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 }
      },
      {
        "id": 7,
        "title": "Monitoring Performance",
        "type": "timeseries",
        "targets": [
          { "expr": "rate(data_points_processed_total[5m])", "legendFormat": "Data Points/Min" },
          { "expr": "monitoring_cycle_duration_sum / monitoring_cycle_duration_count", "legendFormat": "Avg Cycle Duration" }
        ],
        "fieldConfig": {
          "defaults": {
            "color": { "mode": "palette-classic" },
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "linear",
              "barAlignment": 0,
              "lineWidth": 2,
              "fillOpacity": 10,
              "gradientMode": "none",
              "spanNulls": false,
              "showPoints": "auto",
              "pointSize": 5,
              "stacking": { "mode": "none", "group": "A" },
              "axisPlacement": "auto",
              "axisLabel": "",
              "scaleDistribution": { "type": "linear" },
              "axisCenteredZero": false,
              "hideFrom": { "tooltip": false, "viz": false, "legend": false }
            }
          }
        },
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 16 }
      }
    ],
    "time": { "from": "now-1h", "to": "now" },
    "timepicker": {},
    "templating": { "list": [] },
    "refresh": "5s",
    "schemaVersion": 35,
    "version": 0,
    "uid": "ml-drift-monitoring"
  }
}
