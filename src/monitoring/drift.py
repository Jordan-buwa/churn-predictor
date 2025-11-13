# src/monitoring/drift.py

import pandas as pd
import numpy as np
import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from src.data_pipeline.data_simulation import RealisticDataSimulator, DriftedDataSimulator
from src.monitoring.utils import calculate_psi, detect_covariate_shift


class LiveDataDriftMonitor:
    """
    Class for real-time drift monitoring in production data streams.
    """

    def __init__(self, reference_data: pd.DataFrame, target_column: str):
        self.reference_data = reference_data
        self.target_column = target_column
        self.is_running = False
        self.logger = self._setup_logger()
        self.logger.info("Initialized LiveDataDriftMonitor")

    def _setup_logger(self):
        logger = logging.getLogger("LiveDataDriftMonitor")
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    # Core Drift Monitoring Logic
    def monitor_live_drift(self, lookback_hours: int = 1) -> Dict[str, Any]:
        """
        Monitor live drift by comparing the latest production data with reference.
        """
        self.logger.info(f"Running drift detection for last {lookback_hours} hours...")

        try:
            # Simulated data ingestion
            live_data = self._simulate_live_data()

            # Run individual drift checks
            feature_drift = self._analyze_feature_drift(live_data)
            target_drift = self._analyze_target_drift(live_data)
            distribution_drift = self._analyze_distribution_drift(live_data)

            # Summarize drift
            analysis_summary = self._summarize_drift(
                feature_drift, target_drift, distribution_drift
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "feature_drift": feature_drift,
                "target_drift": target_drift,
                "distribution_drift": distribution_drift,
                "analysis_summary": analysis_summary,
                "alerts": self._generate_alerts(analysis_summary),
                "sample_size": len(live_data),
            }

        except Exception as e:
            self.logger.error(f"Drift monitoring failed: {e}")
            raise

    def _simulate_live_data(self) -> pd.DataFrame:
        """
        Generate live (drifted) data using DriftedDataSimulator.
        """
        # Initialize the drift simulator
        drift_simulator = DriftedDataSimulator(random_state=42)
        drift_simulator.configure_drift(
            target_drift_strength=0.3,
            feature_drift_strength=0.4,
            categorical_drift_strength=0.5,
        )

        # Fit the simulator using the reference data
        reference_path = "data/raw/telco_churn.csv"  # adjust if your path differs
        reference_data = drift_simulator.load_and_clean_data(reference_path)

        # Generate drifted data
        live_data = reference_data.sample(frac=1, replace=False).copy()
        live_data = live_data.sample(n=min(1000, len(live_data)), random_state=42)  # Simulated drifted stream

        # Optionally log shape and drift simulation info
        self.logger.info(f"Simulated live drifted data shape: {live_data.shape}")

        return live_data

    def _analyze_feature_drift(self, live_data: pd.DataFrame) -> Dict:
        drifted_features = []
        psi_scores = {}
        for col in live_data.columns:
            if col == self.target_column:
                continue
            psi = calculate_psi(self.reference_data[col], live_data[col])
            psi_scores[col] = psi
            if psi > 0.2:
                drifted_features.append(col)
        return {
            "drifted_features": drifted_features,
            "feature_scores": psi_scores,
            "drift_detected": len(drifted_features) > 0,
        }

    def _analyze_target_drift(self, live_data: pd.DataFrame) -> Dict:
        psi = calculate_psi(
            self.reference_data[self.target_column],
            live_data[self.target_column],
        )
        return {
            "drift_detected": psi > 0.1,
            "drift_metrics": {"psi_score": psi},
        }

    def _analyze_distribution_drift(self, live_data: pd.DataFrame) -> Dict:
        result = detect_covariate_shift(self.reference_data, live_data)
        return {
            "drift_detected": result["shift_detected"],
            "covariate_shift_detected": result["shift_detected"],
        }

    def _summarize_drift(self, feature_drift, target_drift, distribution_drift) -> Dict:
        if (
            target_drift["drift_detected"]
            or distribution_drift["drift_detected"]
            or feature_drift["drift_detected"]
        ):
            overall_status = (
                "CRITICAL"
                if target_drift["drift_detected"]
                else "WARNING"
                if feature_drift["drift_detected"]
                else "STABLE"
            )
        else:
            overall_status = "STABLE"

        return {
            "overall_drift_status": overall_status,
            "target_drift_detected": target_drift["drift_detected"],
            "feature_drift_detected": feature_drift["drift_detected"],
            "distribution_drift_detected": distribution_drift["drift_detected"],
        }

    def _generate_alerts(self, summary: Dict) -> list:
        alerts = []
        if summary["overall_drift_status"] == "CRITICAL":
            alerts.append("🚨 CRITICAL: Target drift detected.")
        elif summary["overall_drift_status"] == "WARNING":
            alerts.append("⚠️ Warning: Feature or distribution drift detected.")
        return alerts

    # Continuous Monitoring and Action System
    def start_continuous_monitoring(self):
        """
        Start continuous drift monitoring in a background thread.
        Monitors every 5 minutes and triggers alerts if needed.
        """
        self.is_running = True
        self.logger.info("Starting continuous drift monitoring")

        def monitoring_loop():
            while self.is_running:
                try:
                    results = self.monitor_live_drift(lookback_hours=1)

                    if self._requires_action(results):
                        self._trigger_alerts(results)

                    self._log_monitoring_results(results)

                except Exception as e:
                    self.logger.error(f"Monitoring cycle failed: {e}")

                time.sleep(300)  # every 5 minutes

        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_running = False
        self.logger.info("Stopped continuous drift monitoring")

    def _requires_action(self, results: Dict) -> bool:
        """
        Check if detected drift requires immediate action.
        """
        summary = results["analysis_summary"]
        return (
            summary["overall_drift_status"] == "CRITICAL"
            or summary["target_drift_detected"]
        )

    def _trigger_alerts(self, results: Dict):
        """
        Trigger alerts for significant drift events.
        """
        alerts = results.get("alerts", [])
        for alert in alerts:
            if "🚨" in alert or "CRITICAL" in alert:
                self._send_critical_alert(alert, results)

    def _send_critical_alert(self, alert: str, results: Dict):
        """
        Send critical alert (Slack, PagerDuty, etc.)
        """
        slack_message = {
            "text": f"🚨 DRIFT ALERT: {alert}",
            "attachments": [
                {
                    "text": f"Status: {results['analysis_summary']['overall_drift_status']}",
                    "color": "danger",
                }
            ],
        }

        try:
            # requests.post(self.config['slack_webhook'], json=slack_message)
            self.logger.info(f"CRITICAL ALERT: {alert}")
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    def _log_monitoring_results(self, results: Dict):
        """
        Log monitoring results in a JSONL file for traceability.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "status": results["analysis_summary"]["overall_drift_status"],
            "feature_drift": results["feature_drift"]["drift_detected"],
            "target_drift": results["target_drift"]["drift_detected"],
            "sample_size": results.get("sample_size", 0),
        }

        log_file = Path("monitoring/monitoring_log.jsonl")
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
