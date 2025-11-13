# src/monitoring/grafana_exporter.py
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import logging
from typing import Dict
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Enum

# Import your drift monitor
from src.monitoring.drift import LiveDataDriftMonitor

class SimpleGrafanaExporter:
    """
    Simple Grafana exporter using your drift monitoring.
    """
    
    def __init__(self, monitoring_interval: int = 300, prometheus_port: int = 8000):
        self.monitoring_interval = monitoring_interval
        self.prometheus_port = prometheus_port
        self.is_running = False
        
        # Load reference data
        self.reference_data = pd.read_csv("data/raw/simulated_realistic_sample.csv")
        
        # Initialize your drift monitor
        self.drift_monitor = LiveDataDriftMonitor(
            reference_data=self.reference_data,
            target_column="churn"
        )
        
        # Setup Prometheus metrics
        self._setup_prometheus_metrics()
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('GrafanaExporter')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for Grafana"""
        # Drift status
        self.drift_status = Enum('drift_status', 'Current drift status', 
                               states=['STABLE', 'WARNING', 'CRITICAL'])
        
        # Feature drift
        self.feature_drift_count = Gauge('feature_drift_count', 'Number of drifted features')
        self.feature_drift_ratio = Gauge('feature_drift_ratio', 'Ratio of drifted features')
        
        # Target drift
        self.target_drift_detected = Gauge('target_drift_detected', 'Target drift detected')
        self.target_psi_score = Gauge('target_psi_score', 'Target PSI score')
        
        # Distribution drift
        self.distribution_drift_detected = Gauge('distribution_drift_detected', 'Distribution drift detected')
        self.covariate_shift_ratio = Gauge('covariate_shift_ratio', 'Covariate shift ratio')
        
        # Performance
        self.monitoring_cycle_duration = Histogram('monitoring_cycle_duration', 'Cycle duration in seconds')
        self.data_points_processed = Counter('data_points_processed', 'Total data points processed')
        self.alerts_triggered = Counter('alerts_triggered', 'Total alerts triggered', ['alert_type'])
        
        # System metrics
        self.uptime_seconds = Gauge('uptime_seconds', 'System uptime in seconds')
        self.last_successful_check = Gauge('last_successful_check', 'Timestamp of last successful check')
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        start_time = time.time()
        
        try:
            # Monitor live drift
            with self.monitoring_cycle_duration.time():
                results = self.drift_monitor.monitor_live_drift(lookback_hours=1)
            
            # Update Prometheus metrics
            self._update_metrics(results)
            
            self.data_points_processed.inc(results.get('sample_size', 0))
            self.last_successful_check.set_to_current_time()
            
            self.logger.info(f"Cycle completed: {results['analysis_summary']['overall_drift_status']}")
            
        except Exception as e:
            self.logger.error(f"Error in cycle: {e}")
            self.drift_status.state('CRITICAL')
    
    def _update_metrics(self, results: Dict):
        """Update Prometheus metrics with results"""
        # Overall status
        self.drift_status.state(results['analysis_summary']['overall_drift_status'])
        
        # Feature drift
        feature_drift = results['feature_drift']
        drifted_features = len(feature_drift.get('drifted_features', []))
        total_features = len(feature_drift.get('feature_scores', {}))
        
        self.feature_drift_count.set(drifted_features)
        self.feature_drift_ratio.set(drifted_features / total_features if total_features > 0 else 0)
        
        # Target drift
        target_drift = results['target_drift']
        self.target_drift_detected.set(1 if target_drift['drift_detected'] else 0)
        self.target_psi_score.set(target_drift.get('drift_metrics', {}).get('psi_score', 0))
        
        # Distribution drift
        distribution_drift = results['distribution_drift']
        self.distribution_drift_detected.set(1 if distribution_drift['drift_detected'] else 0)
        
        # Covariate shift (simplified)
        covariate_shift = 1.0 if distribution_drift.get('covariate_shift_detected', False) else 0.0
        self.covariate_shift_ratio.set(covariate_shift)
        
        # Alert metrics
        alerts = results.get('alerts', [])
        for alert in alerts:
            if 'FEATURE DRIFT' in alert:
                self.alerts_triggered.labels(alert_type='feature_drift').inc()
            elif 'TARGET DRIFT' in alert:
                self.alerts_triggered.labels(alert_type='target_drift').inc()
            elif 'DISTRIBUTION DRIFT' in alert:
                self.alerts_triggered.labels(alert_type='distribution_drift').inc()
    
    def start(self):
        """Start the exporter"""
        self.is_running = True
        
        # Start Prometheus server
        start_http_server(self.prometheus_port)
        self.logger.info(f"Prometheus metrics on port {self.prometheus_port}")
        
        # Start uptime counter
        self._start_uptime_counter()
        
        # Run first cycle
        self.run_monitoring_cycle()
        
        # Start monitoring loop
        def monitor_loop():
            while self.is_running:
                self.run_monitoring_cycle()
                time.sleep(self.monitoring_interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
        self.logger.info(f"Grafana exporter started (interval: {self.monitoring_interval}s)")
    
    def _start_uptime_counter(self):
        """Start uptime counter in background"""
        def uptime_loop():
            start_time = time.time()
            while self.is_running:
                self.uptime_seconds.set(time.time() - start_time)
                time.sleep(1)
        
        uptime_thread = threading.Thread(target=uptime_loop, daemon=True)
        uptime_thread.start()
    
    def stop(self):
        """Stop the exporter"""
        self.is_running = False
        self.logger.info("Grafana exporter stopped")

def main():
    """Main function to start everything"""
    exporter = SimpleGrafanaExporter(
        monitoring_interval=300,  # 5 minutes
        prometheus_port=8000
    )
    
    print("Starting Simple Grafana Drift Monitor")
    print("Metrics: http://localhost:8000/metrics")
    print("Interval: 300 seconds")
    print("Monitoring: Feature, Target, Distribution Drift")
    print("\nPress Ctrl+C to stop")
    
    try:
        exporter.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n Stopping...")
        exporter.stop()

if __name__ == "__main__":
    main()