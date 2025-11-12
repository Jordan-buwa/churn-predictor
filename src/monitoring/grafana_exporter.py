import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import logging
from typing import Dict
import json
from pathlib import Path

# Import your existing drift module
from drift import ComprehensiveDriftMonitor
from data_source import DataSourceHandler 

# Prometheus for Grafana
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Enum

class SimpleGrafanaExporter:
    """
    Simple Grafana exporter using your existing drift.py
    """
    
    def __init__(self, monitoring_interval: int = 300, prometheus_port: int = 8000):
        self.monitoring_interval = monitoring_interval
        self.prometheus_port = prometheus_port
        self.is_running = False
        
        # Loading data
        self.reference_data = pd.read_csv("data/raw/simulated_realistic_sample.csv")
        
        # Initializing drift monitor
        self.drift_monitor = ComprehensiveDriftMonitor(
            reference_data=self.reference_data,
            target_column="churn"
        )
        
        # Setting up Prometheus metrics
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
        """Simple metrics for Grafana"""
        # Drift status
        self.drift_status = Enum('drift_status', 'Current drift status', 
                               states=['STABLE', 'WARNING', 'CRITICAL'])
        
        # Feature drift
        self.feature_drift_count = Gauge('feature_drift_count', 'Number of drifted features')
        self.feature_drift_ratio = Gauge('feature_drift_ratio', 'Ratio of drifted features')
        
        # Target drift
        self.target_drift_detected = Gauge('target_drift_detected', 'Target drift detected')
        
        # Distribution drift
        self.distribution_drift_detected = Gauge('distribution_drift_detected', 'Distribution drift detected')
        
        # Performance
        self.monitoring_cycle_duration = Histogram('monitoring_cycle_duration', 'Cycle duration in seconds')
        self.data_points_processed = Counter('data_points_processed', 'Total data points processed')
    
    def get_current_data(self):
        """Get current production data - simple version"""
        try:
            return pd.read_csv("data/raw/simulated_drifted_sample.csv")
        except:
            # Fallback to reference data
            return self.reference_data.copy()
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        start_time = time.time()
        
        try:
            current_data = self.get_current_data()
            
            with self.monitoring_cycle_duration.time():
                results = self.drift_monitor.comprehensive_drift_analysis(
                    current_data=current_data,
                    timestamp=datetime.now()
                )
            
            # Updating Prometheus metrics
            self._update_metrics(results, current_data)
            
            self.data_points_processed.inc(len(current_data))
            self.logger.info(f"Cycle completed: {results['analysis_summary']['overall_drift_status']}")
            
        except Exception as e:
            self.logger.error(f"Error in cycle: {e}")
            self.drift_status.state('CRITICAL')
    
    def _update_metrics(self, results: Dict, current_data: pd.DataFrame):
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
        self.target_drift_detected.set(1 if results['target_drift']['drift_detected'] else 0)
        
        # Distribution drift
        self.distribution_drift_detected.set(1 if results['distribution_drift']['drift_detected'] else 0)
    
    def start(self):
        """Start the exporter"""
        self.is_running = True
        
        # Starting Prometheus server
        start_http_server(self.prometheus_port)
        self.logger.info(f"Prometheus metrics on port {self.prometheus_port}")
        
        # Running first cycle
        self.run_monitoring_cycle()
        
        # Starting monitoring loop
        def monitor_loop():
            while self.is_running:
                self.run_monitoring_cycle()
                time.sleep(self.monitoring_interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
        self.logger.info(f"Grafana exporter started (interval: {self.monitoring_interval}s)")
    
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
    print("Using your existing drift.py")
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