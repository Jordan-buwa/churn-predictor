# monitoring/grafana_exporter.py
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json
from pathlib import Path
from advanced_drift import AdvancedDriftMonitor
from data_source import DataSourceHandler
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Enum
import warnings
warnings.filterwarnings('ignore')

class GrafanaDriftExporter:
    """
    Exports drift metrics in Prometheus format for Grafana dashboard.
    """
    def __init__(self, 
                 reference_source_config: Dict,
                 production_source_config: Dict,
                 monitoring_interval: int = 300,
                 prometheus_port: int = 8000,
                 random_state: int = 42):
        
        self.data_handler = DataSourceHandler()
        self.monitoring_interval = monitoring_interval
        self.prometheus_port = prometheus_port
        self.random_state = random_state
        self.is_running = False
        
        # Loading reference data
        self.reference_data = self.data_handler.load_reference_data(reference_source_config)
        
        # Storing source configurations
        self.reference_source_config = reference_source_config
        self.production_source_config = production_source_config
        
        # Initializing drift monitor
        self.drift_monitor = AdvancedDriftMonitor(
            reference_data=self.reference_data,
            target_column="churn"
        )
        
        # Initializing Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Monitoring state
        self.monitoring_history = []
        
        # Setup logging
        self.logger = self._setup_logger()
        
        self.logger.info("GrafanaDriftExporter initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger('GrafanaDriftExporter')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for Grafana."""
        
        # Overall drift metrics
        self.drift_severity = Gauge('drift_severity', 'Overall drift severity', ['severity_level'])
        self.drift_status = Enum('drift_status', 'Current drift status', states=['STABLE', 'WARNING', 'CRITICAL'])
        self.drift_score = Gauge('drift_score', 'Overall drift score')
        
        # Feature drift metrics
        self.feature_drift_detected = Gauge('feature_drift_detected', 'Feature drift detected')
        self.feature_drift_count = Gauge('feature_drift_count', 'Number of drifted features')
        self.feature_drift_ratio = Gauge('feature_drift_ratio', 'Ratio of drifted features')
        
        # Target drift metrics
        self.target_drift_detected = Gauge('target_drift_detected', 'Target drift detected')
        self.target_psi_score = Gauge('target_psi_score', 'Target PSI score')
        self.target_proportion_change = Gauge('target_proportion_change', 'Target proportion change')
        
        # Distribution drift metrics
        self.distribution_drift_detected = Gauge('distribution_drift_detected', 'Distribution drift detected')
        self.covariate_shift_ratio = Gauge('covariate_shift_ratio', 'Covariate shift ratio')
        self.correlation_change_count = Gauge('correlation_change_count', 'Number of features with correlation changes')
        
        # Statistical test metrics
        self.ks_test_pvalue = Gauge('ks_test_pvalue', 'KS test p-value', ['feature'])
        self.psi_score = Gauge('psi_score', 'PSI score', ['feature'])
        self.js_divergence = Gauge('js_divergence', 'Jensen-Shannon divergence', ['feature'])
        
        # Performance metrics
        self.monitoring_cycle_duration = Histogram('monitoring_cycle_duration', 'Duration of monitoring cycle in seconds')
        self.data_points_processed = Counter('data_points_processed', 'Total data points processed')
        self.alerts_triggered = Counter('alerts_triggered', 'Total alerts triggered', ['alert_type'])
        
        # System metrics
        self.uptime_seconds = Gauge('uptime_seconds', 'System uptime in seconds')
        self.last_successful_check = Gauge('last_successful_check', 'Timestamp of last successful check')
    
    def get_current_production_data(self) -> pd.DataFrame:
        """Get current production data."""
        return self.data_handler.get_current_production_data(self.production_source_config)
    
    def run_monitoring_cycle(self):
        """Run monitoring cycle and export metrics."""
        start_time = time.time()
        
        try:
            self.logger.info("Running monitoring cycle for Grafana...")
            
            # Getting current production data
            current_data = self.get_current_production_data()
            
            # Running comprehensive drift analysis
            with self.monitoring_cycle_duration.time():
                results = self.drift_monitor.comprehensive_drift_analysis(
                    current_data=current_data,
                    timestamp=datetime.now()
                )
            
            # Updating Prometheus metrics
            self._update_prometheus_metrics(results, current_data)
            
            # Updating monitoring history
            self.monitoring_history.append({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'data_shape': current_data.shape
            })
            
            # Keeping history within limits
            if len(self.monitoring_history) > 1000:
                self.monitoring_history = self.monitoring_history[-1000:]
            
            # Updating system metrics
            self.last_successful_check.set_to_current_time()
            self.data_points_processed.inc(len(current_data))
            
            self.logger.info(f"Monitoring cycle completed. Status: {results['analysis_summary']['overall_drift_status']}")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
            # Set error state in metrics
            self.drift_status.state('CRITICAL')
    
    def _update_prometheus_metrics(self, results: Dict, current_data: pd.DataFrame):
        """Update all Prometheus metrics with current results."""
        
        # Overall drift metrics
        severity_map = {'CRITICAL': 3, 'WARNING': 2, 'NONE': 1, 'STABLE': 1}
        overall_severity = severity_map.get(results['analysis_summary']['overall_drift_status'], 1)
        
        self.drift_severity.labels(severity_level=results['analysis_summary']['overall_drift_status']).set(overall_severity)
        self.drift_status.state(results['analysis_summary']['overall_drift_status'])
        self.drift_score.set(overall_severity / 3.0)  # Normalize to 0-1
        
        # Feature drift metrics
        feature_drift = results['feature_drift']
        drifted_features = [f for f in feature_drift.get('drifted_features', []) 
                          if f.get('severity') in ['WARNING', 'CRITICAL']]
        
        self.feature_drift_detected.set(1 if feature_drift['drift_detected'] else 0)
        self.feature_drift_count.set(len(drifted_features))
        
        total_features = len(feature_drift.get('feature_scores', {}))
        drift_ratio = len(drifted_features) / total_features if total_features > 0 else 0
        self.feature_drift_ratio.set(drift_ratio)
        
        # Target drift metrics
        target_drift = results['target_drift']
        self.target_drift_detected.set(1 if target_drift['drift_detected'] else 0)
        self.target_psi_score.set(target_drift.get('drift_metrics', {}).get('psi_score', 0))
        
        max_prop_change = target_drift.get('drift_metrics', {}).get('max_proportion_change', 0)
        self.target_proportion_change.set(max_prop_change)
        
        # Distribution drift metrics
        distribution_drift = results['distribution_drift']
        self.distribution_drift_detected.set(1 if distribution_drift['drift_detected'] else 0)
        
        covariate_shift = distribution_drift.get('covariate_shift', {})
        self.covariate_shift_ratio.set(covariate_shift.get('anomaly_ratio', 0))
        
        correlation_changes = distribution_drift.get('correlation_changes', {})
        self.correlation_change_count.set(correlation_changes.get('significant_changes', 0))
        
        # Feature-level statistical metrics
        feature_scores = feature_drift.get('feature_scores', {})
        for feature_name, scores in list(feature_scores.items())[:20]:  # Limit to top 20 features
            if 'ks_pvalue' in scores:
                self.ks_test_pvalue.labels(feature=feature_name).set(scores['ks_pvalue'])
            if 'psi' in scores:
                self.psi_score.labels(feature=feature_name).set(scores['psi'])
            if 'js_divergence' in scores:
                self.js_divergence.labels(feature=feature_name).set(scores['js_divergence'])
        
        # Alert metrics
        alerts = results.get('alerts', [])
        for alert in alerts:
            alert_type = self._classify_alert_type(alert)
            self.alerts_triggered.labels(alert_type=alert_type).inc()
    
    def _classify_alert_type(self, alert: str) -> str:
        """Classify alert type for metrics."""
        if 'FEATURE DRIFT' in alert:
            return 'feature_drift'
        elif 'TARGET DRIFT' in alert:
            return 'target_drift'
        elif 'DISTRIBUTION DRIFT' in alert:
            return 'distribution_drift'
        elif 'CRITICAL' in alert:
            return 'critical'
        else:
            return 'info'
    
    def start_monitoring(self):
        """Start the monitoring service and Prometheus server."""
        self.is_running = True
        
        # Starting Prometheus HTTP server
        start_http_server(self.prometheus_port)
        self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        
        # Running initial monitoring cycle
        self.run_monitoring_cycle()
        
        # Starting uptime counter
        self._start_uptime_counter()
        
        # Starting scheduled monitoring
        def monitoring_loop():
            while self.is_running:
                self.run_monitoring_cycle()
                time.sleep(self.monitoring_interval)
        
        # Starting monitoring in background thread
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info(f"Grafana drift monitoring started (interval: {self.monitoring_interval}s)")
    
    def _start_uptime_counter(self):
        """Start uptime counter in background."""
        def uptime_loop():
            start_time = time.time()
            while self.is_running:
                self.uptime_seconds.set(time.time() - start_time)
                time.sleep(1)
        
        uptime_thread = threading.Thread(target=uptime_loop, daemon=True)
        uptime_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring service."""
        self.is_running = False
        self.logger.info("Grafana drift monitoring stopped")


# Main execution
def main():
    """Start Grafana-compatible drift monitoring."""
    exporter = GrafanaDriftExporter(
        reference_source_config={
            'type': 'csv',
            'path': 'data/raw/simulated_realistic_sample.csv'
        },
        production_source_config={
            'type': 'csv',
            'path': 'data/raw/simulated_drifted_sample.csv'
        },
        monitoring_interval=300,  # 5 minutes
        prometheus_port=8000
    )
    
    print("=== Starting Grafana-Compatible Drift Monitoring ===")
    print("Prometheus metrics available at: http://localhost:8000/metrics")
    print("Monitoring interval: 300 seconds")
    print("Drift types: Feature, Target, Distribution")
    print("\nTo set up Grafana:")
    print("1. Add Prometheus data source: http://localhost:8000")
    print("2. Import the dashboard JSON provided below")
    print("3. Start monitoring!")
    
    try:
        exporter.start_monitoring()
        
        # Keeping the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n Stopping monitoring...")
        exporter.stop_monitoring()
        print("Monitoring stopped")

if __name__ == "__main__":
    main()