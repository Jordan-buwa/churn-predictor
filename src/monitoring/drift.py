# monitoring/live_monitor.py
import pandas as pd
import numpy as np
import time
import schedule
import threading
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import json
from pathlib import Path
from drift import ComprehensiveDriftMonitor
import warnings
warnings.filterwarnings('ignore')

class LiveDriftMonitor:
    """
    Live drift monitoring service that runs continuously and provides real-time alerts.
    """
    
    def __init__(self, reference_data_path: str, 
                 monitoring_interval: int = 300,  # 5 minutes
                 dashboard_refresh: int = 30,     # 30 seconds
                 random_state: int = 42):
        """
        Initializing live monitoring service.
        Parameters:
        - reference_data_path: Path to reference data
        - monitoring_interval: Seconds between monitoring cycles
        - dashboard_refresh: Seconds between dashboard updates
        """
        self.reference_data = pd.read_csv(reference_data_path)
        self.monitoring_interval = monitoring_interval
        self.dashboard_refresh = dashboard_refresh
        self.random_state = random_state
        self.is_running = False
        
        # Initialize drift monitor
        self.drift_monitor = ComprehensiveDriftMonitor(
            reference_data=self.reference_data,
            target_column="churn",
            drift_threshold=0.05
        )
        
        # Monitoring state
        self.monitoring_history = []
        self.alerts = []
        self.current_status = "UNKNOWN"
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Create dashboard directory
        self.dashboard_dir = Path("monitoring/dashboard")
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("LiveDriftMonitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for live monitoring."""
        logger = logging.getLogger('LiveDriftMonitor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_current_production_data(self) -> pd.DataFrame:
        """
        Simulate getting current production data.
        In reality, this would connect to your database, API, or data stream.
        """
        try:
            # Simulate: sometimes return realistic data, sometimes drifted data
            if np.random.random() > 0.7:  # 30% chance of drifted data
                current_data = pd.read_csv("data/raw/simulated_drifted_sample.csv")
                self.logger.info("Using drifted data sample")
            else:
                current_data = pd.read_csv("data/raw/simulated_realistic_sample.csv")
                self.logger.info("Using realistic data sample")
            
            # Add some real-time noise to simulate actual production data
            for col in current_data.select_dtypes(include=[np.number]).columns:
                if col != 'churn':
                    noise = np.random.normal(0, 0.01, len(current_data))
                    current_data[col] = current_data[col] * (1 + noise)
            
            return current_data.sample(n=min(1000, len(current_data)), random_state=self.random_state)
            
        except Exception as e:
            self.logger.error(f"Error getting production data: {e}")
            # Return a small sample from reference data as fallback
            return self.reference_data.sample(n=100, random_state=self.random_state)
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle."""
        try:
            self.logger.info("Starting monitoring cycle...")
            
            # Get current production data
            current_data = self.get_current_production_data()
            
            # Run comprehensive drift analysis
            results = self.drift_monitor.comprehensive_drift_analysis(
                current_data=current_data,
                timestamp=datetime.now()
            )
            
            # Update monitoring history
            self.monitoring_history.append({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'data_shape': current_data.shape
            })
            
            # Keep only last 1000 cycles to prevent memory issues
            if len(self.monitoring_history) > 1000:
                self.monitoring_history = self.monitoring_history[-1000:]
            
            # Update current status
            self.current_status = results['analysis_summary']['overall_drift_status']
            
            # Process alerts
            self._process_alerts(results)
            
            # Update dashboard
            self._update_dashboard(results)
            
            self.logger.info(f"Monitoring cycle completed. Status: {self.current_status}")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
    
    def _process_alerts(self, results: Dict):
        """Process and store alerts from drift analysis."""
        current_alerts = results.get('alerts', [])
        
        # Only store new critical alerts
        critical_alerts = [alert for alert in current_alerts if '🚨' in alert or 'CRITICAL' in alert]
        
        for alert in critical_alerts:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'alert': alert,
                'severity': 'CRITICAL',
                'status': results['analysis_summary']['overall_drift_status']
            }
            
            # Avoid duplicate alerts in short time window
            if not self._is_duplicate_alert(alert_record):
                self.alerts.append(alert_record)
                self.logger.warning(f"🚨 ALERT: {alert}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _is_duplicate_alert(self, new_alert: Dict) -> bool:
        """Check if similar alert was recently raised."""
        recent_threshold = datetime.now() - timedelta(minutes=30)
        
        for alert in self.alerts[-10:]:  # Check last 10 alerts
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if (alert_time > recent_threshold and 
                alert['alert'] == new_alert['alert']):
                return True
        return False
    
    def _update_dashboard(self, results: Dict):
        """Update the real-time dashboard files."""
        try:
            # Create simplified dashboard data
            dashboard_data = {
                'last_updated': datetime.now().isoformat(),
                'current_status': self.current_status,
                'drift_metrics': {
                    'feature_drift': results['analysis_summary']['feature_drift_detected'],
                    'target_drift': results['analysis_summary']['target_drift_detected'],
                    'distribution_drift': results['analysis_summary']['distribution_drift_detected'],
                    'severity': results['analysis_summary']['drift_severity']
                },
                'recent_alerts': self.alerts[-5:],  # Last 5 alerts
                'monitoring_stats': {
                    'total_cycles': len(self.monitoring_history),
                    'data_points_processed': sum([h['data_shape'][0] for h in self.monitoring_history[-10:]])
                }
            }
            
            # Save dashboard data as JSON
            dashboard_file = self.dashboard_dir / "live_dashboard.json"
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            # Create HTML dashboard
            self._create_html_dashboard(dashboard_data, results)
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard: {e}")
    
    def _create_html_dashboard(self, dashboard_data: Dict, results: Dict):
        """Create a simple HTML dashboard for real-time viewing."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Live Drift Monitoring Dashboard</title>
            <meta http-equiv="refresh" content="{self.dashboard_refresh}">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .status-critical {{ color: #d63031; font-weight: bold; }}
                .status-warning {{ color: #e17055; font-weight: bold; }}
                .status-stable {{ color: #00b894; font-weight: bold; }}
                .alert {{ background: #ffeaa7; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .metric {{ background: #dfe6e9; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            </style>
        </head>
        <body>
            <h1>🔍 Live Drift Monitoring Dashboard</h1>
            <p>Last updated: {dashboard_data['last_updated']}</p>
            
            <div class="dashboard">
                <div>
                    <h2>Current Status</h2>
                    <div class="status-{dashboard_data['current_status'].lower()}">
                        Overall Status: {dashboard_data['current_status']}
                    </div>
                    <div class="metric">
                        <strong>Severity:</strong> {dashboard_data['drift_metrics']['severity']}
                    </div>
                    
                    <h3>Drift Detection</h3>
                    <div class="metric">
                        Feature Drift: {'🚨 DETECTED' if dashboard_data['drift_metrics']['feature_drift'] else '✅ STABLE'}
                    </div>
                    <div class="metric">
                        Target Drift: {'🚨 DETECTED' if dashboard_data['drift_metrics']['target_drift'] else '✅ STABLE'}
                    </div>
                    <div class="metric">
                        Distribution Drift: {'🚨 DETECTED' if dashboard_data['drift_metrics']['distribution_drift'] else '✅ STABLE'}
                    </div>
                </div>
                
                <div>
                    <h2>Recent Alerts</h2>
                    {"".join([f'<div class="alert">{alert["alert"]}<br><small>{alert["timestamp"]}</small></div>' 
                             for alert in dashboard_data['recent_alerts']]) 
                             or '<div class="metric">No recent alerts</div>'}
                    
                    <h3>Monitoring Statistics</h3>
                    <div class="metric">
                        Total Cycles: {dashboard_data['monitoring_stats']['total_cycles']}
                    </div>
                    <div class="metric">
                        Recent Data Points: {dashboard_data['monitoring_stats']['data_points_processed']}
                    </div>
                </div>
            </div>
            
            <hr>
            <p><small>Auto-refreshing every {self.dashboard_refresh} seconds</small></p>
        </body>
        </html>
        """
        
        dashboard_file = self.dashboard_dir / "index.html"
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
    
    def start_monitoring(self):
        """Start the live monitoring service."""
        self.is_running = True
        self.logger.info(f"Starting live drift monitoring (interval: {self.monitoring_interval}s)")
        
        # Run initial monitoring cycle
        self.run_monitoring_cycle()
        
        # Start scheduled monitoring
        def monitoring_loop():
            while self.is_running:
                self.run_monitoring_cycle()
                time.sleep(self.monitoring_interval)
        
        # Start monitoring in background thread
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Live monitoring started successfully")
    
    def stop_monitoring(self):
        """Stop the live monitoring service."""
        self.is_running = False
        self.logger.info("Live monitoring stopped")
    
    def get_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            'is_running': self.is_running,
            'current_status': self.current_status,
            'total_cycles': len(self.monitoring_history),
            'recent_alerts': len([a for a in self.alerts if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]),
            'last_update': self.monitoring_history[-1]['timestamp'] if self.monitoring_history else None
        }


# Web API for real-time access
from flask import Flask, jsonify, send_file

app = Flask(__name__)
live_monitor = None

@app.route('/api/status')
def get_status():
    """API endpoint to get current monitoring status."""
    if live_monitor:
        return jsonify(live_monitor.get_status())
    return jsonify({'error': 'Monitor not initialized'})

@app.route('/api/alerts')
def get_alerts():
    """API endpoint to get recent alerts."""
    if live_monitor:
        return jsonify({'alerts': live_monitor.alerts[-10:]})
    return jsonify({'error': 'Monitor not initialized'})

@app.route('/api/dashboard')
def get_dashboard():
    """API endpoint to serve the dashboard."""
    return send_file('monitoring/dashboard/index.html')

@app.route('/api/data')
def get_dashboard_data():
    """API endpoint to get dashboard data."""
    try:
        with open('monitoring/dashboard/live_dashboard.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({'error': 'Dashboard data not available'})

def start_web_api(host='0.0.0.0', port=5000):
    """Start the web API for real-time access."""
    app.run(host=host, port=port, debug=False)

# Main execution
def main():
    """Start live monitoring with web interface."""
    global live_monitor
    
    print("=== Starting Live Drift Monitoring ===")
    
    # Initialize live monitor
    live_monitor = LiveDriftMonitor(
        reference_data_path="data/raw/simulated_realistic_sample.csv",
        monitoring_interval=300,  # 5 minutes
        dashboard_refresh=30      # 30 seconds
    )
    
    # Start monitoring
    live_monitor.start_monitoring()
    
    print("Live monitoring started")
    print("Dashboard available at: http://localhost:5000/api/dashboard")
    print("API endpoints:")
    print("   - http://localhost:5000/api/status")
    print("   - http://localhost:5000/api/alerts") 
    print("   - http://localhost:5000/api/data")
    print("\nPress Ctrl+C to stop monitoring")
    
    try:
        # Start web API
        start_web_api()
    except KeyboardInterrupt:
        print("\n Stopping live monitoring...")
        live_monitor.stop_monitoring()
        print("Live monitoring stopped")

if __name__ == "__main__":
    main()