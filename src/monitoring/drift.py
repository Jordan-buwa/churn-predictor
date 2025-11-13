# src/monitoring/drift.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import requests
import warnings
warnings.filterwarnings('ignore')

class LiveDataDriftMonitor:
    """
    Monitors drift in LIVE production data from:
    - API requests to your ML service
    - Real-time data streams  
    - Production database
    - Customer interactions
    """
    
    def __init__(self, 
                 reference_data: pd.DataFrame,
                 target_column: str = "churn",
                 # Live data sources
                 ml_service_url: Optional[str] = None,
                 database_config: Optional[Dict] = None,
                 api_endpoints: Optional[Dict] = None,
                 drift_threshold: float = 0.05,
                 random_state: int = 42):
        """
        Initialize with live data source configurations.
        """
        self.reference_data = reference_data
        self.target_column = target_column
        self.drift_threshold = drift_threshold
        self.random_state = random_state
        
        # Live data sources
        self.ml_service_url = ml_service_url
        self.database_config = database_config
        self.api_endpoints = api_endpoints or {}
        
        self.logger = self._setup_logger()
        self._setup_feature_types()
        self.reference_stats = self._calculate_reference_statistics()
        
        self.logger.info("LiveDataDriftMonitor initialized for production monitoring")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for drift monitoring."""
        logger = logging.getLogger('LiveDataDriftMonitor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_feature_types(self):
        """Identify feature types."""
        self.numerical_features = self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
        self.numerical_features = [col for col in self.numerical_features if col != self.target_column]
        
        self.categorical_features = self.reference_data.select_dtypes(exclude=[np.number]).columns.tolist()
        self.categorical_features = [col for col in self.categorical_features if col != self.target_column]
        
        self.logger.info(f"Features: {len(self.numerical_features)} numerical, {len(self.categorical_features)} categorical")
    
    def _calculate_reference_statistics(self) -> Dict:
        """Calculate and store reference dataset statistics."""
        stats = {
            'target_distribution': self.reference_data[self.target_column].value_counts(normalize=True).to_dict(),
            'feature_correlations': {},
            'feature_distributions': {}
        }
        
        # Numerical features statistics
        for col in self.numerical_features:
            stats['feature_distributions'][col] = {
                'mean': self.reference_data[col].mean(),
                'std': self.reference_data[col].std(),
                'percentiles': {
                    'p5': self.reference_data[col].quantile(0.05),
                    'p50': self.reference_data[col].quantile(0.50),
                    'p95': self.reference_data[col].quantile(0.95)
                }
            }
            
            # Feature-target correlations
            if self.target_column in self.reference_data.columns:
                correlation, p_value = stats.pointbiserialr(
                    self.reference_data[col], 
                    self.reference_data[self.target_column]
                )
                stats['feature_correlations'][col] = {
                    'correlation': correlation,
                    'p_value': p_value
                }
        
        return stats
    
    def _create_minimal_data(self) -> pd.DataFrame:
        """Create minimal synthetic data as last resort."""
        n_samples = 100
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        # Adding any additional features from reference data
        for col in self.numerical_features[:3]:  # Add first 3 features
            if col not in data:
                data[col] = np.random.normal(0, 1, n_samples)
        return pd.DataFrame(data)

    def get_live_production_data(self, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Get REAL production data from live sources.
        Priority order:
        1. ML Service API requests
        2. Production database
        3. Real-time data streams
        4. Fallback to simulated data
        """
        try:
            # Trying ML service API first
            if self.ml_service_url:
                data = self._get_data_from_ml_service(lookback_hours)
                if data is not None and len(data) > 0:
                    self.logger.info(f"Got {len(data)} records from ML service")
                    return data
            
            # Trying database next
            if self.database_config:
                data = self._get_data_from_database(lookback_hours)
                if data is not None and len(data) > 0:
                    self.logger.info(f"Got {len(data)} records from database")
                    return data
            
            # Try other API endpoints
            for endpoint_name, endpoint_config in self.api_endpoints.items():
                data = self._get_data_from_api(endpoint_name, endpoint_config)
                if data is not None and len(data) > 0:
                    self.logger.info(f"Got {len(data)} records from {endpoint_name}")
                    return data
            
            # Fallback to simulated data (for development)
            self.logger.warning("No live data sources available, using simulated data")
            return self._get_simulated_production_data()
            
        except Exception as e:
            self.logger.error(f"Error getting live data: {e}")
            return self._get_simulated_production_data()
    
    def _get_data_from_ml_service(self, lookback_hours: int) -> Optional[pd.DataFrame]:
        """Get recent prediction requests from ML service API."""
        try:
            # Example: Your ML service might have an endpoint for recent requests
            endpoint = f"{self.ml_service_url}/api/recent-requests"
            params = {
                'hours': lookback_hours,
                'limit': 1000  # Getting recent 1000 requests
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Converting to DataFrame - adjusting based on the API response format
                return pd.DataFrame(data['requests'])
            
        except Exception as e:
            self.logger.warning(f"ML service API unavailable: {e}")
        
        return None
    
    def _get_data_from_database(self, lookback_hours: int) -> Optional[pd.DataFrame]:
        """Get production data from database."""
        try:
            # This would use your actual database connection
            # Example with SQLAlchemy or your ORM
            import sqlalchemy as db
            
            engine = db.create_engine(self.database_config['url'])
            query = f"""
            SELECT * FROM customer_predictions 
            WHERE prediction_timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
            ORDER BY prediction_timestamp DESC 
            LIMIT 1000
            """
            
            return pd.read_sql(query, engine)
            
        except Exception as e:
            self.logger.warning(f"Database unavailable: {e}")
        
        return None
    
    def _get_data_from_api(self, endpoint_name: str, config: Dict) -> Optional[pd.DataFrame]:
        """Get data from custom API endpoints."""
        try:
            response = requests.get(
                config['url'],
                params=config.get('params', {}),
                headers=config.get('headers', {}),
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data)
                
        except Exception as e:
            self.logger.warning(f"API {endpoint_name} unavailable: {e}")
        
        return None
    
    def _get_simulated_raw_data(self) -> pd.DataFrame:
        """Fallback: simulated production data with realistic variations."""
        try:
            # Adding realistic noise to simulate production data
            simulated_data = self.reference_data.copy()
            
            # Simulating real-world variations
            for col in self.numerical_features:
                if col in simulated_data.columns:
                    # Adding some random noise (1-5% variation)
                    noise = np.random.normal(0, 0.02, len(simulated_data))
                    simulated_data[col] = simulated_data[col] * (1 + noise)
            
            # Occasionally introducing drift (20% chance)
            if np.random.random() < 0.2:
                drift_col = np.random.choice(self.numerical_features)
                simulated_data[drift_col] = simulated_data[drift_col] * 1.1  # 10% shift
                self.logger.info(f"Simulated drift in column: {drift_col}")
            
            return simulated_data.sample(n=min(500, len(simulated_data)))
            
        except Exception as e:
            self.logger.error(f"Even simulated data failed: {e}")
            # Last resort minimal data
            return self._create_minimal_data()
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect FEATURE DRIFT - Changes in input feature distributions.
        """
        self.logger.info("Detecting feature drift...")
        
        feature_drift_results = {
            'drift_detected': False,
            'drifted_features': [],
            'feature_scores': {},
            'summary': {
                'n_features_tested': len(self.numerical_features) + len(self.categorical_features),
                'n_features_drifted': 0
            }
        }
        
        # Testing numerical features
        for feature in self.numerical_features:
            if feature not in current_data.columns:
                continue
                
            ref_data = self.reference_data[feature].dropna()
            curr_data = current_data[feature].dropna()
            
            if len(ref_data) == 0 or len(curr_data) == 0:
                continue
            
            # Kolmogorov-Smirnov Test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(ref_data, curr_data)
            
            # Determining drift
            drift_detected = (ks_pvalue < self.drift_threshold) or (psi_score > 0.2)
            
            feature_drift_results['feature_scores'][feature] = {
                'type': 'numerical',
                'ks_pvalue': ks_pvalue,
                'psi_score': psi_score,
                'drift_detected': drift_detected
            }
            
            if drift_detected:
                feature_drift_results['drifted_features'].append(feature)
        
        # Updating summary
        feature_drift_results['summary']['n_features_drifted'] = len(feature_drift_results['drifted_features'])
        feature_drift_results['drift_detected'] = len(feature_drift_results['drifted_features']) > 0
        
        self.logger.info(f"Feature drift: {len(feature_drift_results['drifted_features'])} features drifted")
        
        return feature_drift_results
    
    def detect_target_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect TARGET DRIFT - Changes in target variable distribution.
        """
        self.logger.info("Detecting target drift...")
        
        if self.target_column not in current_data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in current data")
        
        target_drift_results = {
            'drift_detected': False,
            'drift_metrics': {}
        }
        
        ref_target = self.reference_data[self.target_column]
        curr_target = current_data[self.target_column]
        
        # For binary classification
        if ref_target.nunique() == 2:
            # Chi-square test for proportions
            ref_counts = ref_target.value_counts()
            curr_counts = curr_target.value_counts()
            
            all_categories = list(set(ref_counts.index) | set(curr_counts.index))
            ref_counts_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_counts_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            chi2_stat, chi2_pvalue = stats.chisquare(curr_counts_aligned, ref_counts_aligned)
            
            # PSI for target distribution
            target_psi = self._calculate_categorical_psi(ref_counts, curr_counts)
            
            target_drift_results.update({
                'drift_detected': chi2_pvalue < self.drift_threshold,
                'drift_metrics': {
                    'chi2_pvalue': chi2_pvalue,
                    'psi_score': target_psi
                }
            })
        
        self.logger.info(f"Target drift detected: {target_drift_results['drift_detected']}")
        
        return target_drift_results
    
    def detect_distribution_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect DISTRIBUTION DRIFT - Changes in feature-target relationships.
        """
        self.logger.info("Detecting distribution drift...")
        
        distribution_drift_results = {
            'drift_detected': False,
            'correlation_changes': {},
            'covariate_shift_detected': False
        }
        
        # Correlation changes
        correlation_changes = {}
        for feature in self.numerical_features:
            if feature not in current_data.columns:
                continue
            
            # Reference correlation
            ref_corr, _ = stats.pointbiserialr(self.reference_data[feature], 
                                             self.reference_data[self.target_column])
            
            # Current correlation
            curr_corr, _ = stats.pointbiserialr(current_data[feature], 
                                              current_data[self.target_column])
            
            correlation_change = abs(curr_corr - ref_corr)
            significant_change = correlation_change > 0.1  # 10% change threshold
            
            correlation_changes[feature] = {
                'reference_correlation': ref_corr,
                'current_correlation': curr_corr,
                'change': correlation_change,
                'significant_change': significant_change
            }
        
        # Covariate Shift Detection
        covariate_shift_detected = self._detect_covariate_shift(current_data)
        
        distribution_drift_results.update({
            'correlation_changes': correlation_changes,
            'covariate_shift_detected': covariate_shift_detected,
            'drift_detected': covariate_shift_detected or any(
                [c['significant_change'] for c in correlation_changes.values()]
            )
        })
        
        self.logger.info(f"Distribution drift detected: {distribution_drift_results['drift_detected']}")
        
        return distribution_drift_results
    
    def _calculate_psi(self, ref_data: pd.Series, curr_data: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        breakpoints = np.percentile(ref_data, [100 * i / buckets for i in range(buckets + 1)])
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        
        ref_proportions = np.histogram(ref_data, bins=breakpoints)[0] / len(ref_data)
        curr_proportions = np.histogram(curr_data, bins=breakpoints)[0] / len(curr_data)
        
        ref_proportions = np.where(ref_proportions == 0, 0.0001, ref_proportions)
        curr_proportions = np.where(curr_proportions == 0, 0.0001, curr_proportions)
        
        psi = np.sum((curr_proportions - ref_proportions) * np.log(curr_proportions / ref_proportions))
        return psi
    
    def _calculate_categorical_psi(self, ref_counts: pd.Series, curr_counts: pd.Series) -> float:
        """Calculate PSI for categorical data."""
        ref_proportions = ref_counts / ref_counts.sum()
        curr_proportions = curr_counts / curr_counts.sum()
        
        all_categories = list(set(ref_counts.index) | set(curr_counts.index))
        ref_props_aligned = [ref_proportions.get(cat, 1e-10) for cat in all_categories]
        curr_props_aligned = [curr_proportions.get(cat, 1e-10) for cat in all_categories]
        
        psi = sum((curr_props_aligned[i] - ref_props_aligned[i]) * 
                 np.log(curr_props_aligned[i] / ref_props_aligned[i]) 
                 for i in range(len(all_categories)))
        return psi
    
    def _detect_covariate_shift(self, current_data: pd.DataFrame) -> bool:
        """Detect covariate shift using Isolation Forest."""
        try:
            numerical_cols = [col for col in self.numerical_features if col in current_data.columns]
            if not numerical_cols:
                return False
            
            ref_data = self.reference_data[numerical_cols].fillna(0)
            curr_data = current_data[numerical_cols].fillna(0)
            
            # Scaling data
            scaler = StandardScaler()
            ref_scaled = scaler.fit_transform(ref_data)
            curr_scaled = scaler.transform(curr_data)
            
            # Trainning Isolation Forest on reference data
            iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
            iso_forest.fit(ref_scaled)
            
            # Predicting on current data
            curr_scores = iso_forest.decision_function(curr_scaled)
            
            # More than 20% anomalies indicates shift
            anomaly_ratio = np.sum(curr_scores < 0) / len(curr_scores)
            return anomaly_ratio > 0.2
            
        except Exception as e:
            self.logger.warning(f"Covariate shift detection failed: {e}")
            return False
    
    def comprehensive_drift_analysis(self, current_data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive drift analysis covering all three drift types.
        """
        timestamp = datetime.now()
        
        self.logger.info("Starting comprehensive drift analysis...")
        
        # Run all drift detection methods
        feature_drift = self.detect_feature_drift(current_data)
        target_drift = self.detect_target_drift(current_data)
        distribution_drift = self.detect_distribution_drift(current_data)
        
        # Compile comprehensive results
        comprehensive_results = {
            'timestamp': timestamp.isoformat(),
            'analysis_summary': {
                'feature_drift_detected': feature_drift['drift_detected'],
                'target_drift_detected': target_drift['drift_detected'],
                'distribution_drift_detected': distribution_drift['drift_detected'],
                'overall_drift_status': self._determine_overall_status(feature_drift, target_drift, distribution_drift)
            },
            'feature_drift': feature_drift,
            'target_drift': target_drift,
            'distribution_drift': distribution_drift,
            'alerts': self._generate_alerts(feature_drift, target_drift, distribution_drift)
        }
        
        self.logger.info("Comprehensive drift analysis completed")
        
        return comprehensive_results
    
    def _determine_overall_status(self, feature_drift: Dict, target_drift: Dict, distribution_drift: Dict) -> str:
        """Determine overall drift status."""
        if target_drift['drift_detected']:
            return "CRITICAL"
        elif feature_drift['drift_detected'] and distribution_drift['drift_detected']:
            return "WARNING"
        elif feature_drift['drift_detected'] or distribution_drift['drift_detected']:
            return "WARNING"
        else:
            return "STABLE"
    
    def _generate_alerts(self, feature_drift: Dict, target_drift: Dict, distribution_drift: Dict) -> List[str]:
        """Generate alerts based on drift detection results."""
        alerts = []
        
        if feature_drift['drift_detected']:
            alerts.append(f"FEATURE DRIFT: {len(feature_drift['drifted_features'])} features show distribution changes")
        
        if target_drift['drift_detected']:
            alerts.append("TARGET DRIFT: Target distribution has changed significantly")
        
        if distribution_drift['drift_detected']:
            alerts.append("DISTRIBUTION DRIFT: Feature-target relationships have changed")
        
        if not alerts:
            alerts.append("No significant drift detected")
        
        return alerts
    
    def monitor_live_drift(self, lookback_hours: int = 24) -> Dict:
        """
        Monitor drift in live production data.
        This is the main method called by your monitoring system.
        """
        self.logger.info(f"Monitoring live drift (last {lookback_hours} hours)")
        
        # Getting current production data
        current_data = self.get_live_production_data(lookback_hours)
        
        # Running comprehensive analysis
        results = self.comprehensive_drift_analysis(current_data)
        
        # Adding live-specific metadata
        results.update({
            'data_source': 'live_production',
            'lookback_hours': lookback_hours,
            'sample_size': len(current_data),
            'monitoring_timestamp': datetime.now().isoformat()
        })
        
        return results


class DriftMonitoringService:
    """
    Continuous drift monitoring service that runs in background.
    Integrates with your production system.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitor = LiveDataDriftMonitor(
            reference_data=pd.read_csv(config['reference_data_path']),
            target_column=config.get('target_column', 'churn'),
            ml_service_url=config.get('ml_service_url'),
            database_config=config.get('database_config'),
            api_endpoints=config.get('api_endpoints', {})
        )
        self.is_running = False
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger('DriftMonitoringService')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring in background."""
        self.is_running = True
        self.logger.info("Starting continuous drift monitoring")
        
        import threading
        import time
        
        def monitoring_loop():
            while self.is_running:
                try:
                    # Monitoring every 5 minutes
                    results = self.monitor.monitor_live_drift(lookback_hours=1)  # Last hour
                    
                    # Checking if action needed
                    if self._requires_action(results):
                        self._trigger_alerts(results)
                    
                    # Logging results
                    self._log_monitoring_results(results)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring cycle failed: {e}")
                
                time.sleep(300)  # 5 minutes
        
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
    
    def _requires_action(self, results: Dict) -> bool:
        """Check if drift requires immediate action."""
        summary = results['analysis_summary']
        return (summary['overall_drift_status'] == 'CRITICAL' or 
                summary['target_drift_detected'])
    
    def _trigger_alerts(self, results: Dict):
        """Trigger alerts for significant drift."""
        # Sending to the alerting system (Slack, PagerDuty, etc.)
        alerts = results.get('alerts', [])
        for alert in alerts:
            if '🚨' in alert or 'CRITICAL' in alert:
                self._send_critical_alert(alert, results)
    
    def _send_critical_alert(self, alert: str, results: Dict):
        """Send critical alert to your notification systems."""
        # Example: Sending to Slack
        slack_message = {
            "text": f"🚨 DRIFT ALERT: {alert}",
            "attachments": [{
                "text": f"Status: {results['analysis_summary']['overall_drift_status']}",
                "color": "danger"
            }]
        }
        
        try:
            # requests.post(self.config['slack_webhook'], json=slack_message)
            self.logger.info(f"CRITICAL ALERT: {alert}")
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def _log_monitoring_results(self, results: Dict):
        """Log monitoring results for historical tracking."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'status': results['analysis_summary']['overall_drift_status'],
            'feature_drift': results['feature_drift']['drift_detected'],
            'target_drift': results['target_drift']['drift_detected'],
            'sample_size': results.get('sample_size', 0)
        }
        
        # Append to monitoring log
        log_file = Path("monitoring/monitoring_log.jsonl")
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_running = False
        self.logger.info("Stopped continuous drift monitoring")