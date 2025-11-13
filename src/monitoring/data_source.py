# src/monitoring/data_source.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
from pathlib import Path

class DataSourceHandler:
    """
    Data source handler for loading reference and production data.
    """
    
    def load_reference_data(self, source_config: Dict) -> pd.DataFrame:
        """
        Load reference data from various sources.
        
        Parameters:
        - source_config: Dictionary containing source configuration
        """
        source_type = source_config.get('type', 'csv').lower()
        
        try:
            if source_type == 'csv':
                return self._load_from_csv(source_config)
            elif source_type == 'database':
                return self._load_from_database(source_config)
            elif source_type == 'api':
                return self._load_from_api(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            print(f"Error loading reference data: {e}")
            raise
    
    def _load_from_csv(self, config: Dict) -> pd.DataFrame:
        """Load data from CSV file."""
        file_path = config['path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Loaded reference data from CSV: {df.shape}")
        return df
    
    def _load_from_database(self, config: Dict) -> pd.DataFrame:
        """Load data from database."""
        try:
            import sqlalchemy as db
            
            engine = db.create_engine(config['url'])
            query = config.get('query', 'SELECT * FROM training_data LIMIT 1000')
            
            df = pd.read_sql(query, engine)
            print(f"Loaded reference data from database: {df.shape}")
            return df
            
        except ImportError:
            raise ImportError("sqlalchemy is required for database connections")
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")
    
    def _load_from_api(self, config: Dict) -> pd.DataFrame:
        """Load data from API endpoint."""
        try:
            import requests
            
            response = requests.get(
                config['url'],
                params=config.get('params', {}),
                headers=config.get('headers', {})
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            print(f"Loaded reference data from API: {df.shape}")
            return df
            
        except ImportError:
            raise ImportError("requests is required for API connections")
        except Exception as e:
            raise ConnectionError(f"API connection failed: {e}")
    
    def get_current_production_data(self, source_config: Dict) -> pd.DataFrame:
        """
        Get current production data for monitoring.
        """
        source_type = source_config.get('type', 'csv').lower()
        
        try:
            if source_type == 'csv':
                return self._get_current_from_csv(source_config)
            elif source_type == 'database':
                return self._get_current_from_database(source_config)
            elif source_type == 'api':
                return self._get_current_from_api(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            print(f"Error getting production data: {e}")
            return self._get_fallback_data()
    
    def _get_current_from_csv(self, config: Dict) -> pd.DataFrame:
        """Get current production data from CSV."""
        file_path = config.get('path', 'data/raw/simulated_drifted_sample.csv')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Production CSV not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Loaded production data from CSV: {df.shape}")
        return df
    
    def _get_current_from_database(self, config: Dict) -> pd.DataFrame:
        """Get current production data from database."""
        try:
            import sqlalchemy as db
            
            engine = db.create_engine(config['url'])
            query = config.get('query', """
                SELECT * FROM customer_predictions 
                WHERE prediction_timestamp >= NOW() - INTERVAL '1 hour'
                LIMIT 1000
            """)
            
            df = pd.read_sql(query, engine)
            print(f"Loaded production data from database: {df.shape}")
            return df
            
        except Exception as e:
            raise ConnectionError(f"Database connection failed: {e}")
    
    def _get_current_from_api(self, config: Dict) -> pd.DataFrame:
        """Get current production data from API."""
        try:
            import requests
            
            response = requests.get(
                config['url'],
                params=config.get('params', {}),
                headers=config.get('headers', {}),
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            print(f"Loaded production data from API: {df.shape}")
            return df
            
        except Exception as e:
            raise ConnectionError(f"API connection failed: {e}")
    
    def _get_fallback_data(self) -> pd.DataFrame:
        """Fallback data when no sources are available."""
        print("Using fallback simulated data")
        
        # Create simple synthetic data
        n_samples = 200
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        }
        return pd.DataFrame(data)