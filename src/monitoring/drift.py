import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance

# Prometheus client
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# Database helpers from your repo (uses connection pool)
from src.api.utils.database import get_db_connection  # (contextmanager)
from src.api.db import UserRole, User, get_db  # SQLModel session helper and User model
from sqlmodel import Session, select

# Preprocessor used in prediction.py — use the same to align features
from src.data_pipeline.preprocess import ProductionPreprocessor
from src.api.utils.config import APIConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(ch)


# Prometheus metrics (created once) 
PROM_METRIC_PREFIX = "model_drift"
g_drift_percentage = Gauge(f"{PROM_METRIC_PREFIX}_drift_percentage", "Fraction of features flagged as drifting (0-1)")
g_total_features = Gauge(f"{PROM_METRIC_PREFIX}_total_features", "Total features evaluated for drift")
g_drifted_features = Gauge(f"{PROM_METRIC_PREFIX}_drifted_features", "Number of features flagged as drifting")
# Per-feature gauges will be registered dynamically
_PER_FEATURE_GAUGES: Dict[str, Dict[str, Gauge]] = {}

def _ensure_feature_gauges(feature_name: str):
    """Create per-feature gauges for psi/ks/wasserstein/p_value if not present."""
    if feature_name in _PER_FEATURE_GAUGES:
        return _PER_FEATURE_GAUGES[feature_name]
    prefix = f"{PROM_METRIC_PREFIX}_feature_{feature_name}"
    gauges = {
        "drift": Gauge(f"{prefix}_drift_flag", f"Drift flag (0/1) for {feature_name}"),
        "p_value": Gauge(f"{prefix}_pvalue", f"Statistical test p-value for {feature_name}"),
        "wasserstein": Gauge(f"{prefix}_wasserstein", f"Wasserstein distance for {feature_name}"),
        "ref_mean": Gauge(f"{prefix}_ref_mean", f"Reference mean for {feature_name}"),
        "prod_mean": Gauge(f"{prefix}_prod_mean", f"Production mean for {feature_name}")
    }
    _PER_FEATURE_GAUGES[feature_name] = gauges
    return gauges


class DriftDetector:
    """
    Drift detector that:
      - uses preprocessed training CSV as reference
      - reads production data from your database (customer_data.features JSON or numeric columns)
      - supports row-by-row mode (read id > last_processed_id) and batch mode (read recent batch_id)
      - enforces admin-only access when loading production data
      - exports Prometheus metrics
    """

    # Reasonable defaults chosen as you requested
    DRIFT_WARNING = 0.15
    DRIFT_CRITICAL = 0.30
    BATCH_SIZE = 500  # rows to read at a time for row-by-row mode
    LOOKBACK_INTERVAL = timedelta(hours=1)  # for batch mode: consider last 1 hour by default

    # drift_state table name (created if not exists)
    DRIFT_STATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS drift_state (
        id SERIAL PRIMARY KEY,
        last_processed_id BIGINT,
        last_batch_id VARCHAR(255),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, reference_path: Optional[str] = None, artifacts_path: Optional[str] = None):
        self.config = APIConfig() if 'APIConfig' in globals() else None
        # default reference CSV path (from your message)
        self.reference_path = reference_path or os.path.join("data", "processed", "processed_data.csv")
        # Preprocessor used to align DB features to model features
        artifacts = artifacts_path or (self.config.preprocessing_artifacts_path if self.config else None)
        self.processor = ProductionPreprocessor(artifacts_path=artifacts) if artifacts else None

    #  Utility / DB helpers
    def _ensure_drift_state_table(self):
        """Create drift_state table if missing using get_db_connection()."""
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(self.DRIFT_STATE_TABLE_SQL)
                cur.close()
        except Exception as e:
            logger.error(f"Unable to ensure drift_state table: {e}")
            raise

    def _get_last_state(self) -> Dict[str, Any]:
        """Return the last processed id and batch id stored in drift_state (or defaults)."""
        self._ensure_drift_state_table()
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT last_processed_id, last_batch_id, updated_at FROM drift_state ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            cur.close()
        if not row or row[0] is None and row[1] is None:
            return {"last_processed_id": 0, "last_batch_id": None, "updated_at": None}
        return {"last_processed_id": row[0], "last_batch_id": row[1], "updated_at": row[2]}

    def _update_state(self, last_processed_id: Optional[int] = None, last_batch_id: Optional[str] = None):
        """Insert new drift_state row (simple append-only ledger style)."""
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO drift_state (last_processed_id, last_batch_id, updated_at) VALUES (%s, %s, CURRENT_TIMESTAMP)",
                (last_processed_id, last_batch_id)
            )
            cur.close()

    def _check_admin(self, user_id: int):
        """Verify user is admin using SQLModel User model (get_db yields sessions)."""
        # get_db is defined in your src.api.db and returns a generator yielding Session
        try:
            # try to use SQLModel session helper if available
            for s in get_db():
                session: Session = s
                break
            else:
                raise RuntimeError("No DB session available for user check")
            user = session.get(User, user_id)
            if user is None:
                raise PermissionError("User not found")
            if user.role != UserRole.ADMIN:
                raise PermissionError("User is not admin")
            return True
        except Exception as e:
            logger.error(f"Admin check failed: {e}")
            raise

    # Data loaders
    def load_reference(self) -> pd.DataFrame:
        """Load reference preprocessed CSV into a DataFrame (used for drift)."""
        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(f"Reference file not found at {self.reference_path}")

        df_ref = pd.read_csv(self.reference_path)
        logger.info(f"Loaded reference (training) data from {self.reference_path} shape={df_ref.shape}")
        return df_ref

    def _rows_to_df(self, rows: List[Dict]) -> pd.DataFrame:
        """
        Convert DB rows to dataframe aligned with preprocessor:
        - rows expected as dicts with a 'features' JSONB column OR direct columns.
        - ensure we produce a dataframe with the same feature columns used during training.
        """
        # Extract features dicts
        extracted = []
        for r in rows:
            # prefer 'features' JSONB column
            features = r.get("features")
            if isinstance(features, str):
                try:
                    features = json.loads(features)
                except Exception:
                    # fallback: if the row contains many columns, reconstruct
                    features = {k: v for k, v in r.items() if k not in ("id", "customer_id", "timestamp", "batch_id", "source", "features")}
            if features is None:
                # fallback to straight row values (filter metadata)
                features = {k: v for k, v in r.items() if k not in ("id", "customer_id", "timestamp", "batch_id", "source", "features")}
            extracted.append(features)

        if not extracted:
            return pd.DataFrame()

        df = pd.DataFrame(extracted)
        # If processor is available, we use it to get consistent feature set (preferred)
        if self.processor:
            # Processor expects a DF with raw columns — taking a copy and running preprocess
            try:
                processed = self.processor.preprocess(df.copy())
                feature_names = self.processor.get_feature_names()
                processed = processed[feature_names]
                return processed
            except Exception as e:
                logger.warning(f"Processor preprocessing failed, falling back to raw df: {e}")
        # Otherwise, return df and rely on caller to align columns
        return df

    def load_production_rows_row_by_row(self, batch_size: int = None) -> Tuple[pd.DataFrame, int]:
        """
        Row-by-row ingestion mode: read rows with id > last_processed_id using drift_state marker.
        Returns (df, last_id_read)
        """
        batch_size = batch_size or self.BATCH_SIZE
        state = self._get_last_state()
        last_id = state["last_processed_id"] or 0

        sql = """
        SELECT id, customer_id, features, timestamp, batch_id
        FROM customer_data
        WHERE id > %s
        ORDER BY id
        LIMIT %s
        """
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, (last_id, batch_size))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            cur.close()

        if not rows:
            logger.info("No new rows found for row-by-row mode.")
            return pd.DataFrame(), last_id

        df = self._rows_to_df(rows)
        new_last_id = max(r["id"] for r in rows)
        logger.info(f"Loaded {len(rows)} rows (id {last_id+1}..{new_last_id}) in row-by-row mode.")
        return df, new_last_id

    def load_production_rows_batch_mode(self, lookback: Optional[timedelta] = None, limit: int = 1000) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Batch ingestion mode: read the most recent batch rows.
        Strategy:
          - If ingestion logs (ingestion_logs) exists, pick latest batch_id in last lookback interval.
          - Else fall back to selecting rows with timestamp within lookback interval.
        Returns (df, batch_id_used)
        """
        lookback = lookback or self.LOOKBACK_INTERVAL
        since = datetime.utcnow() - lookback

        # Attempt 1: find latest batch_id from ingestion_logs within window
        with get_db_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT batch_id FROM ingestion_logs WHERE timestamp >= %s ORDER BY timestamp DESC LIMIT 1",
                    (since,)
                )
                row = cur.fetchone()
            except Exception:
                row = None
            cur.close()

        batch_id = row[0] if row else None

        rows = []
        with get_db_connection() as conn:
            cur = conn.cursor()
            if batch_id:
                cur.execute(
                    "SELECT id, customer_id, features, timestamp, batch_id FROM customer_data WHERE batch_id = %s ORDER BY timestamp LIMIT %s",
                    (batch_id, limit)
                )
            else:
                cur.execute(
                    "SELECT id, customer_id, features, timestamp, batch_id FROM customer_data WHERE timestamp >= %s ORDER BY timestamp LIMIT %s",
                    (since, limit)
                )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            cur.close()

        if not rows:
            logger.info("No rows found in batch mode for the given window.")
            return pd.DataFrame(), batch_id

        df = self._rows_to_df(rows)
        logger.info(f"Loaded {len(rows)} rows for batch_id={batch_id} (batch mode).")
        return df, batch_id


    # Statistical tests
    @staticmethod
    def _numeric_drift(ref: pd.Series, prod: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        if ref.dropna().shape[0] < 2 or prod.dropna().shape[0] < 2:
            return {"p_value": 1.0, "drift": False, "wasserstein": 0.0}
        stat, p_value = ks_2samp(ref.dropna(), prod.dropna())
        wd = wasserstein_distance(ref.dropna(), prod.dropna())
        return {"p_value": float(p_value), "drift": p_value < alpha, "wasserstein": float(wd)}

    @staticmethod
    def _categorical_drift(ref: pd.Series, prod: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
        ref_counts = ref.fillna("__MISSING__").value_counts()
        prod_counts = prod.fillna("__MISSING__").value_counts()
        categories = list(set(ref_counts.index).union(prod_counts.index))
        ref_freq = [ref_counts.get(cat, 0) for cat in categories]
        prod_freq = [prod_counts.get(cat, 0) for cat in categories]
        # If any row sum is zero, chi2_contingency will throw; guard for tiny sizes
        try:
            stat, p_value, _, _ = chi2_contingency([ref_freq, prod_freq])
        except Exception:
            return {"p_value": 1.0, "drift": False, "categories": categories}
        return {"p_value": float(p_value), "drift": p_value < alpha, "categories": categories}

    # Main run function
    def run_drift_check(
        self,
        user_id: int,
        mode: str = "row_by_row",  # "row_by_row" or "batch"
        alpha: float = 0.05,
        batch_size: Optional[int] = None,
        lookback: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Run drift detection end-to-end:
         - ensure admin
         - load reference from CSV
         - load production according to mode
         - align columns via ProductionPreprocessor (if available)
         - compute per-feature drift
         - update drift_state (last_processed_id or last_batch_id)
         - push Prometheus metrics (Gauges)
         - return structured result
        """
        # step 1: admin check
        self._check_admin(user_id)

        # step 2: reference data
        df_ref = self.load_reference()
        if df_ref.empty:
            raise RuntimeError("Reference data is empty")

        # step 3: production data
        if mode == "row_by_row":
            df_prod, new_last_id = self.load_production_rows_row_by_row(batch_size=batch_size or self.BATCH_SIZE)
            last_batch_id = None
        elif mode == "batch":
            df_prod, last_batch_id = self.load_production_rows_batch_mode(lookback=lookback, limit=batch_size or 1000)
            new_last_id = None
        else:
            raise ValueError("mode must be 'row_by_row' or 'batch'")

        if df_prod.empty:
            # push zeroed metrics and return early
            logger.info("No production rows to evaluate; returning empty result.")
            g_total_features.set(0)
            g_drifted_features.set(0)
            g_drift_percentage.set(0.0)
            return {"message": "no_production_data", "drift_summary": {}, "total_features": 0}

        # step 4: Aligning columns — keeping only columns that exist in data reference (so same schema)
        common_cols = [c for c in df_ref.columns if c in df_prod.columns]
        if not common_cols:
            # try if reference contains only processed feature columns and df_prod needs same processing
            # If processor available, attempt to re-process production to feature set (already done in _rows_to_df if processor exists)
            common_cols = [c for c in df_ref.columns if c in df_prod.columns]
        df_ref_common = df_ref[common_cols].copy()
        df_prod_common = df_prod[common_cols].copy()

        # step 5: Compute drift per column
        drift_summary = {}
        for col in common_cols:
            try:
                if pd.api.types.is_numeric_dtype(df_ref_common[col]) or np.issubdtype(df_ref_common[col].dtype, np.number):
                    metrics = self._numeric_drift(df_ref_common[col], df_prod_common[col], alpha=alpha)
                    summary = {
                        "type": "numerical",
                        "p_value": metrics["p_value"],
                        "drift": bool(metrics["drift"]),
                        "wasserstein": metrics["wasserstein"],
                        "ref_mean": float(df_ref_common[col].mean()) if not df_ref_common[col].dropna().empty else None,
                        "prod_mean": float(df_prod_common[col].mean()) if not df_prod_common[col].dropna().empty else None,
                        "ref_std": float(df_ref_common[col].std()) if not df_ref_common[col].dropna().empty else None,
                        "prod_std": float(df_prod_common[col].std()) if not df_prod_common[col].dropna().empty else None,
                    }
                else:
                    metrics = self._categorical_drift(df_ref_common[col].astype(str), df_prod_common[col].astype(str), alpha=alpha)
                    summary = {
                        "type": "categorical",
                        "p_value": metrics["p_value"],
                        "drift": bool(metrics["drift"]),
                        "categories": metrics.get("categories"),
                        "ref_mode": df_ref_common[col].mode().iloc[0] if not df_ref_common[col].mode().empty else None,
                        "prod_mode": df_prod_common[col].mode().iloc[0] if not df_prod_common[col].mode().empty else None
                    }
            except Exception as e:
                logger.exception(f"Error computing drift for column {col}: {e}")
                summary = {"type": "error", "error": str(e)}
            drift_summary[col] = summary

            # update per-feature prometheus metrics
            gauges = _ensure_feature_gauges(col)
            gauges["drift"].set(1.0 if drift_summary[col].get("drift") else 0.0)
            gauges["p_value"].set(drift_summary[col].get("p_value") or 1.0)
            gauges["wasserstein"].set(drift_summary[col].get("wasserstein") or 0.0)
            if "ref_mean" in drift_summary[col]:
                gauges["ref_mean"].set(drift_summary[col].get("ref_mean") or 0.0)
                gauges["prod_mean"].set(drift_summary[col].get("prod_mean") or 0.0)

        total_features = len(drift_summary)
        drifted_features = sum(1 for v in drift_summary.values() if v.get("drift"))
        drift_percentage = (drifted_features / total_features) if total_features else 0.0

        # step 6: update summary prometheus metrics
        g_total_features.set(total_features)
        g_drifted_features.set(drifted_features)
        g_drift_percentage.set(drift_percentage)

        # step 7: update state table
        if mode == "row_by_row" and new_last_id:
            self._update_state(last_processed_id=new_last_id)
        elif mode == "batch" and last_batch_id:
            self._update_state(last_batch_id=last_batch_id)

        # step 8: decide suggested action (no automatic retrain)
        suggestion = "no_action"
        if drift_percentage >= self.DRIFT_CRITICAL:
            suggestion = "critical: investigate - consider retraining with training+new_data or new_only depending on business"
        elif drift_percentage >= self.DRIFT_WARNING:
            suggestion = "warning: monitor and consider periodic retraining"

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "total_features": total_features,
            "drifted_features": drifted_features,
            "drift_percentage": drift_percentage,
            "thresholds": {
                "warning": self.DRIFT_WARNING,
                "critical": self.DRIFT_CRITICAL
            },
            "suggestion": suggestion,
            "drift_summary": drift_summary
        }

        logger.info(f"Drift run complete: {drifted_features}/{total_features} drifted ({drift_percentage:.2%})")
        return result

    
    # Prometheus endpoint helper
    @staticmethod
    def prometheus_metrics_response():
        """Return bytes response for Prometheus scrape (for web framework to serve)."""
        payload = generate_latest()
        return payload, CONTENT_TYPE_LATEST