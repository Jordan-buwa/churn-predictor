from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from src.api.utils.database import get_db_connection, save_customer_data, save_batch_customer_data, generate_batch_id
from src.api.utils.customer_data import CustomerData, BatchCustomerData
from fastapi.responses import PlainTextResponse
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
from datetime import datetime, UTC
import io

# --- Initialization ---
router = APIRouter(prefix="/ingest")
logger = logging.getLogger(__name__)

# Configure logging (note: in a large app, this should be done in main.py)
logging.basicConfig(
    filename='src/api/utils/logs/ingest.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# --- Pydantic Models ---

class IngestResponse(BaseModel):
    """Response model for data ingestion"""
    success: bool
    message: str
    records_processed: int
    records_saved: int
    records_failed: int
    failed_records: Optional[List[Dict]] = None
    batch_id: Optional[str] = None

# --- Data Parsing & Preprocessing Utilities ---


def parse_csv_file(file_content: bytes) -> pd.DataFrame:
    """Parse CSV file content into a Pandas DataFrame."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))[:100]
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")

def parse_excel_file(file_content: bytes) -> pd.DataFrame:
    """Parse Excel file content into a Pandas DataFrame."""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"Error parsing Excel file: {str(e)}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names - handle common variations for Cell2Cell dataset."""
    # First ensure all column names are strings
    df.columns = [str(col) for col in df.columns]
    
    # Now we can use .str accessor
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    column_mapping = {
        'customerid': 'customer_id', 'cust_id': 'customer_id', 'id': 'customer_id',
        'tenure': 'months', 'month': 'months', 'phone': 'phones',
        'minutes_of_use': 'mou', 'out_calls': 'outcalls', 'in_calls': 'incalls',
        'peak_vce': 'peakvce', 'off_peak_vce': 'opeakvce', 'drop_vce': 'dropvce',
        'blck_vce': 'blckvce', 'unans_vce': 'unansvce', 'three_way': 'threeway',
        'call_wait': 'callwait', 'call_fwd': 'callfwdv', 'drop_blk': 'dropblk',
        'cust_care': 'custcare', 'mou_rec': 'mourec',
        'recurring_charge': 'recchrge', 'total_revenue': 'revenue',
        'change_mou': 'changem', 'change_revenue': 'changer',
        'eqp_days': 'eqpdays', 'equipment_days': 'eqpdays',
        'actv_subs': 'actvsubs', 'uniq_subs': 'uniqsubs', 'web_cap': 'webcap',
        'age_1': 'age1', 'age_2': 'age2',
        'credit_a': 'credita', 'credit_aa': 'creditaa', 'credit_cd': 'creditcd',
        'ret_calls': 'retcalls', 'ret_accpt': 'retaccpt',
        'mail_ord': 'mailord', 'mail_res': 'mailres', 'pc_own': 'pcown',
        'prizm': 'prizm_cluster', 'marital': 'marital_status', 'own_rent': 'ownrent',
        'new_cell_y': 'newcelly', 'new_cell_n': 'newcelln',
        'set_prcm': 'setprcm', 'set_prc': 'setprc', 'ret_call': 'retcall',
    }

    df.rename(columns=column_mapping, inplace=True)
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that should not be stored (e.g., indices, internal IDs)."""
    drop_cols = {"unnamed:_0", "x", "customer",
                 "traintest", "row_id", "index", "unnamed", 'churndep'}

    # Find intersection of columns to drop and existing columns
    cols_to_drop = list(drop_cols.intersection(df.columns))

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.debug(f"Dropped columns: {cols_to_drop}")

    return df

def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns to string '0' or '1'."""
    boolean_columns = [
        'customer', 'traintest', "incmiss", 'prizmrur', 'prizmub', 'prizmtwn',
        'occprof', 'occcler', 'occstud', 'occcrft', 'occhmkr', 'occret',
        'refurb', 'webcap', 'children', 'truck', 'rv', 'mcycle', 'occself',
        'marryun', 'marryyes', 'mcycle', 'mailflag', 'ownrent',
        'credita', 'creditaa', 'creditcd', 'mailord', 'mailres',
        'travel', 'pcown', 'newcelly', 'newcelln',
        'refer', 'setprcm', 'retcall', 'churn'
    ]
    cols = ['retcalls', 'retaccpt', 'churndep']
    for col in cols:
        if col in df.columns:
            series = df[col]
            series = series.astype(str).str.strip()
        df[col] = series
    for col in boolean_columns:
        if col in df.columns:
            try:
                # Ensure we're working with a Series, not a DataFrame
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    # This shouldn't happen, but handle it just in case
                    logger.warning(f"Column {col} is a DataFrame, not a Series")
                    continue
                
                # First, convert everything to string and clean
                series = series.astype(str).str.strip()
                
                # Map various representations to '0' or '1' strings
                mapping = {
                    # True values -> '1'
                    'Yes': '1', 'yes': '1', 'YES': '1', 'Y': '1', 'y': '1', 
                    'True': '1', 'true': '1', 'TRUE': '1', 'T': '1', 't': '1',
                    '1': '1', 1: '1', 1.0: '1', True: '1',
                    
                    # False values -> '0'
                    'No': '0', 'no': '0', 'NO': '0', 'N': '0', 'n': '0',
                    'False': '0', 'false': '0', 'FALSE': '0', 'F': '0', 'f': '0',
                    '0': '0', 0: '0', 0.0: '0', False: '0',
                    
                    # Empty/NaN -> '0'
                    'nan': '0', 'NaN': '0', 'None': '0', 'none': '0', 
                    'null': '0', 'Null': '0', 'NULL': '0',
                    '': '0', ' ': '0'
                }
                
                # Apply mapping and fill any remaining with '0'
                df[col] = series.map(mapping).fillna('0')
                
                logger.debug(f"Converted {col} to string boolean: {df[col].unique()[:5]}")

                
            except Exception as e:
                logger.error(f"Error converting boolean column {col}: {str(e)}")
                # If conversion fails, set all to '0' as default
                df[col] = '0'

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns, coercing errors and filling NaN where appropriate."""
    # Float columns (can contain NaNs which become None in Pydantic)
    float_columns = [
        'mou', 'outcalls', 'incalls', 'peakvce', 'opeakvce', 'dropvce',
        'blckvce', 'unansvce', 'threeway', 'callwait', 'callfwdv', 'dropblk',
        'custcare', 'mourec', 'recchrge', 'directas', 'overage', 'roam',
        'changem', 'changer', 'revenue', 'setprc'
    ]

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Integer columns (filled with 0 as they often represent counts or ages)
    int_columns = [
        'months', 'phones', 'eqpdays', 'models', 'actvsubs', 'uniqsubs',
        'age1', 'age2', 'income'
    ]

    for col in int_columns:
        if col in df.columns:
            # Coerce to numeric, fill NaT with 0, then convert to integer
            df[col] = pd.to_numeric(
                df[col], errors='coerce').fillna(0).astype(int)

    return df


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize categorical columns."""
    categorical_columns = ['prizm_cluster', 'occupation', 'marital_status']

    for col in categorical_columns:
        if col in df.columns:
            try:
                # Ensure we're working with a Series
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    logger.warning(f"Column {col} is a DataFrame, not a Series")
                    continue
                    
                # Convert to string, standardize, and replace empty strings with None (NaN)
                series = series.astype(str).str.strip().str.lower()
                series = series.replace({'nan': None, 'none': None, '': None})
                df[col] = series
            except Exception as e:
                logger.error(f"Error cleaning categorical column {col}: {str(e)}")
                continue

    return df


def process_ingestion_data(df: pd.DataFrame, source: str) -> List[CustomerData]:
    """
    Applies the full data cleaning pipeline to a DataFrame and converts it 
    into a list of validated CustomerData objects.
    """
    logger.info("Starting DataFrame preprocessing pipeline.")

    df = normalize_column_names(df)
    df = convert_boolean_columns(df)
    df = convert_numeric_columns(df)
    df = clean_categorical_columns(df)

    # Prepare for Pydantic conversion
    customers = []
    failed_rows = []

    current_time = datetime.utcnow().isoformat()
    logger.info(f"DataFrame shape after preprocessing: {df.columns}, {df.shape}")
    for idx, row in df.iterrows():
        try:
            # Convert row to dict, handling NaN values which Pydantic Optional[T] handles as None
            row_dict = row.to_dict()
            row_dict = {k: (v if pd.notna(v) else None)
                        for k, v in row_dict.items()}

            # Add metadata
            row_dict['source'] = source
            row_dict['timestamp'] = current_time

            # Create and validate customer object using Pydantic
            customer = CustomerData(**row_dict)
            customers.append(customer)

        except Exception as e:
            # Catch Pydantic validation errors
            logger.warning(f"Validation error on row {idx}: {str(e)}")
            failed_rows.append({
                'row_index': idx,
                'error': str(e),
                'data': {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()}
            })
            continue

    logger.info(
        f"Preprocessing finished. {len(customers)} records ready for saving, {len(failed_rows)} failed validation.")
    return customers, failed_rows

# --- Logging (Kept in ingest.py for simplicity but often a DB util) ---


def log_ingestion(batch_id: str, source: str, processed: int, saved: int,
                  failed: int, file_name: Optional[str] = None,
                  status: str = "success", error: Optional[str] = None):
    """Log ingestion activity to the database."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO ingestion_logs (
                        batch_id, source, records_processed, records_saved,
                        records_failed, file_name, status, error_message
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (batch_id) DO UPDATE SET
                        records_processed = EXCLUDED.records_processed,
                        records_saved = EXCLUDED.records_saved,
                        records_failed = EXCLUDED.records_failed,
                        status = EXCLUDED.status,
                        error_message = EXCLUDED.error_message
                """, (batch_id, source, processed, saved, failed, file_name, status, error))
    except Exception as e:
        logger.error(f"Error logging ingestion to DB: {str(e)}")

# --- API Endpoints ---


@router.post("/single", response_model=IngestResponse)
async def ingest_single_record(data: CustomerData):
    """Ingest a single customer record via POST request."""
    batch_id = generate_batch_id()

    try:
        data.source = "api"
        data.timestamp = datetime.utcnow().isoformat()
        success = save_customer_data(data, batch_id)

        if success:
            log_ingestion(batch_id, "api", 1, 1, 0)
            logger.info(
                f"Successfully ingested single record: {data.customer_id}")

            return IngestResponse(
                success=True,
                message="Record ingested successfully",
                records_processed=1,
                records_saved=1,
                records_failed=0,
                batch_id=batch_id
            )
        else:
            log_ingestion(batch_id, "api", 1, 0, 1,
                          error="Failed to save record to DB")
            raise HTTPException(
                status_code=500, detail="Failed to save record to database")

    except Exception as e:
        logger.error(f"Error ingesting single record: {str(e)}")
        log_ingestion(batch_id, "api", 1, 0, 1, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=IngestResponse)
async def ingest_batch_records(data: BatchCustomerData):
    """Ingest multiple customer records via POST request."""
    batch_id = generate_batch_id()

    try:
        processed = len(data.customers)
        # Set metadata for all customers
        for customer in data.customers:
            customer.source = "api_batch"
            customer.timestamp = datetime.utcnow().isoformat()
            customer.batch_id = batch_id

        # Use bulk insert (assuming save_batch_customer_data handles the DB connection)
        saved, failed = save_batch_customer_data(data.customers, batch_id)

        log_ingestion(batch_id, "api_batch", processed, saved, failed)
        logger.info(
            f"Batch ingestion completed: {saved}/{processed} records saved")

        return IngestResponse(
            success=True,
            message="Batch ingestion completed",
            records_processed=processed,
            records_saved=saved,
            records_failed=failed,
            batch_id=batch_id
        )

    except Exception as e:
        logger.error(f"Error in batch ingestion: {str(e)}")
        log_ingestion(batch_id, "api_batch", 0, 0, 0, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/csv", response_model=IngestResponse)
async def ingest_csv_file(
    file: UploadFile = File(..., description="CSV file to ingest"),
    source: str = Form(default="csv_upload",
                       description="Data source identifier")
):
    """Ingest customer data from CSV file."""
    batch_id = generate_batch_id()

    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400, detail="File must be a CSV file (.csv)")

    try:
        content = await file.read()
        df = parse_csv_file(content)

        # --- Consolidated Data Processing ---
        customers, validation_failed_rows = process_ingestion_data(df, source)

        processed = len(df)
        db_saved = 0
        db_failed = 0

        # Use bulk insert
        if customers:
            db_saved, db_failed = save_batch_customer_data(customers, batch_id)

        total_saved = db_saved
        total_failed = db_failed + len(validation_failed_rows)

        log_ingestion(batch_id, source, processed, total_saved,
                      total_failed, file_name=file.filename)
        logger.info(
            f"CSV ingestion completed: {total_saved}/{processed} records saved from {file.filename}")

        response = IngestResponse(
            success=True,
            message="CSV file processed successfully",
            records_processed=processed,
            records_saved=total_saved,
            records_failed=total_failed,
            batch_id=batch_id
        )

        if validation_failed_rows:
            # Only include Pydantic validation failures for now. DB insertion failures are harder to return.
            response.failed_records = validation_failed_rows

        return response

    except Exception as e:
        error_message = f"Error processing CSV: {str(e)}"
        logger.error(error_message)
        log_ingestion(batch_id, source, 0, 0, 0,
                      file_name=file.filename, status="failed", error=error_message)
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/excel", response_model=IngestResponse)
async def ingest_excel_file(
    file: UploadFile = File(..., description="Excel file to ingest"),
    source: str = Form(default="excel_upload",
                       description="Data source identifier")
):
    """Ingest customer data from Excel file (.xlsx, .xls)."""
    batch_id = generate_batch_id()

    filename_lower = file.filename.lower()
    if not (filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls')):
        raise HTTPException(
            status_code=400, detail="File must be an Excel file (.xlsx or .xls)")

    try:
        content = await file.read()
        df = parse_excel_file(content)

        # --- Consolidated Data Processing ---
        customers, validation_failed_rows = process_ingestion_data(df, source)

        processed = len(df)
        db_saved = 0
        db_failed = 0

        # Use bulk insert
        if customers:
            db_saved, db_failed = save_batch_customer_data(customers, batch_id)

        total_saved = db_saved
        total_failed = db_failed + len(validation_failed_rows)

        log_ingestion(batch_id, source, processed, total_saved,
                      total_failed, file_name=file.filename)
        logger.info(
            f"Excel ingestion completed: {total_saved}/{processed} records saved from {file.filename}")

        response = IngestResponse(
            success=True,
            message="Excel file processed successfully",
            records_processed=processed,
            records_saved=total_saved,
            records_failed=total_failed,
            batch_id=batch_id
        )

        if validation_failed_rows:
            response.failed_records = validation_failed_rows

        return response

    except Exception as e:
        error_message = f"Error processing Excel: {str(e)}"
        logger.error(error_message)
        log_ingestion(batch_id, source, 0, 0, 0,
                      file_name=file.filename, status="failed", error=error_message)
        raise HTTPException(status_code=500, detail=error_message)


@router.get("/stats")
async def get_ingestion_stats(limit: int = 10):
    """Get ingestion statistics and recent logs."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get total records
            cursor.execute("SELECT COUNT(*) as total FROM customer_data")
            total_records = cursor.fetchone()['total']

            # Get records by source
            cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM customer_data 
                GROUP BY source
            """)
            by_source = {row['source']: row['count']
                         for row in cursor.fetchall()}

            # Get recent ingestion logs
            cursor.execute("""
                SELECT * FROM ingestion_logs 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, (limit,))
            recent_logs = [dict(row) for row in cursor.fetchall()]

            # Get ingestion summary
            cursor.execute("""
                SELECT 
                    SUM(records_processed) as total_processed,
                    SUM(records_saved) as total_saved,
                    SUM(records_failed) as total_failed
                FROM ingestion_logs
            """)
            summary = dict(cursor.fetchone())

            cursor.close()

            return {
                "total_records": total_records,
                "records_by_source": by_source,
                "ingestion_summary": summary,
                "recent_logs": recent_logs
            }

    except Exception as e:
        logger.error(f"Error getting ingestion stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/{batch_id}")
async def get_batch_details(batch_id: str):
    """Get details of a specific ingestion batch."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                "SELECT * FROM ingestion_logs WHERE batch_id = %s", (batch_id,))
            log = cursor.fetchone()

            if not log:
                raise HTTPException(status_code=404, detail="Batch not found")

            cursor.execute("""
                SELECT * FROM customer_data 
                WHERE batch_id = %s
                ORDER BY created_at
            """, (batch_id,))
            records = [dict(row) for row in cursor.fetchall()]

            cursor.close()

            return {
                "batch_info": dict(log),
                "records": records,
                "record_count": len(records)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/batch/{batch_id}")
async def delete_batch(batch_id: str):
    """Delete a specific ingestion batch."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM customer_data WHERE batch_id = %s", (batch_id,))
            deleted_records = cursor.rowcount

            cursor.execute(
                "DELETE FROM ingestion_logs WHERE batch_id = %s", (batch_id,))

            conn.commit()
            cursor.close()

            if deleted_records == 0:
                raise HTTPException(
                    status_code=404, detail="Batch not found or no records associated")

            logger.info(
                f"Deleted batch {batch_id} with {deleted_records} records")

            return {
                "success": True,
                "message": f"Deleted batch {batch_id}",
                "records_deleted": deleted_records
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_data(
    format: str = "csv",
    source: Optional[str] = None,
    # Changed type hint for better validation
    start_date: Optional[datetime] = None,
    # Changed type hint for better validation
    end_date: Optional[datetime] = None
):
    """Export ingested data, supporting CSV or JSON format."""
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM customer_data WHERE 1=1"
            params = []

            if source:
                query += " AND source = %s"
                params.append(source)

            if start_date:
                # Use ISO format string for PostgreSQL timestamp comparison
                query += " AND created_at >= %s"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND created_at <= %s"
                params.append(end_date.isoformat())

            query += " ORDER BY created_at DESC"

            df = pd.read_sql_query(query, conn, params=params)

            if format.lower() == "csv":
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                return PlainTextResponse(
                    content=csv_buffer.getvalue(),
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": "attachment; filename=customer_data.csv"}
                )

            elif format.lower() == "json":
                return {
                    "format": "json",
                    "data": df.to_dict(orient='records'),
                    "record_count": len(df)
                }

            else:
                raise HTTPException(
                    status_code=400, detail="Invalid format requested. Must be 'csv' or 'json'")

    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customers")
async def get_customers(limit: int = 100, offset: int = 0):
    """Fetch paginated list of customers."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute(
                "SELECT * FROM customer_data ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (limit, offset)
            )
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()

            return {"records": rows, "count": len(rows)}

    except Exception as e:
        logger.error(f"Error fetching customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))
