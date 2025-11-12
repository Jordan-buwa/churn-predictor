from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from src.api.utils.database import get_db_connection, save_customer_data, save_batch_customer_data, generate_batch_id
from fastapi.responses import PlainTextResponse
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
from datetime import datetime
import io

router = APIRouter(prefix="/ingest")
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='src/api/utils/logs/ingest.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# Comprehensive Cell2Cell Customer Data Model
class Cell2CellCustomerData(BaseModel):
    """Complete Cell2Cell customer data model with all features"""
    
    # Basic identifiers
    customer_id: str
    
    # Customer Info
    months: Optional[int] = None
    phones: Optional[int] = None
    
    # Usage & Call Behavior
    mou: Optional[float] = None
    outcalls: Optional[float] = None
    incalls: Optional[float] = None
    peakvce: Optional[float] = None
    opeakvce: Optional[float] = None
    dropvce: Optional[float] = None
    blckvce: Optional[float] = None
    unansvce: Optional[float] = None
    threeway: Optional[float] = None
    callwait: Optional[float] = None
    callfwdv: Optional[float] = None
    dropblk: Optional[float] = None
    custcare: Optional[float] = None
    mourec: Optional[float] = None
    
    # Billing & Revenue
    recchrge: Optional[float] = None
    directas: Optional[float] = None
    overage: Optional[float] = None
    roam: Optional[float] = None
    changem: Optional[float] = None
    changer: Optional[float] = None
    revenue: Optional[float] = None
    
    # Device & Plan
    eqpdays: Optional[int] = None
    models: Optional[int] = None
    actvsubs: Optional[int] = None
    uniqsubs: Optional[int] = None
    refurb: Optional[int] = None
    webcap: Optional[int] = None
    
    # Demographics
    age1: Optional[int] = None
    age2: Optional[int] = None
    children: Optional[int] = None
    income: Optional[int] = None
    truck: Optional[int] = None
    rv: Optional[int] = None
    mcycle: Optional[int] = None
    
    # Credit & Retention
    credita: Optional[int] = None
    creditaa: Optional[int] = None
    creditcd: Optional[int] = None
    retcalls: Optional[int] = None
    retaccpt: Optional[int] = None
    
    # Marketing & Behavior
    mailord: Optional[int] = None
    mailres: Optional[int] = None
    travel: Optional[int] = None
    pcown: Optional[int] = None
    
    # Categorical Data
    prizm_cluster: Optional[str] = None
    occupation: Optional[str] = None
    marital_status: Optional[str] = None
    ownrent: Optional[int] = None
    
    # New Cell & Referral
    newcelly: Optional[int] = None
    newcelln: Optional[int] = None
    refer: Optional[int] = None
    
    # Pricing & Retention
    setprcm: Optional[int] = None
    setprc: Optional[float] = None
    retcall: Optional[int] = None
    
    # Target variable
    churn: Optional[int] = None
    
    # Metadata
    source: Optional[str] = "api"
    timestamp: Optional[str] = None
    batch_id: Optional[str] = None

class BatchCell2CellData(BaseModel):
    """Batch data model for multiple Cell2Cell records"""
    customers: List[Cell2CellCustomerData]

class IngestResponse(BaseModel):
    """Response model for data ingestion"""
    success: bool
    message: str
    records_processed: int
    records_saved: int
    records_failed: int
    failed_records: Optional[List[Dict]] = None
    batch_id: Optional[str] = None

def parse_csv_file(file_content: bytes) -> pd.DataFrame:
    """Parse CSV file content"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")

def parse_excel_file(file_content: bytes) -> pd.DataFrame:
    """Parse Excel file content"""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise ValueError(f"Error parsing Excel file: {str(e)}")
    
def log_ingestion(batch_id: str, source: str, processed: int, saved: int, 
                  failed: int, file_name: Optional[str] = None, 
                  status: str = "success", error: Optional[str] = None):
    """Log ingestion activity"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
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
            cursor.close()
    except Exception as e:
        logger.error(f"Error logging ingestion: {str(e)}")

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names - handle common variations for Cell2Cell dataset"""
    # Normalize to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Comprehensive column mapping for Cell2Cell dataset
    column_mapping = {
        # Basic identifiers
        'customerid': 'customer_id',
        'customer_id': 'customer_id',
        'cust_id': 'customer_id',
        'id': 'customer_id',
        
        # Customer Info
        'tenure': 'months',
        'month': 'months',
        'phone': 'phones',
        
        # Usage metrics
        'mou': 'mou',
        'minutes_of_use': 'mou',
        'out_calls': 'outcalls',
        'in_calls': 'incalls',
        'peak_vce': 'peakvce',
        'off_peak_vce': 'opeakvce',
        'drop_vce': 'dropvce',
        'blck_vce': 'blckvce',
        'unans_vce': 'unansvce',
        'three_way': 'threeway',
        'call_wait': 'callwait',
        'call_fwd': 'callfwdv',
        'drop_blk': 'dropblk',
        'cust_care': 'custcare',
        'mou_rec': 'mourec',
        
        # Billing & Revenue
        'recurring_charge': 'recchrge',
        'direct_as': 'directas',
        'overage': 'overage',
        'roam': 'roam',
        'change_mou': 'changem',
        'change_revenue': 'changer',
        'revenue': 'revenue',
        'total_revenue': 'revenue',
        
        # Device & Plan
        'eqp_days': 'eqpdays',
        'equipment_days': 'eqpdays',
        'models': 'models',
        'actv_subs': 'actvsubs',
        'uniq_subs': 'uniqsubs',
        'refurb': 'refurb',
        'web_cap': 'webcap',
        
        # Demographics
        'age1': 'age1',
        'age_1': 'age1',
        'age2': 'age2',
        'age_2': 'age2',
        'children': 'children',
        'income': 'income',
        'truck': 'truck',
        'rv': 'rv',
        'mcycle': 'mcycle',
        
        # Credit
        'credit_a': 'credita',
        'credit_aa': 'creditaa',
        'credit_cd': 'creditcd',
        
        # Retention
        'ret_calls': 'retcalls',
        'ret_accpt': 'retaccpt',
        
        # Marketing
        'mail_ord': 'mailord',
        'mail_res': 'mailres',
        'travel': 'travel',
        'pc_own': 'pcown',
        
        # Categorical
        'prizm': 'prizm_cluster',
        'occupation': 'occupation',
        'marital': 'marital_status',
        'own_rent': 'ownrent',
        
        # New Cell
        'new_cell_y': 'newcelly',
        'new_cell_n': 'newcelln',
        'refer': 'refer',
        
        # Pricing
        'set_prcm': 'setprcm',
        'set_prc': 'setprc',
        'ret_call': 'retcall',
        
        # Target
        'churn': 'churn',
        'churndep': 'churn'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that should not be stored"""
    drop_cols = ["unnamed:_0", "x", "customer", "traintest", "row_id", "index", "unnamed"]
    
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"Dropped column: {col}")
    
    return df

def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean columns appropriately"""
    boolean_columns = [
        'refurb', 'webcap', 'children', 'truck', 'rv', 'mcycle', 
        'credita', 'creditaa', 'creditcd', 'mailord', 'mailres', 
        'travel', 'pcown', 'ownrent', 'newcelly', 'newcelln', 
        'refer', 'setprcm', 'retcall', 'churn'
    ]
    
    for col in boolean_columns:
        if col in df.columns:
            # Handle various boolean representations
            df[col] = df[col].map({
                'Yes': 1, 'yes': 1, 'YES': 1, 'Y': 1, 'y': 1, '1': 1, True: 1, 1.0: 1,
                'No': 0, 'no': 0, 'NO': 0, 'N': 0, 'n': 0, '0': 0, False: 0, 0.0: 0
            })
            # Fill NaN with 0 for boolean columns
            df[col] = df[col].fillna(0).astype(int)
    
    return df

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns, handling NaN values"""
    # Float columns
    float_columns = [
        'mou', 'outcalls', 'incalls', 'peakvce', 'opeakvce', 'dropvce', 
        'blckvce', 'unansvce', 'threeway', 'callwait', 'callfwdv', 'dropblk',
        'custcare', 'mourec', 'recchrge', 'directas', 'overage', 'roam',
        'changem', 'changer', 'revenue', 'setprc'
    ]
    
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Integer columns
    int_columns = [
        'months', 'phones', 'eqpdays', 'models', 'actvsubs', 'uniqsubs',
        'age1', 'age2', 'income', 'retcalls', 'retaccpt'
    ]
    
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df

def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize categorical columns"""
    categorical_columns = ['prizm_cluster', 'occupation', 'marital_status']
    
    for col in categorical_columns:
        if col in df.columns:
            # Convert to string and handle NaN
            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
            # Strip whitespace and convert to lowercase
            df[col] = df[col].str.strip().str.lower()
            # Replace empty strings with None
            df[col] = df[col].replace('', None)
    
    return df

def dataframe_to_cell2cell_data(df: pd.DataFrame, source: str) -> List[Cell2CellCustomerData]:
    """Convert DataFrame to list of Cell2CellCustomerData objects"""
    customers = []
    failed_rows = []
    
    for idx, row in df.iterrows():
        try:
            # Convert row to dict and handle NaN values
            row_dict = row.to_dict()
            row_dict = {k: (v if pd.notna(v) else None) for k, v in row_dict.items()}
            
            # Add metadata
            row_dict['source'] = source
            row_dict['timestamp'] = datetime.utcnow().isoformat()
            
            # Create customer object
            customer = Cell2CellCustomerData(**row_dict)
            customers.append(customer)
        
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {str(e)}")
            failed_rows.append({
                'row_index': idx,
                'error': str(e),
                'data': {k: (v if pd.notna(v) else None) for k, v in row.to_dict().items()}
            })
            continue
    
    return customers, failed_rows

@router.post("/single", response_model=IngestResponse)
async def ingest_single_record(data: Cell2CellCustomerData):
    """
    Ingest a single customer record via POST request.
    
    Args:
        data: Cell2Cell customer data object with all features
    
    Returns:
        Ingestion response with status
    """
    batch_id = generate_batch_id()
    
    try:
        data.source = "api"
        data.timestamp = datetime.utcnow().isoformat()
        data.batch_id = batch_id
        
        success = save_customer_data(data, batch_id)
        
        if success:
            log_ingestion(batch_id, "api", 1, 1, 0)
            logger.info(f"Successfully ingested single record: {data.customer_id}")
            
            return IngestResponse(
                success=True,
                message="Record ingested successfully",
                records_processed=1,
                records_saved=1,
                records_failed=0,
                batch_id=batch_id
            )
        else:
            log_ingestion(batch_id, "api", 1, 0, 1, error="Failed to save record")
            raise HTTPException(status_code=500, detail="Failed to save record")
    
    except Exception as e:
        logger.error(f"Error ingesting single record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=IngestResponse)
async def ingest_batch_records(data: BatchCell2CellData):
    """
    Ingest multiple customer records via POST request.
    
    Args:
        data: Batch customer data object
    
    Returns:
        Ingestion response with statistics
    """
    batch_id = generate_batch_id()
    
    try:
        # Set metadata for all customers
        for customer in data.customers:
            customer.source = "api_batch"
            customer.timestamp = datetime.utcnow().isoformat()
            customer.batch_id = batch_id
        
        # Use bulk insert
        saved, failed = save_batch_customer_data(data.customers, batch_id)
        processed = len(data.customers)
        
        log_ingestion(batch_id, "api_batch", processed, saved, failed)
        logger.info(f"Batch ingestion completed: {saved}/{processed} records saved")
        
        return IngestResponse(
            success=True,
            message=f"Batch ingestion completed",
            records_processed=processed,
            records_saved=saved,
            records_failed=failed,
            batch_id=batch_id
        )
    
    except Exception as e:
        logger.error(f"Error in batch ingestion: {str(e)}")
        log_ingestion(batch_id, "api_batch", 0, 0, len(data.customers), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/csv", response_model=IngestResponse)
async def ingest_csv_file(
    file: UploadFile = File(..., description="CSV file to ingest"),
    source: str = Form(default="csv", description="Data source identifier")
):
    """
    Ingest customer data from CSV file.
    Supports full Cell2Cell dataset with all features.
    
    Args:
        file: CSV file upload
        source: Source identifier
    
    Returns:
        Ingestion response with statistics
    """
    batch_id = generate_batch_id()
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        content = await file.read()
        
        df = parse_csv_file(content)
        logger.info(f"Parsed CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Comprehensive data preprocessing
        df = normalize_column_names(df)
        df = drop_unnecessary_columns(df)
        df = convert_boolean_columns(df)
        df = convert_numeric_columns(df)
        df = clean_categorical_columns(df)
        
        # Convert to customer objects
        customers, failed_rows = dataframe_to_cell2cell_data(df, source)
        processed = len(df)
        saved = len(customers)
        failed = len(failed_rows)
        
        # Use bulk insert
        if customers:
            saved_count, failed_count = save_batch_customer_data(customers, batch_id)
            saved = saved_count
            failed = failed_count + len(failed_rows)
        
        log_ingestion(batch_id, source, processed, saved, failed, file_name=file.filename)
        logger.info(f"CSV ingestion completed: {saved}/{processed} records saved from {file.filename}")
        
        response = IngestResponse(
            success=True,
            message=f"CSV file processed successfully",
            records_processed=processed,
            records_saved=saved,
            records_failed=failed,
            batch_id=batch_id
        )
        
        if failed_rows:
            response.failed_records = failed_rows
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        log_ingestion(batch_id, source, 0, 0, 0, 
                     file_name=file.filename, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@router.post("/excel", response_model=IngestResponse)
async def ingest_excel_file(
    file: UploadFile = File(..., description="Excel file to ingest"),
    source: str = Form(default="excel", description="Data source identifier")
):
    """
    Ingest customer data from Excel file (.xlsx, .xls).
    
    Args:
        file: Excel file upload
        source: Source identifier
    
    Returns:
        Ingestion response with statistics
    """
    batch_id = generate_batch_id()
    
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
    try:
        content = await file.read()
        
        df = parse_excel_file(content)
        logger.info(f"Parsed Excel with {len(df)} rows and columns: {list(df.columns)}")
        
        # Apply same preprocessing as CSV
        df = normalize_column_names(df)
        df = drop_unnecessary_columns(df)
        df = convert_boolean_columns(df)
        df = convert_numeric_columns(df)
        df = clean_categorical_columns(df)
        
        customers, failed_rows = dataframe_to_cell2cell_data(df, source)
        processed = len(df)
        saved = len(customers)
        failed = len(failed_rows)
        
        # Use bulk insert
        if customers:
            saved_count, failed_count = save_batch_customer_data(customers, batch_id)
            saved = saved_count
            failed = failed_count + len(failed_rows)
        
        log_ingestion(batch_id, source, processed, saved, failed, file_name=file.filename)
        logger.info(f"Excel ingestion completed: {saved}/{processed} records saved from {file.filename}")
        
        response = IngestResponse(
            success=True,
            message=f"Excel file processed successfully",
            records_processed=processed,
            records_saved=saved,
            records_failed=failed,
            batch_id=batch_id
        )
        
        if failed_rows:
            response.failed_records = failed_rows
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        log_ingestion(batch_id, source, processed, saved, failed,
                     file_name=file.filename, status="failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing Excel: {str(e)}")

@router.get("/stats")
async def get_ingestion_stats(limit: int = 10):
    """
    Get ingestion statistics and recent logs.
    
    Args:
        limit: Number of recent logs to return
    
    Returns:
        Ingestion statistics
    """
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
            by_source = {row['source']: row['count'] for row in cursor.fetchall()}
            
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
    """
    Get details of a specific ingestion batch.
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        Batch details including records
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM ingestion_logs 
                WHERE batch_id = %s
            """, (batch_id,))
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
    """
    Delete a specific ingestion batch.
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        Deletion confirmation
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM customer_data WHERE batch_id = %s", (batch_id,))
            deleted_records = cursor.rowcount
            
            cursor.execute("DELETE FROM ingestion_logs WHERE batch_id = %s", (batch_id,))
            
            cursor.close()
            
            if deleted_records == 0:
                raise HTTPException(status_code=404, detail="Batch not found")
            
            logger.info(f"Deleted batch {batch_id} with {deleted_records} records")
            
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
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Export ingested data.
    
    Args:
        format: Export format (csv or json)
        source: Filter by source
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
    
    Returns:
        Exported data
    """
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM customer_data WHERE 1=1"
            params = []
            
            if source:
                query += " AND source = %s"
                params.append(source)
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            query += " ORDER BY created_at DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if format.lower() == "csv":
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                return {
                    "format": "csv",
                    "data": csv_buffer.getvalue(),
                    "record_count": len(df)
                }
            else:
                return {
                    "format": "json",
                    "data": df.to_dict(orient='records'),
                    "record_count": len(df)
                }
    
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/customers")

async def get_customers(limit: int = 100, offset: int = 0):

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
