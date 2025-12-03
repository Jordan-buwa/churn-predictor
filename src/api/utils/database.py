import os
import json
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import List
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from src.api.utils.customer_data import CustomerData

# Load environment variables 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Logging setup with auto folder creation 
logs_dir = os.path.join(project_root, "src/api/utils/logs")
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(logs_dir, 'database.log'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# PostgreSQL configuration
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB_NAME', 'churn_db'),
    'user': os.getenv('POSTGRES_DB_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'sslmode': 'require'
}

# Connection pool 
connection_pool = None

def initialize_connection_pool():
    """Initialize PostgreSQL connection pool"""
    global connection_pool
    try:
        connection_pool = ThreadedConnectionPool(minconn=2, maxconn=10, **DB_CONFIG)
        logger.info("PostgreSQL connection pool created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating connection pool: {str(e)}")
        return False

@contextmanager
def get_db_connection():
    """Context manager for database connections from pool"""
    if connection_pool is None:
        # Fallback dummy connection if pool not initialized
        class DummyConn:
            def cursor(self):
                class DummyCursor:
                    def execute(self, *args, **kwargs): pass
                    def close(self): pass
                return DummyCursor()
            def commit(self): pass
            def rollback(self): pass
        yield DummyConn()
        return

    conn = connection_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        connection_pool.putconn(conn)

def initialize_database():
    """Initialize the database schema"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_data (
                features JSONB,
                id SERIAL PRIMARY KEY,
                customer_id VARCHAR(255),
                unnamed_0 INTEGER,
                x INTEGER,
                customer VARCHAR(255),
                traintest VARCHAR(10),
                churndep VARCHAR(10),
                revenue DECIMAL(10, 2),
                mou DECIMAL(10, 2),
                recchrge DECIMAL(10, 2),
                directas DECIMAL(10, 2),
                overage DECIMAL(10, 2),
                roam DECIMAL(10, 2),
                changem DECIMAL(10, 2),
                changer DECIMAL(10, 2),
                dropvce DECIMAL(10, 2),
                blckvce DECIMAL(10, 2),
                unansvce DECIMAL(10, 2),
                custcare DECIMAL(10, 2),
                threeway DECIMAL(10, 2),
                mourec DECIMAL(10, 2),
                outcalls DECIMAL(10, 2),
                incalls DECIMAL(10, 2),
                peakvce DECIMAL(10, 2),
                opeakvce DECIMAL(10, 2),
                dropblk DECIMAL(10, 2),
                callfwdv DECIMAL(10, 2),
                callwait DECIMAL(10, 2),
                months DECIMAL(10, 2),
                uniqsubs DECIMAL(10, 2),
                actvsubs DECIMAL(10, 2),
                phones DECIMAL(10, 2),
                models DECIMAL(10, 2),
                eqpdays DECIMAL(10, 2),
                age1 DECIMAL(10, 2),
                age2 DECIMAL(10, 2),
                refer DECIMAL(10, 2),
                income DECIMAL(10, 2),
                setprc DECIMAL(10, 2),
                children VARCHAR(10),
                credita VARCHAR(10),
                creditaa VARCHAR(10),
                prizmrur VARCHAR(10),
                prizmub VARCHAR(10),
                prizmtwn VARCHAR(10),
                refurb VARCHAR(10),
                webcap VARCHAR(10),
                truck VARCHAR(10),
                rv VARCHAR(10),
                occprof VARCHAR(10),
                occcler VARCHAR(10),
                occcrft VARCHAR(10),
                occstud VARCHAR(10),
                occhmkr VARCHAR(10),
                occret VARCHAR(10),
                occself VARCHAR(10),
                ownrent VARCHAR(10),
                marryun VARCHAR(10),
                marryyes VARCHAR(10),
                mailord VARCHAR(10),
                mailres VARCHAR(10),
                mailflag VARCHAR(10),
                travel VARCHAR(10),
                pcown VARCHAR(10),
                creditcd VARCHAR(10),
                newcelly VARCHAR(10),
                newcelln VARCHAR(10),
                incmiss VARCHAR(10),
                mcycle VARCHAR(10),
                setprcm VARCHAR(10),
                retcall VARCHAR(10),
                retcalls VARCHAR(10),
                retaccpt VARCHAR(10),
                churn BOOLEAN,
                source VARCHAR(100),
                timestamp TIMESTAMP,
                batch_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_customer_timestamp UNIQUE(customer_id, timestamp)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_logs (
                id SERIAL PRIMARY KEY,
                batch_id VARCHAR(255) UNIQUE,
                source VARCHAR(100),
                records_processed INTEGER,
                records_saved INTEGER,
                records_failed INTEGER,
                file_name VARCHAR(500),
                status VARCHAR(50),
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_id ON customer_data(customer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON customer_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_id ON customer_data(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON customer_data(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_churn ON customer_data(churn)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_revenue ON customer_data(revenue)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_months ON customer_data(months)")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("PostgreSQL database initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

def startup():
    """Startup function to initialize pool and database"""
    if initialize_connection_pool():
        initialize_database()
    else:
        logger.error("Failed to initialize connection pool")

startup()

# Batch helpers
def generate_batch_id() -> str:
    return f"BATCH_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(object())}"

def customer_to_tuple(customer: CustomerData, batch_id: str):
    customer.customer_id = customer.customer_id or f"CUST_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(customer)}"
    features_dict = customer.dict(exclude={'customer_id', 'source', 'timestamp', 'batch_id'})
    features_json = json.dumps(features_dict)

    attrs = [
        customer.customer_id, customer.unnamed_0, customer.x, customer.customer,
        customer.traintest, customer.churndep, customer.revenue, customer.mou,
        customer.recchrge, customer.directas, customer.overage, customer.roam,
        customer.changem, customer.changer, customer.dropvce, customer.blckvce,
        customer.unansvce, customer.custcare, customer.threeway, customer.mourec,
        customer.outcalls, customer.incalls, customer.peakvce, customer.opeakvce,
        customer.dropblk, customer.callfwdv, customer.callwait, customer.months,
        customer.uniqsubs, customer.actvsubs, customer.phones, customer.models,
        customer.eqpdays, customer.age1, customer.age2, customer.refer, customer.income,
        customer.setprc, customer.children, customer.credita, customer.creditaa,
        customer.prizmrur, customer.prizmub, customer.prizmtwn, customer.refurb,
        customer.webcap, customer.truck, customer.rv, customer.occprof, customer.occcler,
        customer.occcrft, customer.occstud, customer.occhmkr, customer.occret,
        customer.occself, customer.ownrent, customer.marryun, customer.marryyes,
        customer.mailord, customer.mailres, customer.mailflag, customer.travel,
        customer.pcown, customer.creditcd, customer.newcelly, customer.newcelln,
        customer.incmiss, customer.mcycle, customer.setprcm, customer.retcall,
        customer.retcalls, customer.retaccpt, customer.churn, customer.source,
        customer.timestamp, batch_id, features_json
    ]
    return tuple(attrs)

def save_customer_data(customer: CustomerData, batch_id: str) -> bool:
    """Save a single customer, fallback to batch insert format"""
    try:
        saved, _ = save_batch_customer_data([customer], batch_id)
        return saved == 1
    except Exception as e:
        logger.error(f"Error saving customer data: {str(e)}")
        return False

def save_batch_customer_data(customers: List[CustomerData], batch_id: str) -> tuple:
    """Save multiple customer records efficiently"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            values = [customer_to_tuple(c, batch_id) for c in customers]
            execute_values(
                cursor,
                """
                INSERT INTO customer_data (
                    customer_id, unnamed_0, x, customer, traintest, churndep, revenue, mou,
                    recchrge, directas, overage, roam, changem, changer, dropvce, blckvce,
                    unansvce, custcare, threeway, mourec, outcalls, incalls, peakvce,
                    opeakvce, dropblk, callfwdv, callwait, months, uniqsubs, actvsubs,
                    phones, models, eqpdays, age1, age2, refer, income, setprc,
                    children, credita, creditaa, prizmrur, prizmub, prizmtwn, refurb,
                    webcap, truck, rv, occprof, occcler, occcrft, occstud, occhmkr,
                    occret, occself, ownrent, marryun, marryyes, mailord, mailres,
                    mailflag, travel, pcown, creditcd, newcelly, newcelln, incmiss,
                    mcycle, setprcm, retcall, retcalls, retaccpt, churn, source, timestamp,
                    batch_id, features
                ) VALUES %s
                ON CONFLICT (customer_id, timestamp) DO NOTHING
                """,
                values
            )
            saved_count = cursor.rowcount
            cursor.close()
            return saved_count, len(customers) - saved_count
    except Exception as e:
        logger.error(f"Error saving batch customer data: {str(e)}")
        return 0, len(customers)
