import os
import subprocess
import time


# Initialize Airflow DB
print("Initializing Airflow database...")
os.system("airflow db init")

# Enable RBAC & Authentication
print("Enabling RBAC and authentication...")

os.environ["AIRFLOW__WEBSERVER__RBAC"] = "True"
os.environ["AIRFLOW__WEBSERVER__AUTHENTICATE"] = "True"
os.environ["AIRFLOW__WEBSERVER__AUTH_BACKEND"] = "airflow.api.auth.backend.basic_auth"

# Create Admin user
print("Creating Admin user...")
subprocess.run([
    "airflow", "users", "create",
    "--username", "admin",
    "--password", "strong_admin_password",
    "--firstname", "Admin",
    "--lastname", "User",
    "--role", "Admin",
    "--email", "admin@example.com"
], check=False)

# Create Data Scientist user
print("Creating Data Scientist user...")
subprocess.run([
    "airflow", "users", "create",
    "--username", "data_scientist",
    "--password", "strong_ds_password",
    "--firstname", "Data",
    "--lastname", "Scientist",
    "--role", "User",
    "--email", "datascientist@example.com"
], check=False)


print("Waiting for Postgres to be ready...")
time.sleep(5)  # Simple wait — docker-compose will usually handle it

POSTGRES_COMMAND_PREFIX = [
    "psql",
    "-h", os.getenv("POSTGRES_HOST", "postgres"),
    "-U", os.getenv("POSTGRES_USER", "postgres"),
    "-d", os.getenv("POSTGRES_DB", "mlops")
]


def run_psql(sql):
    """Utility helper for running inline SQL commands."""
    try:
        subprocess.run(
            POSTGRES_COMMAND_PREFIX + ["-c", sql],
            check=True
        )
    except Exception as e:
        print(f"Warning: Could not run SQL command: {e}")


print("Creating combined model training table if not exists...")

# Combined table 
run_psql("""
CREATE TABLE IF NOT EXISTS combined_training_data (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50),
    customer_id VARCHAR(50),
    name TEXT,
    age INT,
    gender TEXT,
    account_balance FLOAT,
    credit_score INT,
    transaction_history JSONB,
    churn_label INT,
    ingestion_timestamp TIMESTAMP DEFAULT NOW()
);
""")

print("Creating batch tracking table if not exists...")

# Batch tracking table
run_psql("""
CREATE TABLE IF NOT EXISTS batch_tracking (
    batch_id VARCHAR(50) PRIMARY KEY,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
""")


print("Migration complete! Airflow + RBAC + users + training tables are ready.")
