import os
import subprocess

# Initialize Airflow DB
print("Initializing Airflow database...")
os.system("airflow db init")

#  Enable RBAC and authentication (optional if set in airflow.yaml/env)
print("Enabling RBAC and authentication...")

# If using environment variables for RBAC/auth, must be set
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
], check=True)

# Create a Data Scientist user (optional)
print("Creating Data Scientist user...")
subprocess.run([
    "airflow", "users", "create",
    "--username", "data_scientist",
    "--password", "strong_ds_password",
    "--firstname", "Data",
    "--lastname", "Scientist",
    "--role", "User",
    "--email", "datascientist@example.com"
], check=True)

print("Migration complete! Airflow is ready with RBAC and users set up.")
