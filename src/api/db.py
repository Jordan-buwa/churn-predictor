from sqlalchemy import create_engine, Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
# fastapi-users is optional for tests; import only when available
SQLAlchemyBaseUserTableUUID = None
if os.getenv("ENVIRONMENT", "development") != "test":
    try:
        from fastapi_users.db import SQLAlchemyBaseUserTableUUID  # type: ignore
    except ImportError:
        SQLAlchemyBaseUserTableUUID = None

load_dotenv()

# Allow overriding the database URL explicitly for tests and deployments
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME", "churn_db")
    POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    DATABASE_URL = f"postgresql://{POSTGRES_DB_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

if SQLAlchemyBaseUserTableUUID is not None:
    class User(SQLAlchemyBaseUserTableUUID, Base):
        pass
else:
    # Minimal fallback User model for test environment without fastapi-users
    class User(Base):
        __tablename__ = "users"
        id = Column(String, primary_key=True)
        email = Column(String, unique=True, nullable=False)
        hashed_password = Column(String, nullable=False)
        is_active = Column(Boolean, default=True)
        is_superuser = Column(Boolean, default=False)
        is_verified = Column(Boolean, default=False)