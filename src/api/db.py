from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
load_dotenv()
POSTGRES_HOST = os.getenv(POSTGRES_HOST, "localhost")
POSTGRES_DB_NAME = os.getenv(POSTGRES_DB_NAME, "churn_db")
POSTGRES_DB_USER = os.getenv(POSTGRES_DB_USER, "user")
POSTGRES_PASSWORD = os.getenv(POSTGRES_PASSWORD, "postgres")
DATABASE_URL =  "postgresql://{POSTGRES_DB_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)