from sqlalchemy import create_engine, select, Column, String, Enum as SQLAlchemyEnum
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from pwdlib import PasswordHash
from typing import Optional
from pydantic import ConfigDict, EmailStr, BaseModel
from sqlmodel import SQLModel, Field
from datetime import datetime

import os
from dotenv import load_dotenv

load_dotenv()

from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    SUPERVISOR = "supervisor"
    GUEST = "guest"


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    phone: str
    password: str
    role: UserRole


class UserRead(BaseModel):
    id: int
    username: str
    email: EmailStr
    phone: str
    role: UserRole
    model_config = ConfigDict(from_attributes=True)


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None

# Database configuration
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
# User model for authentication
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column(String(50), nullable=False, index=True))
    email: EmailStr = Field(sa_column=Column("email", String(255), unique=True, index=True))
    phone: str = Field(sa_column=Column(String(20), unique=True, index=True)) 
    password: str = Field(sa_column=Column(String(255), nullable=False))
    role: UserRole = Field(sa_column=Column(SQLAlchemyEnum(UserRole), nullable=False)) 
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True) 


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Default admin user 
pwd = PasswordHash.recommended()  

def create_admin():
    # Getting the next session of the generator
    db = next(get_db())
    try:
        existing_admin = db.exec(select(User).where(User.email == "admin@example.com")).first()
        
        if not existing_admin:
            admin = User(
                username="Admin",
                phone="+221783832653",
                email="admin@example.com",
                password=pwd.hash("admin"), 
                role=UserRole.ADMIN,
            )
            db.add(admin)
            db.commit()
            print("Default admin created!")
        else:
            print("Admin already exists!")
    except IntegrityError as e:
        db.rollback()
        print("Admin creation error :", e)
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    create_admin()


