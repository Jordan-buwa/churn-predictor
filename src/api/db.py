# from sqlalchemy import create_engine, select, Column, String, Enum as SQLAlchemyEnum
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.exc import IntegrityError
# from pwdlib import PasswordHash
# from typing import Optional
# from pydantic import ConfigDict, EmailStr, BaseModel
# from sqlmodel import SQLModel, Field, Session
# from datetime import datetime
# import os
# from dotenv import load_dotenv
# from enum import Enum

# load_dotenv()

# class UserRole(str, Enum):
#     ADMIN = "admin"
#     MANAGER = "manager"
#     SUPERVISOR = "supervisor"
#     GUEST = "guest"

# class UserCreate(BaseModel):
#     username: str
#     email: EmailStr
#     phone: str
#     password: str
#     role: UserRole

# class UserRead(BaseModel):
#     id: int
#     username: str
#     email: EmailStr
#     phone: str
#     role: UserRole
#     model_config = ConfigDict(from_attributes=True)

# class UserUpdate(BaseModel):
#     username: Optional[str] = None
#     email: Optional[EmailStr] = None
#     phone: Optional[str] = None
#     password: Optional[str] = None
#     role: Optional[str] = None

# # Database configuration
# DATABASE_URL = os.getenv("DATABASE_URL")
# if not DATABASE_URL:
#     POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
#     POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME", "churn_db")
#     POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER", "user")
#     POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
#     DATABASE_URL = f"postgresql://{POSTGRES_DB_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB_NAME}?sslmode=require"

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # User model for authentication - CORRIGÉ
# class User(SQLModel, table=True):
#     __tablename__ = "user"

#     id: int | None = Field(default=None, primary_key=True)
#     username: str = Field(sa_column=Column(String(50), nullable=False, index=True))
#     email: str = Field(sa_column=Column(String(255), unique=True, index=True))
#     phone: str = Field(sa_column=Column(String(20), unique=True, index=True))
#     password: str = Field(sa_column=Column(String(255), nullable=False))
#     role: UserRole = Field(sa_column=Column(SQLAlchemyEnum(UserRole), nullable=False))
#     created_at: datetime | None = Field(default_factory=datetime.utcnow)
#     is_active: bool = Field(default=True)

# # Database dependency - CORRIGÉ pour utiliser Session de SQLModel
# def get_db():
#     db = Session(engine)
#     try:
#         yield db
#     finally:
#         db.close()

# # Créer les tables
# def create_tables():
#     SQLModel.metadata.create_all(engine)

# # Default admin user
# pwd = PasswordHash.recommended()

# def create_admin():
#     db = Session(engine)
#     try:
#         existing_admin = db.exec(select(User).where(User.email == "admin@example.com")).first()

#         if not existing_admin:
#             admin = User(
#                 username="Admin",
#                 phone="+221783832653",
#                 email="admin@example.com",
#                 password=pwd.hash("admin"),
#                 role=UserRole.ADMIN,
#             )
#             db.add(admin)
#             db.commit()
#             print("Default admin created!")
#         else:
#             print("Admin already exists!")
#     except IntegrityError as e:
#         db.rollback()
#         print("Admin creation error :", e)
#     except Exception as e:
#         db.rollback()
#         print(f"Error: {e}")
#     finally:
#         db.close()
# create_tables()

from sqlalchemy import create_engine, select, Enum as SQLAlchemyEnum
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from pwdlib import PasswordHash
from typing import Optional, Annotated
from pydantic import ConfigDict, EmailStr, BaseModel
from sqlmodel import SQLModel, Field, Session
from datetime import datetime
import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()


class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    SUPERVISOR = "supervisor"
    GUEST = "guest"

# --- Schemas ---


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
    role: Optional[UserRole] = None  # Use the Enum here

# --- Database Setup ---


# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME", "churn_db")
    POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    # Note: sslmode=require might need adjustment based on your environment (e.g., local vs cloud DB)
    DATABASE_URL = f"postgresql://{POSTGRES_DB_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB_NAME}?sslmode=require"

engine = create_engine(DATABASE_URL)

# --- SQLModel Table Definition ---


class User(SQLModel, table=True):
    # Using standard SQLModel fields (which uses SQLAlchemy internally)
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, max_length=50)
    email: str = Field(unique=True, index=True, max_length=255)
    phone: str = Field(unique=True, index=True, max_length=20)
    password: str = Field(max_length=255)  # Hashed password
    role: UserRole = Field(sa_column=SQLAlchemyEnum(
        UserRole), default=UserRole.GUEST)
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

# --- Utility Functions ---


def get_db():
    """Database dependency function using SQLModel's Session."""
    with Session(engine) as session:
        yield session


def create_tables():
    """Creates all database tables defined by SQLModel metadata."""
    SQLModel.metadata.create_all(engine)


pwd_context = PasswordHash.recommended()


def create_admin():
    """Checks for and creates a default admin user if one does not exist."""
    create_tables()  # Ensure tables exist before querying/inserting

    db = Session(engine)
    try:
        existing_admin = db.exec(select(User).where(
            User.email == "admin@example.com")).first()

        if not existing_admin:
            # Use the imported pwd_context for hashing
            hashed_password = pwd_context.hash("admin")
            admin = User(
                username="Admin",
                phone="+221783832653",
                email="admin@example.com",
                password=hashed_password,
                role=UserRole.ADMIN,
            )
            db.add(admin)
            db.commit()
            print("Default admin created!")
        else:
            print("Admin already exists!")
    except IntegrityError as e:
        db.rollback()
        print("Admin creation error (IntegrityError):", e)
    except Exception as e:
        db.rollback()
        print(f"Error during admin creation: {e}")
    finally:
        db.close()
create_tables()
