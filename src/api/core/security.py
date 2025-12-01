import os
from datetime import datetime, timedelta, UTC
from typing import Optional
from jose import jwt
from pwdlib import PasswordHash
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SECRET_KEY = os.getenv("API_KEY_SECRET", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd = PasswordHash.recommended()

# --- Password Functions ---


def hash_password(password: str) -> str:
    """Hashes a plain text password."""
    return pwd.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored hash."""
    return pwd.verify(plain_password, hashed_password)

# --- JWT Functions ---


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(
            UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire.timestamp()})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decodes a JWT token."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
