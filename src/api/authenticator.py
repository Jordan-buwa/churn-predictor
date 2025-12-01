import jwt
import os
# from sqlalchemy.orm import Session
# Changed from sqlalchemy.orm.Session to sqlmodel.Session for consistency
from sqlmodel import Session, select
from dotenv import load_dotenv
from fastapi import HTTPException, Security, APIRouter, Depends, status
from src.api.db import User, get_db, UserRole
from pwdlib import PasswordHash
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Annotated  # Import Annotated for modern dependency syntax

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("API_KEY_SECRET", "your-secret-key-here")
ALGORITHM = "HS256"

# Security
security = HTTPBearer()
router = APIRouter()


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials,
                             SECRET_KEY, algorithms=[ALGORITHM])
        # Use str() for comparison consistency as JWT subject is stringified user ID
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    # db.get expects int, so we convert back
    user = db.get(User, int(user_id))
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# --- NEW ADMIN-ONLY DEPENDENCY ---
def admin_only_access(current_user: User = Depends(get_current_active_user)):
    """Dependency to enforce that the user has the 'admin' role."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Admin access is required to perform this action."
        )
    return current_user  # Return user object for clarity, though True would suffice


# Export dependencies for use in other routes
current_active_user = get_current_active_user
AdminRequired = Annotated[User, Depends(admin_only_access)]


# import os
# from sqlmodel import Session
# from fastapi import HTTPException, Depends, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from datetime import datetime, timedelta
# from typing import Annotated

# from src.api.db import User, get_db, UserRole
# # Import necessary components from the new security module
# from src.api.core.security import decode_token, create_access_token

# # Security scheme instance
# security = HTTPBearer()


# async def get_current_user(
#     credentials: HTTPAuthorizationCredentials = Depends(security),
#     db: Session = Depends(get_db)
# ) -> User:
#     """Authenticates user via JWT token and retrieves the User object."""
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )

#     try:
#         # Decode token using helper function
#         payload = decode_token(credentials.credentials)
#         user_id: str = payload.get("sub")

#         if user_id is None:
#             raise credentials_exception
#     except Exception:  # Catches both JWTError and decode_token errors
#         raise credentials_exception

#     # db.get expects int, so we convert back
#     user = db.get(User, int(user_id))
#     if user is None:
#         raise credentials_exception

#     return user


# async def get_current_active_user(current_user: User = Depends(get_current_user)):
#     """Ensures the authenticated user is active."""
#     if not current_user.is_active:
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user


# # --- ADMIN-ONLY DEPENDENCY ---
# def admin_only_access(current_user: User = Depends(get_current_active_user)):
#     """Dependency to enforce that the user has the 'admin' role."""
#     if current_user.role != UserRole.ADMIN:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Forbidden: Admin access is required to perform this action."
#         )
#     return current_user


# # Export dependencies for use in other routes (AdminRequired is the clean alias)
# current_active_user = get_current_active_user
# AdminRequired = Annotated[User, Depends(admin_only_access)]
