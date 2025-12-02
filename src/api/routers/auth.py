import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Form, Query
from sqlmodel import Session, select
from datetime import timedelta
from typing import Optional
# Import necessary components from other modules
from src.api.db import User, get_db, UserCreate, UserRead, UserUpdate
from src.api.authenticator import get_current_user, get_current_active_user, admin_only_access
# Import security helpers for hashing and token creation
from src.api.core.security import (
    create_access_token, verify_password, hash_password, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Registration route
@router.post("/register", response_model=UserRead)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.exec(
        select(User).where(
            (User.email == user_data.email) | (
                User.username == user_data.username)
        )
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )

    # Create new user
    hashed_password = hash_password(user_data.password)

    user = User(
        username=user_data.username,
        email=user_data.email,
        phone=user_data.phone,
        password=hashed_password,
        role=user_data.role,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user

# Login route


@router.post("/login")
def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Search by username, since that is what the login form provides
    user = db.exec(select(User).where(User.username == username)).first()

    # Check password using imported security helper
    if not user or not verify_password(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserRead.model_validate(user)
    }

# Refresh token


@router.post("/refresh")
def refresh_token(current_user: User = Depends(get_current_user)):
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(current_user.id)}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

# User information and update routes


@router.get("/me", response_model=UserRead)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return UserRead.model_validate(current_user)


@router.put("/me", response_model=UserRead)
async def update_user_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    update_data = user_update.model_dump(exclude_unset=True)

    # If password is being updated, hash it
    if "password" in update_data:
        update_data["password"] = hash_password(update_data["password"])

    for field, value in update_data.items():
        setattr(current_user, field, value)

    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return UserRead.model_validate(current_user)


@router.get("/users", response_model=list[UserRead])
def list_users(db: Session = Depends(get_db), admin: User = Depends(admin_only_access)):
    """Admin-only endpoint to list all API users."""
    users = db.exec(select(User)).all()
    return [UserRead.model_validate(u) for u in users]


@router.put("/users/{user_id}", response_model=UserRead)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_access)
):
    """Admin-only endpoint to update a user."""
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    update_data = user_update.model_dump(exclude_unset=True)
    
    # If password is being updated, hash it
    if "password" in update_data and update_data["password"]:
        update_data["password"] = hash_password(update_data["password"])
    elif "password" in update_data:
        # Remove password if it's empty
        del update_data["password"]
    
    # Check if email/username already exists (excluding current user)
    if "email" in update_data:
        existing = db.exec(
            select(User).where(
                User.email == update_data["email"],
                User.id != user_id
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    if "username" in update_data:
        existing = db.exec(
            select(User).where(
                User.username == update_data["username"],
                User.id != user_id
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    for field, value in update_data.items():
        setattr(user, field, value)
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserRead.model_validate(user)

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_access)
):
    """Admin-only endpoint to delete a user."""
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent self-deletion
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    db.delete(user)
    db.commit()
    
    return {
        "status": "success",
        "message": f"User {user.username} deleted successfully"
    }

@router.get("/users/search")
async def search_users(
    query: Optional[str] = Query(None, description="Search by username, email, or phone"),
    role: Optional[str] = Query(None, description="Filter by role"),
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_access)
):
    """Admin-only endpoint to search and filter users."""
    stmt = select(User)
    
    if query:
        stmt = stmt.where(
            (User.username.ilike(f"%{query}%")) |
            (User.email.ilike(f"%{query}%")) |
            (User.phone.ilike(f"%{query}%"))
        )
    
    if role:
        stmt = stmt.where(User.role == role)
    
    users = db.exec(stmt).all()
    return [UserRead.model_validate(u) for u in users]

@router.patch("/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_access)
):
    """Admin-only endpoint to toggle user active status."""
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent deactivating self
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    user.is_active = not user.is_active
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {
        "status": "success",
        "message": f"User {user.username} is now {'active' if user.is_active else 'inactive'}",
        "user": UserRead.model_validate(user)
    }