import uuid
import os
from fastapi import APIRouter, Depends
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users.manager import BaseUserManager, UUIDIDMixin
from fastapi_users import schemas as fau_schemas
from sqlalchemy.orm import Session

from src.api.db import User, get_db

def get_jwt_strategy() -> JWTStrategy:
    secret = os.getenv("AUTH_SECRET", "CHANGE_ME")
    if secret == "CHANGE_ME":
        print("⚠️  WARNING: Using default AUTH_SECRET - set AUTH_SECRET environment variable for production")
    return JWTStrategy(secret=secret, lifetime_seconds=3600)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

def get_user_db(session: Session = Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User)

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    def __init__(self, user_db: SQLAlchemyUserDatabase):
        super().__init__(user_db)
        self.reset_password_token_secret = os.getenv("AUTH_SECRET", "CHANGE_ME")
        self.verification_token_secret = os.getenv("AUTH_SECRET", "CHANGE_ME")

    async def on_after_register(self, user: User, request=None):
        print(f"User {user.email} has registered.")
        return

def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

router = APIRouter()

# Export a dependency to require an authenticated, active user
current_active_user = fastapi_users.current_user(active=True)

router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

class UserRead(fau_schemas.BaseUser[uuid.UUID]):
    pass

class UserCreate(fau_schemas.BaseUserCreate):
    pass

class UserUpdate(fau_schemas.BaseUserUpdate):
    pass

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)