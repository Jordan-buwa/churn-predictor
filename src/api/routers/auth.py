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

from src.api.db import User, SessionLocal

def get_jwt_strategy() -> JWTStrategy:
    secret = os.getenv("AUTH_SECRET", "CHANGE_ME")
    return JWTStrategy(secret=secret, lifetime_seconds=3600)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_db(session: Session = Depends(get_db)):
    yield SQLAlchemyUserDatabase(session, User)

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    def __init__(self, user_db: SQLAlchemyUserDatabase):
        super().__init__(user_db)
        self.reset_password_token_secret = os.getenv("AUTH_SECRET", "CHANGE_ME")
        self.verification_token_secret = os.getenv("AUTH_SECRET", "CHANGE_ME")

    async def on_after_register(self, user: User, request=None):
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

router.include_router(
    fastapi_users.get_register_router(fau_schemas.UserRead, fau_schemas.UserCreate),
    prefix="/auth",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_verify_router(),
    prefix="/auth",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_users_router(fau_schemas.UserRead, fau_schemas.UserUpdate),
    prefix="/users",
    tags=["users"],
)