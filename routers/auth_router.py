import os
import uuid
import logging
import bcrypt
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from jose import jwt

from db.mongo import users_col

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Auth"])

JWT_SECRET   = os.getenv("JWT_SECRET",   "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING")
JWT_ALGO     = os.getenv("JWT_ALGO",     "HS256")
JWT_EXPIRE_H = int(os.getenv("JWT_EXPIRE_H", "24"))


# ── Request / Response models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    EmailStr
    password: str
    name:     str = ""


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user_id:      str
    email:        str
    name:         str
    role:         str


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain[:72].encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain[:72].encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_H)
    payload = {
        "sub":   user_id,
        "email": email,
        "role":  role,
        "exp":   expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest):
    """Register a new recruiter account."""
    existing = await users_col().find_one({"email": body.email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user_id = str(uuid.uuid4())
    user_doc = {
        "user_id":         user_id,
        "email":           body.email,
        "name":            body.name,
        "hashed_password": hash_password(body.password),
        "role":            "recruiter",
        "created_at":      datetime.now(timezone.utc),
    }
    await users_col().insert_one(user_doc)
    log.info("Recruiter registered: %s", body.email)

    token = create_access_token(user_id, body.email, "recruiter")
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        email=body.email,
        name=body.name,
        role="recruiter",
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """Authenticate a recruiter and return a JWT."""
    user = await users_col().find_one({"email": body.email})
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(user["user_id"], user["email"], user["role"])
    log.info("Recruiter logged in: %s", body.email)
    return TokenResponse(
        access_token=token,
        user_id=user["user_id"],
        email=user["email"],
        name=user.get("name", ""),
        role=user["role"],
    )