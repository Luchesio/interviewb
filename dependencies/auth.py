"""
JWT dependency — use as a FastAPI Depends() on protected routes.

Usage:
    from dependencies.auth import get_current_recruiter, CurrentRecruiter

    @router.get("/protected")
    async def protected(user: CurrentRecruiter):
        return {"user_id": user["user_id"]}
"""

import os
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION_USE_LONG_RANDOM_STRING")
JWT_ALGO   = os.getenv("JWT_ALGO",   "HS256")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_recruiter(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Decode and validate the JWT.  Raises 401 on any failure.
    Returns a dict with user_id, email, role.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        user_id: str = payload.get("sub")
        email:   str = payload.get("email")
        role:    str = payload.get("role")
        if not user_id or role != "recruiter":
            raise credentials_exc
        return {"user_id": user_id, "email": email, "role": role}
    except JWTError:
        raise credentials_exc


# Convenient type alias for route signatures
CurrentRecruiter = Annotated[dict, Depends(get_current_recruiter)]