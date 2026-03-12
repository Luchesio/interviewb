"""
MongoDB async client (Motor).
Exposes one function per collection; call init_db() once at startup.

ENV vars:
  MONGODB_URL  – connection string (default: mongodb://localhost:27017)
  MONGODB_DB   – database name    (default: ai_interview)
"""

import os
import logging
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
    AsyncIOMotorGridFSBucket,
)

log = logging.getLogger(__name__)

_client:   AsyncIOMotorClient   | None = None
_database: AsyncIOMotorDatabase | None = None
_gridfs:   AsyncIOMotorGridFSBucket | None = None


def _get_db() -> AsyncIOMotorDatabase:
    if _database is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _database


async def init_db() -> None:
    """
    Connect to MongoDB and create all required indexes.
    Call this once from the FastAPI lifespan event.
    """
    global _client, _database, _gridfs

    url  = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    name = os.getenv("MONGODB_DB",  "ai_interview")

    _client   = AsyncIOMotorClient(url)
    _database = _client[name]
    _gridfs   = AsyncIOMotorGridFSBucket(_database, bucket_name="interview_media")
    log.info("Connected to MongoDB: %s / %s", url, name)

    await _create_indexes()


async def _create_indexes() -> None:
    db = _get_db()

    # ── interview_tokens ──────────────────────────────────────────────────────
    await db["interview_tokens"].create_index(
        "expires_at", expireAfterSeconds=0, name="ttl_token_expiry",
    )
    await db["interview_tokens"].create_index("token",          unique=True)
    await db["interview_tokens"].create_index("application_id")

    # ── sessions ──────────────────────────────────────────────────────────────
    await db["sessions"].create_index(
        "updated_at", expireAfterSeconds=86400, name="ttl_session_expiry",
    )
    await db["sessions"].create_index("session_id", unique=True)

    # ── jobs ──────────────────────────────────────────────────────────────────
    await db["jobs"].create_index("job_id",       unique=True)
    await db["jobs"].create_index("recruiter_id")

    # ── applications ──────────────────────────────────────────────────────────
    await db["applications"].create_index("application_id", unique=True)
    await db["applications"].create_index("job_id")
    await db["applications"].create_index("candidate_email")

    # ── reports ───────────────────────────────────────────────────────────────
    await db["reports"].create_index("report_id",  unique=True)
    await db["reports"].create_index("job_id")
    await db["reports"].create_index("session_id")

    # ── users (NEW) ───────────────────────────────────────────────────────────
    await db["users"].create_index("user_id", unique=True)
    await db["users"].create_index("email",   unique=True)

    # ── media (NEW) ───────────────────────────────────────────────────────────
    await db["media"].create_index("media_id",   unique=True)
    await db["media"].create_index("session_id")

    log.info("MongoDB indexes ensured.")


# ── Collection accessors ──────────────────────────────────────────────────────

def sessions_col() -> AsyncIOMotorCollection:
    return _get_db()["sessions"]

def jobs_col() -> AsyncIOMotorCollection:
    return _get_db()["jobs"]

def applications_col() -> AsyncIOMotorCollection:
    return _get_db()["applications"]

def tokens_col() -> AsyncIOMotorCollection:
    return _get_db()["interview_tokens"]

def reports_col() -> AsyncIOMotorCollection:
    return _get_db()["reports"]

def users_col() -> AsyncIOMotorCollection:
    """NEW: Recruiter user accounts."""
    return _get_db()["users"]

def media_col() -> AsyncIOMotorCollection:
    """NEW: Media recording metadata."""
    return _get_db()["media"]

async def get_gridfs() -> AsyncIOMotorGridFSBucket:
    """NEW: GridFS bucket for storing interview media blobs."""
    if _gridfs is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _gridfs