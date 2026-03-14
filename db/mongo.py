"""
MongoDB connection and collection accessors.

GridFS has been removed — interview recordings are now stored on Cloudinary.
Only the Cloudinary URL is persisted in the media collection.
"""

import os
import logging
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)

log = logging.getLogger(__name__)

_client:   AsyncIOMotorClient   | None = None
_database: AsyncIOMotorDatabase | None = None


def _get_db() -> AsyncIOMotorDatabase:
    if _database is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")
    return _database


async def init_db() -> None:
    """
    Connect to MongoDB and create all required indexes.
    Call this once from the FastAPI lifespan event.
    """
    global _client, _database

    url  = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    name = os.getenv("MONGODB_DB",  "ai_interview")

    _client   = AsyncIOMotorClient(url)
    _database = _client[name]
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

    # ── users ─────────────────────────────────────────────────────────────────
    await db["users"].create_index("user_id", unique=True)
    await db["users"].create_index("email",   unique=True)

    # ── media (Cloudinary URLs only — no binary data) ─────────────────────────
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
    return _get_db()["users"]

def media_col() -> AsyncIOMotorCollection:
    """Stores Cloudinary URLs and metadata — no binary blobs."""
    return _get_db()["media"]