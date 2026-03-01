"""
Session store backed by MongoDB (Motor).
Replaces the old Redis / in-memory session_store.py.

Each document in the `sessions` collection:
  {
    "session_id": "...",
    "updated_at": <datetime>,   ← TTL index uses this field
    ... all InterviewSession fields ...
  }
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from models.interview import InterviewSession
from db.mongo import sessions_col

log = logging.getLogger(__name__)


async def store_session(session: InterviewSession) -> None:
    """Create or overwrite an interview session document."""
    doc = session.model_dump()
    doc["updated_at"] = datetime.now(timezone.utc)

    await sessions_col().replace_one(
        {"session_id": session.session_id},
        doc,
        upsert=True,
    )


async def fetch_session(session_id: str) -> Optional[InterviewSession]:
    """Return a session or None if not found."""
    doc = await sessions_col().find_one({"session_id": session_id})
    if not doc:
        return None
    doc.pop("_id", None)
    doc.pop("updated_at", None)
    return InterviewSession.model_validate(doc)


async def delete_session(session_id: str) -> None:
    await sessions_col().delete_one({"session_id": session_id})