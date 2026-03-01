"""
MongoDB document models for the recruiter/candidate flow.
All stored via Motor (async pymongo).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ── Job ───────────────────────────────────────────────────────────────────────

class Job(BaseModel):
    job_id:          str
    recruiter_id:    str = "default"          # extend when auth is added
    job_title:       str
    job_description: str
    created_at:      datetime = Field(default_factory=datetime.utcnow)
    is_active:       bool = True


# ── Application ───────────────────────────────────────────────────────────────

class ApplicationStatus(str, Enum):
    PENDING   = "pending"
    INVITED   = "invited"   # email sent
    STARTED   = "started"   # verify endpoint hit
    COMPLETED = "completed"


class Application(BaseModel):
    application_id:  str
    job_id:          str
    candidate_name:  str
    candidate_email: str
    resume_text:     str
    session_id:      Optional[str] = None   # set when interview is created
    status:          ApplicationStatus = ApplicationStatus.PENDING
    created_at:      datetime = Field(default_factory=datetime.utcnow)


# ── Interview invite token ────────────────────────────────────────────────────

class InterviewToken(BaseModel):
    token:          str
    application_id: str
    job_id:         str
    session_id:     str
    expires_at:     datetime       # 72-hour TTL; MongoDB TTL index on this field
    used:           bool = False


# ── Report (stored after interview completes) ─────────────────────────────────

class StoredReport(BaseModel):
    report_id:       str
    session_id:      str
    application_id:  str
    job_id:          str
    candidate_name:  str
    candidate_email: str
    report:          dict[str, Any]   # raw JSON from generate_report()
    created_at:      datetime = Field(default_factory=datetime.utcnow)