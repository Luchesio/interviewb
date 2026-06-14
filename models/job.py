"""
MongoDB document models for the recruiter/candidate flow.
All stored via Motor (async pymongo).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


# ── Job ───────────────────────────────────────────────────────────────────────

class Job(BaseModel):
    job_id:           str
    recruiter_id:     str
    job_title:        str
    job_description:  str
    custom_questions: List[str] = Field(default_factory=list)  # Feature 4
    created_at:       datetime  = Field(default_factory=datetime.utcnow)
    is_active:        bool      = True


# ── Application ───────────────────────────────────────────────────────────────

class ApplicationStatus(str, Enum):
    PENDING   = "pending"
    INVITED   = "invited"
    STARTED   = "started"
    COMPLETED = "completed"
    REJECTED  = "rejected"   # Added: candidate did not pass resume screening


class Application(BaseModel):
    application_id:  str
    job_id:          str
    candidate_name:  str
    candidate_email: str
    resume_text:     str
    session_id:      Optional[str] = None
    status:          ApplicationStatus = ApplicationStatus.PENDING
    language:        str = "en"                     # Feature 6
    created_at:      datetime = Field(default_factory=datetime.utcnow)


# ── Interview invite token ────────────────────────────────────────────────────

class InterviewToken(BaseModel):
    token:          str
    application_id: str
    job_id:         str
    # session_id is created lazily when the candidate first opens the link, so
    # the heavy intro/question generation is deferred out of the /apply request.
    session_id:       Optional[str] = None
    duration_minutes: int           = 15
    expires_at:       datetime
    used:             bool = False


# ── Report (stored after interview completes) ─────────────────────────────────

class StoredReport(BaseModel):
    report_id:       str
    session_id:      str
    application_id:  str
    job_id:          str
    candidate_name:  str
    candidate_email: str
    report:          dict[str, Any]
    media_id:        Optional[str]  = None   # Feature 3: linked recording
    media_type:      Optional[str]  = None
    created_at:      datetime = Field(default_factory=datetime.utcnow)


# ── Recruiter user account ────────────────────────────────────────────────────

class RecruiterUser(BaseModel):
    user_id:           str
    email:             str
    name:              str = ""
    hashed_password:   str
    role:              str = "recruiter"
    slack_webhook_url: Optional[str] = None  # Feature 7
    created_at:        datetime = Field(default_factory=datetime.utcnow)