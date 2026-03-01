"""
Candidate router — public endpoints.

POST /apply                      – submit application (name, email, resume PDF, job_id)
GET  /interview/verify?token=xxx – validate token → load session → redirect to interview
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from db.mongo import applications_col, tokens_col, jobs_col
from models.job import Application, ApplicationStatus, InterviewToken
from services.ai_service import generate_questions_intro
from services.interview_service import create_session
from services.email_service import send_interview_invite
from util.file_util import extract_text, validate_file

log    = logging.getLogger(__name__)
router = APIRouter(tags=["Candidate"])

TOKEN_TTL_HOURS = 72


# ── 1. Candidate applies ──────────────────────────────────────────────────────

@router.post("/apply", status_code=201)
async def apply(
    job_id:          str        = Form(...),
    candidate_name:  str        = Form(...),
    candidate_email: str        = Form(...),
    resume:          UploadFile = File(...),
    duration_minutes: int       = Form(15),
):
    """
    Public apply endpoint.
    1. Validates the job exists.
    2. Extracts resume text.
    3. Calls generate_questions_intro to get candidate name / intro / first question.
    4. Creates InterviewSession + Application document in MongoDB.
    5. Generates a secure 72-hour token.
    6. Sends invite email with verification link.
    """
    # ── Validate job ──────────────────────────────────────────────────────────
    job = await jobs_col().find_one({"job_id": job_id, "is_active": True})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or no longer active")

    # ── Validate & extract resume ─────────────────────────────────────────────
    await validate_file(resume)
    resume_text = await extract_text(resume)

    # ── Generate intro + first question ──────────────────────────────────────
    ai_resp = await generate_questions_intro(
        job_title       = job["job_title"],
        job_description = job["job_description"],
        resume_text     = resume_text,
        candidate_name  = candidate_name,   # pass the real name to the AI
    )

    # ── Create interview session ──────────────────────────────────────────────
    session = await create_session(
        job_title        = job["job_title"],
        job_description  = job["job_description"],
        resume_text      = resume_text,
        candidate_name   = ai_resp.get("candidate_name", candidate_name),
        intro_text       = ai_resp.get("introText", ""),
        first_question   = ai_resp.get("first_question", ""),
        duration_minutes = duration_minutes,
    )

    # ── Create Application document ───────────────────────────────────────────
    application = Application(
        application_id  = str(uuid.uuid4()),
        job_id          = job_id,
        candidate_name  = candidate_name,
        candidate_email = candidate_email,
        resume_text     = resume_text,
        session_id      = session.session_id,
        status          = ApplicationStatus.INVITED,
    )
    await applications_col().insert_one(application.model_dump())

    # ── Generate secure token ─────────────────────────────────────────────────
    raw_token  = str(uuid.uuid4()) + str(uuid.uuid4()).replace("-", "")
    expires_at = datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)

    interview_token = InterviewToken(
        token          = raw_token,
        application_id = application.application_id,
        job_id         = job_id,
        session_id     = session.session_id,
        expires_at     = expires_at,
    )
    await tokens_col().insert_one(interview_token.model_dump())

    # ── Send invite email ─────────────────────────────────────────────────────
    email_ok = await send_interview_invite(
        to_email       = candidate_email,
        candidate_name = candidate_name,
        job_title      = job["job_title"],
        token          = raw_token,
    )
    if not email_ok:
        log.warning("Email delivery failed for application %s", application.application_id)

    return {
        "message":        "Application received. Interview invite sent to your email.",
        "application_id": application.application_id,
        # Only return token in dev when email is not configured
        # Remove this in production!
        "_dev_token":     raw_token if not email_ok else None,
    }


# ── 2. Verify token & load session ───────────────────────────────────────────

@router.get("/interview/verify")
async def verify_token(token: str):
    """
    Validates the interview token.
    - Marks the Application as STARTED.
    - Marks the token as used.
    - Returns session bootstrap data so the Angular app can launch the WS flow.
    """
    now = datetime.now(timezone.utc)

    token_doc = await tokens_col().find_one({"token": token})
    if not token_doc:
        raise HTTPException(status_code=404, detail="Invalid or expired interview link")

    if token_doc.get("used"):
        raise HTTPException(status_code=410, detail="This interview link has already been used")

    expires_at = token_doc["expires_at"]
    # Motor returns naive datetime from MongoDB; make timezone-aware if needed
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now > expires_at:
        raise HTTPException(status_code=410, detail="This interview link has expired")

    # ── Mark token used ───────────────────────────────────────────────────────
    await tokens_col().update_one(
        {"token": token},
        {"$set": {"used": True}},
    )

    # ── Mark application started ──────────────────────────────────────────────
    app_id = token_doc["application_id"]
    await applications_col().update_one(
        {"application_id": app_id},
        {"$set": {"status": ApplicationStatus.STARTED.value}},
    )

    session_id = token_doc["session_id"]

    # ── Fetch session bootstrap data (same as /interview/start/{session_id}) ──
    from services.interview_service import get_session
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")

    return {
        "session_id":       session.session_id,
        "introText":        session.introText,
        "firstQuestion":    session.questions[0] if session.questions else "",
        "candidateName":    session.candidate_name,
        "durationMinutes":  session.duration_minutes,
        "application_id":   app_id,
    }