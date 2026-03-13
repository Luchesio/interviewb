"""
Candidate router — public endpoints.

POST /apply                              – submit application
GET  /interview/verify?token=xxx         – validate token → load session
GET  /candidate/applications?email=x     – candidate dashboard (Feature 2)
POST /interview/upload-media             – upload WebRTC recording (Feature 3)

Changes (new features):
  • /apply now screens the resume against the job BEFORE generating interview
    questions. If the candidate is NOT a fit, they receive a rejection email
    and the endpoint returns early (no interview session / token created).
    If the candidate IS a fit, they receive a shortlist email that also
    contains the interview link (the old plain-invite email is replaced).

  • The 10-minute email delay is now handled entirely by Resend's scheduledAt
    field — no asyncio.sleep, no background task, no in-process timer.
    This makes the endpoint fully serverless-safe (Vercel, Railway, etc.).
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from db.mongo import applications_col, tokens_col, jobs_col, reports_col, media_col
from models.job import Application, ApplicationStatus, InterviewToken
from services.ai_service import generate_questions_intro, screen_resume
from services.interview_service import create_session
from services.email_service import send_candidate_fit_email
from util.file_util import extract_text, validate_file

log    = logging.getLogger(__name__)
router = APIRouter(tags=["Candidate"])

TOKEN_TTL_HOURS   = 72
MAX_MEDIA_BYTES   = 500 * 1024 * 1024   # 500 MB
FIT_EMAIL_DELAY_M = 10                  # minutes — passed to Resend scheduledAt


# ── 1. Candidate applies ──────────────────────────────────────────────────────

@router.post("/apply", status_code=201)
async def apply(
    job_id:           str        = Form(...),
    candidate_name:   str        = Form(...),
    candidate_email:  str        = Form(...),
    resume:           UploadFile = File(...),
    duration_minutes: int        = Form(15),
    language:         str        = Form("en"),
):
    """
    Public apply endpoint.

    Flow:
      1. Validate the job exists and is active.
      2. Extract resume text.
      3. Screen resume against the job via AI.
      4a. NOT FIT  → persist application as REJECTED, schedule rejection email
                     via Resend (10-minute delay), return immediately.
      4b. FIT      → generate interview session + token, schedule shortlist email
                     (with interview link) via Resend (10-minute delay), return.

    The 10-minute delay is handled by Resend's scheduledAt field — the API
    call returns instantly and Resend delivers at the right time. No background
    tasks or asyncio.sleep are needed, making this fully serverless-safe.
    """
    # ── Step 1: validate job ──────────────────────────────────────────────────
    job = await jobs_col().find_one({"job_id": job_id, "is_active": True})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or no longer active")

    # ── Step 2: extract resume text ───────────────────────────────────────────
    await validate_file(resume)
    resume_text = await extract_text(resume)

    # ── Step 3: AI resume screening ───────────────────────────────────────────
    screening = await screen_resume(
        job_title       = job["job_title"],
        job_description = job["job_description"],
        resume_text     = resume_text,
    )
    is_fit = screening["is_fit"]
    score  = screening["score"]

    log.info(
        "Resume screening: candidate=%s job=%s is_fit=%s score=%s",
        candidate_email, job_id, is_fit, score,
    )

    # ── Step 4a: NOT a fit ────────────────────────────────────────────────────
    if not is_fit:
        application = Application(
            application_id  = str(uuid.uuid4()),
            job_id          = job_id,
            candidate_name  = candidate_name,
            candidate_email = candidate_email,
            resume_text     = resume_text,
            session_id      = None,
            status          = ApplicationStatus.REJECTED,
            language        = language,
        )
        await applications_col().insert_one(application.model_dump())

        # Schedule rejection email via Resend's scheduledAt — no sleep, no background task.
        ok = await send_candidate_fit_email(
            to_email       = candidate_email,
            candidate_name = candidate_name,
            job_title      = job["job_title"],
            is_fit         = False,
            token          = None,
            delay_minutes  = FIT_EMAIL_DELAY_M,
        )
        if not ok:
            log.warning("Rejection email scheduling failed for application %s", application.application_id)
        else:
            log.info("Rejection email scheduled (%dm delay) for application %s", FIT_EMAIL_DELAY_M, application.application_id)

        return {
            "message":        "Thank you for applying! We are reviewing your application and will be in touch within the next few minutes.",
            "application_id": application.application_id,
            "screening":      {"is_fit": False, "score": score},
        }

    # ── Step 4b: IS a fit — generate interview session ────────────────────────
    ai_resp = await generate_questions_intro(
        job_title        = job["job_title"],
        job_description  = job["job_description"],
        resume_text      = resume_text,
        candidate_name   = candidate_name,
        language         = language,
        custom_questions = job.get("custom_questions", []),
    )

    session = await create_session(
        job_title        = job["job_title"],
        job_description  = job["job_description"],
        resume_text      = resume_text,
        candidate_name   = ai_resp.get("candidate_name", candidate_name),
        intro_text       = ai_resp.get("introText", ""),
        first_question   = ai_resp.get("first_question", ""),
        duration_minutes = duration_minutes,
        language         = language,
    )

    application = Application(
        application_id  = str(uuid.uuid4()),
        job_id          = job_id,
        candidate_name  = candidate_name,
        candidate_email = candidate_email,
        resume_text     = resume_text,
        session_id      = session.session_id,
        status          = ApplicationStatus.INVITED,
        language        = language,
    )
    await applications_col().insert_one(application.model_dump())

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

    # Schedule shortlist email via Resend's scheduledAt — no sleep, no background task.
    ok = await send_candidate_fit_email(
        to_email       = candidate_email,
        candidate_name = candidate_name,
        job_title      = job["job_title"],
        is_fit         = True,
        token          = raw_token,
        delay_minutes  = FIT_EMAIL_DELAY_M,
    )
    if not ok:
        log.warning("Shortlist email scheduling failed for application %s", application.application_id)
    else:
        log.info("Shortlist email scheduled (%dm delay) for application %s", FIT_EMAIL_DELAY_M, application.application_id)

    return {
        "message":        "Thank you for applying! We are reviewing your application and will be in touch within the next few minutes.",
        "application_id": application.application_id,
        "screening":      {"is_fit": True, "score": score},
    }


# ── 2. Verify token & load session ───────────────────────────────────────────

@router.get("/interview/verify")
async def verify_token(token: str):
    """Validates the interview token and returns session bootstrap data."""
    now = datetime.now(timezone.utc)

    token_doc = await tokens_col().find_one({"token": token})
    if not token_doc:
        raise HTTPException(status_code=404, detail="Invalid or expired interview link")

    if token_doc.get("used"):
        raise HTTPException(status_code=410, detail="This interview link has already been used")

    expires_at = token_doc["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if now > expires_at:
        raise HTTPException(status_code=410, detail="This interview link has expired")

    await tokens_col().update_one({"token": token}, {"$set": {"used": True}})

    app_id = token_doc["application_id"]
    await applications_col().update_one(
        {"application_id": app_id},
        {"$set": {"status": ApplicationStatus.STARTED.value}},
    )

    session_id = token_doc["session_id"]

    from services.interview_service import get_session
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")

    return {
        "session_id":      session.session_id,
        "introText":       session.introText,
        "firstQuestion":   session.questions[0] if session.questions else "",
        "candidateName":   session.candidate_name,
        "durationMinutes": session.duration_minutes,
        "application_id":  app_id,
        "language":        getattr(session, "language", "en"),
    }


# ── 3. Candidate dashboard ────────────────────────────────────────────────────

@router.get("/candidate/applications")
async def get_candidate_applications(email: str):
    """
    Return all applications and their statuses for a candidate email.
    No auth required — candidates use their email address to look up status.
    """
    cursor = applications_col().find(
        {"candidate_email": email},
        {"_id": 0, "resume_text": 0},
        sort=[("created_at", -1)],
    )
    applications = await cursor.to_list(length=100)

    enriched = []
    for app in applications:
        entry = dict(app)

        job = await jobs_col().find_one({"job_id": app["job_id"]}, {"job_title": 1, "_id": 0})
        entry["job_title"] = job.get("job_title", "Unknown Role") if job else "Unknown Role"

        if app.get("session_id"):
            report_doc = await reports_col().find_one(
                {"session_id": app["session_id"]},
                {
                    "_id": 0,
                    "report_id": 1,
                    "report.score": 1,
                    "report.hiring_recommendation": 1,
                    "report.soft_skills.communication_score": 1,
                    "report.soft_skills.confidence": 1,
                    "created_at": 1,
                },
            )
            entry["report_summary"] = report_doc if report_doc else None
        else:
            entry["report_summary"] = None

        enriched.append(entry)

    return {"email": email, "applications": enriched, "total": len(enriched)}


# ── 4. Upload media recording ─────────────────────────────────────────────────

@router.post("/interview/upload-media", status_code=201)
async def upload_media(
    session_id: str        = Form(...),
    media_type: str        = Form("video"),
    file:       UploadFile = File(...),
):
    content = await file.read()
    size_mb  = len(content) / (1024 * 1024)

    if len(content) > MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max 500 MB, got {size_mb:.1f} MB)")

    allowed_types = {"video/webm", "video/mp4", "audio/webm", "audio/ogg", "audio/wav"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")