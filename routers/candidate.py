"""
Candidate router — public endpoints.

POST /apply                              – submit application
GET  /jobs/{job_id}/public               – public job details for apply page
GET  /interview/verify?token=xxx         – validate token → load session
GET  /candidate/applications?email=x     – candidate dashboard
POST /interview/upload-media             – upload recording to Cloudinary
GET  /interview/media/{media_id}         – redirect to Cloudinary URL
"""

import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from db.mongo import applications_col, tokens_col, jobs_col, reports_col, media_col
from models.job import Application, ApplicationStatus, InterviewToken
from services.ai_service import generate_questions_intro, screen_resume
from services.interview_service import create_session
from services.email_service import send_candidate_fit_email
from util.file_util import extract_text, validate_file

log    = logging.getLogger(__name__)
router = APIRouter(tags=["Candidate"])

TOKEN_TTL_HOURS   = 72
FIT_EMAIL_DELAY_M = 10


# ── 0. Public job details (no auth) ──────────────────────────────────────────

@router.get("/jobs/{job_id}/public")
async def get_public_job(job_id: str):
    """
    Public endpoint — returns minimal job info needed to render the apply page.
    No authentication required.  Only active jobs are returned.
    """
    doc = await jobs_col().find_one(
        {"job_id": job_id, "is_active": True},
        {"_id": 0, "job_id": 1, "job_title": 1, "job_description": 1},
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found or no longer active")
    return doc


# ── 1. Candidate applies ──────────────────────────────────────────────────────

@router.post("/apply", status_code=201)
async def apply(
    job_id:            str        = Form(...),
    candidate_name:    str        = Form(...),
    candidate_email:   str        = Form(...),
    resume:            UploadFile = File(...),
    duration_minutes:  int        = Form(15),
    language:          str        = Form("en"),
    # ── New fields ────────────────────────────────────────────────────────────
    phone:             str        = Form(""),
    linkedin_url:      str        = Form(""),
    portfolio_url:     str        = Form(""),
    years_experience:  str        = Form(""),
    current_location:  str        = Form(""),
    cover_letter:      str        = Form(""),
):
    job = await jobs_col().find_one({"job_id": job_id, "is_active": True})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or no longer active")

    await validate_file(resume)
    resume_text = await extract_text(resume)

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

    # Extra fields dict — stored alongside the application
    extra_fields = {
        "phone":            phone,
        "linkedin_url":     linkedin_url,
        "portfolio_url":    portfolio_url,
        "years_experience": years_experience,
        "current_location": current_location,
        "cover_letter":     cover_letter,
    }

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
        app_doc = application.model_dump()
        app_doc.update(extra_fields)
        await applications_col().insert_one(app_doc)

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
    app_doc = application.model_dump()
    app_doc.update(extra_fields)
    await applications_col().insert_one(app_doc)

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
                },
            )
            if report_doc:
                entry["report_id"]              = report_doc.get("report_id")
                entry["score"]                  = report_doc.get("report", {}).get("score")
                entry["hiring_recommendation"]  = report_doc.get("report", {}).get("hiring_recommendation")
        enriched.append(entry)

    return {"applications": enriched}


# ── 4. Upload recording to Cloudinary ────────────────────────────────────────

@router.post("/interview/upload-media")
async def upload_media(
    session_id:  str        = Form(...),
    media_type:  str        = Form("webcam"),
    file:        UploadFile = File(...),
):
    from services.cloudinary_service import upload_video
    import uuid as _uuid

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    url = await upload_video(content, filename=file.filename or "recording.webm")

    media_id  = str(_uuid.uuid4())
    media_doc = {
        "media_id":   media_id,
        "session_id": session_id,
        "media_type": media_type,
        "url":        url,
        "created_at": datetime.now(timezone.utc),
    }
    await media_col().insert_one(media_doc)
    log.info("Media uploaded: media_id=%s session=%s type=%s", media_id, session_id, media_type)
    return {"media_id": media_id, "url": url}


# ── 5. Redirect to Cloudinary URL ────────────────────────────────────────────

@router.get("/interview/media/{media_id}")
async def get_media(media_id: str):
    doc = await media_col().find_one({"media_id": media_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Media not found")
    return RedirectResponse(url=doc["url"])