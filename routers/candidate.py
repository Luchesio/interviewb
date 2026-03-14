"""
Candidate router — public endpoints.

POST /apply                              – submit application
GET  /interview/verify?token=xxx         – validate token → load session
GET  /candidate/applications?email=x     – candidate dashboard
POST /interview/upload-media             – upload recording to Cloudinary
GET  /interview/media/{media_id}         – redirect to Cloudinary URL

Recording storage:
  Videos (webcam + screen share) are uploaded directly to Cloudinary.
  Only the Cloudinary secure_url is stored in MongoDB — no binary data
  ever touches the database, keeping MongoDB usage minimal.
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


# ── 4. Save media URL — frontend uploads directly to Cloudinary ──────────────
#
# Video blobs exceed Vercel's 4.5 MB serverless function limit so they must
# never pass through this server.  The Angular frontend uploads the blob
# straight to Cloudinary (unsigned preset) and then sends only the resulting
# URL here.  This endpoint stores that URL in MongoDB — a tiny JSON payload
# that is well within all platform limits.

class SaveMediaUrlRequest(BaseModel):
    session_id:     str
    media_type:     str           # "webcam" | "screen"
    cloudinary_url: str
    size_bytes:     int = 0


@router.post("/interview/save-media-url", status_code=201)
async def save_media_url(body: SaveMediaUrlRequest):
    """
    Persist the Cloudinary URL returned by a direct browser → Cloudinary upload.
    No file data passes through this endpoint — only a short URL string.
    """
    if not body.cloudinary_url.startswith("https://res.cloudinary.com/"):
        raise HTTPException(status_code=400, detail="Invalid Cloudinary URL.")

    media_id  = str(uuid.uuid4())
    media_doc = {
        "media_id":       media_id,
        "session_id":     body.session_id,
        "media_type":     body.media_type,
        "cloudinary_url": body.cloudinary_url,
        "size_bytes":     body.size_bytes,
        "uploaded_at":    datetime.now(timezone.utc),
    }
    await media_col().insert_one(media_doc)

    url_field = "webcam_url" if body.media_type == "webcam" else "screen_url"
    await reports_col().update_one(
        {"session_id": body.session_id},
        {"$set": {url_field: body.cloudinary_url}},
    )

    log.info(
        "Media URL saved: %s session=%s type=%s",
        media_id, body.session_id, body.media_type,
    )
    return {"media_id": media_id, "cloudinary_url": body.cloudinary_url}


# ── 5. Get media — redirect to Cloudinary URL ────────────────────────────────

@router.get("/interview/media/{media_id}")
async def get_media_url(media_id: str):
    """Redirect to the Cloudinary URL — video streams directly from Cloudinary."""
    media_doc = await media_col().find_one({"media_id": media_id})
    if not media_doc:
        raise HTTPException(status_code=404, detail="Media not found")

    cloudinary_url = media_doc.get("cloudinary_url")
    if not cloudinary_url:
        raise HTTPException(status_code=404, detail="Recording URL not available")

    return RedirectResponse(url=cloudinary_url, status_code=302)