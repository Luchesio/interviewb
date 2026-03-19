"""
Interview router — HTTP polling engine (Vercel-compatible).

Why HTTP instead of WebSocket?
  Vercel serverless functions cannot hold persistent WebSocket connections.
  The platform rejects the HTTP Upgrade handshake immediately.
  Replacing the WebSocket with a request/response HTTP flow makes the backend
  fully Vercel-compatible with zero infrastructure changes.

Turn-based flow:
  POST /interview/answer  — candidate submits a transcribed answer (or skip).
                            Backend saves it, generates the next question via
                            GPT, and returns it in the same response.
  POST /interview/end     — candidate or frontend timer ends the interview.
  GET  /interview/report/{session_id} — unchanged, generates and emails report.

The countdown timer runs on the frontend (it already did). The backend still
validates session expiry on every /answer call so the time limit is enforced
server-side too.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from models.interview import InteriewStatusEnum
from services.ai_service import (
    generate_questions_intro,
    generate_next_question,
    generate_report,
    transcribe_audio,
)
from services.interview_service import (
    create_session,
    get_session,
    save_answer,
    append_question,
    mark_completed,
    start_timer,
    build_conversation_history,
)
from services.soft_skill_analyzer import aggregate_nlp_metrics
from util.file_util import extract_text, validate_file
from db.mongo import reports_col, applications_col, jobs_col, users_col

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/interview", tags=["Interview"])

MIN_DURATION = 15
MAX_DURATION = 20


# ── Request / Response models ─────────────────────────────────────────────────

class AnswerRequest(BaseModel):
    session_id: str
    answer:     Optional[str]   = None   # None or empty → treated as skip
    skip:       bool            = False
    duration:   Optional[float] = None   # seconds candidate spoke


class AnswerResponse(BaseModel):
    interview_ended:   bool
    next_question:     Optional[str] = None
    question_index:    Optional[int] = None
    seconds_remaining: Optional[int] = None
    reason:            Optional[str] = None  # "time_up" | "completed"


class EndRequest(BaseModel):
    session_id: str


class SaveMediaUrlBody(BaseModel):
    session_id:     str
    media_type:     str
    cloudinary_url: str
    size_bytes:     int


# ── 1. Legacy generate session ────────────────────────────────────────────────

@router.post("/generate-question")
async def generate_questions(
    job_title:        str        = Form(...),
    job_description:  str        = Form(...),
    resume:           UploadFile = File(...),
    duration_minutes: int        = Form(30),
    language:         str        = Form("en"),
):
    if not (MIN_DURATION <= duration_minutes <= MAX_DURATION):
        raise HTTPException(
            status_code=400,
            detail=f"duration_minutes must be between {MIN_DURATION} and {MAX_DURATION}",
        )

    await validate_file(resume)
    resume_text = await extract_text(resume)

    ai_resp = await generate_questions_intro(
        job_title       = job_title,
        job_description = job_description,
        resume_text     = resume_text,
        language        = language,
    )

    session = await create_session(
        job_title        = job_title,
        job_description  = job_description,
        resume_text      = resume_text,
        candidate_name   = ai_resp.get("candidate_name", "Candidate"),
        intro_text       = ai_resp.get("introText", ""),
        first_question   = ai_resp.get("first_question", ""),
        duration_minutes = duration_minutes,
        language         = language,
    )

    return {"session_id": session.session_id}


# ── 2. Start metadata ─────────────────────────────────────────────────────────

@router.get("/start/{session_id}")
async def start_interview(session_id: str):
    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        raise HTTPException(status_code=404, detail="Session not found")

    # Start the server-side countdown (idempotent — safe to call multiple times)
    await start_timer(session)
    session = await get_session(session_id)

    return {
        "introText":        session.introText,
        "firstQuestion":    session.questions[0] if session.questions else "",
        "candidateName":    session.candidate_name,
        "durationMinutes":  session.duration_minutes,
        "language":         getattr(session, "language", "en"),
        "secondsRemaining": int(session.seconds_remaining()),
    }


# ── 3. Answer — core of the HTTP polling loop ─────────────────────────────────

@router.post("/answer", response_model=AnswerResponse)
async def submit_answer(body: AnswerRequest):
    """
    Receive a candidate answer (or skip), save it, generate and return the
    next question — all in a single request/response cycle.

    Replaces the entire WebSocket conversation loop.
    """
    session = await get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status == InteriewStatusEnum.COMPLETED:
        return AnswerResponse(interview_ended=True, reason="completed")

    # Start the server-side timer on the first answer if not already started.
    # The frontend never calls GET /start — it uses the session snapshot directly.
    # Without this, expires_at stays 0 and seconds_remaining() returns the full
    # duration on every answer, which causes the timer to appear to reset.
    if session.expires_at == 0:
        await start_timer(session)
        session = await get_session(body.session_id)

    # Server-side time guard
    if session.is_time_up():
        await mark_completed(session)
        return AnswerResponse(interview_ended=True, reason="time_up")

    # Load job metadata for question generation.
    # custom_questions live on job_doc and aren't stored on the session.
    app_doc  = await applications_col().find_one({"session_id": body.session_id})
    job_doc  = await jobs_col().find_one(
        {"job_id": (app_doc or {}).get("job_id")}
    ) if app_doc else None
    custom_q = (job_doc or {}).get("custom_questions", [])
    language = getattr(session, "language", "en")

    # Snapshot these before save_answer mutates session.current_index
    history       = build_conversation_history(session)
    next_q_number = session.current_index + 1
    secs_left     = session.seconds_remaining()

    # Run save_answer (MongoDB write) and generate_next_question (GPT) in
    # parallel — saves the full DB write latency (~100-200 ms) on every turn
    next_question, _ = await asyncio.gather(
        generate_next_question(
            job_title            = session.job_title,
            job_description      = session.job_description,
            resume_text          = session.resume_text,
            conversation_history = history,
            question_number      = next_q_number,
            seconds_remaining    = secs_left,
            language             = language,
            custom_questions     = custom_q,
        ),
        save_answer(
            answer_text      = None if body.skip else body.answer,
            skip             = body.skip,
            session          = session,
            duration_seconds = body.duration,
        ),
    )

    # Refresh session after save_answer has mutated and persisted it
    session = await get_session(body.session_id)

    # GPT takes several seconds — re-check time after the call
    if not next_question or session.is_time_up():
        await mark_completed(session)
        return AnswerResponse(interview_ended=True, reason="time_up")

    await append_question(session, next_question)

    return AnswerResponse(
        interview_ended   = False,
        next_question     = next_question,
        question_index    = next_q_number,
        seconds_remaining = int(session.seconds_remaining()),
    )


# ── 4. End interview ──────────────────────────────────────────────────────────

@router.post("/end")
async def end_interview_http(body: EndRequest):
    """Called when the candidate clicks 'End Interview' or the frontend timer expires."""
    session = await get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != InteriewStatusEnum.COMPLETED:
        await mark_completed(session)
    return {"interview_ended": True}


# ── 5. Whisper transcription ──────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("en")):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    transcript = await transcribe_audio(
        audio_bytes,
        filename=audio.filename or "audio.webm",
        language=language,
    )
    return {"transcript": transcript}


# ── 6. Legacy PUT /end fallback ───────────────────────────────────────────────

@router.put("/end/{session_id}")
async def end_interview_legacy(session_id: str):
    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        raise HTTPException(status_code=404, detail="Session not found")
    await mark_completed(session)
    return {"interviewEnded": True}


# ── 7. Report ─────────────────────────────────────────────────────────────────

@router.get("/report/{session_id}")
async def report(session_id: str):
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    answers_payload = [a.model_dump() for a in session.answers]
    language        = getattr(session, "language", "en")
    nlp_metrics     = aggregate_nlp_metrics(answers_payload)

    result = await generate_report(
        answers_payload,
        session.duration_minutes,
        language    = language,
        nlp_metrics = nlp_metrics,
    )

    app_doc = await applications_col().find_one({"session_id": session_id})

    report_id     = str(uuid.uuid4())
    stored_report = {
        "report_id":       report_id,
        "session_id":      session_id,
        "application_id":  app_doc["application_id"]  if app_doc else None,
        "job_id":          app_doc["job_id"]           if app_doc else getattr(session, "job_id", None),
        "candidate_name":  app_doc["candidate_name"]   if app_doc else session.candidate_name,
        "candidate_email": app_doc["candidate_email"]  if app_doc else "",
        "job_title":       session.job_title,
        "report":          result,
        "answers":         answers_payload,
        "language":        language,
        "created_at":      datetime.utcnow(),
    }

    await reports_col().replace_one(
        {"session_id": session_id},
        stored_report,
        upsert=True,
    )

    if app_doc:
        await applications_col().update_one(
            {"application_id": app_doc["application_id"]},
            {"$set": {"status": "completed"}},
        )

    if app_doc:
        job_doc = await jobs_col().find_one({"job_id": app_doc["job_id"]})
        if job_doc:
            recruiter_id = job_doc.get("recruiter_id", "")
            recruiter    = await users_col().find_one({"user_id": recruiter_id})

            if recruiter and recruiter.get("email"):
                from services.email_service import send_report_to_recruiter
                ok = await send_report_to_recruiter(
                    to_email       = recruiter["email"],
                    recruiter_name = recruiter.get("name") or recruiter.get("email", "Recruiter"),
                    candidate_name = app_doc["candidate_name"],
                    job_title      = session.job_title,
                    report         = result,
                    report_id      = report_id,
                    job_id         = app_doc["job_id"],
                )
                if ok:
                    log.info("Report email sent → %s | report_id=%s",
                             recruiter["email"], report_id)
                else:
                    log.warning("Report email failed → %s | report_id=%s",
                                recruiter["email"], report_id)
            else:
                log.warning("Recruiter %s has no email — report not emailed.", recruiter_id)

            from routers.recruiter import notify_new_report
            await notify_new_report(
                job_id         = app_doc["job_id"],
                candidate_name = app_doc["candidate_name"],
                job_title      = session.job_title,
                recruiter_id   = recruiter_id,
            )

    log.info("Report saved: report_id=%s session=%s", report_id, session_id)

    return {
        "report_id": report_id,
        "message":   "Your interview is complete. Thank you for your time! The results have been sent to the hiring team.",
    }


# ── 8. Save Cloudinary media URL ─────────────────────────────────────────────

@router.post("/save-media-url")
async def save_media_url(body: SaveMediaUrlBody):
    from db.mongo import media_col
    from datetime import timezone

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

    log.info("Media URL saved: %s session=%s type=%s",
             media_id, body.session_id, body.media_type)
    return {"media_id": media_id, "cloudinary_url": body.cloudinary_url}