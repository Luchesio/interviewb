"""
Interview router — HTTP-based interview engine (Vercel-compatible).

Architecture change (v4 — Vercel fix):
  WebSockets are NOT supported by Vercel serverless functions for Python/FastAPI.
  The previous WS-based flow was killed by Vercel's function execution timeout
  mid-interview, leaving the frontend permanently stuck on "Processing your answer".

  The Q&A loop is now driven by a simple REST endpoint:
    POST /interview/answer/{session_id}
  The frontend POSTs the transcript → the backend saves it, generates the next
  question via GPT, and returns it in the same HTTP response. No persistent
  connection required; every call is a normal short-lived serverless invocation.

  The WebSocket endpoint is kept below but is no longer used by the frontend.

Other notes:
  - Timer is maintained client-side; seconds_remaining is returned as a
    server-side sanity-check value only.
  - Report endpoint emails the full report to the recruiter who owns the job.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter, File, Form, HTTPException, UploadFile,
    WebSocket, WebSocketDisconnect,
)
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

    # Start the server-side timer on first load
    await start_timer(session)

    return {
        "introText":       session.introText,
        "firstQuestion":   session.questions[0] if session.questions else "",
        "candidateName":   session.candidate_name,
        "durationMinutes": session.duration_minutes,
        "language":        getattr(session, "language", "en"),
    }


# ── 3. HTTP answer submission — main Q&A endpoint (replaces WebSocket) ────────

class AnswerRequest(BaseModel):
    text:     Optional[str]   = None   # None when skip=True
    duration: Optional[float] = None   # seconds the candidate spoke
    skip:     bool            = False


@router.post("/answer/{session_id}")
async def submit_answer(session_id: str, body: AnswerRequest):
    """
    Receive one candidate answer → save it → generate & return the next question.

    Returns one of:
      { "type": "question",   "text": "...", "index": N, "seconds_remaining": N }
      { "type": "time_up" }
      { "type": "completed" }
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status == InteriewStatusEnum.COMPLETED:
        return {"type": "completed"}

    # Defensive: start the server-side timer if it hasn't been started yet.
    # start_timer is idempotent — it only writes when expires_at == 0.
    # This covers paths (e.g. verify-token flow) that bypass GET /start/.
    await start_timer(session)

    if session.is_time_up():
        await mark_completed(session)
        return {"type": "time_up"}

    # Save answer FIRST, then fetch the updated session so the AI sees the
    # latest answer in the conversation history.
    await save_answer(body.text, body.skip, session, body.duration)
    session = await get_session(session_id)

    if not session or session.is_time_up():
        if session:
            await mark_completed(session)
        return {"type": "time_up"}

    # Load custom questions from the job posting
    app_doc  = await applications_col().find_one({"session_id": session_id})
    job_doc  = await jobs_col().find_one(
        {"job_id": (app_doc or {}).get("job_id")}
    ) if app_doc else None
    custom_q = (job_doc or {}).get("custom_questions", [])
    language = getattr(session, "language", "en")

    history        = build_conversation_history(session)
    next_q_number  = session.current_index + 1
    secs_remaining = session.seconds_remaining()

    next_question = await generate_next_question(
        job_title            = session.job_title,
        job_description      = session.job_description,
        resume_text          = session.resume_text,
        conversation_history = history,
        question_number      = next_q_number,
        seconds_remaining    = secs_remaining,
        language             = language,
        custom_questions     = custom_q,
    )

    if not next_question or session.is_time_up():
        await mark_completed(session)
        return {"type": "time_up"}

    await append_question(session, next_question)

    return {
        "type":              "question",
        "text":              next_question,
        "index":             next_q_number,
        "seconds_remaining": int(session.seconds_remaining()),
    }


# ── 4. WebSocket — kept for reference, NOT used by the frontend ───────────────
#
#  Vercel kills Python serverless WebSocket connections at its function timeout,
#  causing the frontend to hang on "Processing your answer" indefinitely.
#  Use POST /answer/{session_id} instead (endpoint #3 above).

@router.websocket("/ws/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        await websocket.send_json({"type": "error", "detail": "Session not found"})
        await websocket.close()
        return

    app_doc  = await applications_col().find_one({"session_id": session_id})
    job_doc  = await jobs_col().find_one(
        {"job_id": (app_doc or {}).get("job_id")}
    ) if app_doc else None
    custom_q = (job_doc or {}).get("custom_questions", [])
    language = getattr(session, "language", "en")

    await start_timer(session)
    session = await get_session(session_id)

    first_q = session.questions[0] if session.questions else "Tell me about yourself."
    await websocket.send_json({
        "type":              "question",
        "text":              first_q,
        "index":             session.current_index + 1,
        "seconds_remaining": int(session.seconds_remaining()),
    })

    tick_task = asyncio.create_task(_tick_loop(websocket, session_id))

    try:
        while True:
            session = await get_session(session_id)
            if not session or session.status == InteriewStatusEnum.COMPLETED:
                await websocket.send_json({"type": "completed"})
                break

            if session.is_time_up():
                await mark_completed(session)
                await websocket.send_json({"type": "time_up"})
                break

            data     = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "end":
                await mark_completed(session)
                await websocket.send_json({"type": "completed"})
                break

            if msg_type not in ("answer", "skip"):
                continue

            is_skip     = (msg_type == "skip")
            answer_text = data.get("text") if not is_skip else None
            duration    = data.get("duration")

            await save_answer(answer_text, is_skip, session, duration)
            session = await get_session(session_id)

            if not session or session.is_time_up():
                if session:
                    await mark_completed(session)
                await websocket.send_json({"type": "time_up"})
                break

            history        = build_conversation_history(session)
            next_q_number  = session.current_index + 1
            secs_remaining = session.seconds_remaining()

            next_question = await generate_next_question(
                job_title            = session.job_title,
                job_description      = session.job_description,
                resume_text          = session.resume_text,
                conversation_history = history,
                question_number      = next_q_number,
                seconds_remaining    = secs_remaining,
                language             = language,
                custom_questions     = custom_q,
            )

            if not next_question or session.is_time_up():
                await mark_completed(session)
                await websocket.send_json({"type": "time_up"})
                break

            await append_question(session, next_question)
            await websocket.send_json({
                "type":              "question",
                "text":              next_question,
                "index":             next_q_number,
                "seconds_remaining": int(session.seconds_remaining()),
            })

    except WebSocketDisconnect:
        log.info("WS disconnected: session=%s", session_id)
    except Exception as exc:
        log.exception("WS error session=%s: %s", session_id, exc)
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        tick_task.cancel()


async def _tick_loop(websocket: WebSocket, session_id: str) -> None:
    try:
        while True:
            await asyncio.sleep(25)
            session = await get_session(session_id)
            if not session or session.status == InteriewStatusEnum.COMPLETED:
                break
            secs = int(session.seconds_remaining())
            await websocket.send_json({"type": "tick", "seconds_remaining": secs})
            if secs <= 0:
                break
    except (asyncio.CancelledError, Exception):
        pass


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


# ── 6. Force-end fallback ─────────────────────────────────────────────────────

@router.put("/end/{session_id}")
async def end_interview(session_id: str):
    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        raise HTTPException(status_code=404, detail="Session not found")
    await mark_completed(session)
    return {"interviewEnded": True}


# ── 7. Report — generate, persist, email recruiter ───────────────────────────

@router.get("/report/{session_id}")
async def report(session_id: str):
    """
    Generate the AI report with NLP, persist it, then:
      - Email the full report to the recruiter who owns the job.
      - Send a Slack notification (if configured).
      - Return only { report_id, message } to the frontend.
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    answers_payload = [a.model_dump() for a in session.answers]
    language        = getattr(session, "language", "en")

    nlp_metrics = aggregate_nlp_metrics(answers_payload)

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
                    log.info(
                        "Report email sent → recruiter %s | report_id=%s",
                        recruiter["email"], report_id,
                    )
                else:
                    log.warning(
                        "Report email failed → recruiter %s | report_id=%s",
                        recruiter["email"], report_id,
                    )
            else:
                log.warning(
                    "Recruiter %s has no email address on record — report not emailed.",
                    recruiter_id,
                )

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