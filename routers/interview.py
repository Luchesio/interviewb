"""
Interview router — timer-driven interview session.
Updated to:
  - Accept session_id from token-verify flow (no more combined form here)
  - Persist generated reports to MongoDB reports collection
  - Keep all existing AI / WebSocket / Whisper logic unchanged

WebSocket message protocol
──────────────────────────
Client → Server:
  { "type": "answer", "text": "...", "duration": 12.3 }
  { "type": "skip" }
  { "type": "end" }

Server → Client:
  { "type": "question", "text": "...", "index": 1, "seconds_remaining": 1740 }
  { "type": "tick",     "seconds_remaining": 1739 }   ← every 30 s
  { "type": "time_up"  }                               ← timer expired
  { "type": "completed" }
  { "type": "error",   "detail": "..." }
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import (
    APIRouter, File, Form, HTTPException, UploadFile,
    WebSocket, WebSocketDisconnect,
)

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
from util.file_util import extract_text, validate_file
from db.mongo import reports_col, applications_col

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/interview", tags=["Interview"])

MIN_DURATION = 15
MAX_DURATION = 20


# ── 1. Legacy generate session (kept for backwards compat / direct testing) ───
@router.post("/generate-question")
async def generate_questions(
    job_title:        str        = Form(...),
    job_description:  str        = Form(...),
    resume:           UploadFile = File(...),
    duration_minutes: int        = Form(30),
):
    if not (MIN_DURATION <= duration_minutes <= MAX_DURATION):
        raise HTTPException(status_code=400,
            detail=f"duration_minutes must be between {MIN_DURATION} and {MAX_DURATION}")

    await validate_file(resume)
    resume_text = await extract_text(resume)

    ai_resp = await generate_questions_intro(
        job_title       = job_title,
        job_description = job_description,
        resume_text     = resume_text,
    )

    session = await create_session(
        job_title        = job_title,
        job_description  = job_description,
        resume_text      = resume_text,
        candidate_name   = ai_resp.get("candidate_name", "Candidate"),
        intro_text       = ai_resp.get("introText", ""),
        first_question   = ai_resp.get("first_question", ""),
        duration_minutes = duration_minutes,
    )

    return {"session_id": session.session_id}


# ── 2. Start metadata (REST — called before WS opens) ─────────────────────────
@router.get("/start/{session_id}")
async def start_interview(session_id: str):
    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "introText":        session.introText,
        "firstQuestion":    session.questions[0] if session.questions else "",
        "candidateName":    session.candidate_name,
        "durationMinutes":  session.duration_minutes,
    }


# ── 3. WebSocket — main conversation channel ───────────────────────────────────
@router.websocket("/ws/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        await websocket.send_json({"type": "error", "detail": "Session not found"})
        await websocket.close()
        return

    # Start the timer the moment the WS connection is accepted
    await start_timer(session)
    session = await get_session(session_id)   # re-fetch with expires_at set

    # Push Q1 immediately
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
            if session and session.is_time_up():
                await mark_completed(session)
                await websocket.send_json({"type": "time_up"})
                break

            data     = await websocket.receive_json()
            msg_type = data.get("type")

            session = await get_session(session_id)
            if not session or session.status == InteriewStatusEnum.COMPLETED:
                await websocket.send_json({"type": "completed"})
                break

            if msg_type == "end":
                await mark_completed(session)
                await websocket.send_json({"type": "completed"})
                break

            if msg_type in ("answer", "skip"):
                is_skip     = (msg_type == "skip")
                answer_text = data.get("text") if not is_skip else None
                duration    = data.get("duration")

                await save_answer(answer_text, is_skip, session, duration)
                session = await get_session(session_id)

                if session.is_time_up():
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
    """Send a time tick to the client every 30 seconds."""
    try:
        while True:
            await asyncio.sleep(30)
            session = await get_session(session_id)
            if not session or session.status == InteriewStatusEnum.COMPLETED:
                break
            secs = int(session.seconds_remaining())
            await websocket.send_json({"type": "tick", "seconds_remaining": secs})
            if secs <= 0:
                break
    except (asyncio.CancelledError, Exception):
        pass


# ── 4. Whisper transcription ──────────────────────────────────────────────────
@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    transcript = await transcribe_audio(audio_bytes, filename=audio.filename or "audio.webm")
    return {"transcript": transcript}


# ── 5. Force-end fallback ─────────────────────────────────────────────────────
@router.put("/end/{session_id}")
async def end_interview(session_id: str):
    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        raise HTTPException(status_code=404, detail="Session not found")
    await mark_completed(session)
    return {"interviewEnded": True}


# ── 6. Report — generate, persist, return ─────────────────────────────────────
@router.get("/report/{session_id}")
async def report(session_id: str):
    """
    Generate the AI report, persist it to MongoDB linked to job_id,
    and return it to the candidate's frontend.
    """
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    answers_payload = [a.model_dump() for a in session.answers]
    result          = await generate_report(answers_payload, session.duration_minutes)

    # ── Look up the application linked to this session ────────────────────────
    app_doc = await applications_col().find_one({"session_id": session_id})

    report_id   = str(uuid.uuid4())
    stored_report = {
        "report_id":       report_id,
        "session_id":      session_id,
        "application_id":  app_doc["application_id"]  if app_doc else None,
        "job_id":          app_doc["job_id"]           if app_doc else getattr(session, "job_id", None),
        "candidate_name":  app_doc["candidate_name"]   if app_doc else session.candidate_name,
        "candidate_email": app_doc["candidate_email"]  if app_doc else "",
        "job_title":       session.job_title,
        "report":          result,
        "answers":         answers_payload,     # full Q&A stored for recruiter view
        "created_at":      datetime.utcnow(),
    }

    # Upsert so calling the endpoint twice doesn't create duplicates
    await reports_col().replace_one(
        {"session_id": session_id},
        stored_report,
        upsert=True,
    )

    # If application exists, mark it as completed
    if app_doc:
        await applications_col().update_one(
            {"application_id": app_doc["application_id"]},
            {"$set": {"status": "completed"}},
        )

    log.info("Report saved: report_id=%s session=%s", report_id, session_id)
    return {"result": result, "report_id": report_id}