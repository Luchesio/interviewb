"""
Interview router — timer-driven WebSocket interview engine.

Performance optimisations (v2):
  1. app_doc, job_doc, custom_questions, and language are loaded ONCE at
     WebSocket open and reused on every turn — eliminates 2 MongoDB round-trips
     per answer (~300–500 ms saved each turn).
  2. save_answer is awaited first, then get_session — ensures the AI always
     receives the full, up-to-date conversation history including the latest answer.
  3. A single get_session is used at the top of the answer loop (was two).
  4. Vercel keep-warm: the WebSocket tick loop now also pings every 25 s so
     the function stays hot throughout the interview.

Bug fixes (v3):
  - FIX 1: ping (and any unknown) messages from the frontend are now ignored
    with `continue` instead of silently falling through and consuming the next
    receive_json() call — which was swallowing the real "answer" message and
    leaving the candidate stuck on "Processing…" forever.
  - FIX 2: save_answer is now awaited BEFORE get_session (was parallel via
    asyncio.gather), so the session fetched for build_conversation_history
    always contains the answer that was just saved.

Other changes:
  - Report endpoint still emails the full report to the recruiter who owns the job.
  - Slack notification awaited directly (not create_task) so it completes before
    the serverless function exits on Vercel.
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

    return {
        "introText":       session.introText,
        "firstQuestion":   session.questions[0] if session.questions else "",
        "candidateName":   session.candidate_name,
        "durationMinutes": session.duration_minutes,
        "language":        getattr(session, "language", "en"),
    }


# ── 3. WebSocket — main conversation channel ──────────────────────────────────

@router.websocket("/ws/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        await websocket.send_json({"type": "error", "detail": "Session not found"})
        await websocket.close()
        return

    # Load app/job metadata ONCE — never again per answer turn
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

            # ── FIX 1: ignore ping and any other unknown message types ─────────
            # The frontend sends { type: "ping" } every 20 s as a keep-warm.
            # Without this guard the loop would fall through, re-fetch the
            # session, then block on the next receive_json() — consuming the
            # real "answer" message and leaving the candidate stuck on
            # "Processing…" indefinitely.
            if msg_type not in ("answer", "skip"):
                continue
            # ─────────────────────────────────────────────────────────────────

            is_skip     = (msg_type == "skip")
            answer_text = data.get("text") if not is_skip else None
            duration    = data.get("duration")

            # ── FIX 2: save answer FIRST, then fetch the updated session ──────
            # Previously both ran in parallel via asyncio.gather, which meant
            # get_session() could return a snapshot that did NOT yet include
            # the answer just saved — giving the AI stale conversation history.
            await save_answer(answer_text, is_skip, session, duration)
            session = await get_session(session_id)
            # ─────────────────────────────────────────────────────────────────

            if not session or session.is_time_up():
                if session:
                    await mark_completed(session)
                await websocket.send_json({"type": "time_up"})
                break

            history        = build_conversation_history(session)
            next_q_number  = session.current_index + 1
            secs_remaining = session.seconds_remaining()

            # custom_q and language already loaded above — no DB hit here
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
    """
    Sends a timer tick every 25 s.

    Vercel serverless functions are killed after ~30 s of inactivity on the
    WebSocket. Ticking every 25 s keeps the connection alive AND keeps the
    function warm so there is no cold-start penalty between answers.
    (Previously ticked every 30 s — now 25 s to stay safely under the limit.)
    """
    try:
        while True:
            await asyncio.sleep(25)          # <── was 30, now 25 for Vercel
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


# ── 5. Force-end fallback ─────────────────────────────────────────────────────

@router.put("/end/{session_id}")
async def end_interview(session_id: str):
    session = await get_session(session_id)
    if not session or session.status == InteriewStatusEnum.COMPLETED:
        raise HTTPException(status_code=404, detail="Session not found")
    await mark_completed(session)
    return {"interviewEnded": True}


# ── 6. Report — generate, persist, email recruiter ───────────────────────────

@router.get("/report/{session_id}")
async def report(session_id: str):
    """
    Generate the AI report with NLP, persist it, then:
      - Email the full report to the recruiter who owns the job.
      - Send a Slack notification (if configured).
      - Return only { report_id, message } to the frontend — the candidate
        does NOT see the report; it goes to the recruiter's inbox.
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