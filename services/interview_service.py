"""
Interview service — async session CRUD with timer support.
"""

import uuid
import time
import logging
from typing import Optional

from models.interview import Answer, InterviewSession, InteriewStatusEnum
from store.session_store import store_session, fetch_session
from services.soft_skill_analyzer import analyse_answer

log = logging.getLogger(__name__)


async def create_session(
    job_title:        str   = "",
    job_description:  str   = "",
    resume_text:      str   = "",
    candidate_name:   str   = "Candidate",
    intro_text:       str   = "",
    first_question:   str   = "",
    duration_minutes: int   = 30,
) -> InterviewSession:
    session = InterviewSession(
        session_id       = str(uuid.uuid4()),
        job_title        = job_title,
        job_description  = job_description,
        resume_text      = resume_text,
        candidate_name   = candidate_name,
        introText        = intro_text,
        questions        = [first_question] if first_question else [],
        duration_minutes = duration_minutes,
        expires_at       = 0.0,   # set when the WS connection opens
    )
    await store_session(session)
    return session


async def get_session(session_id: str) -> Optional[InterviewSession]:
    return await fetch_session(session_id)


async def start_timer(session: InterviewSession) -> None:
    """Call this once when the WebSocket connection is accepted (interview begins)."""
    if session.expires_at == 0:
        session.expires_at = time.time() + session.duration_minutes * 60
        await store_session(session)


async def save_answer(
    answer_text:      Optional[str],
    skip:             bool,
    session:          InterviewSession,
    duration_seconds: Optional[float] = None,
) -> None:
    question = session.questions[session.current_index]
    metrics  = analyse_answer(None if skip else answer_text)

    session.answers.append(Answer(
        question                = question,
        answer                  = None if skip else answer_text,
        skip                    = skip,
        answer_duration_seconds = duration_seconds,
        **metrics,
    ))
    session.current_index += 1
    await store_session(session)


async def append_question(session: InterviewSession, question: str) -> None:
    session.questions.append(question)
    await store_session(session)


async def mark_completed(session: InterviewSession) -> None:
    session.status = InteriewStatusEnum.COMPLETED
    await store_session(session)


def build_conversation_history(session: InterviewSession) -> list[dict]:
    return [{"question": a.question, "answer": a.answer} for a in session.answers]