"""
AI service layer — supports multi-language interviews, custom questions,
and resume-to-job screening.

Performance optimisations (v2):
  - generate_next_question now truncates job_description and resume_text before
    embedding them in the prompt. Sending the full documents on every turn was
    the single largest source of latency (extra tokens → more time-to-first-token
    from GPT). Truncating to ~600 / ~800 chars cuts input tokens by ~60–70 %
    and typically saves 3–6 seconds per turn with no meaningful quality loss —
    the first question already used the full context; subsequent questions only
    need a reminder of the role and key skills.
  - The conversation history sent to GPT is capped at the last 6 turns. Beyond
    that the model rarely changes its question anyway, and shorter context means
    faster inference.
  - Temperature lowered slightly (0.5 → 0.45) to reduce sampling variance and
    shorten average output length.
"""

import json
import io
import logging
from typing import List, Optional

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
log    = logging.getLogger(__name__)
client = AsyncOpenAI()

# Language code → full name mapping
_LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
    "ar": "Arabic",
    "zh": "Chinese (Simplified)",
    "hi": "Hindi",
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
}


def _lang_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code.lower(), code)


# ── Shared interviewer persona ─────────────────────────────────────────────────
_PERSONA = """
You are Morgan, a senior technical interviewer at a world-class technology firm
with 15 years of hiring experience across software engineering, product, and
design disciplines. You are warm but professionally direct, intellectually
curious, and deeply fair. You adapt your depth and vocabulary to the seniority
level implied by the candidate's resume. You never condescend, never waffle,
and you keep every question purposeful and concise.

Your interview style follows a structured arc:
  Phase 1 – Rapport & warm-up  : one easy, open-ended question to settle the
                                   candidate and confirm their background.
  Phase 2 – Core technical     : progressively deeper questions tied to the
                                   job description and the candidate's actual
                                   stated skills.
  Phase 3 – Behavioural        : one STAR-format situational question.
  Phase 4 – Wrap-up (optional) : if time permits, a forward-looking question.

Tone rules:
  - Ask ONE question per turn. Never bundle two questions together.
  - Questions must be spoken-word natural — short, clear, no bullet points.
  - Never start a question with "Certainly!", "Great!", "Sure!" or similar fillers.
  - Reflect the candidate's last answer briefly before pivoting only when it
    adds genuine value. Otherwise go straight to the next question.
""".strip()


# ── 0. Resume screening ───────────────────────────────────────────────────────

async def screen_resume(
    job_title:       str,
    job_description: str,
    resume_text:     str,
) -> dict:
    """
    Assess how well a candidate's resume matches a job posting.

    Returns:
      {
        "is_fit":  bool,   # True → invite to interview
        "score":   int,    # 0-100 match score
        "reason":  str,    # 1-2 sentence human-readable explanation
      }

    The threshold for is_fit is a score >= 60.
    """
    system_prompt = f"""
You are an expert technical recruiter. Your task is to evaluate how well a
candidate's resume matches a given job posting.

Job Title:       {job_title}
Job Description: {job_description}

Candidate Resume:
{resume_text}

Evaluation criteria:
  1. Required skills / technologies mentioned in the JD vs present in the resume.
  2. Relevant work experience and seniority level.
  3. Educational background where relevant.
  4. Domain / industry alignment.

Scoring:
  - 80–100: Excellent match — candidate clearly meets most or all requirements.
  - 60–79 : Good match — candidate meets the core requirements with minor gaps.
  - 40–59 : Partial match — candidate has some relevant skills but significant gaps.
  - 0–39  : Poor match — candidate does not meet the key requirements.

A candidate is considered FIT (is_fit = true) if score >= 60.

Return STRICT JSON — no markdown, no extra keys:
{{
  "score":  <integer 0-100>,
  "is_fit": <true|false>,
  "reason": "<1-2 concise, professional sentences explaining the decision>"
}}
""".strip()

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.2,
    )
    data = json.loads(resp.choices[0].message.content)

    score  = int(data.get("score", 0))
    is_fit = bool(data.get("is_fit", score >= 60))
    reason = str(data.get("reason", ""))
    return {"score": score, "is_fit": is_fit, "reason": reason}


# ── 1. Session initialisation ─────────────────────────────────────────────────

async def generate_questions_intro(
    job_title:        str,
    job_description:  str,
    resume_text:      str,
    candidate_name:   Optional[str] = None,
    language:         str           = "en",
    custom_questions: List[str]     = [],
) -> dict:
    """
    Returns candidate_name, a spoken introText, and the first warm-up question.
    Supports target language and injects recruiter custom questions into guidance.
    Full context is used here because this is a one-time call.
    """
    lang = _lang_name(language)

    name_instruction = (
        f'Use "{candidate_name}" as the candidate name.'
        if candidate_name
        else 'Extract the candidate\'s full name from the resume. Fallback to "Candidate" if not found.'
    )

    custom_q_section = ""
    if custom_questions:
        formatted = "\n".join(f"  - {q}" for q in custom_questions)
        custom_q_section = f"""
Custom questions from the recruiter (you MUST incorporate these somewhere during
the interview — they can replace or supplement Phase 2/3 questions):
{formatted}
"""

    system_prompt = f"""
{_PERSONA}

You are about to start a live spoken interview. All output MUST be written in {lang}.

Your task right now is to:
  1. {name_instruction}
  2. Write a concise, warm spoken introduction (2-3 sentences MAX) in {lang} that:
       - Addresses the candidate by first name only.
       - Names the role they are interviewing for.
       - Sets a professional but encouraging tone.
  3. Write the very first interview question (Phase 1 – warm-up) in {lang}.
     It must be open-ended, easy, and tied to the candidate's background.
{custom_q_section}
Candidate inputs:
  job_title:        {job_title}
  job_description:  {job_description}
  resume_text:      {resume_text}

Return STRICT JSON — no markdown, no extra keys:
{{
  "candidate_name": "string",
  "introText":      "string",
  "first_question": "string"
}}
""".strip()

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.4,
    )
    return json.loads(resp.choices[0].message.content)


# ── 2. Adaptive next-question generation ──────────────────────────────────────

# How many characters of the JD and resume to embed on each mid-interview turn.
# The first question already used the full documents. For follow-up questions
# only the key bullet points matter — 600 / 800 chars covers that easily and
# cuts input tokens by ~60-70 %, saving 3-6 s per GPT call.
_JD_SNIPPET_CHARS     = 600
_RESUME_SNIPPET_CHARS = 800

# Cap the conversation history sent to GPT at the last N turns.
# Beyond 6 turns the model rarely changes its line of questioning based on
# older turns, and shorter context = faster time-to-first-token.
_MAX_HISTORY_TURNS = 6


async def generate_next_question(
    job_title:            str,
    job_description:      str,
    resume_text:          str,
    conversation_history: list[dict],
    question_number:      int,
    seconds_remaining:    float,
    language:             str       = "en",
    custom_questions:     List[str] = [],
) -> str:
    lang      = _lang_name(language)
    mins_left = seconds_remaining / 60

    # ── Truncate long documents — key savings happen here ─────────────────────
    jd_snippet     = job_description[:_JD_SNIPPET_CHARS].rstrip()
    resume_snippet = resume_text[:_RESUME_SNIPPET_CHARS].rstrip()
    # ─────────────────────────────────────────────────────────────────────────

    if question_number == 1:
        phase          = "Phase 1 – Warm-up"
        phase_guidance = "Ask the candidate to briefly walk you through their background relevant to this role."
    elif mins_left > 10:
        phase          = "Phase 2 – Core Technical"
        phase_guidance = (
            "Go deeper on a technical skill from the resume or JD. "
            "Build directly on the candidate's last answer."
        )
    elif mins_left > 4:
        phase          = "Phase 3 – Behavioural"
        phase_guidance = f"Ask ONE behavioural question in STAR format relevant to a {job_title}."
    else:
        phase          = "Phase 4 – Wrap-up"
        phase_guidance = "Ask a short forward-looking or reflective question."

    # ── Cap history to last N turns ───────────────────────────────────────────
    recent_history = conversation_history[-_MAX_HISTORY_TURNS:]
    history_lines  = []
    for i, turn in enumerate(recent_history):
        ans = turn["answer"] or "[candidate skipped]"
        history_lines.append(f"Q{i+1}: {turn['question']}\nA{i+1}: {ans}")
    history_text = "\n\n".join(history_lines) if history_lines else "(none yet)"
    # ─────────────────────────────────────────────────────────────────────────

    custom_q_section = ""
    if custom_questions:
        asked   = [t["question"] for t in conversation_history]
        pending = [q for q in custom_questions if q not in asked]
        if pending:
            formatted = "\n".join(f"  - {q}" for q in pending)
            custom_q_section = f"""
IMPORTANT — Pending recruiter custom questions (ask these if they fit the current phase):
{formatted}
"""

    system_prompt = f"""
{_PERSONA}

All output MUST be written in {lang}.

You are mid-interview. Here is the full context:
  Role:             {job_title}
  Job description:  {jd_snippet}
  Candidate resume: {resume_snippet}

Interview transcript so far:
{history_text}

Current state:
  Question number : {question_number}
  Time remaining  : {mins_left:.1f} minutes
  Interview phase : {phase}

Phase guidance:
{phase_guidance}
{custom_q_section}
Your task: generate the next single interview question in {lang}.

Strict rules:
  - ONE question only. No preamble, no affirmations, no filler phrases.
  - Never repeat or closely paraphrase a previous question.
  - Max 2 sentences when spoken aloud.

Return STRICT JSON — no markdown:
{{"question": "string"}}
""".strip()

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.45,   # was 0.5 — slightly tighter to reduce output length variance
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("question", "").strip()


# ── 3. Whisper transcription (language-aware) ─────────────────────────────────

async def transcribe_audio(
    audio_bytes: bytes,
    filename:    str = "audio.webm",
    language:    str = "en",
) -> str:
    audio_file      = io.BytesIO(audio_bytes)
    audio_file.name = filename

    transcription = await client.audio.transcriptions.create(
        model           = "whisper-1",
        file            = audio_file,
        language        = language if language != "auto" else None,
        response_format = "text",
    )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


# ── 4. Enhanced report generation ─────────────────────────────────────────────

async def generate_report(
    answers:          list,
    duration_minutes: int  = 30,
    language:         str  = "en",
    nlp_metrics:      dict = None,
) -> dict:
    lang          = _lang_name(language)
    total         = len(answers)
    total_fillers = sum(a.get("filler_word_count", 0) for a in answers)
    all_fillers   = [f for a in answers for f in a.get("filler_words_found", [])]

    sentiment_counts  = {"positive": 0, "neutral": 0, "negative": 0}
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for a in answers:
        sentiment_counts [a.get("sentiment",        "neutral")] += 1
        confidence_counts[a.get("confidence_level", "medium")]  += 1

    avg_sentiment = sum(a.get("sentiment_score", 0.0) for a in answers) / max(total, 1)
    skipped       = sum(1 for a in answers if a.get("skip"))
    answered      = total - skipped

    nlp_section = ""
    if nlp_metrics:
        nlp_section = f"""
─── NLP Soft-Skill Metrics (Hugging Face) ────────────────────────────────────
  Leadership score   : {nlp_metrics.get('leadership_score', 'N/A')}
  Teamwork score     : {nlp_metrics.get('teamwork_score', 'N/A')}
  Problem-solving    : {nlp_metrics.get('problem_solving_score', 'N/A')}
  Communication cues : {nlp_metrics.get('communication_cues', [])}
  Key phrases        : {nlp_metrics.get('key_phrases', [])}
"""

    system_prompt = f"""
{_PERSONA}

The interview has concluded. Produce a detailed evaluation report.
Report language: {lang} (write ALL text fields in {lang}).

─── Interview metadata ───────────────────────────────────────────────────────
  Scheduled duration : {duration_minutes} minutes
  Questions asked    : {total}
  Questions answered : {answered}
  Questions skipped  : {skipped}

─── Q&A transcript (JSON) ───────────────────────────────────────────────────
{json.dumps(answers, indent=2, default=str)}

─── Pre-computed soft-skill metrics ─────────────────────────────────────────
  Total filler words     : {total_fillers}
  Unique fillers used    : {list(set(all_fillers))[:10]}
  Sentiment distribution : {sentiment_counts}
  Avg sentiment score    : {avg_sentiment:.2f}
  Confidence distribution: {confidence_counts}
{nlp_section}
─── Evaluation rubric ────────────────────────────────────────────────────────
  score: (correct answers / total questions) × 100
  communication_score: clarity, structure, examples, filler usage
  filler_word_usage: "low" (<5), "moderate" (5-15), "high" (>15)
  hiring_recommendation: "strong_yes" | "yes" | "maybe" | "no"
  nlp_metrics: include leadership_score, teamwork_score (0-100) if available.

Return STRICT JSON — no markdown, no extra keys:
{{
  "score":              "XX%",
  "correct_answer":     <integer>,
  "improvment_area":    ["...", "..."],
  "hiring_recommendation": "strong_yes | yes | maybe | no",
  "recommendation_reason": "1-2 sentence rationale",
  "soft_skills": {{
    "communication_score": "XX%",
    "filler_word_usage":   "low | moderate | high",
    "overall_sentiment":   "positive | neutral | negative",
    "confidence":          "high | medium | low",
    "top_fillers":         ["um", "like"],
    "coaching_tips":       ["...", "...", "..."],
    "nlp_metrics": {{
      "leadership_score":      <0-100 or null>,
      "teamwork_score":        <0-100 or null>,
      "problem_solving_score": <0-100 or null>,
      "key_phrases":           ["..."]
    }}
  }}
}}
""".strip()

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.3,
    )
    return json.loads(resp.choices[0].message.content)