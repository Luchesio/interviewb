"""
AI service layer — professional prompt rewrite + timer-aware question generation.
Updated: generate_questions_intro now accepts an optional candidate_name param
         so the real name from the apply form is passed to the AI.
"""

import json
import io
import logging

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
client = AsyncOpenAI()


# ── Shared interviewer persona (injected into every prompt) ───────────────────
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
                                   stated skills. Never ask about skills not on
                                   the resume or in the JD.
  Phase 3 – Behavioural        : one STAR-format situational question
                                   (Situation / Task / Action / Result) drawn
                                   from a real scenario the candidate would
                                   face in this role.
  Phase 4 – Wrap-up (optional) : if time permits, a forward-looking question
                                   ("Where do you see this skill taking you?").

Tone rules:
  - Ask ONE question per turn. Never bundle two questions together.
  - Questions must be spoken-word natural — short, clear, no bullet points.
  - Never start a question with "Certainly!", "Great!", "Sure!" or similar
    filler affirmations. Get straight to the question.
  - Reflect the candidate's last answer briefly before pivoting only when it
    adds genuine value (e.g. "You mentioned X — let's dig into that.").
    Otherwise go straight to the next question.
""".strip()


# ── 1. Session initialisation ─────────────────────────────────────────────────
async def generate_questions_intro(
    job_title:       str,
    job_description: str,
    resume_text:     str,
    candidate_name:  str | None = None,   # ← NEW: real name from apply form
) -> dict:
    """
    Returns candidate_name, a spoken introText, and the first warm-up question.
    Only Q1 is generated here; all subsequent questions are adaptive (per turn).
    If candidate_name is supplied, the AI uses it directly and does not extract
    from the resume (avoids mismatches).
    """
    name_instruction = (
        f'Use "{candidate_name}" as the candidate name — do NOT try to extract it from the resume.'
        if candidate_name
        else 'Extract the candidate\'s full name from the resume. Fallback to "Candidate" if not found.'
    )

    system_prompt = f"""
{_PERSONA}

You are about to start a live spoken interview. Your task right now is to:
  1. {name_instruction}
  2. Write a concise, warm spoken introduction (2-3 sentences MAX) that will
     be read aloud by a text-to-speech engine. It must:
       - Address the candidate by first name only.
       - Name the role they are interviewing for.
       - Set a professional but encouraging tone.
       - NOT include any interviewer self-introduction — the candidate already
         knows they are talking to an AI interviewer.
  3. Write the very first interview question (Phase 1 – warm-up). It must be:
       - Open-ended and easy.
       - Directly tied to the candidate's background (e.g. "Tell me about your
         most recent project with [technology from resume].").
       - No longer than 2 sentences when spoken aloud.

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
async def generate_next_question(
    job_title: str,
    job_description: str,
    resume_text: str,
    conversation_history: list[dict],
    question_number: int,
    seconds_remaining: float,
) -> str:
    mins_left = seconds_remaining / 60

    if question_number == 1:
        phase = "Phase 1 – Warm-up"
        phase_guidance = (
            "Ask the candidate to briefly walk you through their background "
            "relevant to this role. Keep it open-ended."
        )
    elif mins_left > 10:
        phase = "Phase 2 – Core Technical"
        phase_guidance = (
            "Go deeper on a technical skill from the resume or JD. "
            "Build directly on the candidate's last answer — probe a gap if "
            "the answer was vague, or escalate difficulty if it was strong."
        )
    elif mins_left > 4:
        phase = "Phase 3 – Behavioural"
        phase_guidance = (
            "Ask ONE behavioural question in STAR format relevant to this role. "
            "Example: 'Tell me about a time when you had to [scenario].' "
            f"Choose a scenario that a {job_title} would genuinely face."
        )
    else:
        phase = "Phase 4 – Wrap-up"
        phase_guidance = (
            "Ask a short forward-looking or reflective question. "
            f"Example: 'What's one area of {job_title} work you're actively "
            "trying to level up in?' Keep it light — we're near the end."
        )

    history_lines = []
    for i, turn in enumerate(conversation_history):
        ans = turn["answer"] or "[candidate skipped]"
        history_lines.append(f"Q{i+1}: {turn['question']}\nA{i+1}: {ans}")
    history_text = "\n\n".join(history_lines) if history_lines else "(none yet)"

    system_prompt = f"""
{_PERSONA}

You are mid-interview. Here is the full context:

  Role:             {job_title}
  Job description:  {job_description}
  Candidate resume: {resume_text}

Interview transcript so far:
{history_text}

Current state:
  Question number : {question_number}
  Time remaining  : {mins_left:.1f} minutes
  Interview phase : {phase}

Phase guidance:
{phase_guidance}

Your task: generate the next single interview question.

Strict rules:
  - ONE question only. No preamble, no affirmations, no filler phrases.
  - It must follow naturally from the candidate's last answer.
    • If the last answer was strong → escalate depth or move to the next phase.
    • If the last answer was vague or incomplete → ask a precise follow-up.
    • If the candidate skipped → pivot to a distinct but related topic from
      the JD or resume without drawing attention to the skip.
  - Never repeat or closely paraphrase a previous question.
  - Max 2 sentences when spoken aloud.
  - Do NOT ask about technologies or experiences NOT mentioned in the resume
    or job description.

Return STRICT JSON — no markdown:
{{"question": "string"}}
""".strip()

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.5,
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("question", "").strip()


# ── 3. Whisper transcription ──────────────────────────────────────────────────
async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    audio_file      = io.BytesIO(audio_bytes)
    audio_file.name = filename

    transcription = await client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en",
        response_format="text",
    )
    return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()


# ── 4. Enhanced report ────────────────────────────────────────────────────────
async def generate_report(answers: list, duration_minutes: int = 30) -> dict:
    total         = len(answers)
    total_fillers = sum(a.get("filler_word_count", 0) for a in answers)
    all_fillers   = [f for a in answers for f in a.get("filler_words_found", [])]

    sentiment_counts  = {"positive": 0, "neutral": 0, "negative": 0}
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for a in answers:
        sentiment_counts [a.get("sentiment",       "neutral")] += 1
        confidence_counts[a.get("confidence_level","medium")]  += 1

    avg_sentiment = sum(a.get("sentiment_score", 0.0) for a in answers) / max(total, 1)
    skipped       = sum(1 for a in answers if a.get("skip"))
    answered      = total - skipped

    system_prompt = f"""
{_PERSONA}

The interview has concluded. Your task is to produce a fair, detailed, and
actionable evaluation report for the candidate and their hiring manager.

─── Interview metadata ───────────────────────────────────────────────────────
  Scheduled duration : {duration_minutes} minutes
  Questions asked    : {total}
  Questions answered : {answered}
  Questions skipped  : {skipped}

─── Q&A transcript (JSON) ───────────────────────────────────────────────────
{json.dumps(answers, indent=2, default=str)}

─── Pre-computed soft-skill metrics ─────────────────────────────────────────
  Total filler words        : {total_fillers}
  Unique fillers used       : {list(set(all_fillers))[:10]}
  Sentiment distribution    : {sentiment_counts}
  Avg sentiment score       : {avg_sentiment:.2f}  (−1 = very negative, +1 = very positive)
  Confidence distribution   : {confidence_counts}

─── Evaluation rubric ────────────────────────────────────────────────────────
Technical scoring:
  - An answer is "correct" if the candidate covered ≥ 70% of the expected key
    concepts for that question given the role and JD. Be fair but honest.
  - A skipped question always scores 0.
  - Score = (correct answers / total questions) × 100, expressed as "XX%".

Soft-skill scoring:
  communication_score: holistic score (0-100%) weighing clarity, structure,
    use of concrete examples, and absence of excessive filler words.
  filler_word_usage  : "low" (<5 total), "moderate" (5-15), "high" (>15).
  overall_sentiment  : dominant label from the distribution above.
  confidence         : dominant label from the distribution above.

Improvement areas: max 5 bullet points. Be specific — name the actual concept
  or skill gap, not generic advice like "study more".

Coaching tips (soft skills): max 3, each 1 sentence, immediately actionable.

Hiring recommendation:
  "strong_yes" | "yes" | "maybe" | "no"
  Base this on technical score + communication quality + engagement level.

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
    "coaching_tips":       ["...", "...", "..."]
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