"""
AI service layer — Gemini edition.

Supports multi-language interviews, custom questions, resume-to-job screening,
speech-to-text (STT) and text-to-speech (TTS) — all powered by Google Gemini
via the `google-genai` SDK.

Migration notes (OpenAI → Gemini):
  - Text + JSON tasks (resume screen, question generation, report) now use a
    single Gemini model (default ``gemini-3.5-flash``) through one helper,
    ``_generate_json``. OpenAI's ``response_format={"type": "json_object"}`` maps
    to Gemini's ``response_mime_type="application/json"`` (+ an optional
    ``response_schema`` for the flat payloads, which guarantees well-formed JSON).
  - STT: Whisper is replaced by Gemini audio understanding. Per Google's guidance
    we send audio **inline** when the request is below the 20 MB limit and fall
    back to the **Files API upload** for anything larger. Interview answers are a
    few seconds long, so the inline path is taken in practice; the upload path is
    a safety net.
  - TTS: previously the browser's Web Speech API spoke each question client-side.
    We now generate natural speech server-side with ``gemini-3.1-flash-tts-preview``
    and stream WAV audio to the browser. The frontend keeps the browser voice as
    an automatic fallback if this endpoint is unavailable.

Performance optimisations carried over from the OpenAI version:
  - generate_next_question truncates job_description / resume_text and caps the
    conversation history sent to the model, cutting input tokens and latency.

Environment variables:
  GEMINI_API_KEY      — required (the SDK also accepts GOOGLE_API_KEY).
  GEMINI_TEXT_MODEL   — optional, default "gemini-3.5-flash".
  GEMINI_STT_MODEL    — optional, default "gemini-3.5-flash".
  GEMINI_TTS_MODEL    — optional, default "gemini-3.1-flash-tts-preview".
  GEMINI_TTS_VOICE    — optional, default "Kore".
"""

import io
import os
import json
import wave
import base64
import logging
import tempfile
from typing import List, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# The SDK reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
client = genai.Client()

# ── Model configuration (override via env without touching code) ──────────────
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash")
STT_MODEL  = os.getenv("GEMINI_STT_MODEL",  "gemini-3.5-flash")
TTS_MODEL  = os.getenv("GEMINI_TTS_MODEL",  "gemini-3.1-flash-tts-preview")
TTS_VOICE  = os.getenv("GEMINI_TTS_VOICE",  "Kore")

# Google's documented threshold for inline audio vs. the Files API upload.
_INLINE_AUDIO_LIMIT = 20 * 1024 * 1024  # 20 MB

# TTS output is raw PCM: 16-bit, 24 kHz, mono.
_TTS_SAMPLE_RATE  = 24_000
_TTS_SAMPLE_WIDTH = 2
_TTS_CHANNELS     = 1

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

# Language code → BCP-47 tag for TTS. Gemini auto-detects language from the text,
# so this is a hint only; unknown codes simply fall through to auto-detection.
_LANGUAGE_BCP47 = {
    "en": "en-US",
    "fr": "fr-FR",
    "es": "es-ES",
    "de": "de-DE",
    "pt": "pt-BR",
    "ar": "ar-XA",
    "zh": "cmn-CN",
    "hi": "hi-IN",
}


def _lang_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code.lower(), code)


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _generate_json(
    prompt:      str,
    *,
    schema:      Optional[type] = None,
    temperature: float          = 0.4,
) -> dict:
    """
    Run a single-turn Gemini generation that returns strict JSON.

    `prompt` must instruct the model to emit JSON (every caller below does).
    When `schema` (a Pydantic model) is supplied the SDK constrains generation to
    that shape — used for the flat payloads. The report payload has nullable
    nested fields, so it relies on the prompt alone (no schema).
    """
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=temperature,
    )
    if schema is not None:
        config.response_schema = schema

    resp = await client.aio.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=config,
    )

    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response (possibly blocked).")
    return json.loads(text)


def _pcm_to_wav(
    pcm:          bytes,
    *,
    channels:     int = _TTS_CHANNELS,
    rate:         int = _TTS_SAMPLE_RATE,
    sample_width: int = _TTS_SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw little-endian PCM (what the TTS model returns) in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


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
    prompt = f"""
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

    data = await _generate_json(prompt, temperature=0.2)

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

    prompt = f"""
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

    return await _generate_json(prompt, temperature=0.4)


# ── 2. Adaptive next-question generation ──────────────────────────────────────

# How many characters of the JD and resume to embed on each mid-interview turn.
# The first question already used the full documents. For follow-up questions
# only the key bullet points matter — 600 / 800 chars covers that easily and
# cuts input tokens significantly, saving time per model call.
_JD_SNIPPET_CHARS     = 600
_RESUME_SNIPPET_CHARS = 800

# Cap the conversation history sent to the model at the last N turns.
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

    prompt = f"""
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

    data = await _generate_json(prompt, temperature=0.45)
    return data.get("question", "").strip()


# ── 3. Speech-to-text (Gemini audio understanding) ────────────────────────────

async def transcribe_audio(
    audio_bytes: bytes,
    filename:    str = "audio.webm",
    language:    str = "en",
) -> str:
    """
    Transcribe spoken audio with Gemini.

    Below the 20 MB inline limit the audio is passed as inline bytes (fast, no
    extra round-trip). Larger payloads are uploaded via the Files API first.
    Interview answers are short, so the inline path is the norm.
    """
    if not audio_bytes:
        return ""

    mime_type = _guess_audio_mime(filename)

    if language and language.lower() != "auto":
        lang_clause = f"The expected language is {_lang_name(language)}."
    else:
        lang_clause = "Detect the spoken language automatically."

    instruction = (
        "Transcribe the speech in this audio clip verbatim. "
        f"{lang_clause} "
        "Output ONLY the transcript text — no preamble, no speaker labels, no "
        "quotation marks, no commentary. If there is no intelligible speech, "
        "output nothing."
    )

    if len(audio_bytes) < _INLINE_AUDIO_LIMIT:
        contents = [
            instruction,
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ]
        resp = await client.aio.models.generate_content(
            model=STT_MODEL,
            contents=contents,
        )
    else:
        # ── Large file: upload via the Files API, then reference it ───────────
        suffix   = os.path.splitext(filename)[1] or ".webm"
        tmp_path = None
        uploaded = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            uploaded = await client.aio.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            resp = await client.aio.models.generate_content(
                model=STT_MODEL,
                contents=[instruction, uploaded],
            )
        finally:
            if uploaded is not None:
                try:
                    await client.aio.files.delete(name=uploaded.name)
                except Exception:  # cleanup is best-effort
                    pass
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    transcript = (resp.text or "").strip()
    # Defensively strip wrapping quotes the model occasionally adds.
    if len(transcript) >= 2 and transcript[0] in "\"'" and transcript[-1] == transcript[0]:
        transcript = transcript[1:-1].strip()
    return transcript


def _guess_audio_mime(filename: str) -> str:
    name = (filename or "").lower()
    if   name.endswith(".webm"): return "audio/webm"
    elif name.endswith(".ogg"):  return "audio/ogg"
    elif name.endswith(".mp4") or name.endswith(".m4a"): return "audio/mp4"
    elif name.endswith(".mp3"):  return "audio/mp3"
    elif name.endswith(".wav"):  return "audio/wav"
    elif name.endswith(".flac"): return "audio/flac"
    return "audio/webm"


# ── 4. Text-to-speech (Gemini TTS) ────────────────────────────────────────────

async def synthesize_speech(
    text:     str,
    language: str           = "en",
    voice:    Optional[str] = None,
    style:    Optional[str] = None,
) -> bytes:
    """
    Convert text to natural speech and return WAV-encoded bytes (24 kHz mono).

    `voice` defaults to GEMINI_TTS_VOICE. `style` is an optional spoken-delivery
    directive (e.g. "Say warmly and clearly:") — the model follows it rather than
    reading it aloud. We leave it off by default so nothing unintended is spoken.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("synthesize_speech: empty text")

    spoken = f"{style.strip()} {text}" if style else text

    voice_name    = voice or TTS_VOICE
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
        ),
    )
    bcp47 = _LANGUAGE_BCP47.get((language or "en").lower())
    if bcp47:
        speech_config.language_code = bcp47

    resp = await client.aio.models.generate_content(
        model=TTS_MODEL,
        contents=spoken,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        ),
    )

    part = resp.candidates[0].content.parts[0]
    pcm  = part.inline_data.data
    if isinstance(pcm, str):          # defensive: decode if base64-encoded
        pcm = base64.b64decode(pcm)
    if not pcm:
        raise RuntimeError("Gemini TTS returned no audio data.")

    return _pcm_to_wav(pcm)


# ── 5. Enhanced report generation ─────────────────────────────────────────────

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

    # Filler-word and sentiment metrics are English-only. For other languages
    # they were not computed (the numbers above are placeholders), so tell the
    # model to ignore them and assess communication qualitatively instead.
    metrics_supported = (language or "en").lower() == "en"
    metrics_note = "" if metrics_supported else (
        f"\nIMPORTANT — Automated filler-word and sentiment metrics are English-only "
        f"and were NOT computed for this {lang} interview. Ignore the zeroed filler/"
        f"sentiment/confidence numbers above; they are placeholders. Assess "
        f"communication and soft skills qualitatively from the transcript instead. "
        f"In the output set filler_word_usage to \"n/a\", top_fillers to [], and base "
        f"overall_sentiment and confidence on your qualitative read of the answers.\n"
    )

    prompt = f"""
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
{nlp_section}{metrics_note}
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

    # No response_schema here: the report has nullable nested fields, which the
    # Gemini structured-output schema validator rejects (no union/AnyOf support).
    # The prompt fully specifies the shape, exactly as the OpenAI version did.
    return await _generate_json(prompt, temperature=0.3)