"""
Email service — sends all platform emails.

ENV vars:
  RESEND_API_KEY   – use Resend (preferred)
  SENDGRID_API_KEY – use SendGrid (fallback)
  EMAIL_FROM       – sender address (default: noreply@yourdomain.com)
  APP_BASE_URL     – public URL of the frontend (default: http://localhost:4200)

Delay strategy:
  When RESEND_API_KEY is set the 10-minute candidate email delay is handled
  entirely by Resend's scheduledAt field — no asyncio.sleep, no background
  task, no in-process timer. This is safe on serverless platforms (Vercel,
  Railway, etc.) where the process exits as soon as the HTTP response is sent.

  When falling back to SendGrid the email is sent immediately (SendGrid's
  free tier has no scheduled-send API). The delay is simply dropped — the
  candidate receives the email right away instead of after 10 minutes.

Gmail desktop clips emails over ~102 KB. Each builder uses only the CSS it
actually needs (no shared mega-block) and all style strings are minified to
keep every email well under that threshold.
"""

import os
import logging
import httpx
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

RESEND_API_KEY   = os.getenv("RESEND_API_KEY", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")

# Separate verified senders on the ingenium.name.ng domain:
#   • candidate-facing mail (invites, fit/no-fit) → invite@ingenium.name.ng
#   • recruiter-facing mail (weekly digest)       → info@ingenium.name.ng
# Both are overridable via env. EMAIL_FROM is kept for backward compatibility
# and defaults to the candidate sender.
EMAIL_FROM_CANDIDATE = os.getenv("EMAIL_FROM_CANDIDATE", "Ingenium Recruitment <invite@ingenium.name.ng>")
EMAIL_FROM_RECRUITER = os.getenv("EMAIL_FROM_RECRUITER", "Ingenium <info@ingenium.name.ng>")
EMAIL_FROM           = os.getenv("EMAIL_FROM", EMAIL_FROM_CANDIDATE)
APP_BASE_URL         = os.getenv("APP_BASE_URL", "http://localhost:4200")

# ---------------------------------------------------------------------------
# Minified CSS — Ingenium design system (navy #0F213A · gold #C5A059 · white)
# Two lean blocks: one for candidate emails, one for the recruiter report.
# Both kept well under the 102 KB Gmail desktop clip threshold.
# ---------------------------------------------------------------------------

# Candidate emails (invite + fit/no-fit)
_CSS_CANDIDATE = (
    "body{margin:0;padding:0;background:#F4F6F8;font-family:Arial,sans-serif;color:#2C3E50}"
    ".w{max-width:560px;margin:32px auto;background:#fff;border:1px solid #E2E8F0;border-radius:10px;overflow:hidden}"
    ".hd{background:#0F213A;padding:28px 36px;border-bottom:3px solid #C5A059}"
    ".hd h1{margin:0;font-size:1.1rem;font-weight:700;color:#fff;letter-spacing:.03em}"
    ".hd .tag{font-size:.7rem;color:#C5A059;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}"
    ".bd{padding:28px 36px;background:#fff}"
    ".bd p{color:#2C3E50;font-size:.93rem;line-height:1.75;margin:0 0 14px}"
    ".bd strong{color:#0F213A}"
    ".btn{display:inline-block;padding:12px 28px;background:#0F213A;color:#C5A059;"
    "font-weight:700;font-size:.9rem;border-radius:8px;text-decoration:none;"
    "margin:6px 0 18px;border:2px solid #C5A059}"
    ".note{font-size:.79rem;color:#6C7A89}"
    ".note a{color:#1A365D}"
    ".ft{padding:16px 36px;background:#F4F6F8;border-top:1px solid #E2E8F0;"
    "font-size:.74rem;color:#6C7A89;text-align:center}"
)

# Recruiter report email — includes tables, badges, lists
_CSS_RECRUITER = (
    "body{margin:0;padding:0;background:#F4F6F8;font-family:Arial,sans-serif;color:#2C3E50}"
    ".w{max-width:600px;margin:32px auto;background:#fff;border:1px solid #E2E8F0;border-radius:10px;overflow:hidden}"
    ".hd{background:#0F213A;padding:26px 36px;border-bottom:3px solid #C5A059}"
    ".hd h1{margin:0;font-size:1.05rem;font-weight:700;color:#fff;letter-spacing:.03em}"
    ".bd{padding:24px 36px;background:#fff}"
    ".bd p{color:#2C3E50;font-size:.9rem;line-height:1.7;margin:0 0 12px}"
    ".bd strong{color:#0F213A}"
    "table{width:100%;border-collapse:collapse;margin-bottom:16px}"
    "th{background:#0F213A;color:#C5A059;font-size:.72rem;text-transform:uppercase;"
    "letter-spacing:.05em;padding:8px 10px;text-align:left}"
    "td{padding:7px 10px;font-size:.85rem;color:#2C3E50;border-bottom:1px solid #E2E8F0}"
    ".badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.74rem;font-weight:700}"
    ".sy{background:#EDFAF3;color:#2E7D52}"
    ".y{background:#EFF6FF;color:#2563EB}"
    ".mb{background:#FFFBEB;color:#B7791F}"
    ".no{background:#FEF2F2;color:#C0392B}"
    ".lbl{margin:0 0 4px;color:#6C7A89;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em}"
    "ul{padding-left:16px;margin:0 0 14px}"
    "li{color:#2C3E50;font-size:.84rem;margin-bottom:5px;line-height:1.5}"
    ".btn{display:inline-block;padding:11px 26px;background:#0F213A;color:#C5A059;"
    "font-weight:700;font-size:.88rem;border-radius:8px;text-decoration:none;"
    "margin:4px 0 14px;border:2px solid #C5A059}"
    ".note{font-size:.77rem;color:#6C7A89}"
    ".note a{color:#1A365D}"
    ".ft{padding:14px 36px;background:#F4F6F8;border-top:1px solid #E2E8F0;"
    "font-size:.72rem;color:#6C7A89;text-align:center}"
    "code{background:#F4F6F8;color:#0F213A;padding:1px 5px;border-radius:4px;font-size:.75rem}"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _first(name: str) -> str:
    """Return just the first name for a friendlier greeting."""
    return name.split()[0] if name else name


def _badge_html(rec: str) -> str:
    cls   = {"strong_yes": "sy", "yes": "y", "maybe": "mb", "no": "no"}.get(rec, "")
    label = {"strong_yes": "Strong Yes ✅", "yes": "Yes ✓",
             "maybe": "Maybe ⚠️", "no": "No ✗"}.get(rec, rec)
    return f'<span class="badge {cls}">{label}</span>'


# ── 1. Interview invite (candidate) ──────────────────────────────────────────

def _build_invite_html(candidate_name: str, job_title: str, verify_url: str) -> str:
    fn = _first(candidate_name)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<style>{_CSS_CANDIDATE}</style></head><body>'
        f'<div class="w">'
        f'<div class="hd">'
        f'<div class="tag">Ingenium &mdash; Recruitment</div>'
        f'<h1>Interview Invitation &mdash; {job_title}</h1>'
        f'</div>'
        f'<div class="bd">'
        f'<p>Dear <strong>{fn}</strong>,</p>'
        f'<p>Thank you for your application for the <strong>{job_title}</strong> position. '
        f'We are pleased to invite you to the next stage of our recruitment process.</p>'
        f'<p>Please complete the short structured interview at your earliest convenience. '
        f'It takes approximately 15&ndash;20 minutes and can be completed from anywhere &mdash; '
        f'all you need is a quiet space and a working microphone.</p>'
        f'<p>Your interview link is valid for <strong>72 hours</strong>. '
        f'Please use it at a time when you can give it your full attention.</p>'
        f'<a class="btn" href="{verify_url}">Begin Your Interview &rarr;</a>'
        f'<p class="note">Button not working? Copy and paste this link into your browser:<br>'
        f'<a href="{verify_url}">{verify_url}</a></p>'
        f'<p class="note">This link is personal to you, single-use, and expires in 72 hours.</p>'
        f'<p>Should you have any questions, please reply to this email. '
        f'We wish you the very best of luck!</p>'
        f'<p>Kind regards,<br><strong>The Ingenium Recruitment Team</strong></p>'
        f'</div>'
        f'<div class="ft">You received this because you applied for the {job_title} position.'
        f' If this was not you, please disregard this email.</div>'
        f'</div></body></html>'
    )


async def send_interview_invite(
    to_email:       str,
    candidate_name: str,
    job_title:      str,
    token:          str,
) -> bool:
    verify_url = f"{APP_BASE_URL}/interview/verify?token={token}"
    subject    = f"Your Interview Invitation — {job_title}"
    return await _dispatch(to_email, subject,
                           _build_invite_html(candidate_name, job_title, verify_url),
                           dev_label=f"invite → {to_email}")


# ── 2. Candidate fit / no-fit email ──────────────────────────────────────────

def _build_fit_html(candidate_name: str, job_title: str,
                    is_fit: bool, verify_url: str | None) -> str:
    fn = _first(candidate_name)

    if is_fit:
        headline = f"Application Update &mdash; {job_title}"
        body = (
            f'<p>Dear <strong>{fn}</strong>,</p>'
            f'<p>Thank you for your application for the <strong>{job_title}</strong> position. '
            f'We have reviewed your application and are pleased to inform you that you have been '
            f'selected to proceed to the next stage of our recruitment process.</p>'
            f'<p>As part of this stage, we would like you to complete a short structured interview '
            f'at your earliest convenience. It takes approximately 15&ndash;20 minutes and can be '
            f'completed from anywhere &mdash; all you need is a quiet space and a working microphone.</p>'
            f'<p>Your interview link is valid for <strong>72 hours</strong>. '
            f'Please ensure you use it at a time when you are able to give it your full attention.</p>'
            f'<a class="btn" href="{verify_url}">Begin Your Interview &rarr;</a>'
            f'<p class="note">Button not working? Copy and paste this link into your browser:<br>'
            f'<a href="{verify_url}">{verify_url}</a></p>'
            f'<p class="note">This link is personal to you, single-use, and expires in 72 hours.</p>'
            f'<p>Should you have any questions or require any adjustments, '
            f'please do not hesitate to reply to this email.</p>'
            f'<p>We wish you the very best of luck!</p>'
            f'<p>Kind regards,<br><strong>The Ingenium Recruitment Team</strong></p>'
        )
        footer = f"You received this because your application for the {job_title} position was progressed."

    else:
        headline = f"Your Application &mdash; {job_title}"
        body = (
            f'<p>Dear <strong>{fn}</strong>,</p>'
            f'<p>Thank you for taking the time to apply for the <strong>{job_title}</strong> role '
            f'and for the interest you have shown in joining our team.</p>'
            f'<p>After carefully reviewing all applications received, we regret to inform you that '
            f'we will not be progressing with your application on this occasion. '
            f'Please be assured that this decision was not taken lightly, and we appreciate '
            f'the effort you put into your application.</p>'
            f'<p>We would encourage you to continue to follow our openings, as we regularly '
            f'post new roles that may be an excellent fit for your skills and experience.</p>'
            f'<p>We wish you every success with your job search and future career.</p>'
            f'<p>Kind regards,<br><strong>The Ingenium Recruitment Team</strong></p>'
        )
        footer = f"You received this because you applied for the {job_title} position."

    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<style>{_CSS_CANDIDATE}</style></head><body>'
        f'<div class="w">'
        f'<div class="hd">'
        f'<div class="tag">Ingenium &mdash; Recruitment</div>'
        f'<h1>{headline}</h1>'
        f'</div>'
        f'<div class="bd">{body}</div>'
        f'<div class="ft">{footer}</div>'
        f'</div></body></html>'
    )


async def send_candidate_fit_email(
    to_email:        str,
    candidate_name:  str,
    job_title:       str,
    is_fit:          bool,
    token:           str | None = None,
    delay_minutes:   int = 0,
) -> bool:
    """
    Send a fit/no-fit email to the candidate.

    - is_fit=True  → professional shortlist email with the interview link.
    - is_fit=False → warm, professional rejection email.

    delay_minutes: when > 0 and Resend is the active provider, the email is
    scheduled via Resend's scheduledAt field so the delay is handled entirely
    server-side. No asyncio.sleep or background task is needed in the caller.

    Reads as if written by a human recruiter. No AI analysis, screening scores,
    match percentages, gap analysis, or mention of automated screening appears
    anywhere — that information goes to the recruiter report only.
    """
    verify_url = (f"{APP_BASE_URL}/interview/verify?token={token}"
                  if (is_fit and token) else None)
    subject = (f"You Have Been Shortlisted — {job_title}"
               if is_fit else f"Your Application for {job_title}")

    # Compute the scheduled delivery time if a delay is requested
    scheduled_at: datetime | None = None
    if delay_minutes > 0:
        scheduled_at = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)

    return await _dispatch(
        to_email, subject,
        _build_fit_html(candidate_name, job_title, is_fit, verify_url),
        dev_label=f"fit_email(is_fit={is_fit}) → {to_email}",
        scheduled_at=scheduled_at,
    )


# ── 3. Recruiter report email ─────────────────────────────────────────────────

def _build_report_html(
    recruiter_name: str,
    candidate_name: str,
    job_title:      str,
    report:         dict,
    report_id:      str,
    job_id:         str,
) -> str:
    rec        = report.get("hiring_recommendation", "")
    score      = report.get("score", "N/A")
    reason     = report.get("recommendation_reason", "")
    soft       = report.get("soft_skills", {})
    comm_score = soft.get("communication_score", "N/A")
    confidence = soft.get("confidence", "N/A")
    sentiment  = soft.get("overall_sentiment", "N/A")
    filler_use = soft.get("filler_word_usage", "N/A")
    tips       = soft.get("coaching_tips", [])
    gaps       = report.get("improvment_area", [])
    nlp        = soft.get("nlp_metrics", {})

    report_url = f"{APP_BASE_URL}/recruiter/reports/{report_id}"
    tips_html  = "".join(f"<li>{t}</li>" for t in tips) if tips else "<li>N/A</li>"
    gaps_html  = "".join(f"<li>{g}</li>" for g in gaps) if gaps else "<li>None identified</li>"

    nlp_rows = ""
    if nlp:
        nlp_rows = (
            f'<tr><td>Leadership</td><td>{nlp.get("leadership_score", "N/A")}</td></tr>'
            f'<tr><td>Teamwork</td><td>{nlp.get("teamwork_score", "N/A")}</td></tr>'
            f'<tr><td>Problem-Solving</td><td>{nlp.get("problem_solving_score", "N/A")}</td></tr>'
        )

    fn = _first(recruiter_name)

    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<style>{_CSS_RECRUITER}</style></head><body>'
        f'<div class="w">'
        f'<div class="hd">'
        f'<div style="font-size:.68rem;color:#C5A059;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px">Ingenium &mdash; Interview Report</div>'
        f'<h1>{candidate_name} &mdash; {job_title}</h1>'
        f'</div>'
        f'<div class="bd">'
        f'<p>Dear <strong>{fn}</strong>,</p>'
        f'<p>The AI-assisted interview for <strong>{candidate_name}</strong> applying for the '
        f'<strong>{job_title}</strong> role has been completed. Please find the full evaluation summary below.</p>'

        f'<table><thead><tr><th colspan="2">Overall Assessment</th></tr></thead><tbody>'
        f'<tr><td>Recommendation</td><td>{_badge_html(rec)}</td></tr>'
        f'<tr><td>Technical Score</td><td><strong>{score}</strong></td></tr>'
        f'<tr><td>Rationale</td><td>{reason}</td></tr>'
        f'</tbody></table>'

        f'<table><thead><tr><th colspan="2">Soft Skills</th></tr></thead><tbody>'
        f'<tr><td>Communication Score</td><td>{comm_score}</td></tr>'
        f'<tr><td>Confidence Level</td><td>{confidence}</td></tr>'
        f'<tr><td>Overall Sentiment</td><td>{sentiment}</td></tr>'
        f'<tr><td>Filler Word Usage</td><td>{filler_use}</td></tr>'
        f'{nlp_rows}'
        f'</tbody></table>'

        f'<p class="lbl">Areas for Improvement</p><ul>{gaps_html}</ul>'
        f'<p class="lbl">Coaching Tips for Candidate</p><ul>{tips_html}</ul>'

        f'<a class="btn" href="{report_url}">View Full Report &rarr;</a>'
        f'<p class="note">Direct link: <a href="{report_url}">{report_url}</a></p>'
        f'<p>Kind regards,<br><strong>The Ingenium Platform</strong></p>'
        f'</div>'
        f'<div class="ft">'
        f'Ingenium AI Interview Platform &nbsp;&bull;&nbsp; '
        f'Job ID: <code>{job_id}</code> &nbsp;&bull;&nbsp; Report ID: <code>{report_id}</code>'
        f'</div>'
        f'</div></body></html>'
    )


async def send_report_to_recruiter(
    to_email:       str,
    recruiter_name: str,
    candidate_name: str,
    job_title:      str,
    report:         dict,
    report_id:      str,
    job_id:         str,
) -> bool:
    """Email the full interview report to the recruiter who owns the job."""
    subject = f"Interview Report: {candidate_name} — {job_title}"
    return await _dispatch(
        to_email, subject,
        _build_report_html(recruiter_name, candidate_name,
                           job_title, report, report_id, job_id),
        dev_label=f"recruiter_report → {to_email} | candidate={candidate_name}",
        from_email=EMAIL_FROM_RECRUITER,
    )


# ── 4. Recruiter weekly digest ───────────────────────────────────────────────

def _build_weekly_digest_html(recruiter_name: str, period_label: str,
                              rows: list[dict], totals: dict) -> str:
    fn = _first(recruiter_name)

    if rows:
        body_rows = "".join(
            f'<tr>'
            f'<td>{r.get("job_title", "—")}</td>'
            f'<td style="text-align:center"><strong>{r.get("interviewed", 0)}</strong></td>'
            f'<td style="text-align:center">{r.get("applied", 0)}</td>'
            f'</tr>'
            for r in rows
        )
    else:
        body_rows = '<tr><td colspan="3">No active job postings.</td></tr>'

    dash_url = f"{APP_BASE_URL}/recruiter/jobs"

    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<style>{_CSS_RECRUITER}</style></head><body>'
        f'<div class="w">'
        f'<div class="hd">'
        f'<div style="font-size:.68rem;color:#C5A059;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px">Ingenium &mdash; Weekly Summary</div>'
        f'<h1>Your hiring activity &mdash; {period_label}</h1>'
        f'</div>'
        f'<div class="bd">'
        f'<p>Dear <strong>{fn}</strong>,</p>'
        f'<p>Here is your hiring activity across all job postings for the past week.</p>'
        f'<table role="presentation" style="width:100%;border-collapse:separate;border-spacing:10px 0;margin:4px 0 18px">'
        f'<tr>'
        f'<td style="width:50%;background:#F4F6F8;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center">'
        f'<div style="font-size:1.9rem;font-weight:700;color:#0F213A;line-height:1">{totals.get("applied", 0)}</div>'
        f'<div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#6C7A89;margin-top:6px">New Applications</div>'
        f'</td>'
        f'<td style="width:50%;background:#F4F6F8;border:1px solid #E2E8F0;border-radius:8px;padding:16px;text-align:center">'
        f'<div style="font-size:1.9rem;font-weight:700;color:#0F213A;line-height:1">{totals.get("interviewed", 0)}</div>'
        f'<div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#6C7A89;margin-top:6px">Interviews Completed</div>'
        f'</td>'
        f'</tr></table>'
        f'<p class="lbl">Breakdown by job posting</p>'
        f'<table><thead><tr>'
        f'<th>Job Posting</th><th style="text-align:center">Interviewed</th>'
        f'<th style="text-align:center">New Applicants</th>'
        f'</tr></thead><tbody>{body_rows}</tbody></table>'
        f'<p>Open your dashboard to review each candidate&rsquo;s full report.</p>'
        f'<a class="btn" href="{dash_url}">View Dashboard &rarr;</a>'
        f'<p class="note">Direct link: <a href="{dash_url}">{dash_url}</a></p>'
        f'<p>Kind regards,<br><strong>The Ingenium Platform</strong></p>'
        f'</div>'
        f'<div class="ft">'
        f'Ingenium AI Interview Platform &nbsp;&bull;&nbsp; Weekly recruiter summary'
        f'</div>'
        f'</div></body></html>'
    )


async def send_weekly_digest_to_recruiter(
    to_email:       str,
    recruiter_name: str,
    period_label:   str,
    rows:           list[dict],
    totals:         dict,
) -> bool:
    """Email a recruiter the weekly count of candidates per job posting."""
    subject = f"Your Weekly Hiring Summary — {period_label}"
    return await _dispatch(
        to_email, subject,
        _build_weekly_digest_html(recruiter_name, period_label, rows, totals),
        dev_label=f"weekly_digest → {to_email} ({len(rows)} jobs)",
        from_email=EMAIL_FROM_RECRUITER,
    )


# ── Retry helper ──────────────────────────────────────────────────────────────

# Exponential backoff: attempt 1 → wait 30s, attempt 2 → wait 60s, attempt 3 → wait 120s
# NOTE: retries only apply to immediate sends. Scheduled Resend emails do not
# need retries — if the API call succeeds (201) Resend owns the delivery.
_RETRY_DELAYS = [30, 60, 120]


async def _with_retry(fn, label: str) -> bool:
    """
    Call an async send function up to 4 times (1 initial + 3 retries) with
    exponential backoff. Returns True as soon as one attempt succeeds.
    """
    import asyncio
    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            log.warning("[EMAIL] attempt %d for %s — retrying in %ds", attempt, label, delay)
            await asyncio.sleep(delay)
        try:
            success = await fn()
            if success:
                if attempt > 1:
                    log.info("[EMAIL] delivered on attempt %d for %s", attempt, label)
                return True
        except Exception as exc:
            log.warning("[EMAIL] attempt %d error for %s: %s", attempt, label, exc)

    log.error("[EMAIL] all %d attempts failed for %s", len(_RETRY_DELAYS) + 1, label)
    return False


# ── Internal send dispatcher ──────────────────────────────────────────────────

async def _dispatch(
    to:           str,
    subject:      str,
    html:         str,
    dev_label:    str = "",
    scheduled_at: "datetime | None" = None,
    from_email:   str = EMAIL_FROM_CANDIDATE,
) -> bool:
    if RESEND_API_KEY:
        # Scheduled sends go directly — no retry loop needed because if the
        # API call itself succeeds, Resend owns the delivery from that point.
        if scheduled_at:
            return await _send_via_resend(to, subject, html, scheduled_at=scheduled_at, from_email=from_email)
        return await _with_retry(
            lambda: _send_via_resend(to, subject, html, from_email=from_email),
            label=f"resend→{to}",
        )
    if SENDGRID_API_KEY:
        # SendGrid free tier has no scheduled-send API — send immediately.
        if scheduled_at:
            log.info("[EMAIL] SendGrid fallback: scheduled_at ignored, sending immediately to %s", to)
        return await _with_retry(
            lambda: _send_via_sendgrid(to, subject, html, from_email=from_email),
            label=f"sendgrid→{to}",
        )
    log.warning("[EMAIL DEV] %s | From: %s | Subject: %s", dev_label, from_email, subject)
    return True


def _parse_from(from_str: str) -> tuple[str, str | None]:
    """Split a 'Display Name <addr@x>' string into (email, name)."""
    if "<" in from_str and ">" in from_str:
        name = from_str.split("<", 1)[0].strip().strip('"')
        addr = from_str.split("<", 1)[1].split(">", 1)[0].strip()
        return addr, (name or None)
    return from_str.strip(), None


async def _send_via_resend(
    to:           str,
    subject:      str,
    html:         str,
    scheduled_at: "datetime | None" = None,
    from_email:   str = EMAIL_FROM_CANDIDATE,
) -> bool:
    payload: dict = {
        "from":    from_email,
        "to":      [to],
        "subject": subject,
        "html":    html,
    }
    if scheduled_at:
        # Resend expects ISO-8601 UTC, e.g. "2024-09-05T11:52:01.858Z"
        payload["scheduledAt"] = scheduled_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        log.info("Resend: scheduling email to %s at %s", to, payload["scheduledAt"])

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=15,
        )
    if resp.status_code in (200, 201):
        if scheduled_at:
            log.info("Resend: scheduled delivery accepted for %s", to)
        else:
            log.info("Resend: sent to %s", to)
        return True
    # 429 = rate limited, 5xx = server error — both worth retrying on immediate sends
    log.error("Resend HTTP %s for %s: %s", resp.status_code, to, resp.text[:200])
    return False


async def _send_via_sendgrid(to: str, subject: str, html: str,
                             from_email: str = EMAIL_FROM_CANDIDATE) -> bool:
    addr, name = _parse_from(from_email)
    from_obj   = {"email": addr}
    if name:
        from_obj["name"] = name
    payload = {
        "personalizations": [{"to": [{"email": to}]}],
        "from":    from_obj,
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=15,
        )
    if resp.status_code == 202:
        log.info("SendGrid: sent to %s", to)
        return True
    log.error("SendGrid HTTP %s for %s: %s", resp.status_code, to, resp.text[:200])
    return False