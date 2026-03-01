"""
Email service — sends interview invite emails.

Set one of the following ENV vars:
  RESEND_API_KEY   – use Resend (preferred)
  SENDGRID_API_KEY – use SendGrid (fallback)
  EMAIL_FROM       – sender address (default: noreply@yourdomain.com)
  APP_BASE_URL     – public URL of the frontend (default: http://localhost:4200)

If neither key is set the email is only logged (useful in local dev).
"""

import os
import logging
import httpx

log = logging.getLogger(__name__)

RESEND_API_KEY    = os.getenv("RESEND_API_KEY", "")
SENDGRID_API_KEY  = os.getenv("SENDGRID_API_KEY", "")
EMAIL_FROM        = os.getenv("EMAIL_FROM", "noreply@yourdomain.com")
APP_BASE_URL      = os.getenv("APP_BASE_URL", "http://localhost:4200")


def _build_html(candidate_name: str, job_title: str, verify_url: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Your Interview Invitation</title>
  <style>
    body {{ margin:0; padding:0; background:#0d0d0f; font-family:'Segoe UI',Arial,sans-serif; color:#fff; }}
    .wrapper {{ max-width:560px; margin:40px auto; background:rgba(255,255,255,0.04);
                border:1px solid rgba(255,255,255,0.08); border-radius:16px; overflow:hidden; }}
    .header  {{ background:linear-gradient(135deg,#4f9cf9 0%,#7b6ef6 100%); padding:36px 40px; }}
    .header h1 {{ margin:0; font-size:1.5rem; font-weight:700; color:#fff; }}
    .body    {{ padding:36px 40px; }}
    .body p  {{ color:rgba(255,255,255,0.75); font-size:0.95rem; line-height:1.7; margin:0 0 18px; }}
    .btn     {{ display:inline-block; padding:14px 32px; background:linear-gradient(135deg,#4f9cf9,#7b6ef6);
                color:#fff; font-weight:700; font-size:0.95rem; border-radius:10px;
                text-decoration:none; margin:8px 0 24px; }}
    .note    {{ font-size:0.82rem; color:rgba(255,255,255,0.4); }}
    .footer  {{ padding:20px 40px; border-top:1px solid rgba(255,255,255,0.06);
                font-size:0.78rem; color:rgba(255,255,255,0.3); text-align:center; }}
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <h1>🤖 AI Interview Invitation</h1>
    </div>
    <div class="body">
      <p>Hi <strong>{candidate_name}</strong>,</p>
      <p>
        You have been invited to complete an AI-powered interview for the
        <strong>{job_title}</strong> position. The interview is conducted by
        our AI interviewer and takes approximately 15–20 minutes.
      </p>
      <p>Click the button below to begin. The link is valid for <strong>72 hours</strong>.</p>
      <a class="btn" href="{verify_url}">Start My Interview →</a>
      <p class="note">
        If the button doesn't work, paste this link into your browser:<br/>
        <a href="{verify_url}" style="color:#7db8fc;">{verify_url}</a>
      </p>
      <p class="note">This link can only be used once and will expire after 72 hours.</p>
    </div>
    <div class="footer">
      You are receiving this email because you applied for {job_title}.<br/>
      If you did not apply, please disregard this email.
    </div>
  </div>
</body>
</html>"""


async def send_interview_invite(
    to_email:       str,
    candidate_name: str,
    job_title:      str,
    token:          str,
) -> bool:
    """
    Send an interview invitation email.
    Returns True on success, False on failure.
    """
    verify_url = f"{APP_BASE_URL}/interview/verify?token={token}"
    subject    = f"Your interview invitation: {job_title}"
    html_body  = _build_html(candidate_name, job_title, verify_url)

    if RESEND_API_KEY:
        return await _send_via_resend(to_email, subject, html_body)
    if SENDGRID_API_KEY:
        return await _send_via_sendgrid(to_email, subject, html_body)

    # Local dev fallback — just log
    log.warning(
        "[EMAIL DEV] Would send to %s\n  Subject : %s\n  Verify  : %s",
        to_email, subject, verify_url,
    )
    return True


async def _send_via_resend(to: str, subject: str, html: str) -> bool:
    payload = {
        "from":    EMAIL_FROM,
        "to":      [to],
        "subject": subject,
        "html":    html,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
        if resp.status_code in (200, 201):
            log.info("Resend: email sent to %s", to)
            return True
        log.error("Resend error %s: %s", resp.status_code, resp.text)
        return False
    except Exception as exc:
        log.exception("Resend exception: %s", exc)
        return False


async def _send_via_sendgrid(to: str, subject: str, html: str) -> bool:
    payload = {
        "personalizations": [{"to": [{"email": to}]}],
        "from":             {"email": EMAIL_FROM},
        "subject":          subject,
        "content":          [{"type": "text/html", "value": html}],
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
        if resp.status_code == 202:
            log.info("SendGrid: email sent to %s", to)
            return True
        log.error("SendGrid error %s: %s", resp.status_code, resp.text)
        return False
    except Exception as exc:
        log.exception("SendGrid exception: %s", exc)
        return False