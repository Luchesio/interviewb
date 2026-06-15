"""
Recruiter router — protected with JWT auth.

POST /recruiter/jobs                       – create a job posting
GET  /recruiter/jobs                       – list all jobs (for authenticated recruiter)
GET  /recruiter/jobs/{job_id}              – get a single job
DELETE /recruiter/jobs/{job_id}            – deactivate a job
GET  /recruiter/jobs/{job_id}/reports      – list all interview reports for a job
GET  /recruiter/reports/{report_id}        – get a single full report
GET  /recruiter/analytics/{job_id}         – aggregated analytics for a job
POST /recruiter/webhooks                   – configure Slack webhook URL
POST /recruiter/webhooks/test              – send test Slack notification
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, List

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, HttpUrl

from db.mongo import jobs_col, reports_col, users_col, applications_col, media_col, tokens_col, sessions_col
from models.job import Job
from dependencies.auth import get_current_recruiter, CurrentRecruiter
from services.email_service import send_weekly_digest_to_recruiter

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/recruiter", tags=["Recruiter"])


# ── Request bodies ─────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    job_title:        str
    job_description:  str
    custom_questions: List[str] = []   # Feature 4: recruiter-supplied questions


class WebhookRequest(BaseModel):
    slack_webhook_url: str


# ── Job endpoints ─────────────────────────────────────────────────────────────

@router.post("/jobs", status_code=201)
async def create_job(
    body:    CreateJobRequest,
    current: CurrentRecruiter,
):
    """Create a new job posting (auth required)."""
    job = Job(
        job_id           = str(uuid.uuid4()),
        recruiter_id     = current["user_id"],
        job_title        = body.job_title,
        job_description  = body.job_description,
        custom_questions = body.custom_questions,
    )
    await jobs_col().insert_one(job.model_dump())
    log.info("Job created: %s (%s) by %s", job.job_id, job.job_title, current["email"])
    return {"job_id": job.job_id, "job_title": job.job_title}


@router.get("/jobs")
async def list_jobs(current: CurrentRecruiter):
    """Return all of the recruiter's jobs (active and paused), active first."""
    cursor = jobs_col().find(
        {"recruiter_id": current["user_id"]},
        {"_id": 0},
        sort=[("is_active", -1), ("created_at", -1)],
    )
    jobs = await cursor.to_list(length=200)
    return {"jobs": jobs}


class JobStatus(BaseModel):
    is_active: bool


@router.patch("/jobs/{job_id}/status")
async def set_job_status(job_id: str, body: JobStatus, current: CurrentRecruiter):
    """Pause (stop accepting new applicants) or reopen a job posting. Existing
    candidates and reports are untouched."""
    result = await jobs_col().update_one(
        {"job_id": job_id, "recruiter_id": current["user_id"]},
        {"$set": {"is_active": body.is_active}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"is_active": body.is_active}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, current: CurrentRecruiter):
    doc = await jobs_col().find_one({"job_id": job_id, "recruiter_id": current["user_id"]}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return doc


async def _purge_job(job_id: str) -> None:
    """Permanently remove a job posting and everything tied to it."""
    session_ids: list[str] = []
    async for app in applications_col().find({"job_id": job_id}, {"_id": 0, "session_id": 1}):
        sid = app.get("session_id")
        if sid:
            session_ids.append(sid)

    if session_ids:
        await media_col().delete_many({"session_id": {"$in": session_ids}})
        await sessions_col().delete_many({"session_id": {"$in": session_ids}})

    await reports_col().delete_many({"job_id": job_id})
    await tokens_col().delete_many({"job_id": job_id})
    await applications_col().delete_many({"job_id": job_id})
    await jobs_col().delete_one({"job_id": job_id})


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, current: CurrentRecruiter):
    """Permanently delete a job posting and all of its candidates, reports and
    recordings. The recruiter must own the posting."""
    job = await jobs_col().find_one({"job_id": job_id}, {"_id": 0, "recruiter_id": 1})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("recruiter_id") != current["user_id"]:
        raise HTTPException(status_code=403, detail="This job posting isn't yours")
    await _purge_job(job_id)
    return {"deleted": True}


@router.delete("/account")
async def delete_account(current: CurrentRecruiter):
    """Permanently delete the recruiter's account and every job posting,
    candidate, report and recording associated with it."""
    recruiter_id = current["user_id"]
    job_ids = await jobs_col().distinct("job_id", {"recruiter_id": recruiter_id})
    for job_id in job_ids:
        await _purge_job(job_id)
    await users_col().delete_one({"user_id": recruiter_id})
    log.info("Recruiter account deleted: %s (%d jobs purged)", recruiter_id, len(job_ids))
    return {"deleted": True}


@router.get("/overview")
async def overview(current: CurrentRecruiter):
    """At-a-glance totals across all of the recruiter's job postings."""
    recruiter_id = current["user_id"]
    job_ids = await jobs_col().distinct("job_id", {"recruiter_id": recruiter_id})

    active_jobs = await jobs_col().count_documents(
        {"recruiter_id": recruiter_id, "is_active": True}
    )

    candidates = interviewed = recommended = 0
    if job_ids:
        candidates  = await applications_col().count_documents({"job_id": {"$in": job_ids}})
        interviewed = await reports_col().count_documents({"job_id": {"$in": job_ids}})
        recommended = await reports_col().count_documents({
            "job_id": {"$in": job_ids},
            "report.hiring_recommendation": {"$in": ["strong_yes", "yes"]},
        })

    return {
        "active_jobs":  active_jobs,
        "candidates":   candidates,
        "interviewed":  interviewed,
        "recommended":  recommended,
    }


# ── Profile / account settings ────────────────────────────────────────────────

@router.get("/profile")
async def get_profile(current: CurrentRecruiter):
    user = await users_col().find_one(
        {"user_id": current["user_id"]},
        {"_id": 0, "name": 1, "email": 1, "weekly_digest_enabled": 1},
    )
    if not user:
        raise HTTPException(status_code=404, detail="Account not found")
    return {
        "name":  user.get("name", ""),
        "email": user.get("email", ""),
        "weekly_digest_enabled": user.get("weekly_digest_enabled", True),
    }


class ProfileUpdate(BaseModel):
    name: str


@router.patch("/profile")
async def update_profile(body: ProfileUpdate, current: CurrentRecruiter):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name can't be empty")
    await users_col().update_one(
        {"user_id": current["user_id"]}, {"$set": {"name": name}}
    )
    return {"name": name}


class PreferencesUpdate(BaseModel):
    weekly_digest_enabled: bool


@router.patch("/preferences")
async def update_preferences(body: PreferencesUpdate, current: CurrentRecruiter):
    await users_col().update_one(
        {"user_id": current["user_id"]},
        {"$set": {"weekly_digest_enabled": body.weekly_digest_enabled}},
    )
    return {"weekly_digest_enabled": body.weekly_digest_enabled}


class PasswordChange(BaseModel):
    current_password: str
    new_password:     str


@router.post("/change-password")
async def change_password(body: PasswordChange, current: CurrentRecruiter):
    from routers.auth_router import hash_password, verify_password
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    user = await users_col().find_one({"user_id": current["user_id"]})
    if not user or not verify_password(body.current_password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    await users_col().update_one(
        {"user_id": current["user_id"]},
        {"$set": {"hashed_password": hash_password(body.new_password)}},
    )
    return {"changed": True}


# ── Report endpoints ──────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/reports")
async def list_reports_for_job(job_id: str, current: CurrentRecruiter):
    """Return summary list of all completed interview reports for a job."""
    job = await jobs_col().find_one({"job_id": job_id, "recruiter_id": current["user_id"]}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cursor = reports_col().find(
        {"job_id": job_id},
        {
            "_id": 0,
            "report_id":       1,
            "candidate_name":  1,
            "candidate_email": 1,
            "created_at":      1,
            "media_url":       1,
            "report.score":                           1,
            "report.hiring_recommendation":           1,
            "report.recommendation_reason":           1,
            "report.soft_skills.communication_score": 1,
            "report.soft_skills.confidence":          1,
        },
        sort=[("created_at", -1)],
    )
    reports = await cursor.to_list(length=500)
    return {
        "job_id":    job_id,
        "job_title": job.get("job_title", ""),
        "reports":   reports,
        "total":     len(reports),
    }


@router.get("/reports/{report_id}")
async def get_full_report(report_id: str, current: CurrentRecruiter):
    """Return a full interview report including Q&A and soft-skill breakdown."""
    doc = await reports_col().find_one({"report_id": report_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")

    # Recording URLs are stored separately in media_col as ordered segments
    # (progressive upload), keyed by session_id — attach them in order so the
    # recruiter can play them back-to-back.
    session_id = doc.get("session_id")
    if session_id:
        webcam_segs: list[str] = []
        screen_segs: list[str] = []
        cursor = media_col().find(
            {"session_id": session_id},
            {"_id": 0, "media_type": 1, "url": 1, "segment_index": 1},
        ).sort("segment_index", 1)
        async for m in cursor:
            if not m.get("url"):
                continue
            if m.get("media_type") == "screen":
                screen_segs.append(m["url"])
            else:
                webcam_segs.append(m["url"])
        if webcam_segs:
            doc["webcam_urls"] = webcam_segs
            doc["webcam_url"]  = webcam_segs[0]   # backward-compatible single URL
        if screen_segs:
            doc["screen_urls"] = screen_segs
            doc["screen_url"]  = screen_segs[0]

    return doc


# ── Analytics endpoint (Feature 5) ────────────────────────────────────────────

@router.get("/analytics/{job_id}")
async def get_analytics(job_id: str, current: CurrentRecruiter):
    """
    MongoDB aggregation pipeline to compute analytics for a job:
    - total candidates
    - avg technical score
    - hiring recommendation distribution
    - avg communication score
    - skill gap frequency
    """
    job = await jobs_col().find_one({"job_id": job_id, "recruiter_id": current["user_id"]})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pipeline = [
        {"$match": {"job_id": job_id}},
        {
            "$group": {
                "_id": None,
                "total_candidates": {"$sum": 1},
                "avg_score_raw": {
                    "$avg": {
                        "$convert": {
                            "input": {
                                "$rtrim": {
                                    "input": "$report.score",
                                    "chars": "%",
                                }
                            },
                            "to": "double",
                            "onError": 0,
                            "onNull":  0,
                        }
                    }
                },
                "avg_comm_score_raw": {
                    "$avg": {
                        "$convert": {
                            "input": {
                                "$rtrim": {
                                    "input": "$report.soft_skills.communication_score",
                                    "chars": "%",
                                }
                            },
                            "to": "double",
                            "onError": 0,
                            "onNull":  0,
                        }
                    }
                },
                "recommendation_distribution": {
                    "$push": "$report.hiring_recommendation"
                },
                "all_improvement_areas": {
                    "$push": "$report.improvment_area"  # note: typo matches existing schema
                },
                "confidence_distribution": {
                    "$push": "$report.soft_skills.confidence"
                },
                "sentiment_distribution": {
                    "$push": "$report.soft_skills.overall_sentiment"
                },
                # NLP soft skill metrics from enhanced analyzer
                "avg_leadership_score": {
                    "$avg": "$report.soft_skills.nlp_metrics.leadership_score"
                },
                "avg_teamwork_score": {
                    "$avg": "$report.soft_skills.nlp_metrics.teamwork_score"
                },
            }
        },
    ]

    results = await reports_col().aggregate(pipeline).to_list(length=1)

    if not results:
        return {
            "job_id": job_id,
            "job_title": job.get("job_title"),
            "total_candidates": 0,
            "avg_technical_score": 0,
            "avg_communication_score": 0,
            "recommendation_distribution": {},
            "top_skill_gaps": [],
            "confidence_distribution": {},
            "sentiment_distribution": {},
            "avg_leadership_score": 0,
            "avg_teamwork_score": 0,
        }

    row = results[0]

    # Tally distributions
    rec_dist: dict = {}
    for r in row.get("recommendation_distribution", []):
        if r:
            rec_dist[r] = rec_dist.get(r, 0) + 1

    conf_dist: dict = {}
    for c in row.get("confidence_distribution", []):
        if c:
            conf_dist[c] = conf_dist.get(c, 0) + 1

    sent_dist: dict = {}
    for s in row.get("sentiment_distribution", []):
        if s:
            sent_dist[s] = sent_dist.get(s, 0) + 1

    # Flatten and count skill gaps
    gap_freq: dict = {}
    for areas in row.get("all_improvement_areas", []):
        for area in (areas or []):
            gap_freq[area] = gap_freq.get(area, 0) + 1
    top_gaps = sorted(gap_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "job_id":    job_id,
        "job_title": job.get("job_title"),
        "total_candidates": row.get("total_candidates", 0),
        "avg_technical_score":     round(row.get("avg_score_raw", 0), 1),
        "avg_communication_score": round(row.get("avg_comm_score_raw", 0), 1),
        "recommendation_distribution": rec_dist,
        "top_skill_gaps": [{"area": k, "count": v} for k, v in top_gaps],
        "confidence_distribution": conf_dist,
        "sentiment_distribution":  sent_dist,
        "avg_leadership_score": round(row.get("avg_leadership_score") or 0, 1),
        "avg_teamwork_score":   round(row.get("avg_teamwork_score") or 0, 1),
    }


# ── Webhook endpoints (Feature 7) ─────────────────────────────────────────────

@router.post("/webhooks")
async def configure_webhook(body: WebhookRequest, current: CurrentRecruiter):
    """Save a Slack webhook URL for the authenticated recruiter."""
    await users_col().update_one(
        {"user_id": current["user_id"]},
        {"$set": {"slack_webhook_url": body.slack_webhook_url}},
    )
    return {"message": "Slack webhook URL saved."}


@router.post("/webhooks/test")
async def test_webhook(current: CurrentRecruiter):
    """Send a test notification to the configured Slack webhook."""
    user = await users_col().find_one({"user_id": current["user_id"]})
    webhook_url = (user or {}).get("slack_webhook_url")
    if not webhook_url:
        raise HTTPException(status_code=404, detail="No Slack webhook configured. POST /recruiter/webhooks first.")

    ok = await _send_slack_notification(
        webhook_url,
        text="✅ *AI Interview Platform*: Your Slack notifications are configured correctly!",
    )
    if not ok:
        raise HTTPException(status_code=502, detail="Failed to deliver test message to Slack.")
    return {"message": "Test notification sent."}


# ── Internal Slack helper ─────────────────────────────────────────────────────

async def _send_slack_notification(webhook_url: str, text: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(webhook_url, json={"text": text})
        return resp.status_code == 200
    except Exception as exc:
        log.error("Slack webhook error: %s", exc)
        return False


async def notify_new_report(job_id: str, candidate_name: str, job_title: str, recruiter_id: str) -> None:
    """
    Called from interview router after a report is saved.
    Looks up the recruiter's webhook URL and sends a Slack notification.
    """
    user = await users_col().find_one({"user_id": recruiter_id})
    webhook_url = (user or {}).get("slack_webhook_url")
    if not webhook_url:
        return
    text = (
        f"📋 *New interview report available*\n"
        f"• *Candidate:* {candidate_name}\n"
        f"• *Role:* {job_title}\n"
        f"• *Job ID:* `{job_id}`"
    )
    await _send_slack_notification(webhook_url, text)


# ── Weekly digest (cron target) ───────────────────────────────────────────────
#
#  Sends each recruiter a once-a-week summary of how many candidates took an
#  interview (and how many newly applied) per job posting over the last 7 days.
#  Replaces the old per-candidate report emails. Point a scheduled trigger
#  (e.g. Vercel Cron, weekly) at POST /recruiter/weekly-digest.
#
#  If the CRON_SECRET env var is set, the request must include a matching
#  `Authorization: Bearer <CRON_SECRET>` header (Vercel Cron can supply this).

CRON_SECRET = os.getenv("CRON_SECRET", "")
DIGEST_WINDOW_DAYS = 7


@router.api_route("/weekly-digest", methods=["GET", "POST"])
async def send_weekly_digest(authorization: Optional[str] = Header(default=None)):
    if CRON_SECRET:
        if authorization != f"Bearer {CRON_SECRET}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    cutoff = datetime.utcnow() - timedelta(days=DIGEST_WINDOW_DAYS)
    period_label = f"{cutoff.strftime('%b %d')} – {datetime.utcnow().strftime('%b %d, %Y')}"

    # Recruiters who actually own at least one job posting.
    recruiter_ids = await jobs_col().distinct("recruiter_id")

    sent, skipped = 0, 0
    for recruiter_id in recruiter_ids:
        recruiter = await users_col().find_one({"user_id": recruiter_id})
        if not recruiter or not recruiter.get("email"):
            skipped += 1
            continue
        if recruiter.get("weekly_digest_enabled", True) is False:
            skipped += 1
            continue

        jobs = await jobs_col().find(
            {"recruiter_id": recruiter_id, "is_active": True},
            {"_id": 0, "job_id": 1, "job_title": 1},
        ).to_list(length=500)

        rows, totals = [], {"interviewed": 0, "applied": 0}
        for job in jobs:
            jid = job["job_id"]
            interviewed = await reports_col().count_documents(
                {"job_id": jid, "created_at": {"$gte": cutoff}}
            )
            applied = await applications_col().count_documents(
                {"job_id": jid, "created_at": {"$gte": cutoff}}
            )
            rows.append({
                "job_title":   job.get("job_title", "—"),
                "interviewed": interviewed,
                "applied":     applied,
            })
            totals["interviewed"] += interviewed
            totals["applied"]     += applied

        if not rows:
            skipped += 1
            continue

        ok = await send_weekly_digest_to_recruiter(
            to_email       = recruiter["email"],
            recruiter_name = recruiter.get("name") or recruiter.get("email", "Recruiter"),
            period_label   = period_label,
            rows           = rows,
            totals         = totals,
        )
        if ok:
            sent += 1
        else:
            skipped += 1
            log.warning("Weekly digest failed for recruiter %s", recruiter_id)

    log.info("Weekly digest run complete: sent=%d skipped=%d", sent, skipped)
    return {"sent": sent, "skipped": skipped, "period": period_label}