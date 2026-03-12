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

import uuid
import logging
from datetime import datetime
from typing import Optional, List

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl

from db.mongo import jobs_col, reports_col, users_col
from models.job import Job
from dependencies.auth import get_current_recruiter, CurrentRecruiter

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
    """Return all active jobs for the authenticated recruiter."""
    cursor = jobs_col().find(
        {"recruiter_id": current["user_id"], "is_active": True},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    jobs = await cursor.to_list(length=200)
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, current: CurrentRecruiter):
    doc = await jobs_col().find_one({"job_id": job_id, "recruiter_id": current["user_id"]}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return doc


@router.delete("/jobs/{job_id}")
async def deactivate_job(job_id: str, current: CurrentRecruiter):
    result = await jobs_col().update_one(
        {"job_id": job_id, "recruiter_id": current["user_id"]},
        {"$set": {"is_active": False}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deactivated"}


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