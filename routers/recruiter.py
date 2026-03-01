"""
Recruiter router — private endpoints (add auth middleware in production).

POST /recruiter/jobs               – create a job posting
GET  /recruiter/jobs               – list all jobs
GET  /recruiter/jobs/{job_id}      – get a single job
GET  /recruiter/jobs/{job_id}/reports – list all interview reports for a job
GET  /recruiter/reports/{report_id}   – get a single full report
"""

import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.mongo import jobs_col, reports_col
from models.job import Job

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/recruiter", tags=["Recruiter"])


# ── Request body ──────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    job_title:       str
    job_description: str
    recruiter_id:    str = "default"


# ── Job endpoints ─────────────────────────────────────────────────────────────

@router.post("/jobs", status_code=201)
async def create_job(body: CreateJobRequest):
    """Create a new job posting and return the job_id."""
    job = Job(
        job_id          = str(uuid.uuid4()),
        recruiter_id    = body.recruiter_id,
        job_title       = body.job_title,
        job_description = body.job_description,
    )
    await jobs_col().insert_one(job.model_dump())
    log.info("Job created: %s (%s)", job.job_id, job.job_title)
    return {"job_id": job.job_id, "job_title": job.job_title}


@router.get("/jobs")
async def list_jobs(recruiter_id: str = "default"):
    """Return all jobs for a recruiter."""
    cursor = jobs_col().find(
        {"recruiter_id": recruiter_id, "is_active": True},
        {"_id": 0},
        sort=[("created_at", -1)],
    )
    jobs = await cursor.to_list(length=200)
    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    doc = await jobs_col().find_one({"job_id": job_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return doc


# ── Report endpoints ──────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/reports")
async def list_reports_for_job(job_id: str):
    """
    Return summary list of all completed interview reports for a job.
    Each item contains candidate info + top-level scores (not the full transcript).
    """
    # Verify the job exists
    job = await jobs_col().find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    cursor = reports_col().find(
        {"job_id": job_id},
        {
            "_id":            0,
            "report_id":      1,
            "candidate_name": 1,
            "candidate_email":1,
            "created_at":     1,
            # Top-level report scores only (not the full Q&A transcript)
            "report.score":                   1,
            "report.hiring_recommendation":   1,
            "report.recommendation_reason":   1,
            "report.soft_skills.communication_score": 1,
            "report.soft_skills.confidence":          1,
        },
        sort=[("created_at", -1)],
    )
    reports = await cursor.to_list(length=500)
    return {
        "job_id":  job_id,
        "job_title": job.get("job_title", ""),
        "reports": reports,
        "total":   len(reports),
    }


@router.get("/reports/{report_id}")
async def get_full_report(report_id: str):
    """Return a full interview report including Q&A and soft-skill breakdown."""
    doc = await reports_col().find_one({"report_id": report_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")
    return doc