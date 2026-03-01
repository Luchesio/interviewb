from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.mongo import init_db
from routers.interview import router as interview_router
from routers.recruiter import router as recruiter_router
from routers.candidate import router as candidate_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    version="2.0.0",
    title="AI Interview Platform – Adaptive Engine",
    description=(
        "Turn-based adaptive interviewer using OpenAI GPT-4.1-mini + Whisper. "
        "MongoDB session persistence. Recruiter/candidate split flow with email invites."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interview_router)
app.include_router(recruiter_router)
app.include_router(candidate_router)


@app.get("/health")
async def health():
    return {"status": "ok"}