import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from db.mongo import init_db
from routers.interview import router as interview_router
from routers.recruiter import router as recruiter_router
from routers.candidate import router as candidate_router
from routers.auth_router import router as auth_router



# from services.email_service import _build_fit_html
# html = _build_fit_html("John Doe", "AI Backend Engineer", True, "http://localhost/test")
# print(f"{len(html.encode('utf-8')) / 1024:.1f} KB")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up...")
    await init_db()
    log.info("Ready.")
    yield


app = FastAPI(version="3.0.0", title="AI Interview Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler — shows real error in response ──────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error("Unhandled exception on %s\n%s", request.url, tb)
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "traceback": tb,   # remove this line in production
        },
    )


app.include_router(auth_router)
app.include_router(interview_router)
app.include_router(recruiter_router)
app.include_router(candidate_router)


@app.get("/health")
async def health():
    return {"status": "ok"}