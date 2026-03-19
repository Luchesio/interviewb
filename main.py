import logging
import os
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
# IS_PRODUCTION is set automatically by Railway via the NODE_ENV / RAILWAY_*
# environment variables. We check for any of the common indicators so you
# don't have to set anything manually.
_IS_PROD = any([
    os.getenv("RAILWAY_ENVIRONMENT"),       # Railway sets this automatically
    os.getenv("RAILWAY_PROJECT_ID"),        # also set by Railway
    os.getenv("IS_PRODUCTION", "").lower() in ("1", "true", "yes"),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up... (production=%s)", _IS_PROD)
    await init_db()
    log.info("Ready.")
    yield


app = FastAPI(
    version="3.0.0",
    title="AI Interview Platform",
    lifespan=lifespan,
    # Disable the auto-generated /docs and /redoc pages in production
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None if _IS_PROD else "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ──────────────────────────────────────────────────
# In production: log the full traceback server-side (visible in Railway's log
# viewer) but return only a generic message to the client — never expose
# internal stack traces or file paths to end users.
# In development: include the traceback in the response body so you can debug
# without switching to the server logs.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error("Unhandled exception on %s\n%s", request.url, tb)

    if _IS_PROD:
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    # Development only — full traceback in response
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": tb},
    )


app.include_router(auth_router)
app.include_router(interview_router)
app.include_router(recruiter_router)
app.include_router(candidate_router)


@app.get("/health")
async def health():
    return {"status": "ok"}