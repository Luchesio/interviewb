"""
Cloudinary service — uploads interview recordings and returns permanent URLs.

ENV vars required:
  CLOUDINARY_CLOUD_NAME  – your cloud name from the Cloudinary dashboard
  CLOUDINARY_API_KEY     – your API key
  CLOUDINARY_API_SECRET  – your API secret

Upload preset required:
  Name:          interview_recordings
  Signing mode:  Unsigned
  Resource type: Video
  Folder:        interview_recordings

Videos are never stored in MongoDB — only the Cloudinary URL is persisted.
"""

import os
import logging
import hashlib
import hmac
import time
import httpx

log = logging.getLogger(__name__)

CLOUD_NAME  = os.getenv("CLOUDINARY_CLOUD_NAME", "")
API_KEY     = os.getenv("CLOUDINARY_API_KEY", "")
API_SECRET  = os.getenv("CLOUDINARY_API_SECRET", "")

UPLOAD_PRESET = "interview_recordings"
UPLOAD_URL    = f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/video/upload"


async def upload_recording(
    file_bytes:   bytes,
    filename:     str,
    session_id:   str,
    media_type:   str = "webcam",   # "webcam" | "screen"
) -> str | None:
    """
    Upload a video/webm blob to Cloudinary.

    Returns the secure Cloudinary URL on success, or None on failure.
    The public_id is set to session_id + media_type so each recording
    is uniquely addressable and overwritable (e.g. if retried).
    """
    if not CLOUD_NAME:
        log.error("CLOUDINARY_CLOUD_NAME is not set — cannot upload recording")
        return None

    public_id = f"interview_recordings/{session_id}_{media_type}"

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                UPLOAD_URL,
                data={
                    "upload_preset": UPLOAD_PRESET,
                    "public_id":     public_id,
                    "resource_type": "video",
                },
                files={
                    "file": (filename, file_bytes, "video/webm"),
                },
            )

        if resp.status_code == 200:
            data = resp.json()
            url  = data.get("secure_url", "")
            log.info(
                "Cloudinary upload OK: session=%s type=%s url=%s",
                session_id, media_type, url,
            )
            return url

        log.error(
            "Cloudinary upload failed: HTTP %s — %s",
            resp.status_code, resp.text[:300],
        )
        return None

    except Exception as exc:
        log.error("Cloudinary upload exception: %s", exc)
        return None


async def delete_recording(public_id: str) -> bool:
    """
    Delete a recording from Cloudinary by its public_id.
    Uses signed deletion (requires API key + secret).
    """
    if not all([CLOUD_NAME, API_KEY, API_SECRET]):
        log.warning("Cloudinary credentials incomplete — cannot delete recording")
        return False

    timestamp = str(int(time.time()))
    to_sign   = f"public_id={public_id}&timestamp={timestamp}{API_SECRET}"
    signature = hashlib.sha256(to_sign.encode()).hexdigest()

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/video/destroy",
                data={
                    "public_id":  public_id,
                    "api_key":    API_KEY,
                    "timestamp":  timestamp,
                    "signature":  signature,
                },
            )
        if resp.status_code == 200:
            log.info("Cloudinary delete OK: public_id=%s", public_id)
            return True
        log.error("Cloudinary delete failed: HTTP %s — %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        log.error("Cloudinary delete exception: %s", exc)
        return False