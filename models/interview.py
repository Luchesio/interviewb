from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import time


class InteriewStatusEnum(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"


class Answer(BaseModel):
    question:               str
    answer:                 Optional[str] = None
    skip:                   bool          = False
    filler_word_count:      int           = 0
    filler_words_found:     list[str]     = Field(default_factory=list)
    sentiment:              str           = "neutral"
    sentiment_score:        float         = 0.0
    confidence_level:       str           = "medium"
    answer_duration_seconds: Optional[float] = None


class InterviewSession(BaseModel):
    session_id:       str
    job_title:        str = ""
    job_description:  str = ""
    resume_text:      str = ""
    candidate_name:   str = "Candidate"
    introText:        str = ""

    # Timer fields — duration_minutes set at creation, expires_at = Unix timestamp
    duration_minutes: int   = 30
    expires_at:       float = 0.0   # 0 means not yet started (set when WS opens)

    questions: list[str]  = Field(default_factory=list)
    answers:   list[Answer] = Field(default_factory=list)
    current_index: int    = 0

    status: InteriewStatusEnum = InteriewStatusEnum.IN_PROGRESS

    # Convenience helpers (not stored — computed on the fly)
    def seconds_remaining(self) -> float:
        if self.expires_at == 0:
            return self.duration_minutes * 60
        return max(0.0, self.expires_at - time.time())

    def is_time_up(self) -> bool:
        return self.expires_at > 0 and time.time() >= self.expires_at