from typing import Optional
from pydantic import BaseModel

class AnswerResponse(BaseModel):
    interviewEnded: bool
    nextQuestion: Optional[str] = None