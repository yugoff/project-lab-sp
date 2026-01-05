from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    mode: str
    language: str = "ru"

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = []
    confidence: Optional[float] = None
