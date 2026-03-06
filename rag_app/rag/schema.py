from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Passage(BaseModel):
    doc_id: str
    text: str
    source: str
    score: float


class RetrievedContext(BaseModel):
    passages: List[Passage] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    answer: str
    citations: List[Passage] = Field(default_factory=list)
    kg_triplets: List[Dict[str, str]] = Field(default_factory=list)
    meta: Dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_kg: bool = True
    use_history: bool = True
