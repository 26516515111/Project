from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Passage(BaseModel):
    """检索片段数据结构。"""

    doc_id: str
    text: str
    source: str
    score: float


class RetrievedContext(BaseModel):
    """检索上下文，包含多个Passage。"""

    passages: List[Passage] = Field(default_factory=list)


class Answer(BaseModel):
    """问答结果，包含回答与引用信息。"""

    question: str
    answer: str
    citations: List[Passage] = Field(default_factory=list)
    retrieved_chunks: List[Passage] = Field(default_factory=list)
    kg_chunks: List[Passage] = Field(default_factory=list)
    kg_triplets: List[Dict[str, str]] = Field(default_factory=list)
    meta: Dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """查询请求参数。"""

    question: str
    top_k: Optional[int] = None
    use_kg: bool = True
    use_history: bool = True
    session_id: Optional[str] = None
    enable_decompose: Optional[bool] = None
