# models.py
from typing import List, Dict, Optional
from pydantic import BaseModel
import time

# --- 数据接口定义 ---
class Passage(BaseModel):
    text: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    use_kg: bool = True

class Answer(BaseModel):
    answer: str
    citations: List[Passage]
    kg_triplets: List[Dict]
    meta: Dict

# --- 模拟后端 Pipeline ---
class MockPipeline:
    def query(self, req: QueryRequest) -> Answer:
        # 模拟网络延迟和推理时间
        time.sleep(1.5)
        
        # 根据是否使用知识图谱返回不同结果
        kg_info = "结合知识图谱分析，" if req.use_kg else "仅基于文本检索，"
        
        ans_text = f"{kg_info}针对您的故障描述【{req.question}】，建议检查燃油系统和进气系统。（Top K 设为: {req.top_k}）"
        
        citations = [Passage(text="《柴油机维修手册》 第4章: 燃油系统检查规范")] if req.top_k > 0 else []
        triplets = [{"head": "柴油机", "relation": "具有", "tail": "燃油系统"}] if req.use_kg else []
        
        return Answer(
            answer=ans_text,
            citations=citations,
            kg_triplets=triplets,
            meta={"latency": "1.5s", "model": "DeepBlue-V3"}
        )