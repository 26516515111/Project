from fastapi import FastAPI

from rag.pipeline import RagPipeline
from rag.schema import QueryRequest

app = FastAPI()
pipeline = RagPipeline()


@app.post("/rag/query")
def query_rag(req: QueryRequest):
    answer = pipeline.query(req)
    return pipeline.export_answer(answer)
