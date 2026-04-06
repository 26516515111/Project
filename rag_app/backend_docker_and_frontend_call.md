# RAG Backend Docker Packaging + Frontend Call Guide

This document shows how to package the RAG backend as a Docker service and how the frontend should call it.
It does not modify the existing UI. You will add a lightweight API layer that wraps `RagPipeline`.

## 1. Backend API (FastAPI)

Create a minimal API server (example path: `rag_app/api_server.py`):

```python
from fastapi import FastAPI
from rag.pipeline import RagPipeline
from rag.schema import QueryRequest

app = FastAPI()
pipeline = RagPipeline()


@app.post("/rag/query")
def query_rag(req: QueryRequest):
    answer = pipeline.query(req)
    return pipeline.export_answer(answer)
```

Add FastAPI server deps to your backend image (not to the UI):

```
fastapi
uvicorn
```

If you want to keep `requirements.txt` unchanged, you can install these in the Dockerfile.

## 2. Backend Dockerfile

Create a dedicated Dockerfile for the backend (example path: `rag_app/Dockerfile.backend`):

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install fastapi uvicorn

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 3. Build + Run Backend Docker

```bash
docker build -t rag-backend -f rag_app/Dockerfile.backend ./rag_app
```

```bash
docker run --rm -p 8000:8000 \
  -e RAG_USE_RERANKER=false \
  -e RAG_LLM_PROVIDER=ollama \
  -e RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v %cd%\rag_app\data:/app/data \
  rag-backend
```

Notes:

- On macOS/Linux, replace `-v %cd%\rag_app\data:/app/data` with `-v $(pwd)/rag_app/data:/app/data`.
- If you do not use Ollama on the host, set `RAG_LLM_PROVIDER=none` for extractive-only answers.
- Neo4j is external by default. Set `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD` to your instance.

## 4. Frontend Calls Backend API

Endpoint:

```
POST http://<backend-host>:8000/rag/query
Content-Type: application/json
```

Request body example:

```json
{
  "question": "船舶电气设备故障怎么排查？",
  "session_id": "web_001",
  "top_k": 5,
  "use_kg": true,
  "use_history": true,
  "enable_decompose": true
}
```

Response (shape from `rag.schema.Answer`):

```json
{
  "question": "...",
  "answer": "...",
  "citations": [
    {
      "chunk_id": "...",
      "text": "...",
      "score": 0.82,
      "source": "..."
    }
  ],
  "kg_triplets": [],
  "meta": {
    "retriever": "hybrid",
    "top_k": "5",
    "hybrid_top_k": "10",
    "decompose": "True"
  }
}
```

## 5. Frontend Example (Fetch)

```js
const resp = await fetch("http://localhost:8000/rag/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "船舶电气设备故障怎么排查？",
    session_id: "web_001",
    top_k: 5,
    use_kg: true,
    use_history: true,
    enable_decompose: true
  })
});

const data = await resp.json();
console.log(data.answer);
```

## 6. Frontend Docker Notes

If your frontend is also Dockerized, call the backend by service name in the same Docker network:

```
http://rag-backend:8000/rag/query
```

For Docker Compose, ensure both services share the same network.
