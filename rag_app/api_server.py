import re
import threading
import time
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
try:
    from fastapi.middleware.cors import CORSMiddleware
except Exception:
    # 兼容部分环境下 fastapi 子模块解析异常
    from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableLambda

_LANGSERVE_ERR = None
try:
    from langserve import add_routes
    _LANGSERVE_OK = True
except Exception as _e:
    add_routes = None
    _LANGSERVE_OK = False
    _LANGSERVE_ERR = str(_e)

from rag.config import SETTINGS
from rag.pipeline import RagPipeline
from rag.schema import QueryRequest
from update_index_incremental import main as run_incremental_update

app = FastAPI(title="RAG LangServe Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    # 不要 allow_origins=["*"] + allow_credentials=True
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_lock = threading.RLock()
_pipeline: Optional[RagPipeline] = None
_allowed_suffixes = {".txt", ".md", ".pdf"}


def _get_pipeline() -> RagPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RagPipeline()
        return _pipeline


def _reload_pipeline() -> None:
    global _pipeline
    with _pipeline_lock:
        _pipeline = RagPipeline()


def _sanitize_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", str(user_id or "").strip())
    return cleaned[:64] or "anonymous"


def _run_query(payload: dict) -> dict:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    user_id = _sanitize_user_id(payload.get("user_id", ""))
    req = QueryRequest(
        question=question,
        session_id=user_id,
        top_k=payload.get("top_k"),
        use_kg=payload.get("use_kg", True),
        use_llm=payload.get("use_llm", True),
        use_history=payload.get("use_history", True),
        enable_decompose=payload.get("enable_decompose"),
        enable_retrieval_optimization=payload.get("enable_retrieval_optimization"),
        llm_provider=payload.get("llm_provider"),
        llm_model=payload.get("llm_model"),
        llm_api_key=payload.get("llm_api_key"),
        llm_api_base=payload.get("llm_api_base"),
    )
    with _pipeline_lock:
        pipeline = _get_pipeline()
        answer = pipeline.query(req)
        return pipeline.export_answer(answer)


def _build_query_request(payload: dict) -> QueryRequest:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    user_id = _sanitize_user_id(payload.get("user_id", ""))
    return QueryRequest(
        question=question,
        session_id=user_id,
        top_k=payload.get("top_k"),
        use_kg=payload.get("use_kg", True),
        use_llm=payload.get("use_llm", True),
        use_history=payload.get("use_history", True),
        enable_decompose=payload.get("enable_decompose"),
        enable_retrieval_optimization=payload.get("enable_retrieval_optimization"),
        llm_provider=payload.get("llm_provider"),
        llm_model=payload.get("llm_model"),
        llm_api_key=payload.get("llm_api_key"),
        llm_api_base=payload.get("llm_api_base"),
    )


def _sse_json(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


rag_runnable = RunnableLambda(_run_query).with_types(input_type=dict, output_type=dict)
if _LANGSERVE_OK:
    add_routes(app, rag_runnable, path="/rag")


@app.get("/health")
def health():
    return {
        "ok": True,
        "langserve": _LANGSERVE_OK,
        "langserve_error": None if _LANGSERVE_OK else _LANGSERVE_ERR,
        "llm_provider": SETTINGS.llm_provider,
        "llm_model": SETTINGS.llm_model,
        "llm_api_key_set": bool(str(SETTINGS.llm_api_key or "").strip()),
    }


@app.post("/rag/query")
def query_rag(req: QueryRequest):
    with _pipeline_lock:
        pipeline = _get_pipeline()
        answer = pipeline.query(req)
        return pipeline.export_answer(answer)


@app.post("/rag/query/stream")
def query_rag_stream(payload: dict = Body(...)):
    try:
        req = _build_query_request(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        started = time.perf_counter()
        yield _sse_json("meta", {"user_id": req.session_id, "question": req.question, "stream": True})

        payload_ctx = {}
        try:
            with _pipeline_lock:
                pipeline = _get_pipeline()
                token_stream, payload_ctx = pipeline.stream_query_with_payload(req)
                for token in token_stream:
                    if token:
                        yield _sse_json("token", {"text": str(token).replace("\r", "")})
        except Exception as exc:
            yield _sse_json("error", {"message": str(exc), "type": type(exc).__name__})
            return

        references = []
        context = payload_ctx.get("context") if isinstance(payload_ctx, dict) else None
        passages = getattr(context, "passages", []) if context else []
        for p in passages:
            references.append({
                "doc_id": getattr(p, "doc_id", ""),
                "source": getattr(p, "source", ""),
                "score": getattr(p, "score", None),
                "text": getattr(p, "text", ""),
            })

        if references:
            yield _sse_json("references", {"citations": references})

        kg_triplets = payload_ctx.get("kg_triplets") if isinstance(payload_ctx, dict) else []
        if kg_triplets:
            yield _sse_json("kg", {"triplets": kg_triplets})

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        yield _sse_json("done", {"finish_reason": "stop", "elapsed_ms": elapsed_ms, "citation_count": len(references)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/rag/incremental/upload")
async def upload_incremental_file(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    safe_user_id = _sanitize_user_id(user_id)
    original_name = file.filename or "upload.txt"
    suffix = Path(original_name).suffix.lower()
    if suffix not in _allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {suffix}. allowed: {sorted(_allowed_suffixes)}",
        )

    upload_root = Path(SETTINGS.docs_dir) / "incremental" / safe_user_id
    upload_root.mkdir(parents=True, exist_ok=True)
    save_name = f"{int(time.time())}_{Path(original_name).name}"
    save_path = upload_root / save_name

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    save_path.write_bytes(data)

    with _pipeline_lock:
        run_incremental_update()
        _reload_pipeline()

    return {
        "ok": True,
        "user_id": safe_user_id,
        "file_name": save_name,
        "saved_path": str(save_path),
        "message": "incremental file indexed",
    }
