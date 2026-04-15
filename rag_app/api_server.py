import re
import os
import sys
import threading
import time
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from rag.config import SETTINGS
from rag.pipeline import RagPipeline
from rag.schema import QueryRequest
from update_index_incremental import main as run_incremental_update
from db_service import (
    authenticate_user,
    get_user_settings,
    initialize_database,
    list_user_chats,
    replace_user_chats,
    save_user_settings,
)


def _configure_stdio_for_windows() -> None:
    if os.name != "nt":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_for_windows()


app = FastAPI(title="RAG LangServe Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_lock = threading.RLock()
_pipeline: Optional[RagPipeline] = None
_allowed_suffixes = {".txt", ".md", ".pdf"}

initialize_database()


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


@app.on_event("startup")
def on_startup() -> None:
    _start_warmup_async()


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
        enable_parent_retriever=payload.get("enable_parent_retriever"),
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
        enable_parent_retriever=payload.get("enable_parent_retriever"),
    )


def _sse_json(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


rag_runnable = RunnableLambda(_run_query).with_types(input_type=dict, output_type=dict)
add_routes(app, rag_runnable, path="/rag")


@app.post("/rag/query")
def query_rag(req: QueryRequest):
    with _pipeline_lock:
        pipeline = _get_pipeline()
        answer = pipeline.query(req)
        return pipeline.export_answer(answer)


@app.post("/auth/login")
def auth_login(payload: dict):
    user_id = str(payload.get("user_id", "")).strip()
    password = str(payload.get("password", ""))
    if not user_id or not password:
        raise HTTPException(status_code=400, detail="user_id and password are required")
    user = authenticate_user(user_id, password)
    if not user:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    return {
        "ok": True,
        "user": user,
        "settings": get_user_settings(user_id),
        "chats": list_user_chats(user_id),
    }


@app.get("/users/{user_id}/settings")
def read_settings(user_id: str):
    return {"ok": True, "settings": get_user_settings(user_id)}


@app.put("/users/{user_id}/settings")
def update_settings(user_id: str, payload: dict):
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be an object")
    changed_by = str(payload.get("changed_by") or user_id)
    saved = save_user_settings(user_id, settings, changed_by=changed_by)
    return {"ok": True, "settings": saved}


@app.get("/users/{user_id}/chats")
def read_chats(user_id: str):
    return {"ok": True, "chats": list_user_chats(user_id)}


@app.put("/users/{user_id}/chats")
def save_chats(user_id: str, payload: dict):
    chats = payload.get("chats")
    if not isinstance(chats, list):
        raise HTTPException(status_code=400, detail="chats must be an array")
    saved = replace_user_chats(user_id, chats)
    return {"ok": True, "chats": saved}


@app.post("/rag/query/stream")
def query_rag_stream(payload: dict):
    try:
        req = _build_query_request(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        started = time.perf_counter()
        yield _sse_json(
            "meta",
            {
                "user_id": req.session_id,
                "question": req.question,
                "stream": True,
            },
        )
        try:
            with _pipeline_lock:
                pipeline = _get_pipeline()
                token_stream, payload_ctx = pipeline.stream_query_with_payload(req)

                timing = payload_ctx.get("timing") or {}
                t0 = timing.get("t0")
                t1 = timing.get("t1")
                t2 = timing.get("t2")
                retrieve_ms = (
                    int((t1 - t0) * 1000)
                    if isinstance(t0, (int, float)) and isinstance(t1, (int, float))
                    else None
                )
                kg_ms = (
                    int((t2 - t1) * 1000)
                    if isinstance(t1, (int, float)) and isinstance(t2, (int, float))
                    else None
                )

                context_count = len(
                    payload_ctx.get("context", {}).passages
                    if payload_ctx.get("context")
                    else []
                )
                yield _sse_json(
                    "trace",
                    {
                        "stage": "retrieve",
                        "message": "完成知识库检索",
                        "context_count": context_count,
                        "elapsed_ms": retrieve_ms,
                    },
                )

                if req.use_kg:
                    kg_count = len(payload_ctx.get("kg_triplets") or [])
                    yield _sse_json(
                        "trace",
                        {
                            "stage": "kg",
                            "message": "完成知识图谱查询",
                            "kg_triplet_count": kg_count,
                            "elapsed_ms": kg_ms,
                        },
                    )

                yield _sse_json(
                    "trace",
                    {
                        "stage": "generate",
                        "message": "开始答案生成",
                        "mode": "online" if req.use_llm else "offline",
                    },
                )

                for token in token_stream:
                    if not token:
                        continue
                    chunk = str(token).replace("\r", "")
                    yield _sse_json("token", {"text": chunk})
        except Exception as exc:
            yield _sse_json(
                "error",
                {
                    "message": str(exc),
                    "type": type(exc).__name__,
                },
            )
            return
        references = []
        for passage in (
            payload_ctx.get("context", {}).passages
            if payload_ctx.get("context")
            else []
        ):
            references.append(
                {
                    "doc_id": passage.doc_id,
                    "source": passage.source,
                    "score": passage.score,
                    "text": passage.text,
                }
            )
        if references:
            yield _sse_json("references", {"citations": references})
        kg_triplets = payload_ctx.get("kg_triplets") or []
        if kg_triplets:
            yield _sse_json("kg", {"triplets": kg_triplets})
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        yield _sse_json(
            "done",
            {
                "finish_reason": "stop",
                "elapsed_ms": elapsed_ms,
                "citation_count": len(references),
            },
        )

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
    save_path_abs = save_path.resolve()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    save_path.write_bytes(data)

    try:
        with _pipeline_lock:
            stats = run_incremental_update(target_paths=[str(save_path_abs)])
            _reload_pipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"incremental indexing failed: {exc}"
        ) from exc

    return {
        "ok": True,
        "user_id": safe_user_id,
        "file_name": save_name,
        "saved_path": str(save_path),
        "indexed_docs": int((stats or {}).get("indexed_docs", 0)),
        "indexed_chunks": int((stats or {}).get("indexed_chunks", 0)),
        "message": "incremental file indexed",
    }
