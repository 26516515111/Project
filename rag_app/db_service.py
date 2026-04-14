import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from database_models import (
    ChatMessage,
    ChatSession,
    SessionLocal,
    SettingHistory,
    User,
    UserSettings,
    init_db,
    sync_legacy_db,
)


DEFAULT_SETTINGS: Dict[str, Any] = {
    "nickname": "Captain Park",
    "avatar": "",
    "imo": "9876543",
    "email": "captain.park@deepblue.com",
    "emergency": "+86 13800138000",
    "theme": "dark",
    "fontSize": 16,
    "notify": True,
    "autoSave": True,
    "graphOn": True,
    "offlineOn": False,
    "streamSpeed": 50,
    "dbPath": "/mnt/data/local_kb",
    "model": "hybrid",
    "retention": "30",
}


def _hash_password(raw_password: str) -> str:
    return hashlib.sha256(str(raw_password or "").encode("utf-8")).hexdigest()


def _settings_to_dict(settings: UserSettings) -> Dict[str, Any]:
    if not settings:
        return dict(DEFAULT_SETTINGS)
    return {
        "nickname": settings.nickname or "",
        "avatar": settings.avatar or "",
        "imo": settings.imo or "",
        "email": settings.email or "",
        "emergency": settings.emergency or "",
        "theme": settings.theme or "dark",
        "fontSize": int(settings.font_size or 16),
        "notify": bool(settings.notify),
        "autoSave": bool(settings.auto_save),
        "graphOn": bool(settings.graph_on),
        "offlineOn": bool(settings.offline_on),
        "streamSpeed": int(settings.stream_speed or 50),
        "dbPath": settings.db_path or "",
        "model": settings.model or "hybrid",
        "retention": str(settings.retention or "30"),
    }


def _apply_settings_payload(settings: UserSettings, payload: Dict[str, Any]) -> None:
    merged = dict(DEFAULT_SETTINGS)
    merged.update(payload or {})
    settings.nickname = str(merged.get("nickname") or "")
    settings.avatar = str(merged.get("avatar") or "")
    settings.imo = str(merged.get("imo") or "")
    settings.email = str(merged.get("email") or "")
    settings.emergency = str(merged.get("emergency") or "")
    settings.theme = str(merged.get("theme") or "dark")
    settings.font_size = int(merged.get("fontSize") or 16)
    settings.notify = bool(merged.get("notify", True))
    settings.auto_save = bool(merged.get("autoSave", True))
    settings.graph_on = bool(merged.get("graphOn", True))
    settings.offline_on = bool(merged.get("offlineOn", False))
    settings.stream_speed = int(merged.get("streamSpeed") or 50)
    settings.db_path = str(merged.get("dbPath") or "")
    settings.model = str(merged.get("model") or "hybrid")
    settings.retention = str(merged.get("retention") or "30")


def _get_or_create_settings(db, user: User) -> UserSettings:
    settings = db.query(UserSettings).filter(UserSettings.user_id == user.user_id).first()
    if settings:
        return settings
    settings = UserSettings(user_id=user.user_id)
    _apply_settings_payload(settings, DEFAULT_SETTINGS)
    if user.role == 1:
        settings.nickname = "王轮机长"
    db.add(settings)
    db.flush()
    return settings


def _ensure_user(db, user_id: str, password: str, role: int, nickname: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id, password_hash=_hash_password(password), role=int(role))
        db.add(user)
        db.flush()
    settings = _get_or_create_settings(db, user)
    if not settings.nickname:
        settings.nickname = nickname
    return user


def initialize_database() -> None:
    sync_legacy_db()
    init_db()
    with SessionLocal() as db:
        _ensure_user(db, "admin", "123456", 1, "王轮机长")
        _ensure_user(db, "captain_park", "123456", 0, "Captain Park")
        db.commit()
    sync_legacy_db()


def authenticate_user(user_id: str, password: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as db:
        user = db.query(User).filter(User.user_id == str(user_id or "").strip()).first()
        if not user:
            return None
        if user.password_hash != _hash_password(password):
            return None
        settings = _get_or_create_settings(db, user)
        db.commit()
        return {
            "user_id": user.user_id,
            "role": int(user.role or 0),
            "name": settings.nickname or user.user_id,
            "avatar": settings.avatar or "",
        }


def get_user_settings(user_id: str) -> Dict[str, Any]:
    with SessionLocal() as db:
        user = db.query(User).filter(User.user_id == str(user_id or "").strip()).first()
        if not user:
            user = User(user_id=str(user_id or "anonymous"), password_hash=_hash_password("123456"), role=0)
            db.add(user)
            db.flush()
        settings = _get_or_create_settings(db, user)
        db.commit()
        return _settings_to_dict(settings)


def save_user_settings(user_id: str, payload: Dict[str, Any], changed_by: str = "system") -> Dict[str, Any]:
    with SessionLocal() as db:
        user = db.query(User).filter(User.user_id == str(user_id or "").strip()).first()
        if not user:
            user = User(user_id=str(user_id or "anonymous"), password_hash=_hash_password("123456"), role=0)
            db.add(user)
            db.flush()
        settings = _get_or_create_settings(db, user)
        _apply_settings_payload(settings, payload)
        snapshot = _settings_to_dict(settings)
        db.add(
            SettingHistory(
                user_id=user.user_id,
                changed_at=datetime.utcnow(),
                changed_by=str(changed_by or "system"),
                payload=json.dumps(snapshot, ensure_ascii=False),
            )
        )
        db.commit()
        sync_legacy_db()
        return snapshot


def _message_to_dict(msg: ChatMessage) -> Dict[str, Any]:
    citations = []
    kg_triplets = []
    extra = {}
    if msg.citations:
        try:
            citations = json.loads(msg.citations)
        except Exception:
            citations = []
    if msg.kg_triplets:
        try:
            kg_triplets = json.loads(msg.kg_triplets)
        except Exception:
            kg_triplets = []
    if msg.extra_data:
        try:
            extra = json.loads(msg.extra_data)
        except Exception:
            extra = {}
    data = {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "searchProcess": msg.search_process or "",
        "citations": citations,
        "kgTriplets": kg_triplets,
    }
    if isinstance(extra, dict):
        data.update(extra)
    return data


def _session_to_dict(session: ChatSession) -> Dict[str, Any]:
    messages = sorted(session.messages or [], key=lambda m: m.created_at or datetime.utcnow())
    return {
        "id": session.id,
        "title": session.title or "未命名会话",
        "messages": [_message_to_dict(m) for m in messages],
        "createdAt": (session.created_at or datetime.utcnow()).isoformat(),
    }


def list_user_chats(user_id: str) -> List[Dict[str, Any]]:
    uid = str(user_id or "").strip()
    if not uid:
        return []
    with SessionLocal() as db:
        sessions = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == uid)
            .order_by(ChatSession.created_at.desc())
            .all()
        )
        return [_session_to_dict(s) for s in sessions]


def replace_user_chats(user_id: str, chats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uid = str(user_id or "").strip()
    if not uid:
        return []
    with SessionLocal() as db:
        user = db.query(User).filter(User.user_id == uid).first()
        if not user:
            user = User(user_id=uid, password_hash=_hash_password("123456"), role=0)
            db.add(user)
            db.flush()
            _get_or_create_settings(db, user)

        old_sessions = db.query(ChatSession).filter(ChatSession.user_id == uid).all()
        for s in old_sessions:
            db.delete(s)
        db.flush()

        for chat in chats or []:
            session_id = str(chat.get("id") or uuid.uuid4())
            session = ChatSession(
                id=session_id,
                user_id=uid,
                title=str(chat.get("title") or "未命名会话"),
                created_at=datetime.utcnow(),
            )
            db.add(session)
            db.flush()

            for idx, msg in enumerate(chat.get("messages") or []):
                role = str(msg.get("role") or "assistant")
                content = str(msg.get("content") or "")
                if not content:
                    continue
                citations = msg.get("citations")
                kg_triplets = msg.get("kgTriplets")
                known_keys = {
                    "id",
                    "role",
                    "content",
                    "searchProcess",
                    "citations",
                    "kgTriplets",
                }
                extra_data = {k: v for k, v in msg.items() if k not in known_keys}
                message = ChatMessage(
                    id=str(msg.get("id") or f"{session_id}-msg-{idx}-{uuid.uuid4().hex[:8]}"),
                    session_id=session_id,
                    role=role,
                    content=content,
                    search_process=str(msg.get("searchProcess") or ""),
                    citations=json.dumps(citations, ensure_ascii=False) if citations is not None else None,
                    kg_triplets=json.dumps(kg_triplets, ensure_ascii=False) if kg_triplets is not None else None,
                    extra_data=json.dumps(extra_data, ensure_ascii=False) if extra_data else None,
                    created_at=datetime.utcnow(),
                )
                db.add(message)

        db.commit()
        sync_legacy_db()

    return list_user_chats(uid)
