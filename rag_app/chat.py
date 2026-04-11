import streamlit as st
import base64
import time
import re
import json
import uuid
import html
from pathlib import Path
import requests

API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEADING_PATTERN = re.compile(r"^\[(H[1-6])\]\s*(.+?)\s*$")
IMAGE_ATTACHMENT_PATTERN = re.compile(r"^\[(?:IMG|图片附件)\s*(.+?)\]\s*$")

def _resolve_attachment_path(raw_ref: str) -> Path | None:
    ref = (raw_ref or "").strip()
    if not ref: return None
    if ref.startswith("path="):
        parts = ref.split(" alt=", 1)
        ref = parts[0].split("=", 1)[1].strip()
    candidate = Path(ref)
    if candidate.is_absolute(): return candidate if candidate.exists() else None
    normalized = ref[2:] if ref.startswith("./") else ref
    resolved = PROJECT_ROOT / normalized
    if resolved.exists(): return resolved
    if normalized.startswith("images/"):
        image_name = Path(normalized).name
        image_roots = [
            PROJECT_ROOT / "rag_app" / "data" / "KG" / "images",
            PROJECT_ROOT / "data" / "KG" / "images",
        ]
        for root in image_roots:
            if root.exists():
                matches = list(root.rglob(image_name))
                if matches: return matches[0]
    return None

def _image_data_uri(image_path: Path) -> str | None:
    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp",
    }.get(suffix)
    if not mime: return None
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError: return None
    return f"data:{mime};base64,{encoded}"

def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("[H"): return False
    if stripped.count("|") < 1: return False
    parts = [part.strip() for part in stripped.split("|")]
    non_empty = [part for part in parts if part]
    return len(non_empty) >= 2

def _render_table(lines: list[str]) -> str:
    rows = []
    for idx, raw_line in enumerate(lines):
        cells = [html.escape(cell.strip()) for cell in raw_line.split("|")]
        rows.append((idx == 0, cells))
    header_cells = "".join(f"<th>{cell}</th>" for cell in rows[0][1])
    body_rows = []
    for _, cells in rows[1:]:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    body_html = "".join(body_rows)
    return (
        '<div class="chunk-table-wrap"><table class="chunk-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body_html}</tbody></table></div>"
    )

def _render_attachment_block(ref: str) -> str:
    display_ref = html.escape(ref)
    image_path = _resolve_attachment_path(ref)
    if image_path:
        data_uri = _image_data_uri(image_path)
        if data_uri:
            return (
                '<div class="chunk-attachment">'
                f'<div class="chunk-attachment-label" style="font-size:12px; color:#2dd4bf; margin-bottom:6px;">图片附件</div>'
                f'<img class="chunk-image" src="{data_uri}" alt="{display_ref}" />'
                f'<div class="chunk-attachment-path" style="font-size:10px; color:#6b7280; margin-top:4px;">{display_ref}</div>'
                "</div>"
            )
    return (
        '<div class="chunk-attachment chunk-attachment-fallback">'
        '<div class="chunk-attachment-label" style="font-size:12px; color:#2dd4bf;">图片附件 (加载失败)</div>'
        f'<div class="chunk-attachment-path" style="font-size:10px; color:#6b7280;">{display_ref}</div>'
        "</div>"
    )

def _render_structured_content(content: str) -> str:
    lines = content.splitlines()
    parts: list[str] = []
    paragraph_buffer: list[str] = []
    table_buffer: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer: return
        text = "<br>".join(html.escape(line) for line in paragraph_buffer)
        parts.append(f'<div class="chunk-paragraph">{text}</div>')
        paragraph_buffer = []

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer: return
        parts.append(_render_table(table_buffer))
        table_buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_table(); flush_paragraph(); continue

        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            flush_table(); flush_paragraph()
            level, title = heading_match.groups()
            parts.append(f'<div class="chunk-heading {level.lower()}">{html.escape(title)}</div>')
            continue

        attachment_match = IMAGE_ATTACHMENT_PATTERN.match(stripped)
        if attachment_match:
            flush_table(); flush_paragraph()
            parts.append(_render_attachment_block(attachment_match.group(1).strip()))
            continue

        if _looks_like_table_row(stripped):
            flush_paragraph()
            table_buffer.append(stripped)
            continue

        flush_table()
        paragraph_buffer.append(line)

    flush_table()
    flush_paragraph()
    return "".join(parts) or '<div class="chunk-paragraph"></div>'

def _save_current_session():
    if not st.session_state.history: return
    title = "新对话"
    for msg in st.session_state.history:
        if msg["role"] == "user":
            title = msg["content"][:15] + ("..." if len(msg["content"]) > 15 else "")
            break
    found = False
    for s in st.session_state.sessions:
        if s["id"] == st.session_state.current_session_id:
            s["history"] = st.session_state.history.copy()
            s["title"] = title
            found = True
            break
    if not found:
        st.session_state.sessions.append({
            "id": st.session_state.current_session_id,
            "title": title,
            "history": st.session_state.history.copy()
        })

def render_chat():
    # ==================== 初始化 Session 状态 ====================
    if "history" not in st.session_state: st.session_state.history = []
    if "sessions" not in st.session_state: st.session_state.sessions = []
    if "current_session_id" not in st.session_state: st.session_state.current_session_id = str(uuid.uuid4())
    if "is_generating" not in st.session_state: st.session_state.is_generating = False
    if "use_kg" not in st.session_state: st.session_state.use_kg = True
    if "generation_started" not in st.session_state: st.session_state.generation_started = False
    user_info = st.session_state.get("user_info", {"name": "用户", "avatar": "👤"})

    # ==================== 全局 CSS 注入 - 完全复刻原设计 ====================
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    @import url('https://unpkg.com/@phosphor-icons/web@2.0.3/src/all.css');
    
    * { font-family: 'Noto Sans SC', sans-serif !important; box-sizing: border-box; }
    
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu, footer, header, [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    
    /* 全局背景 - 与原设计完全一致 #030d17 */
    .stApp {
        background: #030d17 !important;
        color: #9ca3af !important;
        overflow: hidden !important;
    }
    
    /* 主容器调整 */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* ===== 左侧边栏玻璃态 - 精确复刻 256px ===== */
    [data-testid="stSidebar"] {
        background: rgba(4, 21, 39, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        width: 256px !important;
        min-width: 256px !important;
        max-width: 256px !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        padding: 20px 16px !important;
    }
    
    /* 侧边栏内容区域 */
    [data-testid="stSidebarContent"] {
        background: transparent !important;
    }
    
    /* ===== 主内容区域背景 #041527 ===== */
    section.main {
        background: #041527 !important;
    }
    
    /* ===== 右侧边栏玻璃态 ===== */
    [data-testid="column"]:nth-child(3) {
        background: rgba(4, 21, 39, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* ===== 顶部导航栏玻璃态 ===== */
    .glass-header {
        background: rgba(4, 21, 39, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        height: 64px !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
    }
    
    /* ===== 发光背景 600px ===== */
    .glow-bg {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        width: 600px !important;
        height: 600px !important;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.05) 0%, transparent 70%) !important;
        pointer-events: none !important;
        z-index: 0 !important;
    }
    
    /* ===== 玻璃态卡片 - 精确复刻 rgba(255,255,255,0.02) ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .glass-card:hover {
        border-color: rgba(45, 212, 191, 0.3) !important;
        background: rgba(20, 184, 166, 0.05) !important;
    }
    
    /* ===== 状态点动画 pulse ===== */
    .status-dot {
        width: 8px;
        height: 8px;
        background: #2dd4bf;
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(45, 212, 191, 0.8);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* ===== 进度条 4px #14b8a6 ===== */
    .progress-bar {
        height: 4px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: #14b8a6;
        border-radius: 2px;
        width: 85%;
    }
    
    /* ===== 历史记录项 ===== */
    .history-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #9ca3af;
    }
    
    .history-item:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #e5e7eb;
    }
    
    .history-item.active {
        background: rgba(20, 184, 166, 0.1);
        border: 1px solid rgba(20, 184, 166, 0.2);
        color: #ccfbf1;
    }
    
    /* ===== 用户头像 蓝渐变 ===== */
    .user-avatar {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        font-size: 12px;
    }
    
    /* ===== 图标盒子 ===== */
    .icon-box {
        width: 40px;
        height: 40px;
        background: rgba(20, 184, 166, 0.1);
        border: 1px solid rgba(20, 184, 166, 0.3);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2dd4bf;
    }
    
    /* ===== 主图标容器 80px #0a2530 ===== */
    .main-icon-container {
        width: 80px;
        height: 80px;
        background: rgba(10, 37, 48, 0.8);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    
    .main-icon-container i {
        font-size: 40px;
        color: #2dd4bf;
    }
    
    /* 闪烁徽章 */
    .sparkle-badge {
        position: absolute;
        bottom: -6px;
        right: -6px;
        width: 24px;
        height: 24px;
        background: #0d9488;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid #041527;
        font-size: 12px;
        color: white;
    }
    
    /* ===== 浮动粒子 ===== */
    .floating-particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: rgba(45, 212, 191, 0.5);
        border-radius: 50%;
    }
    
    .particle-1 { top: -16px; left: -16px; }
    .particle-2 { top: 50%; right: -32px; background: rgba(156, 163, 175, 0.5); width: 3px; height: 3px; }
    
    /* ===== 消息气泡 ===== */
    .message-user {
        background: rgba(20, 184, 166, 0.1);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-radius: 12px;
        padding: 16px;
        color: #e5e7eb;
        max-width: 70%;
    }
    
    .message-assistant {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        color: #e5e7eb;
        max-width: 80%;
    }
    
    /* ===== 渲染引擎表格样式 ===== */
    .chunk-table-wrap {
        overflow-x: auto;
        margin: 10px 0;
        border-radius: 8px;
    }
    
    .chunk-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        background: rgba(255,255,255,0.02);
    }
    
    .chunk-table th, .chunk-table td {
        padding: 8px 12px;
        border: 1px solid rgba(45, 212, 191, 0.2);
    }
    
    .chunk-table th {
        background: rgba(20, 184, 166, 0.1);
        font-weight: 600;
        text-align: left;
        color: #2dd4bf;
    }
    
    .chunk-paragraph {
        margin-bottom: 10px;
        line-height: 1.6;
    }
    
    .chunk-heading {
        font-weight: bold;
        margin: 16px 0 8px;
        color: #2dd4bf;
    }
    
    .chunk-heading.h1 { font-size: 1.4em; }
    .chunk-heading.h2 { font-size: 1.2em; }
    .chunk-heading.h3 { font-size: 1.0em; }
    
    .chunk-attachment {
        background: rgba(0,0,0,0.3);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid rgba(45, 212, 191, 0.3);
    }
    
    .chunk-image {
        max-width: 100%;
        border-radius: 6px;
    }
    
    /* ===== 辅助按钮样式 ===== */
    .aux-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.1);
        background: transparent;
        color: #9ca3af;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .aux-btn:hover {
        background: rgba(255,255,255,0.05);
        color: #e5e7eb;
    }
    
    /* ===== 自定义滚动条 4px ===== */
    ::-webkit-scrollbar {
        width: 4px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* ===== Streamlit 组件覆盖 ===== */
    .stButton > button {
        background: #0d9488 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 15px -3px rgba(13, 148, 136, 0.2) !important;
    }
    
    .stButton > button:hover {
        background: #14b8a6 !important;
        box-shadow: 0 0 20px rgba(13, 148, 136, 0.3) !important;
    }
    
    /* 次要按钮 */
    button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #9ca3af !important;
        box-shadow: none !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.05) !important;
        color: #e5e7eb !important;
    }
    
    /* Toggle 开关 */
    .stCheckbox [data-testid="stCheckbox"] {
        margin-top: -4px;
    }
    
    /* 输入框去除边框 */
    .stTextInput > div > div > input {
        background: transparent !important;
        border: none !important;
        color: #e5e7eb !important;
        font-size: 14px !important;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: none !important;
    }
    
    /* Chat input 玻璃态 */
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    .stChatInputContainer:focus-within {
        border-color: rgba(45, 212, 191, 0.4) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.05) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== 左侧边栏 ====================
    with st.sidebar:
        st.markdown("""
        <div style="padding: 4px 0 24px 0;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
                <div class="icon-box">
                    <i class="ph ph-steering-wheel" style="font-size: 20px;"></i>
                </div>
                <div>
                    <h2 style="color: white; font-size: 14px; font-weight: 700; margin: 0; letter-spacing: 0.025em;">智能船舶问答</h2>
                    <p style="color: #6b7280; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; margin: 2px 0 0 0;">DeepBlue AI</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 新建会话按钮
        if st.button("➕ 新建会话", key="new_chat", use_container_width=True, type="primary"):
            _save_current_session()
            st.session_state.history = []
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.is_generating = False
            st.session_state.generation_started = False
            st.rerun()
        
        st.markdown("""
        <p style="color: #6b7280; font-size: 12px; font-weight: 500; margin: 16px 0 12px 8px;">历史记录</p>
        <div style="max-height: calc(100vh - 300px); overflow-y: auto;">
        """, unsafe_allow_html=True)
        
        sessions = st.session_state.get("sessions", [])
        for sess in reversed(sessions[-8:]):
            title = sess.get("title", "未命名会话")[:16]
            if len(sess.get("title", "")) > 16:
                title += "..."
            is_active = (sess["id"] == st.session_state.current_session_id)
            
            item_class = "history-item active" if is_active else "history-item"
            icon_color = "#2dd4bf" if is_active else "#6b7280"
            text_color = "#ccfbf1" if is_active else "#9ca3af"
            
            st.markdown(f"""
            <div class="{item_class}">
                <i class="ph ph-chat-centered-text" style="color: {icon_color}; font-size: 16px;"></i>
                <span style="color: {text_color}; font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{html.escape(title)}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 使用真实按钮来处理点击
            if st.button(f"💬 {title}", key=f"sess_{sess['id']}", use_container_width=True, 
                        type="primary" if is_active else "secondary"):
                _save_current_session()
                st.session_state.history = sess.get("history", [])
                st.session_state.current_session_id = sess["id"]
                st.session_state.is_generating = False
                st.session_state.generation_started = False
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 底部用户信息
        st.markdown(f"""
        <div style="position: fixed; bottom: 0; left: 0; right: 0; padding: 16px; border-top: 1px solid rgba(255,255,255,0.05); background: rgba(4,21,39,0.9); z-index: 100;">
            <div style="display: flex; align-items: center; gap: 12px; padding: 8px; border-radius: 8px; cursor: pointer; transition: background 0.2s;" class="history-item">
                <div class="user-avatar">{user_info['name'][:2]}</div>
                <div style="flex: 1; min-width: 0;">
                    <p style="margin: 0; color: white; font-size: 14px; font-weight: 500;">{html.escape(user_info['name'])}</p>
                    <p style="margin: 0; color: #6b7280; font-size: 11px;">值班人员</p>
                </div>
                <i class="ph ph-gear" style="color: #9ca3af; font-size: 18px;"></i>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== 主内容与右侧栏布局 ====================
    main_container = st.container()
    
    with main_container:
        cols = st.columns([5, 1.8], gap="large") # 第1层列
        
        with cols[0]:
            pad_container = st.container()
            with pad_container: 
                # 顶部导航栏 
                nav_cols = st.columns([8, 1, 0.5]) # 合规的第2层列
                with nav_cols[0]:
                    st.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-top: 8px;">
    <div class="status-dot"></div>
    <h1 style="color: #d1d5db; font-size: 14px; font-weight: 500; margin: 0;">当前会话</h1>
</div>
""", unsafe_allow_html=True)
                with nav_cols[1]:
                    st.button("📤 设置", key="settings_btn", use_container_width=True)
                with nav_cols[2]:
                    st.button("🔔", key="notify_btn", use_container_width=True)
                
                st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 12px 0;'>", unsafe_allow_html=True)
                
                # 主内容展示区域
                if not st.session_state.history and not st.session_state.is_generating:
                    st.markdown('<div class="glow-bg"></div>', unsafe_allow_html=True)
                    
                    # 中央欢迎区域 (去除左侧缩进，防止 Markdown 解析为代码块)
                    st.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 40vh; position: relative;">
    <div style="position: relative; margin-bottom: 32px; margin-top: 5vh;">
        <div class="main-icon-container">
            <i class="ph ph-steering-wheel"></i>
            <div class="sparkle-badge">
                <i class="ph ph-sparkle" style="font-size: 10px;"></i>
            </div>
        </div>
        <div class="floating-particle particle-1"></div>
        <div class="floating-particle particle-2"></div>
    </div>
    <h2 style="color: white; font-size: 24px; font-weight: 700; margin-bottom: 16px; text-align: center;">欢迎使用智能船舶问答系统</h2>
    <p style="color: #9ca3af; font-size: 14px; text-align: center; line-height: 1.6; max-width: 400px;">
        基于知识图谱的深度故障诊断引擎已就绪<br>请告诉我您想了解的船舶设备问题
    </p>
</div>
""", unsafe_allow_html=True)
                    
                    # 修复：打平列层次为 1.5 : 3 : 3 : 1.5（两边留白占位，中间放卡片）
                    card_grid = st.columns([1.5, 3, 3, 1.5], gap="medium")
                    cards = [
                        ("ph-engine", "主机故障诊断", "分析推进系统异常"),
                        ("ph-lightning", "电气系统咨询", "电路故障排查建议"),
                        ("ph-drop", "液压系统分析", "压力与流量优化"),
                        ("ph-wrench", "维护周期建议", "基于历史数据分析"),
                    ]
                    
                    for idx, (icon, title, desc) in enumerate(cards):
                        # 判断放入左侧卡片列还是右侧卡片列（对应 card_grid 的下标 1 和 2）
                        col_idx = 1 if idx % 2 == 0 else 2
                        with card_grid[col_idx]:
                            st.markdown(f"""
<div class="glass-card" style="margin-bottom: 16px;">
    <i class="ph {icon}" style="font-size: 24px; color: #2dd4bf; margin-bottom: 12px; display: block;"></i>
    <h3 style="color: #e5e7eb; font-size: 14px; font-weight: 500; margin: 0 0 4px 0;">{title}</h3>
    <p style="color: #6b7280; font-size: 12px; margin: 0;">{desc}</p>
</div>
""", unsafe_allow_html=True)
                    
                    # 底部输入区域
                    st.markdown("<div style='height: 4vh;'></div>", unsafe_allow_html=True)
                    
                    st.markdown("""
<style>
/* 强制隐藏默认背景并融为一体 */
div[data-testid="stHorizontalBlock"] { align-items: center; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 4px 8px; }
div[data-testid="stHorizontalBlock"] input { background: transparent !important; }
</style>
""", unsafe_allow_html=True)
                        
                    # 修复：打平输入栏列层次，按照 留白:回形针:输入框:按钮:留白 分配比例
                    in_cols = st.columns([1.5, 0.5, 8, 1, 1.5])
                    with in_cols[1]:
                        st.markdown("<div style='text-align:center; margin-top: 8px;'><i class='ph ph-paperclip' style='color:#6b7280; font-size: 20px;'></i></div>", unsafe_allow_html=True)
                    with in_cols[2]:
                        user_input = st.text_input("", placeholder="输入您的问题，如：'对比去年同期的油耗数据'", label_visibility="collapsed", key="welcome_input")
                    with in_cols[3]:
                        if st.button("➤", key="welcome_send", use_container_width=True):
                            if user_input:
                                st.session_state.history.append({"role": "user", "content": user_input})
                                st.session_state.is_generating = True
                                st.session_state.generation_started = False
                                st.rerun()
                        
                    # 快捷操作去缩进
                    st.markdown("""
<div style="display: flex; justify-content: center; gap: 32px; margin-top: 24px;">
    <span class="aux-btn"><i class="ph ph-lightning" style="color: #2dd4bf;"></i> 快速体验</span>
    <span class="aux-btn"><i class="ph ph-chart-line" style="color: #2dd4bf;"></i> 效能评估</span>
    <span class="aux-btn"><i class="ph ph-shield-check" style="color: #2dd4bf;"></i> 安全审计</span>
</div>
<div style="height: 40px;"></div>
""", unsafe_allow_html=True)
                
                else:
                    # ===== 聊天历史逻辑保持不变 =====
                    for msg in st.session_state.history:
                        if msg.get("role") == "user":
                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-end; margin-bottom: 16px;">
                                <div class="message-user">
                                    {html.escape(msg.get("content", ""))}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            safe_content = _render_structured_content(msg.get("content", ""))
                            st.markdown(f"""
                            <div style="display: flex; justify-content: flex-start; margin-bottom: 16px;">
                                <div class="message-assistant">
                                    {safe_content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if "raw_answer" in msg and msg["raw_answer"]:
                                ans_obj = msg["raw_answer"]
                                with st.expander("📚 查看参考来源与知识图谱节点", expanded=False):
                                    if hasattr(ans_obj, "citations") and ans_obj.citations:
                                        st.markdown("**📄 依据文档:**")
                                        for c in ans_obj.citations:
                                            source = getattr(c, "source", None) or getattr(c, "text", "未知来源")
                                            score = getattr(c, "score", None)
                                            st.markdown(f"- {source} (相关度: {score:.2f})" if score else f"- {source}")
                                    
                                    if hasattr(ans_obj, "kg_triplets") and ans_obj.kg_triplets:
                                        st.markdown("**🕸️ 图谱推理路径:**")
                                        for t in ans_obj.kg_triplets:
                                            head = t.get("head", "")
                                            rel = t.get("rel", "") or t.get("relation", "")
                                            tail = t.get("tail", "")
                                            desc = t.get("description", "")
                                            if desc:
                                                st.markdown(f"- `{head}` ➡️ **{rel}** ➡️ `{tail}`<br><span style='color:#888; font-size:12px;'>📝 {desc}</span>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"- `{head}` ➡️ **{rel}** ➡️ `{tail}`")
                    
                    # 生成中状态逻辑保持不变
                    if st.session_state.get("is_generating") and not st.session_state.get("generation_started", False):
                        st.session_state.generation_started = True
                        latest_question = st.session_state.history[-1]["content"]
                        
                        placeholder = st.empty()
                        curr_text = ""
                        
                        with st.spinner("正在检索维修手册与图谱库..."):
                            req_payload = {
                                "question": latest_question.strip(),
                                "top_k": 3,
                                "use_kg": st.session_state.get("use_kg", True),
                                "session_id": st.session_state.get("current_session_id", "default"),
                            }
                            
                            class DummyAnswer:
                                def __init__(self):
                                    self.citations = []
                                    self.kg_triplets = []
                                    self.meta = {}
                            ans = DummyAnswer()
                            st.session_state.partial_raw = ans
                            
                            try:
                                response = requests.post(f"{API_URL}/rag/query/stream", json=req_payload, stream=True, timeout=60)
                                response.raise_for_status()
                                
                                for line in response.iter_lines():
                                    if not line: continue
                                    line_text = line.decode('utf-8')
                                    if line_text.startswith("data: "):
                                        data_str = line_text[6:]
                                        try:
                                            data_json = json.loads(data_str)
                                            if "text" in data_json:
                                                curr_text += data_json["text"]
                                                st.session_state.partial_content = curr_text
                                                safe = html.escape(curr_text).replace("\n", "<br>")
                                                placeholder.markdown(f"""
                                                <div style="display: flex; justify-content: flex-start; margin-bottom: 16px;">
                                                    <div class="message-assistant">
                                                        {safe}▌
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                time.sleep(0.01)
                                            elif "citations" in data_json:
                                                ans.citations = data_json["citations"]
                                            elif "triplets" in data_json:
                                                ans.kg_triplets = data_json["triplets"]
                                            elif "meta" in data_json:
                                                ans.meta = data_json
                                        except json.JSONDecodeError:
                                            pass
                            except Exception as e:
                                st.error(f"引擎响应出错: {str(e)}")
                            
                            st.session_state.is_generating = False
                            st.session_state.generation_started = False
                            st.session_state.partial_content = ""
                            st.session_state.partial_raw = None
                            st.session_state.history.append({
                                "role": "assistant",
                                "content": curr_text,
                                "raw_answer": ans
                            })
                            _save_current_session()
                            st.rerun()
                    
                    # 聊天输入
                    new_input = st.chat_input("输入您的问题，如：'对比去年同期的主机油耗数据'", disabled=st.session_state.get("is_generating", False))
                    if new_input and not st.session_state.get("is_generating"):
                        st.session_state.history.append({"role": "user", "content": new_input})
                        st.session_state.is_generating = True
                        st.session_state.generation_started = False
                        st.session_state.partial_content = ""
                        st.session_state.partial_raw = None
                        st.rerun()
        
        with cols[1]:  # 右侧边栏
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: space-between; margin: 16px 0 24px 0;">
                <h3 style="color: white; font-size: 14px; font-weight: 700; margin: 0; letter-spacing: 0.025em;">系统状况</h3>
                <div style="display: flex; align-items: center; gap: 8px; background: rgba(20, 184, 166, 0.1); padding: 4px 10px; border-radius: 20px; border: 1px solid rgba(20, 184, 166, 0.2);">
                    <div class="status-dot"></div>
                    <span style="color: #2dd4bf; font-size: 11px; font-weight: 500;">运行中</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 知识图谱引擎卡片
            st.markdown("""
            <div class="glass-card" style="margin-bottom: 16px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <span style="color: #9ca3af; font-size: 12px;">知识图谱引擎</span>
                    <i class="ph ph-brain" style="color: #2dd4bf; font-size: 20px;"></i>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: white; margin-bottom: 12px;">就绪</div>
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 故障知识库卡片
            st.markdown("""
            <div class="glass-card" style="margin-bottom: 24px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                    <span style="color: #9ca3af; font-size: 12px;">故障知识库</span>
                    <i class="ph ph-database" style="color: #2dd4bf; font-size: 20px;"></i>
                </div>
                <div style="font-size: 20px; font-weight: 700; color: white; margin-bottom: 8px;">已连接</div>
                <div style="font-size: 12px; color: #6b7280;">
                    <span style="color: #9ca3af; font-weight: 500;">12,847</span> 条历史记录
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 知识图谱分析原生组合实现面板 (避免 Markdown 截断导致样式崩坏)
            kg_container = st.container()
            with kg_container:
                st.markdown("""
                <div style="background: rgba(20, 184, 166, 0.05); border: 1px solid rgba(20, 184, 166, 0.2); border-radius: 12px 12px 0 0; border-bottom: none; padding: 16px 16px 0 16px;">
                </div>
                """, unsafe_allow_html=True)
                
                # 在流式布局中放入并排的内容与 Switch
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown("""
                    <div style="display: flex; align-items: center; gap: 8px; margin-left: 16px; margin-top: -10px;">
                        <i class="ph ph-graph" style="color: #2dd4bf; font-size: 18px;"></i>
                        <span style="color: #e5e7eb; font-size: 14px; font-weight: 500;">知识图谱分析</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    # Switch 紧贴右侧
                    st.session_state.use_kg = st.toggle("KG", value=st.session_state.use_kg, key="kg_toggle", label_visibility="collapsed")
                
                st.markdown("""
                <div style="background: rgba(20, 184, 166, 0.05); border: 1px solid rgba(20, 184, 166, 0.2); border-radius: 0 0 12px 12px; border-top: none; padding: 0 16px 16px 16px;">
                    <p style="color: #6b7280; font-size: 11px; line-height: 1.5; margin: 0;">
                        启用后，AI将结合船舶历史维护记录、设备关联图谱进行多维度故障推理
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # 居底帮助按钮
            st.markdown("<div style='height: 40vh;'></div>", unsafe_allow_html=True)
            if st.button("ℹ️ 帮助与文档", key="help_btn", use_container_width=True):
                st.info("帮助文档功能开发中...")

# 运行应用
if __name__ == "__main__":
    render_chat()