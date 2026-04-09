"""Chat UI rendering with KG-aware rich chunk formatting."""

import html
import base64
import time
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from rag.schema import QueryRequest
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEADING_PATTERN = re.compile(r"^\[(H[1-6])\]\s*(.+?)\s*$")
IMAGE_ATTACHMENT_PATTERN = re.compile(r"^\[(?:IMG|图片附件)\s*(.+?)\]\s*$")


def _resolve_attachment_path(raw_ref: str) -> Path | None:
    ref = (raw_ref or "").strip()
    if not ref:
        return None
    if ref.startswith("path="):
        parts = ref.split(" alt=", 1)
        ref = parts[0].split("=", 1)[1].strip()
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    normalized = ref[2:] if ref.startswith("./") else ref
    resolved = PROJECT_ROOT / normalized
    if resolved.exists():
        return resolved
    if normalized.startswith("images/"):
        image_name = Path(normalized).name
        image_roots = [
            PROJECT_ROOT / "rag_app" / "data" / "KG" / "images",
            PROJECT_ROOT / "data" / "KG" / "images",
        ]
        for root in image_roots:
            if root.exists():
                matches = list(root.rglob(image_name))
                if matches:
                    return matches[0]
    return None


def _image_data_uri(image_path: Path) -> str | None:
    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix)
    if not mime:
        return None
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{mime};base64,{encoded}"


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("[H"):
        return False
    if stripped.count("|") < 1:
        return False
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
                f'<div class="chunk-attachment-label">图片附件</div>'
                f'<img class="chunk-image" src="{data_uri}" alt="{display_ref}" />'
                f'<div class="chunk-attachment-path">{display_ref}</div>'
                "</div>"
            )
    return (
        '<div class="chunk-attachment chunk-attachment-fallback">'
        '<div class="chunk-attachment-label">图片附件</div>'
        f'<div class="chunk-attachment-path">{display_ref}</div>'
        "</div>"
    )


def _render_structured_content(content: str) -> str:
    lines = content.splitlines()
    parts: list[str] = []
    paragraph_buffer: list[str] = []
    table_buffer: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        text = "<br>".join(html.escape(line) for line in paragraph_buffer)
        parts.append(f'<div class="chunk-paragraph">{text}</div>')
        paragraph_buffer = []

    def flush_table() -> None:
        nonlocal table_buffer
        if not table_buffer:
            return
        parts.append(_render_table(table_buffer))
        table_buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_table()
            flush_paragraph()
            continue

        heading_match = HEADING_PATTERN.match(stripped)
        if heading_match:
            flush_table()
            flush_paragraph()
            level, title = heading_match.groups()
            parts.append(
                f'<div class="chunk-heading {level.lower()}">{html.escape(title)}</div>'
            )
            continue

        attachment_match = IMAGE_ATTACHMENT_PATTERN.match(stripped)
        if attachment_match:
            flush_table()
            flush_paragraph()
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
    """将当前对话保存到会话列表中"""
    if not st.session_state.history:
        return
    # 获取标题（取第一个用户问题的前15个字）
    title = "新对话"
    for msg in st.session_state.history:
        if msg["role"] == "user":
            title = msg["content"][:15] + ("..." if len(msg["content"]) > 15 else "")
            break
            
    # 更新或追加到 sessions
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

def _render_bubble(role: str, content: str, user_avatar: str = "👤", bot_avatar: str = "⚓"):
    cls = "user" if role == "user" else "bot"
    avatar = user_avatar if cls == "user" else bot_avatar
    safe = (
        html.escape(content).replace("\n", "<br>")
        if cls == "user"
        else _render_structured_content(content)
    )

    if cls == "user":
        body = f"""
        <div class="chat-wrap">
            <div class="msg-row user">
                <div class="msg-bubble user">{safe}</div>
                <div class="msg-avatar user">{avatar}</div>
            </div>
        </div>
        """
    else:
        body = f"""
        <div class="chat-wrap">
            <div class="msg-row bot">
                <div class="msg-avatar bot">{avatar}</div>
                <div class="msg-bubble bot rich-content">{safe}</div>
            </div>
        </div>
        """
    st.markdown(body, unsafe_allow_html=True)


def _submit_question(question: str):
    q = (question or "").strip()
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        st.rerun()


def _render_icon_input(form_key: str, placeholder: str):
    # 去除断开的 div 包裹，直接使用原生组件
    with st.form(form_key, clear_on_submit=True):
        c1, c2 = st.columns([12, 1])
        with c1:
            q = st.text_input(
                "输入问题",
                label_visibility="collapsed",
                placeholder=placeholder,
            )
        with c2:
            submitted = st.form_submit_button("➤", type="primary", use_container_width=True)
    return q if submitted else None


def _render_kg_toggle():
    # 去除断开的 div 包裹，使用 st 列布局居中
    l, c, r = st.columns([1.2, 2.0, 0.8])
    with c:
        st.checkbox("启用知识图谱进行深度故障诊断", key="use_kg")


def render_chat():
    user = st.session_state.get("user_info", {"name": "用户", "avatar": "👤", "user_id": "N/A"})
    user_name = user.get("name", "用户")
    user_avatar = user.get("avatar", "👤")
    bot_avatar = "⚓"

    st.markdown(
        """
    <style>
        header[data-testid="stHeader"] { background: transparent !important; }
        header[data-testid="stHeader"]::before { display: none !important; }
        .stAppDeployButton, [data-testid="stToolbar"], [data-testid="stActionElements"] { display: none !important; }
        footer { visibility: hidden; }

        .block-container {
            max-width: 100% !important; /* 占满剩余空间 */
            height: calc(100vh - 110px) !important; /* 限制高度，为底部输入框预留约110px的计算空间 */
            overflow-y: auto !important; /* 内容过多时自动出现内部滚动条 */
            padding-bottom: 130px !important; /* 增加底部防遮挡内边距，取代原本的生硬占位块 */
            padding-top: 1rem !important;
        }

        /* 强制覆盖主区域的表单样式，使得它固定在底部不动 */
        .block-container [data-testid="stForm"] {
            position: fixed !important;
            bottom: 40px !important;
            left: calc(50% + 125px) !important; /* 假设侧边栏宽度250px，保持主视图居中 */
            transform: translateX(-50%) !important;
            z-index: 100 !important;
            border: 1px solid #bae6fd !important; /* 淡蓝色边框 */
            border-radius: 20px !important;
            padding: 0.2rem 0.5rem !important;
            background: #f8fafc !important; /* 淡蓝底色 */
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1) !important;
            width: min(1000px, 90%) !important;
            margin: 0 auto !important;
        }
        
        [data-testid="stFormSubmitButton"] button {
            min-width: 44px !important;
            width: 44px !important;
            height: 42px !important;
            border-radius: 50% !important; /* 圆形按钮 */
            padding: 0 !important;
            font-size: 18px !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            background-color: #3b82f6 !important; /* 默认蓝色按钮 */
            color: white !important;
            border: none !important;
        }

        /* 给暂停按钮特制的红色样式，利用它的内容区分，保持圆形 */
        [data-testid="stFormSubmitButton"] button:has(p:contains("⏹")) {
            background-color: #ef4444 !important; /* 红色按钮 */
        }

        /* 隐藏输入框内的 "Press Enter to submit form" 提示语 */
        div[data-testid="InputInstructions"] {
            display: none !important;
        }

        div[data-testid="stPopover"] {
            position: fixed !important;
            top: 15px !important;
            right: 20px !important;
            z-index: 999999 !important;
            width: auto !important;
        }

        /* 侧边栏样式调整：改成淡蓝色 */
        [data-testid="stSidebar"] { 
            background-color: #f0f9ff; /* 极淡的蓝色背景 */
            min-width: 250px !important;
            max-width: 250px !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border: none;
            background-color: #fff;
            color: #0284c7; /* 深蓝色文字 */
            text-align: center;
            justify-content: center;
            padding: 8px 10px !important;
            border-radius: 20px;
            font-size: 14px !important;
            font-weight: 600 !important;
            min-height: 40px !important;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        [data-testid="stSidebar"] .stButton > button:hover { background-color: #e0f2fe; }
        .sidebar-title {
            font-size: 14px;
            color: #0284c7;
            font-weight: 700;
            margin-top: 18px;
            margin-bottom: 8px;
            padding-left: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chat-wrap {
            max-width: 1000px; /* 增大会话区宽度 */
            width: 95%;
            margin: 0 auto 0.55rem auto;
        }
        .msg-row {
            display: flex;
            width: 100%;
            align-items: flex-start;
            gap: 10px;
        }
        .msg-row.user { justify-content: flex-end; }
        .msg-row.bot  { justify-content: flex-start; }

        .msg-avatar {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
            border: 1px solid #e5e7eb;
            background: #fff;
            margin-top: 2px;
        }
        .msg-avatar.bot  { background: #eff6ff; }
        .msg-avatar.user { background: #e0f2fe; }

        .msg-bubble {
            max-width: 78%;
            padding: 0.68rem 0.92rem;
            border-radius: 14px;
            border: 1px solid #e5e7eb;
            line-height: 1.6;
            word-break: break-word;
            white-space: normal;
        }
        .msg-bubble.user {
            background: #e0f2fe; /* 淡蓝色泡泡 */
            border-color: #bae6fd;
            color: #111827;
        }
        .msg-bubble.bot {
            background: #f8fafc;
            color: #111827;
        }
        .msg-bubble.rich-content .chunk-heading {
            font-weight: 700;
            color: #0f172a;
            margin: 0.35rem 0 0.45rem 0;
            line-height: 1.35;
        }
        .msg-bubble.rich-content .h1 { font-size: 1.1rem; color: #0c4a6e; }
        .msg-bubble.rich-content .h2 { font-size: 1.02rem; color: #075985; }
        .msg-bubble.rich-content .h3 { font-size: 0.98rem; color: #0369a1; }
        .msg-bubble.rich-content .h4,
        .msg-bubble.rich-content .h5,
        .msg-bubble.rich-content .h6 {
            font-size: 0.94rem;
            color: #334155;
        }
        .msg-bubble.rich-content .chunk-paragraph {
            margin: 0.28rem 0 0.55rem 0;
        }
        .msg-bubble.rich-content .chunk-table-wrap {
            overflow-x: auto;
            margin: 0.45rem 0 0.7rem 0;
            border: 1px solid #dbeafe;
            border-radius: 10px;
            background: #ffffff;
        }
        .msg-bubble.rich-content .chunk-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.93rem;
        }
        .msg-bubble.rich-content .chunk-table th,
        .msg-bubble.rich-content .chunk-table td {
            padding: 0.45rem 0.6rem;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
            text-align: left;
        }
        .msg-bubble.rich-content .chunk-table th {
            background: #eff6ff;
            color: #0f172a;
            font-weight: 700;
        }
        .msg-bubble.rich-content .chunk-table tr:last-child td {
            border-bottom: none;
        }
        .msg-bubble.rich-content .chunk-attachment {
            margin: 0.45rem 0 0.8rem 0;
            padding: 0.55rem;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            background: #ffffff;
        }
        .msg-bubble.rich-content .chunk-attachment-label {
            font-size: 0.78rem;
            font-weight: 700;
            color: #0369a1;
            margin-bottom: 0.35rem;
        }
        .msg-bubble.rich-content .chunk-image {
            display: block;
            max-width: 100%;
            max-height: 320px;
            margin: 0 auto;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            background: #fff;
        }
        .msg-bubble.rich-content .chunk-attachment-path {
            margin-top: 0.35rem;
            color: #64748b;
            font-size: 0.78rem;
            word-break: break-all;
        }

        .hero-wrap { text-align: center; margin-bottom: 2rem; margin-top: 15vh;}
        .hero-logo {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 12px auto;
        }
        .hero-logo span {
            font-size: 86px;
            line-height: 1;
            filter: drop-shadow(0 8px 10px rgba(0,0,0,0.1));
        }

        /* 修正 Streamlit 原生组件（Expander, Spinner）的对齐，使其与自定义聊天气泡贴合 */
        div[data-testid="stSpinner"],
        div[data-testid="stExpander"] {
            max-width: 1000px !important; /* 匹配 .chat-wrap 的控制宽度 */
            width: 95% !important;
            margin: 0 auto !important;
            padding: 0 !important; /* 移除外层强制 padding，避免宽度计算叠加 */
        }
        
        div[data-testid="stSpinner"] {
            margin-bottom: 10px !important;
            padding-left: 44px !important;  /* Spinner 保留直接内边距 */
        }

        div[data-testid="stExpander"] {
            margin-bottom: 0.7rem !important;
        }

        /* 限定展开框宽度：直接使用 width = 78% 以对齐满宽的 msg-bubble，同时用 margin 偏移头像位置 */
        div[data-testid="stExpander"] > details {
            width: 78% !important;
            max-width: 78% !important;
            margin-left: 44px !important; /* 34px(头像) + 10px(gap) = 44px 偏移量 */
            background-color: #f8fafc !important;
            border-radius: 14px !important;
            border: 1px solid #bae6fd !important; /* 边框色带点淡蓝以融合主题 */
        }
        
        div[data-testid="stExpander"] summary {
            border-radius: 14px !important;
        }
        
        .warning-text {
            position: fixed !important;
            bottom: 15px !important;
            left: calc(50% + 125px) !important;
            transform: translateX(-50%) !important;
            z-index: 100 !important;
            text-align: center;
            font-size: 12px;
            color: #aaa;
            width: 100%;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 注入全局 MutationObserver 实现平滑且实时同步的自适应上滑
    components.html(
        """
        <script>
            const parent = window.parent.document;
            const container = parent.querySelector('.block-container');
            if (container) {
                const observer = new MutationObserver(() => {
                    container.scrollTop = container.scrollHeight;
                });
                observer.observe(container, { childList: true, subtree: true, characterData: true });
                container.scrollTop = container.scrollHeight;
            }
        </script>
        """,
        height=0,
        width=0,
    )

    # 顶部状态栏
    st.markdown("""
        <div style="position: absolute; top: 15px; left: 20px; color: #10b981; font-size: 14px; display: flex; align-items: center; gap: 5px;">
            <div style="width: 8px; height: 8px; background-color: #10b981; border-radius: 50%;"></div>
            DeepBlue在线中...
        </div>
    """, unsafe_allow_html=True)


    # 右上角用户菜单
    with st.popover("...", use_container_width=False):
        st.markdown(f"**{user_name}** ({user.get('user_id', 'N/A')})")
        st.divider()
        if st.button("⚙️ 个人设置", use_container_width=True):
            st.session_state.page = "settings"
            st.rerun()
        if st.button("退出登录", type="primary", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

    # 侧边栏
    with st.sidebar:
        st.markdown(f"<div class='sidebar-title' style='font-size: 18px; margin-bottom: 2rem;'>{bot_avatar} DeepBlue助手</div>", unsafe_allow_html=True)
        
        if st.button("＋ 新建会话"):
            _save_current_session() 
            st.session_state.history = []
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.is_generating = False
            st.rerun()

        st.markdown('<div class="sidebar-title">⏱️ 历史记录</div>', unsafe_allow_html=True)
        
        if not st.session_state.sessions and not st.session_state.history:
            st.markdown("<div style='padding-left: 12px; color: #aaa; font-size: 14px;'>暂无历史记录</div>", unsafe_allow_html=True)
        else:
            for session_info in reversed(st.session_state.sessions):
                is_current = (session_info["id"] == st.session_state.current_session_id)
                btn_type = "primary" if is_current else "secondary"
                
                if st.button(session_info["title"], key=f"hist_{session_info['id']}", type=btn_type):
                    _save_current_session()
                    st.session_state.history = session_info["history"].copy()
                    st.session_state.current_session_id = session_info["id"]
                    st.session_state.is_generating = False
                    st.rerun()
        
        st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True) 
        st.markdown('<div class="sidebar-title" style="margin-top: 50px;">⚙️ 设置</div>', unsafe_allow_html=True)
        st.checkbox("启用知识图谱进行深度故障诊断", key="use_kg")


    # ==========================
    # 初始态：渲染Logo
    # ==========================
    if not st.session_state.history:
        st.markdown(
            f"""
            <div class="hero-wrap">
                <div class="hero-logo"><span>{bot_avatar}</span></div>
                <div style="font-size: 20px; font-weight: 700; color: #333; margin-top: 10px;">
                    你好呀！我是DeepBlue
                </div>
                <div style="font-size: 14px; color: #888; margin-top: 8px;">
                    我可以帮你诊断故障、查阅手册，或者只是陪你聊天~ ⚓！
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================
    # 会话态：消息输出区
    # ==========================
    if st.session_state.history:
        for msg in st.session_state.history:
            _render_bubble(msg["role"], msg["content"], user_avatar=user_avatar, bot_avatar=bot_avatar)
            
            if msg["role"] == "assistant" and "raw_answer" in msg and msg["raw_answer"]:
                ans_obj = msg["raw_answer"]
                if hasattr(ans_obj, "citations") and (ans_obj.citations or ans_obj.kg_triplets):
                    with st.expander("📚 查看参考来源与图谱节点", expanded=False):
                        if ans_obj.citations:
                            st.markdown("**📄 依据文档:**")
                            for c in ans_obj.citations:
                                source = getattr(c, "source", None) or getattr(c, "text", "未知来源")
                                score = getattr(c, "score", None)
                                if score is not None:
                                    st.markdown(f"- {source} (相关度: {score:.2f})")
                                else:
                                    st.markdown(f"- {source}")
                                passage_text = getattr(c, "text", "") or ""
                                if passage_text:
                                    st.markdown(
                                        (
                                            '<div class="chat-wrap" style="width:100%; margin:0.35rem 0 0.8rem 0;">'
                                            '<div class="msg-row bot">'
                                            '<div class="msg-bubble bot rich-content" '
                                            'style="max-width:100%; width:100%;">'
                                            f"{_render_structured_content(passage_text)}"
                                            "</div></div></div>"
                                        ),
                                        unsafe_allow_html=True,
                                    )
                        
                        # 新增：展示 Meta 元数据信息
                        if hasattr(ans_obj, "meta") and ans_obj.meta:
                            st.markdown("**⚙️ 检索配置:**")
                            meta_info = " | ".join([f"{k}: {v}" for k, v in ans_obj.meta.items()])
                            st.caption(f"_{meta_info}_")

                        if ans_obj.kg_triplets:
                            st.markdown("**🕸️ 知识图谱路径:**")
                            for t in ans_obj.kg_triplets:
                                head = t.get("head", "")
                                rel = t.get("rel", "") or t.get("relation", "")
                                tail = t.get("tail", "")
                                desc = t.get("description", "")
                                
                                # 将三元组关系与详细描述组合展示
                                if desc:
                                    st.markdown(
                                        f"- `{head}` ➡️ **{rel}** ➡️ `{tail}` <br>"
                                        f"<span style='color:#888; font-size:12px; margin-left:15px;'>📝 {desc}</span>", 
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(f"- `{head}` ➡️ **{rel}** ➡️ `{tail}`")

    # ==========================
    # 底部输入/中断核心逻辑区
    # ==========================
    if st.session_state.get("is_generating", False):
        
        # 使用 form 固定在底部的“思考中/停止”区 (CSS会让它保持在旧输入框的位置)
        with st.form("stop_form", clear_on_submit=False):
            c1, c2 = st.columns([12, 1]) # 和正常输入框列宽比例保持一致
            with c1:
                st.text_input("提示", value="DeepBlue 正在思考中...", disabled=True, label_visibility="collapsed")
            with c2:
                stop_btn = st.form_submit_button("⏹", type="primary", use_container_width=True)
        
        st.markdown("<div class='warning-text'>AI 生成的内容可能包含错误，请仔细甄别。</div>", unsafe_allow_html=True)

        # 1. 捕捉到了中断点击（通过直接判断 Streamlit 重新执行传入的表单结果）
        if stop_btn:
            st.session_state.is_generating = False
            ans_text = st.session_state.get("partial_content", "")
            ans_text += "\n\n**（用户终止生成！）**"
            ans_obj = st.session_state.get("partial_raw", None)
            
            st.session_state.history.append({"role": "assistant", "content": ans_text, "raw_answer": ans_obj})
            st.session_state.partial_content = ""
            st.session_state.partial_raw = None
            st.session_state.generation_started = False
            _save_current_session()
            st.rerun()

        # 2. 正常流：尚未彻底完成时，继续生成逻辑
        elif not st.session_state.get("generation_started", False):
            st.session_state.generation_started = True
            
            latest_question = st.session_state.history[-1]["content"]
            placeholder = st.empty()
            curr_text = ""
            
            with st.spinner("正在检索维修手册与图谱库..."):
                # 获取复选框当前状态
                is_kg_enabled = st.session_state.get("use_kg", True)
                
                req = QueryRequest(
                    question=latest_question.strip(),
                    top_k=3,
                    use_kg=is_kg_enabled,
                    # 补充动态切换后端常用的 retriever 参数（如果你的后端需要此字段）
                    retriever="hybrid" if is_kg_enabled else "vector", 
                    session_id=st.session_state.get("session_id", "default_user_123"),
                )
                ans = st.session_state.pipeline.query(req)
                
            st.session_state.partial_raw = ans 

            try:
                for ch in ans.answer:
                    curr_text += ch
                    st.session_state.partial_content = curr_text # 实时保存，防止中断时为空
                    
                    safe = html.escape(curr_text).replace("\n", "<br>")
                    
                    placeholder.markdown(
                        f"""
                        <div class="chat-wrap">
                            <div class="msg-row bot">
                                <div class="msg-avatar bot">{bot_avatar}</div>
                                <div class="msg-bubble bot">{safe}▌</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.02)
                    
            except BaseException as e:
                # 捕获因点击停止按钮等造成的 Streamlit rerun 异常向上抛出允许中断页面重绘
                raise e

            # 如未被中断而是正常完成
            st.session_state.is_generating = False
            st.session_state.generation_started = False
            st.session_state.partial_content = ""
            st.session_state.partial_raw = None
            st.session_state.history.append({"role": "assistant", "content": curr_text, "raw_answer": ans})
            _save_current_session()
            st.rerun()

    else:
        # 常态：渲染普通输入框（CSS也会把它锁在同一个位置不会乱跑）
        chat_q = _render_icon_input(
            form_key="chat_global_input",
            placeholder="和DeepBlue说点什么吧... (Shift+Enter 换行)"
        )
        st.markdown("<div class='warning-text'>AI 生成的内容可能包含错误，请仔细甄别。</div>", unsafe_allow_html=True)

        if chat_q:
            st.session_state.history.append({"role": "user", "content": chat_q})
            st.session_state.is_generating = True
            st.session_state.generation_started = False
            st.session_state.partial_content = ""
            st.session_state.partial_raw = None
            st.rerun()
