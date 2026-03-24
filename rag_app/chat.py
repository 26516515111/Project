# chat.py
import html
import time
import streamlit as st
from rag.schema import QueryRequest


def _render_bubble(role: str, content: str, user_avatar: str = "👤", bot_avatar: str = "⚓"):
    cls = "user" if role == "user" else "bot"
    avatar = user_avatar if cls == "user" else bot_avatar
    safe = html.escape(content).replace("\n", "<br>")

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
                <div class="msg-bubble bot">{safe}</div>
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
            max-width: 1120px !important;
            min-height: calc(100vh - 64px);
            display: flex;
            flex-direction: column;
            padding-bottom: 160px !important; 
        }

        /* 强制覆盖全部的表单样式，使得它看起来像个搜索框 */
        [data-testid="stForm"] {
            border: 1px solid #e5e7eb !important;
            border-radius: 14px !important;
            padding: 0.5rem !important;
            background: #fff !important;
            box-shadow: 0 6px 16px rgba(0,0,0,.05) !important;
            width: min(860px, 100%) !important;
            margin: 0 auto !important;
        }
        
        [data-testid="stFormSubmitButton"] button {
            min-width: 44px !important;
            width: 44px !important;
            height: 42px !important;
            border-radius: 10px !important;
            padding: 0 !important;
            font-size: 20px !important;
            font-weight: 700 !important;
            line-height: 1 !important;
        }

        div[data-testid="stPopover"] {
            position: fixed !important;
            top: 15px !important;
            right: 20px !important;
            z-index: 999999 !important;
            width: auto !important;
        }

        [data-testid="stSidebar"] { background-color: #f9f9f9; }
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border: none;
            background-color: transparent;
            color: #333;
            text-align: left;
            justify-content: flex-start;
            padding: 4px 10px !important;
            border-radius: 8px;
            font-size: 13px !important;
            min-height: 34px !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover { background-color: #ededed; }
        .sidebar-title {
            font-size: 12px;
            color: #888;
            font-weight: 600;
            margin-top: 18px;
            margin-bottom: 8px;
            padding-left: 12px;
        }

        .chat-wrap {
            max-width: 860px;
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
        .msg-avatar.user { background: #fff7ed; }

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
            background: #fff1f2;
            border-color: #ffe4e6;
            color: #111827;
        }
        .msg-bubble.bot {
            background: #f8fafc;
            color: #111827;
        }

        .hero-wrap { text-align: center; margin-bottom: 2rem; }
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

        /* 参考来源区域居中 */
        div[data-testid="stExpander"] {
            max-width: 860px;
            margin: 0 auto 0.7rem auto;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 右上角用户菜单
    with st.popover(f"{user_avatar} {user_name}", use_container_width=False):
        st.markdown(f"**工号**: {user.get('user_id', 'N/A')}")
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
        if st.button("➕ 新建对话", type="primary"):
            st.session_state.history = []
            st.rerun()

        st.markdown('<div class="sidebar-title">过去 7 天</div>', unsafe_allow_html=True)
        st.button("解决Streamlit启动错误")
        st.button("主机排气温度高分析")

        st.markdown('<div class="sidebar-title">一月</div>', unsafe_allow_html=True)
        st.button("主配电板绝缘低处理")
        st.button("全面从严治党核心要点")


    # ==========================
    # 初始态：输入居中
    # ==========================
    if not st.session_state.history:
        # 使用原生的空行往下推，制造居中视觉，而不是用断开的 div
        st.write("\n" * 5)
        st.markdown(
            f"""
            <div class="hero-wrap">
                <div class="hero-logo"><span>{bot_avatar}</span></div>
                <div style="font-size: 30px; font-weight: 700; color: #111827;">
                    👋 欢迎回来，{html.escape(user_name)}
                </div>
                <div style="font-size: 15px; color: #6b7280; margin-top: 8px;">
                    有什么可以帮您？我可以协助诊断主机、发电机及辅助设备故障。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        init_q = _render_icon_input(
            form_key="init_chat_form",
            placeholder="在此输入故障现象，例如：柴油机排气温度高..."
        )
        _render_kg_toggle()

        if init_q is not None:
            _submit_question(init_q)

    # ==========================
    # 会话态：消息在上，输入区沉底
    # ==========================
    else:
        for msg in st.session_state.history:
            _render_bubble(msg["role"], msg["content"], user_avatar=user_avatar, bot_avatar=bot_avatar)

            if msg["role"] == "assistant" and "raw_answer" in msg:
                ans_obj = msg["raw_answer"]
                if ans_obj.citations or ans_obj.kg_triplets:
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

                        if ans_obj.kg_triplets:
                            st.markdown("**🕸️ 知识图谱路径:**")
                            for t in ans_obj.kg_triplets:
                                rel = t.get("rel", "") or t.get("relation", "")
                                st.markdown(f"- `{t.get('head', '')}` ➡️ **{rel}** ➡️ `{t.get('tail', '')}`")

        if st.session_state.history and st.session_state.history[-1]["role"] == "user":
            latest_question = st.session_state.history[-1]["content"]

            with st.spinner("正在检索维修手册与图谱库..."):
                req = QueryRequest(
                    question=latest_question.strip(),
                    top_k=3,
                    use_kg=st.session_state.get("use_kg", True),
                    session_id=st.session_state.get("session_id", "default_user_123"),
                )
                ans = st.session_state.pipeline.query(req)

            placeholder = st.empty()
            curr_text = ""
            for ch in ans.answer:
                curr_text += ch
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
                time.sleep(0.01)

            _render_bubble("assistant", ans.answer, user_avatar=user_avatar, bot_avatar=bot_avatar)
            st.session_state.history.append({"role": "assistant", "content": ans.answer, "raw_answer": ans})
            st.rerun()

        # 对会话区的输入框启用悬浮底部的 CSS (直接包裹，不分割 form)
        st.markdown(
            """
            <style>
            /* 沉底输入区的特殊处理 */
            div[data-testid="stVerticalBlock"] > div:last-child {
                position: fixed;
                bottom: 0px;
                left: 50%;
                transform: translateX(-50%);
                width: 100%;
                max-width: 860px;
                z-index: 100;
                padding: 10px 10px 20px 10px;
                background: linear-gradient(to top, #ffffff 85%, rgba(255,255,255,0));
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        chat_q = _render_icon_input(
            form_key="chat_input_form",
            placeholder="在此输入故障现象，例如：柴油机排气温度高..."
        )
        _render_kg_toggle()

        if chat_q is not None:
            _submit_question(chat_q)