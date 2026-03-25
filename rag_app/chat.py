# chat.py
import html
import time
import streamlit as st
import streamlit.components.v1 as components
from rag.schema import QueryRequest
import uuid

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