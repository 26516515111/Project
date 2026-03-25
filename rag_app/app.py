# app.py
import streamlit as st
from auth import render_login
from chat import render_chat
from settings import render_settings
from rag.pipeline import RagPipeline 

# ==========================================
# 1. 核心配置与初始化
# ==========================================
st.set_page_config(
    page_title="DeepBlue - 智能船舶诊断",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "history" not in st.session_state:
    st.session_state.history = []
if "sessions" not in st.session_state:
    st.session_state.sessions = [] # 用于存储所有的历史对话 [{id: str, title: str, history: list}]
if "current_session_id" not in st.session_state:
    import uuid
    st.session_state.current_session_id = str(uuid.uuid4())
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RagPipeline()
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_123"
if "user_info" not in st.session_state:
    st.session_state.user_info = {"name": "访客", "avatar": "👤", "user_id": "N/A"}
if "use_kg" not in st.session_state:
    st.session_state.use_kg = True

# ==========================================
# 2. 页面路由
# ==========================================
if not st.session_state.logged_in: 
    render_login()
else:
    if st.session_state.page == "chat": 
        render_chat()
    elif st.session_state.page == "settings": 
        render_settings()