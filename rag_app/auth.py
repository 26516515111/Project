# auth.py
import streamlit as st
import base64
from pathlib import Path

def db_verify_user(username, password):
    if username == "admin" and password == "123456":
        return {"name": "王轮机长", "avatar": "👨‍✈️", "user_id": "882103"}
    return None

def get_local_image_base64(image_path):
    """读取本地图片并转换为 base64"""
    try:
        # 获取图片的绝对路径
        img_path = Path(__file__).parent / image_path
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        st.warning(f"⚠️ 背景图片未找到：{image_path}，将使用默认背景")
        return None

def render_login():
    # 使用正确的 CSS 注入方式
    bg_image_base64 = get_local_image_base64("images/background.webp")
    
    # 构建背景 CSS
    if bg_image_base64:
        bg_css = f"""
            background:
                linear-gradient(rgba(7, 20, 38, 0.55), rgba(7, 20, 38, 0.55)),
                url("data:image/webp;base64,{bg_image_base64}") center/cover no-repeat fixed;
        """
    else:
        bg_css = """
            background:
                linear-gradient(rgba(7, 20, 38, 0.55), rgba(7, 20, 38, 0.55)),
                url("https://images.unsplash.com/photo-1559620065-9839423c8612?q=80&w=2670&auto=format&fit=crop") center/cover no-repeat fixed;
        """

    st.markdown(
        f"""
        <style>
            .stApp {{ {bg_css} }}

            header[data-testid="stHeader"] {{ background: transparent !important; }}
            header[data-testid="stHeader"]::before {{ display: none !important; }}
            .stAppDeployButton, [data-testid="stToolbar"], [data-testid="stActionElements"] {{ display: none !important; }}
            footer {{ visibility: hidden; }}
            [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}

            .login-panel {{
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #e5e7eb;
                border-radius: 16px;
                padding: 1.4rem;
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
            }}
            .brand-wrap {{
                color: #f8fafc;
                padding: 3rem 1rem 1rem 1rem;
            }}
            .brand-title {{
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }}
            .brand-sub {{
                font-size: 1rem;
                opacity: 0.9;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="brand-wrap">
                <div class="brand-title">⚓ DeepBlue</div>
                <div class="brand-sub">船舶装备故障诊断系统 V3.0</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div style="height: 10vh;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="login-panel">', unsafe_allow_html=True)
        st.markdown("### 登录系统")
        st.caption("请输入账号和密码")

        with st.form("login_form"):
            username = st.text_input("账号 / Username", placeholder="请输入账号")
            password = st.text_input("密码 / Password", type="password", placeholder="请输入密码")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.form_submit_button("登 录 / SIGN IN", type="primary", use_container_width=True):
                user_data = db_verify_user(username, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user_data
                    st.session_state.page = "chat"
                    st.rerun()
                else:
                    st.error("账号或密码错误！")
        st.markdown("</div>", unsafe_allow_html=True)