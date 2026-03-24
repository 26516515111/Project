# settings.py
import streamlit as st

def db_update_user(user_id, new_name, new_password):
    """【预留接口】在此处连接数据库更新用户信息"""
    # 模拟更新成功
    return True

def render_settings():
    user_info = st.session_state.get("user_info", {})

    st.markdown("""
    <style>
        .settings-wrap { max-width: 760px; margin: 0.5rem auto 0 auto; }
        .settings-desc { color: #6b7280; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="settings-wrap">', unsafe_allow_html=True)
    st.markdown("## ⚙️ 个人设置")
    st.markdown('<div class="settings-desc">修改显示姓名与登录密码</div>', unsafe_allow_html=True)

    with st.container(border=True):
        new_name = st.text_input("显示姓名", value=user_info.get("name", ""))
        new_pwd = st.text_input("新密码（不修改请留空）", type="password")
        new_pwd_confirm = st.text_input("确认新密码", type="password")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("保存修改", type="primary", use_container_width=True):
                if new_pwd and new_pwd != new_pwd_confirm:
                    st.error("两次输入的密码不一致！")
                else:
                    success = db_update_user(user_info.get("user_id"), new_name, new_pwd)
                    if success:
                        st.session_state.user_info["name"] = new_name
                        st.success("修改成功！")
        with c2:
            if st.button("返回主界面", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)