import streamlit as st
import base64
from pathlib import Path


def db_verify_user(username, password):
    """验证用户身份 - 保持接口不变"""
    if username == "admin" and password == "123456":
        return {"name": "王轮机长", "avatar": "👨‍✈️", "user_id": "882103"}
    return None


# def get_local_image_base64(image_path):
#     """读取本地图片并转换为 base64 - 保持接口不变"""
#     try:
#         img_path = Path(__file__).parent / image_path
#         with open(img_path, "rb") as image_file:
#             encoded_string = base64.b64encode(image_file.read()).decode()
#         return encoded_string
#     except FileNotFoundError:
#         return None


def render_login():
    """渲染动态海洋主题登录界面 - 保持接口不变"""
    
    # 隐藏Streamlit默认元素并修复全屏样式
    st.markdown("""
    <style>
        header[data-testid="stHeader"] { display: none !important; }
        .stAppDeployButton, [data-testid="stToolbar"] { display: none !important; }
        footer { visibility: hidden; }
        [data-testid="stSidebar"] { display: none !important; }
        .stApp { background: #041527; overflow: hidden; }
        .block-container { 
            max-width: 100vw !important; 
            padding: 0 !important; 
            margin: 0 !important;
        }
        /* 使内部的 HTML 撑满整个屏幕 */
        iframe { 
            border: none !important; 
            width: 100vw !important; 
            height: 100vh !important; 
            position: fixed !important; 
            top: 0 !important; 
            left: 0 !important; 
            z-index: 9999;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 动态海洋登录界面HTML (与选中设计完全一致)
    login_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>智能船舶问答系统 - Intelligent Ship Q&A System</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/@phosphor-icons/web"></script>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        fontFamily: { sans: ['"Noto Sans SC"', 'sans-serif'] },
                        colors: {
                            ocean: { 900: '#041527', 800: '#0a2540', 700: '#11355a' },
                            teal: { 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488' }
                        }
                    }
                }
            }
        </script>
        <style>
            body {
                margin: 0;
                overflow: hidden;
                background: linear-gradient(135deg, #041527 0%, #0a2540 50%, #0d4a3a 100%);
                font-family: 'Noto Sans SC', sans-serif;
            }
            
            .glass-panel {
                background: rgba(10, 37, 64, 0.5);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(45, 212, 191, 0.2);
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
            }
            
            .glass-input {
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: white;
                transition: all 0.3s ease;
                padding: 14px 14px 14px 48px;
                border-radius: 12px;
                width: 100%;
                font-size: 14px;
                outline: none;
            }
            
            .glass-input::placeholder { color: rgba(156, 163, 175, 0.5); }
            
            .glass-input:focus {
                border-color: rgba(45, 212, 191, 0.7);
                box-shadow: 0 0 15px rgba(45, 212, 191, 0.2);
                background: rgba(0, 0, 0, 0.5);
            }
            
            .waves {
                position: absolute;
                width: 100%;
                height: 20vh;
                min-height: 120px;
                max-height: 250px;
                bottom: 0;
                left: 0;
                z-index: 10;
                pointer-events: none;
            }
            
            .parallax > use {
                animation: move-forever 25s cubic-bezier(.55,.5,.45,.5) infinite;
            }
            .parallax > use:nth-child(1) { animation-delay: -2s; animation-duration: 10s; }
            .parallax > use:nth-child(2) { animation-delay: -4s; animation-duration: 14s; }
            .parallax > use:nth-child(3) { animation-delay: -6s; animation-duration: 18s; }
            .parallax > use:nth-child(4) { animation-delay: -8s; animation-duration: 22s; }
            
            @keyframes move-forever {
                0% { transform: translate3d(-90px,0,0); }
                100% { transform: translate3d(85px,0,0); }
            }
            
            .float-anim { animation: floating 8s ease-in-out infinite; }
            .float-anim-slow { animation: floating 10s ease-in-out infinite; }
            
            @keyframes floating {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-20px); }
            }
            
            .radar-spin { animation: spin 6s linear infinite; }
            @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
            
            .particle {
                position: absolute;
                border-radius: 50%;
                background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.8), rgba(45, 212, 191, 0.2));
                box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.3);
                animation: rise linear infinite;
                bottom: -50px;
                pointer-events: none;
                z-index: 5;
            }
            
            @keyframes rise {
                0% { transform: translateY(0) scale(0.8) translateX(0); opacity: 0; }
                10% { opacity: 0.6; }
                50% { transform: translateY(-50vh) scale(1.2) translateX(20px); opacity: 0.4; }
                90% { opacity: 0; }
                100% { transform: translateY(-100vh) scale(1) translateX(-20px); opacity: 0; }
            }
            
            .glow-pulse { animation: pulse-glow 4s ease-in-out infinite; }
            @keyframes pulse-glow {
                0%, 100% { box-shadow: 0 0 30px rgba(45, 212, 191, 0.2); }
                50% { box-shadow: 0 0 50px rgba(45, 212, 191, 0.4); }
            }
            
            .btn-shine {
                position: relative;
                overflow: hidden;
            }
            
            .btn-shine::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 50%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                transform: skewX(45deg);
                animation: shine 3s infinite;
            }
            
            @keyframes shine {
                0%, 100% { left: -100%; }
                20% { left: 200%; }
            }
        </style>
    </head>
    <body class="text-white antialiased h-screen w-screen relative bg-gradient-to-br from-[#041527] via-[#0a2540] to-[#0d4a3a] overflow-hidden flex items-center justify-center">
        
        <!-- 雷达背景 -->
        <div class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full border border-teal-500/10 pointer-events-none z-0">
            <div class="absolute inset-0 rounded-full border border-teal-500/10 scale-75"></div>
            <div class="absolute inset-0 rounded-full border border-teal-500/10 scale-50"></div>
            <div class="absolute inset-0 rounded-full border border-teal-500/20 scale-25"></div>
            <div class="absolute inset-0 rounded-full radar-spin" style="background: conic-gradient(from 0deg, transparent 0deg, transparent 270deg, rgba(45, 212, 191, 0.15) 360deg);"></div>
            <div class="absolute top-0 bottom-0 left-1/2 w-px bg-teal-500/10"></div>
            <div class="absolute left-0 right-0 top-1/2 h-px bg-teal-500/10"></div>
        </div>
        
        <!-- 浮动图标 -->
        <div class="absolute right-[15%] top-[20%] opacity-10 pointer-events-none float-anim">
            <i class="ph ph-anchor text-[180px] text-teal-200"></i>
        </div>
        <div class="absolute left-[10%] bottom-[30%] opacity-10 pointer-events-none float-anim-slow">
            <i class="ph ph-compass text-[240px] text-blue-200"></i>
        </div>
        <div class="absolute right-[25%] bottom-[15%] opacity-5 pointer-events-none float-anim" style="animation-delay: 1s;">
            <i class="ph ph-paper-plane-tilt text-[120px] text-white"></i>
        </div>
        
        <!-- 粒子容器 -->
        <div id="particles-container" class="absolute inset-0 z-0"></div>
        
        <!-- 波浪 -->
        <svg class="waves" xmlns="http://www.w3.org/2000/svg" viewBox="0 24 150 28" preserveAspectRatio="none">
            <defs>
                <path id="gentle-wave" d="M-160 44c30 0 58-18 88-18s 58 18 88 18 58-18 88-18 58 18 88 18 v44h-352z"/>
            </defs>
            <g class="parallax">
                <use xlink:href="#gentle-wave" x="48" y="0" fill="rgba(13, 148, 136, 0.2)"/>
                <use xlink:href="#gentle-wave" x="48" y="3" fill="rgba(17, 53, 90, 0.4)"/>
                <use xlink:href="#gentle-wave" x="48" y="5" fill="rgba(45, 212, 191, 0.1)"/>
                <use xlink:href="#gentle-wave" x="48" y="7" fill="rgba(4, 21, 39, 0.8)"/>
            </g>
        </svg>
        
        <!-- 登录卡片 -->
        <div class="relative z-20 w-full max-w-[420px] px-6 float-anim">
            <div class="absolute -inset-0.5 bg-gradient-to-r from-teal-500 to-blue-500 rounded-3xl blur opacity-20 glow-pulse"></div>
            
            <div class="glass-panel relative rounded-3xl p-10 overflow-hidden">
                <!-- 装饰角 -->
                <div class="absolute top-0 left-0 w-16 h-16 border-t-2 border-l-2 border-teal-400/30 rounded-tl-3xl m-1 pointer-events-none"></div>
                <div class="absolute bottom-0 right-0 w-16 h-16 border-b-2 border-r-2 border-teal-400/30 rounded-br-3xl m-1 pointer-events-none"></div>
                
                <!-- 品牌区 -->
                <div class="text-center mb-8 relative">
                    <div class="relative inline-flex items-center justify-center w-24 h-24 rounded-full bg-[#0a2540]/80 border border-teal-500/40 mb-5 glow-pulse group">
                        <div class="absolute inset-0 rounded-full border border-teal-400/20 animate-ping" style="animation-duration: 3s;"></div>
                        <i class="ph ph-steering-wheel text-5xl text-teal-400 transition-transform duration-700 group-hover:rotate-180"></i>
                    </div>
                    <h1 class="text-3xl font-bold text-white tracking-wide mb-2">智能船舶问答系统</h1>
                    <p class="text-teal-200/60 text-xs font-medium tracking-widest uppercase">Intelligent Ship Q&A System</p>
                </div>
                
                <!-- 表单 -->
                <form id="loginForm" class="space-y-5 relative z-10">
                    
                    <!-- 用户名 -->
                    <div class="group">
                        <label class="block text-xs text-teal-200/80 mb-1.5 ml-1 font-medium tracking-wider">用户名 / USERNAME</label>
                        <div class="relative">
                            <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                <i class="ph ph-user text-teal-500/80 text-xl group-focus-within:text-teal-300 transition-colors"></i>
                            </div>
                            <input type="text" id="username" placeholder="请输入您的账号" class="glass-input" required>
                        </div>
                    </div>
                    
                    <!-- 密码 -->
                    <div class="group">
                        <label class="block text-xs text-teal-200/80 mb-1.5 ml-1 font-medium tracking-wider">密码 / PASSWORD</label>
                        <div class="relative">
                            <div class="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                <i class="ph ph-lock-key text-teal-500/80 text-xl group-focus-within:text-teal-300 transition-colors"></i>
                            </div>
                            <input type="password" id="password" placeholder="••••••••" class="glass-input pr-12 tracking-widest font-mono" required>
                            <button type="button" onclick="togglePassword()" class="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-teal-300 transition-colors">
                                <i id="eyeIcon" class="ph ph-eye-slash text-lg"></i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- 选项 -->
                    <div class="flex items-center justify-between text-sm mt-2">
                        <label class="flex items-center space-x-2 cursor-pointer group">
                            <input type="checkbox" id="remember" class="w-4 h-4 rounded border border-gray-500 bg-transparent accent-teal-500">
                            <span class="text-gray-300 text-xs group-hover:text-white transition-colors">记住密码</span>
                        </label>
                        <a href="#" class="text-xs text-teal-400/80 hover:text-teal-300 transition-colors border-b border-transparent hover:border-teal-300 pb-0.5">忘记密码?</a>
                    </div>
                    
                    <!-- 登录按钮 -->
                    <div class="pt-4">
                        <button type="submit" class="w-full btn-shine relative overflow-hidden rounded-xl bg-gradient-to-r from-teal-600 to-[#0a2540] text-white font-bold py-4 shadow-[0_8px_20px_rgba(13,148,136,0.3)] hover:shadow-[0_8px_25px_rgba(45,212,191,0.5)] transition-all duration-300 transform hover:-translate-y-0.5">
                            <span class="relative z-10 flex items-center justify-center gap-2 text-base tracking-widest">
                                登 录 系 统 <i class="ph ph-arrow-right text-lg font-bold"></i>
                            </span>
                        </button>
                    </div>
                </form>
                
                <!-- 底部信息 -->
                <div class="mt-8 pt-6 border-t border-white/5 text-center flex flex-col items-center">
                    <p class="text-[10px] text-gray-400 tracking-wider mb-2">Powered by Streamlit</p>
                    <div class="flex gap-4">
                        <div class="w-1.5 h-1.5 rounded-full bg-teal-500/50 animate-pulse"></div>
                        <div class="w-1.5 h-1.5 rounded-full bg-teal-500/50 animate-pulse" style="animation-delay: 0.2s;"></div>
                        <div class="w-1.5 h-1.5 rounded-full bg-teal-500/50 animate-pulse" style="animation-delay: 0.4s;"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // 生成气泡粒子
            const container = document.getElementById('particles-container');
            const particleCount = 25;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                const size = Math.random() * 8 + 4 + 'px';
                particle.style.width = size;
                particle.style.height = size;
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDuration = (Math.random() * 15 + 10) + 's';
                particle.style.animationDelay = Math.random() * 10 + 's';
                container.appendChild(particle);
            }
            
            // 密码可见切换
            function togglePassword() {
                const pwd = document.getElementById('password');
                const eye = document.getElementById('eyeIcon');
                if (pwd.type === 'password') {
                    pwd.type = 'text';
                    eye.classList.remove('ph-eye-slash');
                    eye.classList.add('ph-eye');
                } else {
                    pwd.type = 'password';
                    eye.classList.remove('ph-eye');
                    eye.classList.add('ph-eye-slash');
                }
            }
            
            // 表单提交处理 - 修改为无刷新事件推送
            document.getElementById('loginForm').addEventListener('submit', function(e) {
                e.preventDefault();
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                
                // 将数据写入父窗口URL但不执行 reload 刷新
                const url = new URL(window.parent.location);
                url.searchParams.set('login_user', username);
                url.searchParams.set('login_pass', password);
                window.parent.history.pushState({}, '', url);
                
                // 触发 popstate 事件，通知 Streamlit 悄悄更新状态
                window.parent.dispatchEvent(new Event('popstate'));
            });
        </script>
    </body>
    </html>
    """
    
    # 使用components嵌入完整HTML
    st.components.v1.html(login_html, height=800, scrolling=False)
    
    # 处理登录状态
    params = st.query_params
    if "login_user" in params and "login_pass" in params:
        user_data = db_verify_user(params["login_user"], params["login_pass"])
        
        # 验证后立即清除 URL 参数，防止状态残留
        if "login_user" in st.query_params:
            del st.query_params["login_user"]
        if "login_pass" in st.query_params:
            del st.query_params["login_pass"]
            
        if user_data:
            st.session_state.logged_in = True
            st.session_state.user_info = user_data
            st.session_state.page = "chat"
            st.rerun()  # 原生软刷新，零延迟无白屏完成跳转
        else:
            st.error("账号或密码错误！")