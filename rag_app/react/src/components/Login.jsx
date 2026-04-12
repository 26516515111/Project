import { useState } from 'react';

export default function Login({ onLogin }) {
    const [userId, setUserId] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [showPassword, setShowPassword] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        setError('');
        if (userId === "admin" && password === "123456") {
            onLogin({ name: "王轮机长", avatar: "👨‍✈️", user_id: "882103" });
        } else {
            setError('账号或密码错误！');
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#041527] via-[#0a2540] to-[#0d4a3a] overflow-hidden relative font-sans w-full">
            
            {/* 雷达背景 */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full border border-teal-500/10 pointer-events-none z-0">
                <div className="absolute inset-0 rounded-full border border-teal-500/10 scale-75"></div>
                <div className="absolute inset-0 rounded-full border border-teal-500/10 scale-50"></div>
                <div className="absolute inset-0 rounded-full border border-teal-500/20 scale-25"></div>
                <div className="absolute inset-0 rounded-full radar-spin" style={{ background: 'conic-gradient(from 0deg, transparent 0deg, transparent 270deg, rgba(45, 212, 191, 0.15) 360deg)' }}></div>
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-teal-500/10"></div>
                <div className="absolute left-0 right-0 top-1/2 h-px bg-teal-500/10"></div>
            </div>
            
            {/* 浮动图标 */}
            <div className="absolute right-[15%] top-[20%] opacity-10 pointer-events-none float-anim">
                <i className="ph ph-anchor text-[180px] text-teal-200"></i>
            </div>
            <div className="absolute left-[10%] bottom-[30%] opacity-10 pointer-events-none float-anim-slow">
                <i className="ph ph-compass text-[240px] text-blue-200"></i>
            </div>
            <div className="absolute right-[25%] bottom-[15%] opacity-5 pointer-events-none float-anim" style={{ animationDelay: '1s' }}>
                <i className="ph ph-paper-plane-tilt text-[120px] text-white"></i>
            </div>

            {/* 粒子效果 */}
            <div className="absolute inset-0 z-0 pointer-events-none">
                {Array.from({ length: 25 }).map((_, i) => {
                    const size = Math.random() * 8 + 4 + 'px';
                    const style = {
                        width: size,
                        height: size,
                        left: Math.random() * 100 + '%',
                        animationDuration: (Math.random() * 15 + 10) + 's',
                        animationDelay: Math.random() * 10 + 's'
                    };
                    return <div key={i} className="particle" style={style}></div>;
                })}
            </div>
            
            {/* 波浪效果 */}
            <svg className="waves" xmlns="http://www.w3.org/2000/svg" viewBox="0 24 150 28" preserveAspectRatio="none">
                <defs>
                    <path id="gentle-wave" d="M-160 44c30 0 58-18 88-18s 58 18 88 18 58-18 88-18 58 18 88 18 v44h-352z"/>
                </defs>
                <g className="parallax">
                    <use href="#gentle-wave" x="48" y="0" fill="rgba(13, 148, 136, 0.2)"/>
                    <use href="#gentle-wave" x="48" y="3" fill="rgba(17, 53, 90, 0.4)"/>
                    <use href="#gentle-wave" x="48" y="5" fill="rgba(45, 212, 191, 0.1)"/>
                    <use href="#gentle-wave" x="48" y="7" fill="rgba(4, 21, 39, 0.8)"/>
                </g>
            </svg>

            <div className="relative z-20 w-full max-w-[420px] px-6 float-anim">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-teal-500 to-blue-500 rounded-3xl blur opacity-20 glow-pulse"></div>
                
                <div className="max-w-md w-full bg-[#0a2540]/50 backdrop-blur-xl border border-teal-500/20 p-10 rounded-3xl shadow-[0_25px_50px_-12px_rgba(0,0,0,0.7)] relative z-10 overflow-hidden">
                    {/* 装饰角 */}
                    <div className="absolute top-0 left-0 w-16 h-16 border-t-2 border-l-2 border-teal-400/30 rounded-tl-3xl m-1 pointer-events-none"></div>
                    <div className="absolute bottom-0 right-0 w-16 h-16 border-b-2 border-r-2 border-teal-400/30 rounded-br-3xl m-1 pointer-events-none"></div>
                    
                    <div className="text-center mb-8 relative">
                        <div className="relative inline-flex items-center justify-center w-24 h-24 rounded-full bg-[#0a2540]/80 border border-teal-500/40 mb-5 glow-pulse group">
                            <div className="absolute inset-0 rounded-full border border-teal-400/20 animate-ping" style={{ animationDuration: '3s' }}></div>
                            <i className="ph ph-steering-wheel text-5xl text-teal-400 transition-transform duration-700 group-hover:rotate-180"></i>
                        </div>
                        <h2 className="mt-6 text-center text-3xl font-extrabold text-white tracking-wide">
                            智能船舶问答系统
                        </h2>
                        <p className="mt-2 text-center text-xs font-medium tracking-widest text-teal-200/60 uppercase">
                            Intelligent Ship Q&A System
                        </p>
                    </div>
                    {error && (
                        <div className="bg-red-500/20 border-l-4 border-red-500 p-4 mb-4 rounded">
                            <div className="flex">
                                <div className="flex-shrink-0">
                                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <div className="ml-3">
                                    <p className="text-sm text-red-200">{error}</p>
                                </div>
                            </div>
                        </div>
                    )}
                    <form className="space-y-5 relative z-10" onSubmit={handleSubmit}>
                        <div className="space-y-4">
                            <div className="group relative">
                                <label htmlFor="user-id" className="block text-xs text-teal-200/80 mb-1.5 ml-1 font-medium tracking-wider">用户名 / USERNAME</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <i className="ph ph-user text-teal-500/80 text-xl group-focus-within:text-teal-300 transition-colors"></i>
                                    </div>
                                    <input 
                                        id="user-id" 
                                        name="user_id" 
                                        type="text" 
                                        required 
                                        className="w-full bg-black/30 border border-white/10 text-white rounded-xl py-3.5 pr-4 pl-12 text-sm focus:outline-none focus:border-teal-400/70 focus:bg-black/50 transition-all duration-300 placeholder-gray-500/50" 
                                        placeholder="请输入您的账号 (例如: admin)"
                                        value={userId}
                                        onChange={(e) => setUserId(e.target.value)}
                                    />
                                </div>
                            </div>
                            <div className="group relative">
                                <label htmlFor="password" className="block text-xs text-teal-200/80 mb-1.5 ml-1 font-medium tracking-wider">密码 / PASSWORD</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <i className="ph ph-lock-key text-teal-500/80 text-xl group-focus-within:text-teal-300 transition-colors"></i>
                                    </div>
                                    <input 
                                        id="password" 
                                        name="password" 
                                        type={showPassword ? "text" : "password"} 
                                        required 
                                        className="w-full bg-black/30 border border-white/10 text-white rounded-xl py-3.5 pr-12 pl-12 text-sm focus:outline-none focus:border-teal-400/70 focus:bg-black/50 transition-all duration-300 placeholder-gray-500/50 tracking-widest font-mono" 
                                        placeholder="••••••••"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                    />
                                    <button type="button" onClick={() => setShowPassword(!showPassword)} className="absolute inset-y-0 right-0 pr-4 flex items-center text-gray-400 hover:text-teal-300 transition-colors">
                                        <i className={`ph ${showPassword ? 'ph-eye' : 'ph-eye-slash'} text-lg`}></i>
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div className="pt-4">
                            <button 
                                type="submit" 
                                className="w-full btn-shine relative overflow-hidden rounded-xl bg-gradient-to-r from-teal-600 to-[#0a2540] text-white font-bold py-4 shadow-[0_8px_20px_rgba(13,148,136,0.3)] hover:shadow-[0_8px_25px_rgba(45,212,191,0.5)] transition-all duration-300 transform hover:-translate-y-0.5 mt-6"
                            >
                                <span className="relative z-10 flex items-center justify-center gap-2 text-base tracking-widest">
                                    登 录 系 统 <i className="ph ph-arrow-right text-lg font-bold"></i>
                                </span>
                            </button>
                        </div>
                    </form>
                    
                    {/* 底部信息 */}
                    <div className="mt-8 pt-6 border-t border-white/5 text-center flex flex-col items-center">
                        <p className="text-[10px] text-gray-400 tracking-wider mb-2">Powered by React Vite</p>
                        <div className="flex gap-4">
                            <div className="w-1.5 h-1.5 rounded-full bg-teal-500/50 animate-pulse"></div>
                            <div className="w-1.5 h-1.5 rounded-full bg-teal-500/50 animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                            <div className="w-1.5 h-1.5 rounded-full bg-teal-500/50 animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}