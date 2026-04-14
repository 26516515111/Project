// setting.jsx
import { useState, useEffect, useRef } from "react";

// ===================== 通用组件 =====================

const API_HOST = window.location.hostname || "127.0.0.1";
const BASE_URL = `http://${API_HOST}:8000`;
const PROJECT_META = {
    name: "智能船舶问答系统",
    version: "2.4.1",
    build: "8829",
    releaseDate: "2024-01-15",
    frontend: "React + Vite",
    backend: "FastAPI + LangChain RAG",
    apiBase: BASE_URL,
};
const detectSystemDark = () => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
        return false;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
};

const Switch = ({ checked, onChange, disabled }) => (
    <label className="switch-toggle" style={{ position: "relative", display: "inline-block", width: 36, height: 20, cursor: disabled ? "not-allowed" : "pointer" }}>
        <input type="checkbox" checked={checked} onChange={onChange} disabled={disabled} style={{ opacity: 0, width: 0, height: 0 }} />
        <span style={{
            position: "absolute",
            top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: checked ? "#0d9488" : "rgba(255,255,255,0.1)",
            transition: ".4s",
            borderRadius: 20,
            opacity: disabled ? 0.5 : 1
        }}>
            <span style={{
                position: "absolute",
                height: 14, width: 14,
                left: checked ? "calc(100% - 17px)" : 3,
                bottom: 3,
                backgroundColor: "white",
                transition: ".4s",
                borderRadius: "50%"
            }} />
        </span>
    </label>
);

const GlassInput = ({ type = "text", value, onChange, placeholder, disabled, icon }) => (
    <div className="glass-input rounded-xl flex items-center px-3" style={{
        background: "rgba(255, 255, 255, 0.03)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
        backdropFilter: "blur(10px)",
        opacity: disabled ? 0.5 : 1
    }}>
        {icon && <i className={`ph ${icon} text-gray-500 mr-2 text-lg`}></i>}
        <input
            type={type}
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            disabled={disabled}
            className="flex-1 bg-transparent border-none py-2.5 outline-none text-sm text-gray-200 placeholder:text-gray-600 w-full"
            style={{ color: "#e5e7eb" }}
        />
    </div>
);

const GlassSelect = ({ value, onChange, options, icon }) => (
    <div className="glass-input rounded-xl flex items-center pr-3 relative" style={{
        background: "rgba(255, 255, 255, 0.03)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
        backdropFilter: "blur(10px)"
    }}>
        {icon && <i className={`ph ${icon} text-gray-500 ml-3 mr-1 text-lg`}></i>}
        <select
            value={value}
            onChange={onChange}
            className="flex-1 bg-transparent border-none py-2.5 px-3 outline-none text-sm text-gray-200 appearance-none cursor-pointer w-full"
            style={{ color: "#e5e7eb" }}
        >
            {options.map(opt => <option key={opt.value} value={opt.value} style={{ background: "#0a2540", color: "white" }}>{opt.label}</option>)}
        </select>
        <i className="ph ph-caret-down text-gray-500 absolute right-3 pointer-events-none"></i>
    </div>
);

const ConfirmDialog = ({ isOpen, onClose, onConfirm, title, message }) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}>
            <div className="rounded-2xl p-6 w-[400px] shadow-2xl" style={{ background: "#041527", border: "1px solid rgba(255,255,255,0.1)" }}>
                <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                    <i className="ph ph-warning-circle text-orange-500"></i> {title}
                </h3>
                <p className="text-sm mb-6 leading-relaxed" style={{ color: "#9ca3af" }}>{message}</p>
                <div className="flex justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 rounded-lg text-sm transition border hover:bg-white/5"
                        style={{ color: "#d1d5db", borderColor: "transparent" }}
                    >
                        取消
                    </button>
                    <button onClick={onConfirm} className="px-4 py-2 rounded-lg text-sm text-white transition shadow-lg" style={{ background: "rgba(239,68,68,0.8)" }}>
                        确认删除
                    </button>
                </div>
            </div>
        </div>
    );
};

const Toast = ({ show, message }) => (
    <div className={`fixed top-6 right-6 z-50 flex items-center gap-2 px-4 py-2 rounded-lg shadow-lg transition-all duration-300 ${show ? 'translate-y-0 opacity-100' : '-translate-y-4 opacity-0 pointer-events-none'}`}
        style={{ background: "rgba(13,148,136,0.1)", border: "1px solid rgba(45,212,191,0.2)", backdropFilter: "blur(10px)" }}>
        <i className="ph ph-check-circle text-teal-400 text-lg"></i>
        <span className="text-sm" style={{ color: "#99f6e4" }}>{message || "设置已保存"}</span>
    </div>
);

const HelpDocsDialog = ({ isOpen, onClose }) => {
    if (!isOpen) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.65)", backdropFilter: "blur(4px)" }}>
            <div className="rounded-2xl p-6 w-[680px] max-w-[92vw] max-h-[82vh] overflow-y-auto shadow-2xl"
                style={{ background: "#041527", border: "1px solid rgba(255,255,255,0.1)" }}>
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold text-white flex items-center gap-2">
                        <i className="ph ph-book-open text-teal-400"></i>
                        帮助与文档
                    </h3>
                    <button
                        type="button"
                        onClick={onClose}
                        className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-white/10 transition-colors"
                        style={{ color: "#9ca3af" }}
                        title="关闭"
                    >
                        <i className="ph ph-x"></i>
                    </button>
                </div>

                <div className="space-y-4 text-sm">
                    <section className="p-4 rounded-xl" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
                        <p className="text-xs mb-2" style={{ color: "#6b7280" }}>项目版本信息</p>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                            <p style={{ color: "#d1d5db" }}>项目名称：<span style={{ color: "#f3f4f6" }}>{PROJECT_META.name}</span></p>
                            <p style={{ color: "#d1d5db" }}>版本号：<span style={{ color: "#2dd4bf" }}>{PROJECT_META.version}</span></p>
                            <p style={{ color: "#d1d5db" }}>构建号：<span style={{ color: "#f3f4f6" }}>{PROJECT_META.build}</span></p>
                            <p style={{ color: "#d1d5db" }}>发布日期：<span style={{ color: "#f3f4f6" }}>{PROJECT_META.releaseDate}</span></p>
                            <p style={{ color: "#d1d5db" }}>前端技术栈：<span style={{ color: "#f3f4f6" }}>{PROJECT_META.frontend}</span></p>
                            <p style={{ color: "#d1d5db" }}>后端技术栈：<span style={{ color: "#f3f4f6" }}>{PROJECT_META.backend}</span></p>
                        </div>
                    </section>

                    <section className="p-4 rounded-xl" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
                        <p className="text-xs mb-2" style={{ color: "#6b7280" }}>核心接口</p>
                        <div className="space-y-1 text-xs" style={{ color: "#d1d5db" }}>
                            <p><span style={{ color: "#2dd4bf" }}>POST</span> {PROJECT_META.apiBase}/rag/query/stream</p>
                            <p><span style={{ color: "#2dd4bf" }}>POST</span> {PROJECT_META.apiBase}/rag/query</p>
                            <p><span style={{ color: "#2dd4bf" }}>POST</span> {PROJECT_META.apiBase}/rag/incremental/upload</p>
                        </div>
                    </section>

                    <section className="p-4 rounded-xl" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
                        <p className="text-xs mb-2" style={{ color: "#6b7280" }}>使用说明</p>
                        <ul className="text-xs space-y-1 list-disc pl-5" style={{ color: "#d1d5db" }}>
                            <li>问答页面可切换在线/离线模式，并支持知识图谱分析。</li>
                            <li>可在“专家知识库上传”中上传文档，系统会自动切块并增量索引。</li>
                            <li>在线回答可查看后端响应过程，便于排查与调优。</li>
                        </ul>
                    </section>
                </div>

                <div className="mt-5 flex justify-end">
                    <button
                        type="button"
                        onClick={onClose}
                        className="px-4 py-2 rounded-lg text-sm text-white transition-colors"
                        style={{ background: "#0d9488" }}
                    >
                        我知道了
                    </button>
                </div>
            </div>
        </div>
    );
};

// ===================== 主组件 =====================

export default function Setting({
    onBack,
    chats = [],
    onClearChats,
    onDeleteChat,
    userId = "captain_park",
    userRole = 0,
    initialSettings,
    onSettingsChange,
}) {
    const [activeTab, setActiveTab] = useState("personal");
    const isAdmin = Number(userRole) === 1;

    const [showToast, setShowToast] = useState(false);
    const [toastMessage, setToastMessage] = useState("设置已保存"); // 动态提示文案
    const [showConfirm, setShowConfirm] = useState(false);
    const [showHelpDocs, setShowHelpDocs] = useState(false);

    // 关于与帮助面板的模拟状态
    const [isCheckingUpdate, setIsCheckingUpdate] = useState(false);
    const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
    const [feedbackText, setFeedbackText] = useState("");

    // 统一定义呼出提示框的函数
    const triggerToast = (msg = "设置已保存") => {
        setToastMessage(msg);
        setShowToast(true);
        setTimeout(() => setShowToast(false), 3000);
    };

    const avatarInputRef = useRef(null); // 添加用于关联文件上传的 ref
    const kbInputRef = useRef(null); // 新增：用于专家知识库文件上传的 ref

    // 优化：从本地缓存读取已上传的知识库文件列表
    const [files, setFiles] = useState(() => {
        const saved = localStorage.getItem("deepblue_kb_files");
        // 删除测试示例，默认返回空数组
        return saved ? JSON.parse(saved) : [];
    });
    const [kbUploading, setKbUploading] = useState(false);
    const [kbUploadHint, setKbUploadHint] = useState("");

    // 优化：当知识库文件列表发生变化时，同步保存到本地缓存
    useEffect(() => {
        localStorage.setItem("deepblue_kb_files", JSON.stringify(files));
    }, [files]);

    const [settings, setSettings] = useState(() => {
        return initialSettings || {
            // 个人信息
            nickname: "Captain Park",
            avatar: "", // 新增：保存头像的 Base64
            imo: "9876543",
            email: "captain.park@deepblue.com",
            emergency: "+86 13800138000",

            // 系统偏好
            theme: "dark",
            fontSize: 16, // 默认推荐回到16px (1rem基准)
            notify: true,
            autoSave: true,

            // AI 与知识图谱
            graphOn: true,
            offlineOn: false,
            streamSpeed: 50,
            dbPath: "/mnt/data/local_kb",
            model: "hybrid",

            // 隐私与安全
            retention: "30"
        };
    });
    const [systemDark, setSystemDark] = useState(() => detectSystemDark());
    const themePref = settings?.theme || "dark";
    const effectiveTheme = themePref === "system" ? (systemDark ? "dark" : "light") : themePref;
    const themeSurface = effectiveTheme === "light" ? "#f8fafc" : "#030d17";
    const contentSurface = effectiveTheme === "light" ? "#ffffff" : "#041527";
    const isLightUi = effectiveTheme === "light";

    useEffect(() => {
        if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
            return undefined;
        }
        const media = window.matchMedia("(prefers-color-scheme: dark)");
        const onChange = (event) => setSystemDark(Boolean(event.matches));
        setSystemDark(Boolean(media.matches));
        if (typeof media.addEventListener === "function") {
            media.addEventListener("change", onChange);
            return () => media.removeEventListener("change", onChange);
        }
        media.addListener(onChange);
        return () => media.removeListener(onChange);
    }, []);

    useEffect(() => {
        if (initialSettings) {
            setSettings(initialSettings);
        }
    }, [initialSettings]);

    // 监听字号等设置修改，并全局同步
    useEffect(() => {
        // 同步字号到根节点
        document.documentElement.style.fontSize = `${settings.fontSize}px`;
    }, [settings]);

    const updateSetting = (key, value) => {
        setSettings(prev => {
            const next = { ...prev, [key]: value };
            if (onSettingsChange) onSettingsChange(next);
            return next;
        });
    };

    const handleAvatarChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // 检查文件大小 (限制为 2MB)
        if (file.size > 2 * 1024 * 1024) {
            alert("头像文件大小不能超过 2MB");
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            updateSetting("avatar", event.target.result);
            setShowToast(true);
            setTimeout(() => setShowToast(false), 3000);
        };
        reader.readAsDataURL(file);

        e.target.value = ""; // 清空 input 以便下次能选择同一张图
    };

    const handleSave = () => {
        if (onSettingsChange) onSettingsChange(settings);
        setShowToast(true);
        setTimeout(() => setShowToast(false), 3000);
    };

    const removeFile = (id) => setFiles(files.filter(f => f.id !== id));

    // 修改：处理知识库文件真实上传到后端
    const handleKBFileUpload = async (e) => {
        const uploadedFiles = Array.from(e.target.files);
        if (!uploadedFiles.length) return;

        if (!isAdmin) {
            e.target.value = "";
            return;
        }

        const updateFileRecord = (id, patch) => {
            setFiles(prev => prev.map(item => (item.id === id ? { ...item, ...patch } : item)));
        };

        setKbUploading(true);
        setKbUploadHint(`正在处理 0/${uploadedFiles.length} 个文件...`);
        let successCount = 0;

        // 遍历所有选中的文件逐一上传
        for (let i = 0; i < uploadedFiles.length; i++) {
            const file = uploadedFiles[i];
            const fileId = `${Date.now()}-${i}-${Math.random().toString(36).slice(2, 8)}`;
            const fileRecord = {
                id: fileId,
                name: file.name,
                size: (file.size / (1024 * 1024)).toFixed(2) + " MB",
                date: new Date().toISOString().slice(0, 10),
                status: "processing",
                detail: "上传中，正在切块与索引...",
            };

            setFiles(prev => [fileRecord, ...prev]);
            setKbUploadHint(`正在处理 ${i + 1}/${uploadedFiles.length}：${file.name}`);

            const formData = new FormData();
            formData.append("user_id", userId || "admin");
            formData.append("file", file);

            try {
                // 调用后端增量上传接口，后端会自动触发 run_incremental_update() 和知识图谱转化
                const res = await fetch(`${BASE_URL}/rag/incremental/upload`, {
                    method: "POST",
                    body: formData
                });

                const data = await res.json().catch(() => ({}));
                if (res.ok && data?.ok !== false) {
                    successCount += 1;
                    const indexedChunks = Number(data?.indexed_chunks || 0);
                    updateFileRecord(fileId, {
                        status: "success",
                        detail: indexedChunks > 0 ? `索引完成，新增 ${indexedChunks} 个 chunks` : "索引完成",
                    });
                } else {
                    updateFileRecord(fileId, {
                        status: "error",
                        detail: `上传失败：${data?.detail || data?.message || "未知错误"}`,
                    });
                }
            } catch (err) {
                console.error("Upload error:", err);
                updateFileRecord(fileId, {
                    status: "error",
                    detail: `上传异常：${err?.message || "请检查后端运行状态"}`,
                });
            }
        }

        setKbUploading(false);
        setKbUploadHint(`处理完成：成功 ${successCount}/${uploadedFiles.length}`);
        triggerToast(`知识库上传完成（成功 ${successCount}/${uploadedFiles.length}）`);
        setTimeout(() => setKbUploadHint(""), 5000);
        e.target.value = ""; // 清空以便重复上传同名文件
    };

    // 新增：导出所有个人数据 (JSON)
    const handleExportPersonalData = () => {
        const exportData = {
            exportTime: new Date().toISOString(),
            settings: settings,
            history: chats
        };
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `DeepBlue_PersonalData_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setShowToast(true);
        setTimeout(() => setShowToast(false), 3000);
    };

    const tabs = [
        { id: "personal", label: "个人信息", icon: "ph-user" },
        { id: "system", label: "系统偏好", icon: "ph-sliders" },
        { id: "ai", label: "AI 与知识图谱", icon: "ph-brain" },
        { id: "privacy", label: "隐私与安全", icon: "ph-shield-check" },
        { id: "about", label: "关于与帮助", icon: "ph-info" }
    ];

    const currentTabLabel = tabs.find(t => t.id === activeTab)?.label || "";

    // ===================== 渲染辅助函数 =====================

    const renderPersonalInfo = () => (
        <div className="space-y-8 animate-fade-in">
            {/* 头像区域 */}
            <div className="p-6 rounded-2xl flex items-center gap-6" style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div className="relative group cursor-pointer" onClick={() => avatarInputRef.current?.click()}>
                    <div className="w-20 h-20 rounded-full flex items-center justify-center text-2xl font-bold text-white border-4 shadow-lg overflow-hidden"
                        style={{ background: "#3b82f6", borderColor: "#041527" }}>
                        {settings.avatar ? (
                            <img src={settings.avatar} alt="avatar" className="w-full h-full object-cover" />
                        ) : (
                            "CP"
                        )}
                    </div>
                    <div className="absolute inset-0 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity" style={{ background: "rgba(0,0,0,0.5)" }}>
                        <i className="ph ph-camera text-white text-xl"></i>
                    </div>
                </div>
                <div>
                    <h3 className="text-sm font-medium text-white mb-1">个人头像</h3>
                    <p className="text-xs mb-3" style={{ color: "#6b7280" }}>支持 JPG, PNG 格式，文件小于 2MB</p>

                    {/* 隐藏的真实文件输入框 */}
                    <input
                        type="file"
                        accept=".jpg,.jpeg,.png"
                        className="hidden"
                        ref={avatarInputRef}
                        onChange={handleAvatarChange}
                    />

                    <button
                        onClick={() => avatarInputRef.current?.click()}
                        className="px-3 py-1.5 rounded-md text-xs transition-colors border hover:bg-white/5"
                        style={{ borderColor: "rgba(255,255,255,0.1)", color: "#d1d5db" }}
                    >
                        上传新头像
                    </button>
                    {settings.avatar && (
                        <button
                            onClick={() => updateSetting("avatar", "")}
                            className="px-3 py-1.5 ml-2 rounded-md text-xs transition-colors border border-transparent hover:bg-red-500/10 text-red-400"
                        >
                            移除
                        </button>
                    )}
                </div>
            </div>

            {/* 表单字段 */}
            <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                    <label className="text-xs font-medium ml-1" style={{ color: "#9ca3af" }}>显示昵称</label>
                    <GlassInput value={settings.nickname} onChange={e => updateSetting("nickname", e.target.value)} icon="ph-user-focus" />
                </div>

                {/* 权限级别 - 只读显示 */}
                <div className="space-y-2">
                    <label className="text-xs font-medium ml-1" style={{ color: "#9ca3af" }}>权限级别</label>
                    <div className="rounded-xl flex items-center px-3 py-2.5 select-none"
                        style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", opacity: 0.7 }}>
                        <i className="ph ph-shield text-gray-500 mr-2 text-lg"></i>
                        <span className="text-sm flex-1" style={{ color: "#d1d5db" }}>{isAdmin ? "管理者" : "普通用户"}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: "rgba(255,255,255,0.05)", color: "#6b7280" }}>由系统分配</span>
                    </div>
                    <p className="text-[10px] ml-1 mt-1" style={{ color: "#6b7280" }}>权限级别由后端数据库统一管理，无法自行修改（后端标识：{isAdmin ? "1" : "0"}）</p>
                </div>

                <div className="space-y-2">
                    <label className="text-xs font-medium ml-1" style={{ color: "#9ca3af" }}>IMO 编号</label>
                    <GlassInput value={settings.imo} onChange={e => updateSetting("imo", e.target.value)} icon="ph-hash" />
                </div>

                <div className="space-y-2">
                    <label className="text-xs font-medium ml-1" style={{ color: "#9ca3af" }}>电子邮箱</label>
                    <GlassInput type="email" value={settings.email} onChange={e => updateSetting("email", e.target.value)} icon="ph-envelope-simple" />
                </div>

                <div className="space-y-2">
                    <label className="text-xs font-medium ml-1" style={{ color: "#9ca3af" }}>紧急联系电话</label>
                    <GlassInput value={settings.emergency} onChange={e => updateSetting("emergency", e.target.value)} icon="ph-phone" />
                </div>
            </div>
        </div>
    );

    const renderSystemPrefs = () => (
        <div className="space-y-8 animate-fade-in">
            {/* 主题选择 */}
            <section>
                <h3 className="text-sm font-medium mb-4 flex items-center gap-2" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>
                    <i className="ph ph-palette text-teal-400"></i> 主题外观
                </h3>
                <div className="grid grid-cols-3 gap-4">
                    {[
                        { id: "dark", label: "深色模式", icon: "ph-moon-stars" },
                        { id: "light", label: "浅色模式", icon: "ph-sun" },
                        { id: "system", label: "跟随系统", icon: "ph-desktop" }
                    ].map(t => (
                        (() => {
                            const selected = settings.theme === t.id;
                            const activeThemeColor = isLightUi ? "#2563eb" : "#2dd4bf";
                            const idleTextColor = isLightUi ? "#475569" : "#9ca3af";
                            const idleBg = isLightUi ? "rgba(241,245,249,0.9)" : "rgba(255,255,255,0.02)";
                            const idleBorder = isLightUi ? "rgba(148,163,184,0.45)" : "rgba(255,255,255,0.05)";
                            return (
                                <button
                                    key={t.id}
                                    onClick={() => updateSetting("theme", t.id)}
                                    className="p-4 rounded-xl border flex flex-col items-center gap-3 transition-all"
                                    style={{
                                        background: selected ? (isLightUi ? "rgba(37,99,235,0.18)" : "rgba(13,148,136,0.1)") : idleBg,
                                        borderColor: selected ? (isLightUi ? "rgba(37,99,235,0.45)" : "rgba(45,212,191,0.5)") : idleBorder,
                                        boxShadow: selected
                                            ? (isLightUi ? "0 0 12px rgba(37,99,235,0.14)" : "0 0 15px rgba(45,212,191,0.1)")
                                            : "none"
                                    }}
                                >
                                    <div className="w-10 h-10 rounded-full flex items-center justify-center"
                                        style={{
                                            background: selected ? (isLightUi ? "#2563eb" : "#0d9488") : (isLightUi ? "rgba(148,163,184,0.15)" : "rgba(255,255,255,0.05)"),
                                            color: selected ? "#ffffff" : idleTextColor
                                        }}>
                                        <i className={`ph ${t.icon} text-xl`}></i>
                                    </div>
                                    <span className={`text-sm ${selected ? "font-medium" : ""}`} style={{ color: selected ? activeThemeColor : idleTextColor }}>
                                        {t.label}
                                    </span>
                                </button>
                            );
                        })()
                    ))}
                </div>
            </section>

            {/* 字体大小 */}
            <section
                className="p-6 rounded-2xl"
                style={{
                    background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)",
                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                }}
            >
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium flex items-center gap-2" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>
                        <i className="ph ph-text-aa text-teal-400"></i> 界面字号
                    </h3>
                    <span
                        className="text-xs px-2 py-1 rounded"
                        style={{
                            background: isLightUi ? "rgba(14,116,144,0.12)" : "rgba(13,148,136,0.1)",
                            color: isLightUi ? "#0c4a6e" : "#2dd4bf",
                        }}
                    >
                        {settings.fontSize}px
                    </span>
                </div>
                <div className="flex items-center gap-4 mb-4">
                    <span className="text-xs" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>A</span>
                    <input
                        type="range" min="12" max="20" step="1"
                        value={settings.fontSize}
                        onChange={e => updateSetting("fontSize", parseInt(e.target.value))}
                        className="range-slider flex-1"
                        style={{
                            appearance: "none",
                            height: 4,
                            background: "rgba(255,255,255,0.1)",
                            borderRadius: 5,
                            outline: "none"
                        }}
                    />
                    <span className="text-lg" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>A</span>
                </div>
                <div
                    className="p-4 rounded-lg border flex items-start gap-3"
                    style={{
                        background: isLightUi ? "#ffffff" : "rgba(0,0,0,0.2)",
                        borderColor: isLightUi ? "rgba(148,163,184,0.45)" : "rgba(255,255,255,0.05)",
                    }}
                >
                    <i className="ph ph-info mt-1" style={{ color: isLightUi ? "#334155" : "#9ca3af" }}></i>
                    <p style={{ fontSize: `${settings.fontSize}px`, color: isLightUi ? "#0f172a" : "#d1d5db", lineHeight: 1.6 }}>
                        这是预览文本。调整滑块以更改整个问答界面的默认文本大小，确保在不同光线环境下均可清晰阅读。
                    </p>
                </div>
            </section>

            {/* 消息与记录 - 已合并为单一"提示"开关 */}
            <section className="space-y-2">
                <h3 className="text-sm font-medium mb-4 mt-2 px-1" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>消息与记录</h3>

                <div
                    className="flex items-center justify-between p-4 rounded-xl transition-colors hover:bg-white/[0.04]"
                    style={{
                        background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)",
                        border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                    }}
                >
                    <div>
                        <p className="text-sm" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>消息提示</p>
                        <p className="text-xs mt-0.5" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>收到 AI 回复时播放提示音并显示通知</p>
                    </div>
                    <Switch checked={settings.notify} onChange={e => updateSetting("notify", e.target.checked)} disabled={false} />
                </div>

                <div
                    className="flex items-center justify-between p-4 rounded-xl transition-colors hover:bg-white/[0.04]"
                    style={{
                        background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)",
                        border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                    }}
                >
                    <div>
                        <p className="text-sm" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>自动保存历史记录</p>
                        <p className="text-xs mt-0.5" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>退出系统时保留所有未关闭的会话</p>
                    </div>
                    <Switch checked={settings.autoSave} onChange={e => updateSetting("autoSave", e.target.checked)} disabled={false} />
                </div>
            </section>
        </div>
    );

    const renderAISettings = () => (
        <div className="space-y-6 animate-fade-in">
            {/* 知识图谱默认开关 */}
            <div
                className="p-6 rounded-2xl"
                style={{
                    background: isLightUi ? "#ecfeff" : "rgba(13,148,136,0.05)",
                    border: isLightUi ? "1px solid rgba(14,116,144,0.35)" : "1px solid rgba(45,212,191,0.2)",
                }}
            >
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <i className="ph ph-graph text-teal-400 text-xl"></i>
                        <h3 className="text-sm font-medium" style={{ color: isLightUi ? "#0f172a" : "#ffffff" }}>默认开启知识图谱分析</h3>
                    </div>
                    <Switch checked={settings.graphOn} onChange={e => updateSetting("graphOn", e.target.checked)} disabled={false} />
                </div>
                <p className="text-xs pl-7" style={{ color: isLightUi ? "#334155" : "#9ca3af" }}>新建会话时，默认勾选右侧边栏的"知识图谱分析"功能，结合历史记录深度诊断。</p>
            </div>

            {/* 离线模式 */}
            <div
                className="p-6 rounded-2xl"
                style={{
                    background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)",
                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                }}
            >
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <i className="ph ph-wifi-slash text-gray-400 text-xl"></i>
                        <h3 className="text-sm font-medium" style={{ color: isLightUi ? "#0f172a" : "#ffffff" }}>默认离线模式状态</h3>
                    </div>
                    <Switch checked={settings.offlineOn} onChange={e => updateSetting("offlineOn", e.target.checked)} disabled={false} />
                </div>
                <p className="text-xs pl-7 mb-4" style={{ color: isLightUi ? "#334155" : "#9ca3af" }}>在无网络覆盖海域，默认启动本地轻量模型和离线知识库。</p>

                <div className="mt-4 pt-4 pl-7 border-t" style={{ borderColor: isLightUi ? "rgba(148,163,184,0.35)" : "rgba(255,255,255,0.05)" }}>
                    <label className="block text-xs font-medium mb-2" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>本地知识库路径配置</label>
                    <div className="flex gap-2">
                        <div className="flex-1">
                            <GlassInput value={settings.dbPath} onChange={e => updateSetting("dbPath", e.target.value)} icon="ph-folder" disabled={!settings.offlineOn} />
                        </div>
                        <button
                            disabled={!settings.offlineOn}
                            className="px-4 rounded-xl text-xs transition-colors whitespace-nowrap border"
                            style={{
                                background: settings.offlineOn
                                    ? (isLightUi ? "#eef2ff" : "rgba(255,255,255,0.05)")
                                    : (isLightUi ? "#f1f5f9" : "rgba(255,255,255,0.02)"),
                                borderColor: isLightUi ? "rgba(148,163,184,0.45)" : "rgba(255,255,255,0.1)",
                                color: settings.offlineOn ? (isLightUi ? "#0f172a" : "#d1d5db") : "#6b7280",
                                cursor: settings.offlineOn ? "pointer" : "not-allowed",
                                opacity: settings.offlineOn ? 1 : 0.5
                            }}
                        >
                            浏览...
                        </button>
                    </div>
                </div>
            </div>

            {/* 模型选择与流式速度 */}
            <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                    <label className="text-xs font-medium ml-1" style={{ color: isLightUi ? "#334155" : "#9ca3af" }}>默认问答模型</label>
                    <GlassSelect
                        value={settings.model}
                        onChange={e => updateSetting("model", e.target.value)}
                        icon="ph-cpu"
                        options={[
                            { value: "cloud", label: "DeepBlue 云端大模型 (推荐)" },
                            { value: "hybrid", label: "混合智能调度模式" },
                            { value: "local", label: "本地轻量化模型 (7B)" }
                        ]}
                    />
                </div>
                <div className="space-y-3">
                    <div className="flex justify-between items-center ml-1">
                        <label className="text-xs font-medium" style={{ color: isLightUi ? "#334155" : "#9ca3af" }}>流式输出速度</label>
                        <span className="text-[10px]" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>
                            {settings.streamSpeed < 30 ? "较慢" : settings.streamSpeed > 70 ? "极速" : "正常"}
                        </span>
                    </div>
                    <div className="px-2 pt-2">
                        <input
                            type="range" min="10" max="100" step="1"
                            value={settings.streamSpeed}
                            onChange={e => updateSetting("streamSpeed", parseInt(e.target.value))}
                            className="range-slider"
                            style={{
                                appearance: "none",
                                width: "100%",
                                height: 4,
                                background: isLightUi ? "rgba(148,163,184,0.4)" : "rgba(255,255,255,0.1)",
                                borderRadius: 5,
                                outline: "none"
                            }}
                        />
                    </div>
                </div>
            </div>

            {/* 专家知识库上传 - 仅管理者可见 */}
            <div
                className="relative p-6 rounded-2xl overflow-hidden group mt-2"
                style={{
                    background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)",
                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                }}
            >
                <div className="flex justify-between items-center mb-5">
                    <div>
                        <h3 className="text-sm font-medium flex items-center gap-2" style={{ color: isLightUi ? "#0f172a" : "#ffffff" }}>
                            <i className="ph ph-folder-open text-teal-400 text-lg"></i> 专家知识库上传
                        </h3>
                        <p className="text-xs mt-1" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>上传 PDF/Word 等资料以丰富图谱库</p>
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                        <span className="hidden sm:inline" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>当前身份:</span>
                        <div
                            className="px-2.5 py-1 rounded-md border flex items-center gap-1.5"
                            style={{
                                background: isAdmin ? "rgba(13,148,136,0.1)" : "rgba(255,255,255,0.05)",
                                borderColor: isAdmin
                                    ? "rgba(45,212,191,0.3)"
                                    : (isLightUi ? "rgba(148,163,184,0.45)" : "rgba(255,255,255,0.1)"),
                                color: isAdmin ? "#0d9488" : (isLightUi ? "#334155" : "#9ca3af")
                            }}
                        >
                            {isAdmin ? <i className="ph ph-crown"></i> : <i className="ph ph-user"></i>}
                            {isAdmin ? "管理者" : "普通用户"}
                        </div>
                    </div>
                </div>

                {!isAdmin && (
                    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center"
                        style={{ background: "rgba(4,21,39,0.85)", backdropFilter: "blur(4px)" }}>
                        <div className="w-16 h-16 rounded-full flex items-center justify-center mb-3" style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)" }}>
                            <i className="ph ph-lock-key text-3xl" style={{ color: "#6b7280" }}></i>
                        </div>
                        <p className="text-sm font-medium mb-1" style={{ color: "#d1d5db" }}>无权访问此功能</p>
                        <p className="text-xs" style={{ color: "#6b7280" }}>仅限管理员级别的账号上传全局资料</p>
                    </div>
                )}

                <input
                    type="file"
                    multiple
                    accept=".pdf,.doc,.docx,.md,.txt"
                    className="hidden"
                    ref={kbInputRef}
                    disabled={!isAdmin || kbUploading}
                    onChange={handleKBFileUpload}
                />

                <div
                    onClick={() => isAdmin && !kbUploading && kbInputRef.current?.click()}
                    className="border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center transition-colors mb-4 hover:bg-white/5"
                    style={{
                        borderColor: isAdmin ? "rgba(45,212,191,0.3)" : (isLightUi ? "rgba(148,163,184,0.45)" : "rgba(255,255,255,0.05)"),
                        background: isAdmin ? (isLightUi ? "#ecfeff" : "rgba(13,148,136,0.05)") : (isLightUi ? "#f1f5f9" : "rgba(0,0,0,0.2)"),
                        cursor: isAdmin && !kbUploading ? "pointer" : "default",
                        opacity: kbUploading ? 0.85 : 1,
                    }}
                >
                    <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3"
                        style={{
                            background: isAdmin ? "rgba(13,148,136,0.2)" : "rgba(255,255,255,0.05)",
                            color: isAdmin ? "#2dd4bf" : "#4b5563"
                        }}>
                        <i className={`ph ${kbUploading ? "ph-spinner-gap animate-spin" : "ph-upload-simple"} text-2xl`}></i>
                    </div>
                    <p className="text-sm font-medium mb-1" style={{ color: isLightUi ? "#0f172a" : "#d1d5db" }}>
                        {kbUploading ? "正在上传并处理文档..." : "点击或拖拽文件到此处上传"}
                    </p>
                    <p className="text-xs" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>
                        {kbUploading ? "系统正在执行切块、向量化与索引写入" : "支持 PDF, DOCX, MD, TXT (单文件最大 50MB)"}
                    </p>
                </div>

                {isAdmin && (kbUploading || kbUploadHint) && (
                    <div className="mb-4 px-3 py-2 rounded-lg flex items-center gap-2 text-xs"
                        style={{
                            background: kbUploading ? "rgba(56,189,248,0.08)" : "rgba(16,185,129,0.08)",
                            border: kbUploading ? "1px solid rgba(56,189,248,0.25)" : "1px solid rgba(16,185,129,0.25)",
                            color: kbUploading ? "#7dd3fc" : "#6ee7b7",
                        }}
                    >
                        <i className={`ph ${kbUploading ? "ph-spinner-gap animate-spin" : "ph-check-circle"}`}></i>
                        <span>{kbUploadHint || "处理中..."}</span>
                    </div>
                )}

                {isAdmin && files.length > 0 && (
                    <div className="space-y-2 mt-2">
                        <p className="text-xs font-medium mb-2 px-1" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>最近上传</p>
                        {files.map(file => (
                            <div key={file.id} className="flex items-center justify-between p-3 rounded-lg transition-colors group/file hover:bg-white/5"
                                style={{
                                    background: isLightUi ? "#ffffff" : "rgba(0,0,0,0.2)",
                                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                                }}>
                                <div className="flex items-center gap-3">
                                    <i className={`ph ${file.name.toLowerCase().endsWith("pdf") ? "ph-file-pdf text-red-400" : file.name.toLowerCase().endsWith("md") || file.name.toLowerCase().endsWith("txt") ? "ph-file-text text-cyan-400" : "ph-file-doc text-blue-400"} text-2xl`}></i>
                                    <div>
                                        <p className="text-xs font-medium" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>{file.name}</p>
                                        <p className="text-[10px] mt-0.5" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>{file.size} • {file.date}</p>
                                        {file.detail && (
                                            <p className="text-[10px] mt-0.5" style={{ color: file.status === "error" ? "#dc2626" : file.status === "processing" ? "#0369a1" : "#047857" }}>
                                                {file.detail}
                                            </p>
                                        )}
                                    </div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="text-[10px] px-1.5 py-0.5 rounded border"
                                        style={{
                                            color: file.status === "error" ? "#dc2626" : file.status === "processing" ? "#0369a1" : "#047857",
                                            borderColor: file.status === "error" ? "rgba(251,113,133,0.35)" : file.status === "processing" ? "rgba(56,189,248,0.35)" : "rgba(16,185,129,0.35)",
                                            background: file.status === "error" ? "rgba(244,63,94,0.08)" : file.status === "processing" ? "rgba(56,189,248,0.08)" : "rgba(16,185,129,0.08)",
                                        }}
                                    >
                                        {file.status === "error" ? "失败" : file.status === "processing" ? "处理中" : "完成"}
                                    </span>
                                    <button
                                        onClick={() => removeFile(file.id)}
                                        disabled={file.status === "processing"}
                                        className="w-8 h-8 rounded flex items-center justify-center transition-all opacity-0 group-hover/file:opacity-100 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-30 disabled:cursor-not-allowed"
                                        style={{ color: "#6b7280" }}
                                        title="删除文件"
                                    >
                                        <i className="ph ph-trash"></i>
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );

    const renderPrivacySecurity = () => (
        <div className="space-y-6 animate-fade-in">
            {/* 数据清理周期与导出 */}
            <div className="grid grid-cols-2 gap-6 items-end">
                <div className="space-y-2">
                    <label className="text-xs font-medium ml-1" style={{ color: "#9ca3af" }}>会话数据自动清理</label>
                    <GlassSelect
                        value={settings.retention}
                        onChange={e => updateSetting("retention", e.target.value)}
                        icon="ph-clock-counter-clockwise"
                        options={[
                            { value: "7", label: "保留 7 天" },
                            { value: "30", label: "保留 30 天" },
                            { value: "90", label: "保留 90 天" },
                            { value: "never", label: "永不自动清理" }
                        ]}
                    />
                </div>
                <div>
                    <button
                        onClick={handleExportPersonalData}
                        className="w-full h-[42px] flex items-center justify-center gap-2 rounded-xl border text-sm transition-colors hover:bg-white/5"
                        style={{ borderColor: "rgba(255,255,255,0.1)", color: "#d1d5db" }}>
                        <i className="ph ph-export text-lg"></i> 导出所有个人数据 (JSON)
                    </button>
                </div>
            </div>

            {/* 危险操作 - 删除历史记录 */}
            <div className="p-6 rounded-2xl mt-8" style={{ background: "rgba(239,68,68,0.05)", border: "1px solid rgba(239,68,68,0.2)" }}>
                <h3 className="text-sm font-medium mb-2" style={{ color: "#f87171" }}>危险操作</h3>
                <p className="text-xs mb-4" style={{ color: "#6b7280" }}>删除后将无法恢复所有历史问答记录和本地缓存数据。系统偏好设置将被保留。</p>
                <button
                    onClick={() => setShowConfirm(true)}
                    className="px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors"
                    style={{
                        background: "rgba(239,68,68,0.1)",
                        color: "#f87171",
                        border: "1px solid rgba(239,68,68,0.3)"
                    }}
                >
                    <i className="ph ph-trash text-lg"></i> 删除所有历史记录
                </button>
            </div>

            {/* 管理指定的历史记录 */}
            <div
                className="relative p-6 rounded-2xl overflow-hidden group mt-6"
                style={{
                    background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)",
                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                }}
            >
                <h3 className="text-sm font-medium flex items-center gap-2 mb-4" style={{ color: isLightUi ? "#0f172a" : "#ffffff" }}>
                    <i className="ph ph-list-dashes text-teal-400 text-lg"></i> 管理特定会话记录
                </h3>
                {chats.length === 0 ? (
                    <p className="text-xs" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>暂无任何问答历史记录。</p>
                ) : (
                    <div className="space-y-2 max-h-48 overflow-y-auto pr-2" style={{ scrollbarWidth: "thin", scrollbarColor: "rgba(255,255,255,0.1) transparent" }}>
                        {chats.map(chat => (
                            <div key={chat.id} className="flex items-center justify-between p-3 rounded-lg transition-colors group/chat hover:bg-white/5"
                                style={{
                                    background: isLightUi ? "#ffffff" : "rgba(0,0,0,0.2)",
                                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                                }}>
                                <div className="flex items-center gap-3 overflow-hidden">
                                    <i className="ph ph-chat-text text-gray-500 text-lg"></i>
                                    <p className="text-xs font-medium truncate" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>{chat.title}</p>
                                </div>
                                <button
                                    onClick={() => onDeleteChat && onDeleteChat(chat.id)}
                                    className="w-8 h-8 rounded flex items-center justify-center transition-all hover:text-red-400 hover:bg-red-500/10"
                                    style={{ color: isLightUi ? "#334155" : "#6b7280" }}
                                    title="删除该条对话"
                                >
                                    <i className="ph ph-trash"></i>
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );

    const renderAboutHelp = () => (
        <div className="space-y-6 animate-fade-in flex flex-col items-center max-w-lg mx-auto mt-4">
            {/* 系统信息卡片 */}
            <div className="w-full flex flex-col items-center p-8 rounded-3xl relative overflow-hidden mb-2"
                style={{
                    background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)",
                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                }}>
                <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full blur-3xl" style={{ background: "rgba(13,148,136,0.1)" }}></div>

                <div className="w-20 h-20 rounded-2xl flex items-center justify-center mb-4 relative z-10 shadow-lg"
                    style={{ background: "#0a2530", border: "1px solid rgba(45,212,191,0.3)" }}>
                    <i className="ph ph-steering-wheel text-4xl text-teal-400"></i>
                </div>
                <h2 className="text-xl font-bold tracking-wide mb-1 relative z-10" style={{ color: isLightUi ? "#0f172a" : "#ffffff" }}>智能船舶问答系统</h2>
                <p className="text-xs font-mono px-2 py-0.5 rounded border mb-4 relative z-10"
                    style={{
                        background: isLightUi ? "rgba(14,116,144,0.12)" : "rgba(13,148,136,0.1)",
                        color: isLightUi ? "#0c4a6e" : "#2dd4bf",
                        borderColor: isLightUi ? "rgba(14,116,144,0.25)" : "rgba(45,212,191,0.2)",
                    }}>
                    Version 2.4.1 Build 8829
                </p>
                <p className="text-xs" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>上次更新: 2024-01-15</p>

                <div className="w-full h-px my-6" style={{ background: "rgba(255,255,255,0.05)" }}></div>

                <div className="flex gap-4 w-full">
                    <button
                        onClick={() => {
                            setIsCheckingUpdate(true);
                            setTimeout(() => {
                                setIsCheckingUpdate(false);
                                triggerToast("当前已是最新版本 (v2.4.1)");
                            }, 1500);
                        }}
                        disabled={isCheckingUpdate}
                        className="flex-1 py-2.5 rounded-xl text-sm transition-colors flex items-center justify-center gap-2 hover:bg-white/10 disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{
                            background: isLightUi ? "#e2e8f0" : "rgba(255,255,255,0.05)",
                            color: isLightUi ? "#0f172a" : "#d1d5db",
                            border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "none",
                        }}>
                        {isCheckingUpdate ? <i className="ph ph-spinner animate-spin"></i> : <i className="ph ph-arrows-clockwise"></i>}
                        {isCheckingUpdate ? "正在检查..." : "检查更新"}
                    </button>
                    <button
                        onClick={() => setShowHelpDocs(true)}
                        className="flex-1 py-2.5 rounded-xl text-sm text-white transition-colors flex items-center justify-center gap-2 shadow-lg"
                        style={{ background: "#0d9488" }}>
                        <i className="ph ph-book-open"></i> 帮助与文档
                    </button>
                </div>
            </div>

            {/* 意见反馈 */}
            <div className="w-full p-6 rounded-2xl" style={{ background: isLightUi ? "#f8fafc" : "rgba(255,255,255,0.02)", border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)" }}>
                <h3 className="text-sm font-medium mb-4 flex items-center gap-2" style={{ color: isLightUi ? "#0f172a" : "#e5e7eb" }}>
                    <i className="ph ph-chat-text text-teal-400"></i> 意见反馈
                </h3>
                <textarea
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                    className="w-full h-32 rounded-xl p-3 resize-none mb-4 text-sm"
                    style={{
                        background: isLightUi ? "#ffffff" : "rgba(255,255,255,0.03)",
                        border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.1)",
                        backdropFilter: "blur(10px)",
                        color: isLightUi ? "#0f172a" : "#e5e7eb",
                        outline: "none"
                    }}
                    placeholder="请描述您遇到的问题或改进建议..."
                ></textarea>
                <div className="flex justify-end">
                    <button
                        onClick={() => {
                            if (!feedbackText.trim()) {
                                triggerToast("请输入反馈内容后提交");
                                return;
                            }
                            setIsSubmittingFeedback(true);
                            setTimeout(() => {
                                setIsSubmittingFeedback(false);
                                setFeedbackText("");
                                triggerToast("提交成功，感谢您的反馈！");
                            }, 1000);
                        }}
                        disabled={isSubmittingFeedback}
                        className="px-5 py-2 rounded-lg text-sm text-white transition-colors hover:bg-white/20 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{
                            background: isLightUi ? "#0f766e" : "rgba(255,255,255,0.1)",
                            color: "#ffffff",
                        }}>
                        {isSubmittingFeedback && <i className="ph ph-spinner animate-spin"></i>}
                        {isSubmittingFeedback ? "提交中..." : "提交反馈"}
                    </button>
                </div>
            </div>

            {/* 后端连接状态 */}
            <div className="w-full flex items-center justify-between px-4 py-3 rounded-xl"
                style={{
                    background: isLightUi ? "#f8fafc" : "rgba(0,0,0,0.2)",
                    border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.05)",
                }}>
                <span className="text-xs" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>后端连接状态</span>
                <div className="flex items-center gap-2 px-2.5 py-1 rounded-full border"
                    style={{ background: "rgba(13,148,136,0.1)", borderColor: "rgba(45,212,191,0.2)" }}>
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>
                    <span className="text-[11px] font-medium" style={{ color: isLightUi ? "#0f766e" : "#2dd4bf" }}>运行中</span>
                </div>
            </div>
        </div>
    );

    return (
        <div
            className={`flex h-screen w-full ${isLightUi ? "db-setting-light" : ""}`}
            style={{ background: themeSurface, color: "#d1d5db", margin: 0, overflow: "hidden" }}
        >
            {/* ========== 左侧边栏 ========== */}
            <aside className="w-64 z-20 flex flex-col h-full shrink-0"
                style={{
                    background: isLightUi ? "rgba(255, 255, 255, 0.92)" : "rgba(4, 21, 39, 0.7)",
                    backdropFilter: "blur(20px)",
                    borderRight: isLightUi ? "1px solid rgba(148, 163, 184, 0.28)" : "1px solid rgba(255, 255, 255, 0.05)"
                }}>
                <div className="p-5">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                            style={{ background: "#0a2530", border: "1px solid rgba(45,212,191,0.3)" }}>
                            <i className="ph ph-steering-wheel text-teal-400 text-xl"></i>
                        </div>
                        <div>
                            <h2 className="text-sm font-bold text-white tracking-wide">智能船舶问答</h2>
                            <p className="text-[10px] uppercase tracking-wider mt-0.5" style={{ color: "#6b7280" }}>DeepBlue AI</p>
                        </div>
                    </div>

                    <button
                        onClick={onBack}
                        className="w-full py-2.5 px-4 rounded-lg border text-sm font-medium transition-colors flex items-center justify-center gap-2 hover:bg-white/5"
                        style={{
                            borderColor: isLightUi ? "rgba(148,163,184,0.5)" : "rgba(255,255,255,0.1)",
                            color: isLightUi ? "#0f172a" : "#d1d5db",
                            background: isLightUi ? "#f8fafc" : "transparent",
                        }}>
                        <i className="ph ph-arrow-left text-lg"></i>
                        返回聊天
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-3 flex flex-col gap-1 mt-2"
                    style={{ scrollbarWidth: "thin", scrollbarColor: "rgba(255,255,255,0.1) transparent" }}>
                    <p className="px-2 text-xs font-medium mb-2 mt-2" style={{ color: isLightUi ? "#334155" : "#6b7280" }}>设置菜单</p>
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-sm text-left ${activeTab === tab.id
                                ? "bg-teal-500/10 border border-teal-500/20 text-teal-400"
                                : "text-gray-400 hover:text-gray-200 hover:bg-white/5"
                                }`}
                        >
                            <i className={`ph ${tab.icon} text-lg`}></i>
                            {tab.label}
                        </button>
                    ))}
                </div>

                <div className="p-4 border-t shrink-0" style={{ borderColor: "rgba(255,255,255,0.05)" }}>
                    <div className="flex items-center gap-3 p-2 rounded-lg"
                        style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.05)" }}>
                        <div className="w-9 h-9 rounded-lg flex items-center justify-center font-bold text-xs shrink-0 text-white overflow-hidden"
                            style={{ background: "#3b82f6" }}>
                            {settings.avatar ? (
                                <img src={settings.avatar} alt="avatar" className="w-full h-full object-cover" />
                            ) : (
                                "CP"
                            )}
                        </div>
                        <div className="flex-1 overflow-hidden">
                            <p className="text-sm font-medium text-white truncate">{settings.nickname}</p>
                            <p className="text-[11px]" style={{ color: "#6b7280" }}>一级轮机长</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* ========== 主内容区 ========== */}
            <main className="flex-1 flex flex-col relative z-10 overflow-hidden"
                style={{ background: contentSurface }}>
                {/* 背景光效 */}
                <div className="absolute pointer-events-none z-0"
                    style={{
                        top: "50%", left: "50%", transform: "translate(-50%, -50%)",
                        width: "800px", height: "800px",
                        background: isLightUi
                            ? "radial-gradient(circle, rgba(148, 163, 184, 0.16) 0%, transparent 65%)"
                            : "radial-gradient(circle, rgba(20, 184, 166, 0.03) 0%, transparent 60%)"
                    }}></div>

                {/* 头部 */}
                <header className="h-16 flex items-center justify-between px-8 shrink-0 relative z-20"
                    style={{
                        background: isLightUi ? "rgba(255, 255, 255, 0.95)" : "rgba(4, 21, 39, 0.4)",
                        backdropFilter: "blur(10px)",
                        borderBottom: isLightUi ? "1px solid rgba(148, 163, 184, 0.28)" : "1px solid rgba(255, 255, 255, 0.05)"
                    }}>
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                            style={{
                                background: isLightUi ? "#f1f5f9" : "rgba(255,255,255,0.03)",
                                border: isLightUi ? "1px solid rgba(148,163,184,0.45)" : "1px solid rgba(255,255,255,0.1)",
                            }}>
                            <i className={`ph ${tabs.find(t => t.id === activeTab)?.icon}`} style={{ color: isLightUi ? "#334155" : "#9ca3af" }}></i>
                        </div>
                        <h1 className="text-base font-medium" style={{ color: isLightUi ? "#0f172a" : "#f3f4f6" }}>{currentTabLabel}</h1>
                    </div>
                    <button onClick={handleSave} className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white transition-colors shadow-lg"
                        style={{ background: "#0d9488" }}>
                        <i className="ph ph-floppy-disk text-lg"></i> 保存更改
                    </button>
                </header>

                {/* 内容区 */}
                <div className="flex-1 overflow-y-auto p-8 relative z-10"
                    style={{ scrollbarWidth: "thin", scrollbarColor: "rgba(255,255,255,0.1) transparent" }}>
                    <div className="max-w-3xl mx-auto pb-20">
                        {activeTab === "personal" && renderPersonalInfo()}
                        {activeTab === "system" && renderSystemPrefs()}
                        {activeTab === "ai" && renderAISettings()}
                        {activeTab === "privacy" && renderPrivacySecurity()}
                        {activeTab === "about" && renderAboutHelp()}
                    </div>
                </div>
            </main>

            {/* ========== 弹窗与提示 ========== */}
            <ConfirmDialog
                isOpen={showConfirm}
                onClose={() => setShowConfirm(false)}
                onConfirm={() => {
                    if (onClearChats) onClearChats();
                    setShowConfirm(false);
                    triggerToast("已成功清空所有历史记录");
                }}
                title="清空历史记录"
                message="此操作不可逆。是否确认删除本设备上的所有本地问答缓存及图谱分析记录？"
            />

            <Toast show={showToast} message={toastMessage} />
            <HelpDocsDialog isOpen={showHelpDocs} onClose={() => setShowHelpDocs(false)} />

            {/* 动画样式 */}
            <style>{`
        .animate-fade-in {
          animation: fadeIn 0.3s ease-out forwards;
        }
        .db-setting-light .text-white {
          color: #0f172a !important;
        }
        .db-setting-light .text-gray-400,
        .db-setting-light .text-gray-500,
        .db-setting-light .text-gray-600 {
          color: #475569 !important;
        }
        .db-setting-light .text-gray-200,
        .db-setting-light .text-gray-300 {
          color: #1e293b !important;
        }
        .db-setting-light .glass-input {
          background: rgba(255, 255, 255, 0.98) !important;
          border-color: rgba(148, 163, 184, 0.45) !important;
        }
        .db-setting-light .glass-input input,
        .db-setting-light .glass-input select {
          color: #0f172a !important;
        }
        .db-setting-light .glass-input input::placeholder {
          color: #64748b !important;
        }
        .db-setting-light .glass-input select option {
          background: #ffffff !important;
          color: #0f172a !important;
        }
        .db-setting-light [style*="color: #e5e7eb"],
        .db-setting-light [style*="color:#e5e7eb"],
        .db-setting-light [style*="color: #d1d5db"],
        .db-setting-light [style*="color:#d1d5db"] {
          color: #0f172a !important;
        }
        .db-setting-light [style*="color: #9ca3af"],
        .db-setting-light [style*="color:#9ca3af"],
        .db-setting-light [style*="color: #6b7280"],
        .db-setting-light [style*="color:#6b7280"] {
          color: #475569 !important;
        }
        .db-setting-light [style*="background: rgba(255,255,255,0.02)"],
        .db-setting-light [style*="background: rgba(255,255,255,0.03)"],
        .db-setting-light [style*="background: rgba(255,255,255,0.05)"] {
          background: rgba(255, 255, 255, 0.9) !important;
        }
        .db-setting-light [style*="background: rgba(0,0,0,0.2)"] {
          background: rgba(248, 250, 252, 0.95) !important;
        }
        .db-setting-light [style*="border: 1px solid rgba(255,255,255,0.05)"],
        .db-setting-light [style*="border: 1px solid rgba(255,255,255,0.1)"],
        .db-setting-light [style*="border-right: 1px solid rgba(255, 255, 255, 0.05)"],
        .db-setting-light [style*="border-bottom: 1px solid rgba(255, 255, 255, 0.05)"],
        .db-setting-light [style*="border-top: 1px solid rgba(255,255,255,0.05)"] {
          border-color: rgba(148, 163, 184, 0.35) !important;
        }
        .db-setting-light .text-red-400 {
          color: #dc2626 !important;
        }
        .db-setting-light .hover\\:bg-white\\/5:hover {
          background: rgba(148, 163, 184, 0.12) !important;
        }
        .db-setting-light .hover\\:bg-red-500\\/10:hover {
          background: rgba(239, 68, 68, 0.12) !important;
        }
        .db-setting-light .range-slider {
          background: rgba(148, 163, 184, 0.35) !important;
        }
        .db-setting-light .range-slider::-webkit-slider-thumb {
          appearance: none;
          width: 14px;
          height: 14px;
          border-radius: 999px;
          background: #0d9488;
          border: 2px solid #ffffff;
          box-shadow: 0 0 0 1px rgba(13, 148, 136, 0.3);
          cursor: pointer;
        }
        .db-setting-light .range-slider::-moz-range-thumb {
          width: 14px;
          height: 14px;
          border-radius: 999px;
          background: #0d9488;
          border: 2px solid #ffffff;
          box-shadow: 0 0 0 1px rgba(13, 148, 136, 0.3);
          cursor: pointer;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(5px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
        </div>
    );
}
