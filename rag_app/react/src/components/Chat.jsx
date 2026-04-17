import { useState, useRef, useEffect, useCallback } from "react";
import * as echarts from "echarts";
import Setting from "./setting";

// ===================== 模拟后端 API =====================
const API_HOST = window.location.hostname || "127.0.0.1";
const API_CONFIG = {
  baseURL: `http://${API_HOST}:8000`,
  timeoutOnline: 90000,
  timeoutOffline: 30000,
};
const DEFAULT_USER_SETTINGS = {
  nickname: "Captain Park",
  avatar: "",
  imo: "9876543",
  email: "captain.park@deepblue.com",
  emergency: "+86 13800138000",
  theme: "dark",
  fontSize: 16,
  notify: true,
  autoSave: true,
  graphOn: true,
  offlineOn: false,
  streamSpeed: 50,
  dbPath: "/mnt/data/local_kb",
  model: "hybrid",
  retention: "30",
  neo4jUri: "bolt://127.0.0.1:7687",
  neo4jUser: "neo4j",
  neo4jPassword: "",
};
const THEME_SURFACES = {
  dark: {
    shellBg: "#030d17",
    mainBg: "#041527",
    panelBg: "rgba(4, 21, 39, 0.7)",
    panelBorder: "rgba(255, 255, 255, 0.05)",
    headerBg: "rgba(4, 21, 39, 0.4)",
    inputBg: "rgba(255, 255, 255, 0.03)",
    inputBorder: "rgba(255, 255, 255, 0.1)",
    glowBg: "radial-gradient(circle, rgba(20, 184, 166, 0.05) 0%, transparent 70%)",
    footerBg: "linear-gradient(to top, #041527 0%, #041527 60%, transparent 100%)",
  },
  light: {
    shellBg: "#f8fafc",
    mainBg: "#ffffff",
    panelBg: "rgba(255, 255, 255, 0.9)",
    panelBorder: "rgba(148, 163, 184, 0.3)",
    headerBg: "rgba(255, 255, 255, 0.96)",
    inputBg: "rgba(255, 255, 255, 0.98)",
    inputBorder: "rgba(148, 163, 184, 0.3)",
    glowBg: "radial-gradient(circle, rgba(148, 163, 184, 0.2) 0%, transparent 70%)",
    footerBg: "linear-gradient(to top, #ffffff 0%, #ffffff 60%, transparent 100%)",
  },
};
const detectSystemDark = () => {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return false;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
};
const PROJECT_META = {
  name: "智能船舶问答系统",
  version: "2.4.1",
  build: "8829",
  releaseDate: "2024-01-15",
};
const normalizeSettings = (settings = {}) => ({
  ...DEFAULT_USER_SETTINGS,
  ...(settings || {}),
});

const sanitizeUserId = (value = "") => String(value || "").replace(/[^a-zA-Z0-9_-]/g, "_");

const getSessionUserId = (baseUserId, chatId) =>
  `${sanitizeUserId(baseUserId || "captain_park")}_${String(chatId || "session").replace(/[^a-zA-Z0-9_-]/g, "_")}`;

// SSE 流式请求模拟（实际项目中替换为真实的 EventSource）
function* mockStreamResponse(text, isGraphEnabled, isOffline) {
  const isGraphOn = isGraphEnabled;
  const offlineMode = isOffline;

  let res = "";
  if (offlineMode) {
    res += "⚠️ **当前为离线模式**，系统正使用本地缓存模型为您解答。\n\n";
  }

  res += `关于您提问的 "${text}"：\n\n`;

  if (isGraphOn) {
    res += "通过 **知识图谱引擎** 深度关联分析，发现以下隐患路径：\n";
    res += "1. 结合近3个月的维护记录，关联设备存在 85% 的磨损老化概率。\n";
    res += "2. 当前环境参数波动可能触发了安全阈值临界点。\n";
    res += "3. 知识图谱路径搜索返回 12 条相关推理链路，可信度 0.91。\n";
    res += "\n建议采取以下高级排查步骤：\n- 隔离关联的液压分路\n- 对比同型号设备的运行曲线\n- 检查通讯总线的干扰情况";
  } else {
    res += "基于 **常规知识库** 检索标准处理手册：\n";
    res += "常规处理步骤如下：\n";
    res += "1. 确认主控面板的报警代码。\n";
    res += "2. 按照手册执行复位操作。\n";
    res += "3. 若无法恢复，请联系岸基工程师。\n";
    res += "\n建议：开启右侧【知识图谱分析】获取更精准的深层诊断。";
  }

  // 逐字符模拟流式输出
  for (let i = 0; i < res.length; i++) {
    yield res[i];
  }
}

// ===================== 数据模型 =====================
// 清空测试数据
const INITIAL_CHATS = [];

const QUICK_CARDS = [
  { title: "主机故障诊断", desc: "分析推进系统异常", icon: "ph-engine" },
  { title: "电气系统咨询", desc: "电路故障排查建议", icon: "ph-lightning" },
  { title: "液压系统分析", desc: "压力与流量优化", icon: "ph-drop" },
  { title: "维护周期建议", desc: "基于历史数据分析", icon: "ph-wrench" },
];

// ===================== 工具函数 =====================
const escapeHtml = (s = "") =>
  s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

const stripRagArtifacts = (text = "") =>
  text
    // 移除离线抽取模式的固定提示语
    .replace(/^\s*基于检索到的资料，建议如下[：:]\s*/g, "")

    // [H1] [H2]...
    .replace(/\[(?:H\d+|h\d+)\]\s*/g, "")

    // [IMG ...]（有闭合或无闭合都清理）
    .replace(/\[\s*(?:IMG|img)[^\]\n\r]*(?:\]|$)/g, "")

    // IMG path=...（有无方括号都清理，直到行尾）
    .replace(/\b(?:IMG|img)\s*path\s*=\s*(?:data\/KG\/images|images)\/[^\n\r]*/g, "")

    // 裸图片路径（有扩展名）
    .replace(/\b(?:images|data\/KG\/images)\/\S+\.(?:png|jpe?g|gif|webp|bmp)\b/gi, "")

    // 离线抽取内容中常见的哈希图片路径噪声
    .replace(/\b(?:images|data\/KG\/images)\/[a-f0-9]{64}(?:\.(?:png|jpe?g|gif|webp|bmp))?\b/gi, "")
    .replace(/\bimages\/a01b4c420131362891ab24ba13f1823c5b07d9d4cef48f98b71bd28945bef1c7\b/g, "")

    // [图片附件]...
    .replace(/\[图片附件[^\]]*\]/g, "")

    // 清理残留孤立括号（仅限特定无用标记，保留内容展示）
    .replace(/[【】]/g, "")

    // 空白整理
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

// 去掉常见 Markdown 语法（先清理 RAG 标记）
const stripMarkdown = (text = "") =>
  stripRagArtifacts(text) // 关键：这里必须调用
    .replace(/```[\s\S]*?```/g, (m) => m.replace(/```/g, ""))
    .replace(/`([^`]*)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]+\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    // 移除会导致有序列表被破坏的正则
    // .replace(/^\s*\d+\.\s+/gm, "")
    // .replace(/^\s*[-+*]\s+/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/~~([^~]+)~~/g, "$1")
    .replace(/\r/g, "")
    .replace(/\n{3,}/g, "\n\n");

// 渲染为纯文本（保留换行）
const formatText = (text) => {
  const plain = stripMarkdown(text);
  return escapeHtml(plain).replace(/\n/g, "<br>");
};

const normalizeOfflineText = (text = "") => {
  let normalized = stripRagArtifacts(text)
    .replace(/\r/g, "")
    .replace(/\\leq/g, "≤")
    .replace(/\\geq/g, "≥")
    .replace(/\\pm/g, "±")
    .replace(/\\sim/g, "~")
    .replace(/\\times/g, "×")
    .replace(/\\%/g, "%")
    .replace(/\^\{\\circ\}/g, "°")
    .replace(/\\circ/g, "°")
    .replace(/\\mathrm\{([^}]*)\}/g, "$1")
    .replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/\\left|\\right/g, "")
    .replace(/[{}]/g, "")
    .replace(/\\/g, "")
    .replace(/\s+([,.;:，。；：])/g, "$1")
    .replace(/([\u4e00-\u9fa5A-Za-z0-9])\s+([0-9]+[、．.])/g, "$1\n$2")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  normalized = normalized
    .replace(/(\d+[\.．、]\s*[^\s，。；;:：]{2,40})\s+\1/g, "$1")
    .replace(/([一二三四五六七八九十]+、[^\s，。；;:：]{2,40})\s+\1/g, "$1");

  const headingOnlyPattern =
    /^\s*(?:\d+[\.．、]\s*)?(?:[一二三四五六七八九十]+、)?[^\s，。；;:：]{2,30}\s*$/;
  const lines = normalized.split("\n");
  const dedupedLines = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      if (dedupedLines[dedupedLines.length - 1] !== "") dedupedLines.push("");
      continue;
    }

    const prev = dedupedLines[dedupedLines.length - 1] || "";
    if (line === prev) continue;

    let current = line;
    if (prev && headingOnlyPattern.test(prev) && current.startsWith(`${prev} `)) {
      current = current.slice(prev.length).trim();
      if (!current) continue;
    }

    dedupedLines.push(current);
  }

  return dedupedLines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
};

const formatOfflineMarkdown = (text = "") => {
  const cleaned = normalizeOfflineText(text);
  if (!cleaned) return "";

  const codeBlocks = [];
  let html = escapeHtml(cleaned).replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push({ lang: lang || "", code: code || "" });
    return token;
  });

  html = html
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

  const lines = html.split("\n");
  const output = [];
  let listType = null;

  const closeList = () => {
    if (!listType) return;
    output.push(listType === "ol" ? "</ol>" : "</ul>");
    listType = null;
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();

    if (!trimmed) {
      closeList();
      output.push("");
      continue;
    }

    const ulMatch = line.match(/^\s*[-*+]\s+(.+)$/);
    if (ulMatch) {
      if (listType !== "ul") {
        closeList();
        output.push("<ul>");
        listType = "ul";
      }
      output.push(`<li>${ulMatch[1]}</li>`);
      continue;
    }

    const olMatch = line.match(/^\s*\d+[\.．、]\s*(.+)$/);
    if (olMatch) {
      if (listType !== "ol") {
        closeList();
        output.push("<ol>");
        listType = "ol";
      }
      output.push(`<li>${olMatch[1]}</li>`);
      continue;
    }

    closeList();

    if (/^###\s+/.test(trimmed)) {
      output.push(`<h3>${trimmed.replace(/^###\s+/, "")}</h3>`);
    } else if (/^##\s+/.test(trimmed)) {
      output.push(`<h2>${trimmed.replace(/^##\s+/, "")}</h2>`);
    } else if (/^#\s+/.test(trimmed)) {
      output.push(`<h1>${trimmed.replace(/^#\s+/, "")}</h1>`);
    } else if (/^>\s?/.test(trimmed)) {
      output.push(`<blockquote>${trimmed.replace(/^>\s?/, "")}</blockquote>`);
    } else {
      output.push(trimmed);
    }
  }
  closeList();

  let blockHtml = output
    .join("\n")
    .split(/\n{2,}/)
    .map((block) => {
      const t = block.trim();
      if (!t) return "";
      if (/^<(h1|h2|h3|ul|ol|blockquote)/.test(t)) return t;
      if (/^__CODE_BLOCK_\d+__$/.test(t)) return t;
      return `<p>${t.replace(/\n/g, "<br>")}</p>`;
    })
    .filter(Boolean)
    .join("");

  blockHtml = blockHtml.replace(/__CODE_BLOCK_(\d+)__/g, (_, idx) => {
    const item = codeBlocks[Number(idx)] || { lang: "", code: "" };
    const lang = item.lang ? `<span class="md-code-lang">${item.lang}</span>` : "";
    return `<pre class="md-pre">${lang}<code>${item.code}</code></pre>`;
  });

  return blockHtml;
};

// 新增：读取设置工具
const getSystemSettings = () => {
  return { ...DEFAULT_USER_SETTINGS };
};

// 新增：网页标题/标签页闪烁提示（替代提示音）
let flashInterval = null;
const originalTitle = document.title || "智能船舶问答系统";

const notifyNewMessage = (settings) => {
  const current = normalizeSettings(settings || getSystemSettings());
  if (!current.notify) return;
  if (flashInterval) clearInterval(flashInterval);
  let flashFlag = false;
  flashInterval = setInterval(() => {
    document.title = flashFlag ? originalTitle : "【新消息】" + originalTitle;
    flashFlag = !flashFlag;
  }, 500);
  const clearFlash = () => {
    clearInterval(flashInterval);
    flashInterval = null;
    document.title = originalTitle;
    window.removeEventListener("mousemove", clearFlash);
    window.removeEventListener("focus", clearFlash);
    window.removeEventListener("click", clearFlash);
  };
  window.addEventListener("mousemove", clearFlash);
  window.addEventListener("focus", clearFlash);
  window.addEventListener("click", clearFlash);
};

const toMarkdownText = (text = "", isOffline = false) => {
  if (!text) return "";
  return isOffline
    ? normalizeOfflineText(text)
    : stripRagArtifacts(text).replace(/\r/g, "").trim();
};

const shortenLabel = (value = "", max = 10) => {
  const text = String(value || "").trim();
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1)}...`;
};

const formatCitationSnippet = (text = "", maxLen = 180) => {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return "";
  if (clean.length <= maxLen) return clean;
  return `${clean.slice(0, maxLen)}...`;
};

const generateMarkdown = (chat) => {
  const safeTitle = (chat?.title || "未命名会话").replace(/[\r\n]+/g, " ").trim();
  const lines = [
    `# ${safeTitle}`,
    "",
    `- 导出时间：${new Date().toLocaleString("zh-CN", { hour12: false })}`,
    "- 导出格式：Markdown",
    "",
    "---",
    "",
  ];

  (chat?.messages || []).forEach((msg, idx) => {
    const roleLabel = msg.role === "user" ? "用户" : "助手";
    const isOffline =
      msg.role === "ai" &&
      (!!msg.isOffline || /离线模式|\[离线\]/.test(msg.searchProcess || ""));
    const modeLabel = msg.role === "ai" ? (isOffline ? "（离线回答）" : "（在线回答）") : "";
    const content = toMarkdownText(msg.content || "", isOffline);

    lines.push(`## ${idx + 1}. ${roleLabel}${modeLabel}`);
    lines.push("");
    lines.push(content || "_（无内容）_");
    lines.push("");

    if (msg.role === "ai" && Array.isArray(msg.citations) && msg.citations.length > 0) {
      lines.push("### 参考来源");
      msg.citations.forEach((c, i) => {
        const name = c?.name || c?.source || `文档${i + 1}`;
        const url = c?.url || "";
        lines.push(url ? `${i + 1}. [${name}](${url})` : `${i + 1}. ${name}`);
      });
      lines.push("");
    }

    if (msg.role === "ai" && Array.isArray(msg.kgTriplets) && msg.kgTriplets.length > 0) {
      lines.push("### 知识图谱关系");
      msg.kgTriplets.forEach((t) => {
        const head = t?.head || "";
        const rel = t?.rel || t?.relation || "关联";
        const tail = t?.tail || "";
        lines.push(`- ${head} --${rel}--> ${tail}`);
      });
      lines.push("");
    }

    lines.push("---");
    lines.push("");
  });

  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim() + "\n";
};

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
          <button type="button" onClick={onClose} className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-white/10 transition-colors text-gray-400" title="关闭">
            <i className="ph ph-x"></i>
          </button>
        </div>

        <div className="space-y-4 text-sm">
          <section className="p-4 rounded-xl" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
            <p className="text-xs mb-2 text-gray-500">项目版本信息</p>
            <div className="grid grid-cols-2 gap-3 text-xs text-gray-300">
              <p>项目名称：<span className="text-gray-100">{PROJECT_META.name}</span></p>
              <p>版本号：<span className="text-teal-300">{PROJECT_META.version}</span></p>
              <p>构建号：<span className="text-gray-100">{PROJECT_META.build}</span></p>
              <p>发布日期：<span className="text-gray-100">{PROJECT_META.releaseDate}</span></p>
            </div>
          </section>

          <section className="p-4 rounded-xl" style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
            <p className="text-xs mb-2 text-gray-500">核心接口</p>
            <div className="space-y-1 text-xs text-gray-300">
              <p><span className="text-teal-300">POST</span> {API_CONFIG.baseURL}/rag/query/stream</p>
              <p><span className="text-teal-300">POST</span> {API_CONFIG.baseURL}/rag/query</p>
              <p><span className="text-teal-300">POST</span> {API_CONFIG.baseURL}/rag/incremental/upload</p>
            </div>
          </section>
        </div>

        <div className="mt-5 flex justify-end">
          <button type="button" onClick={onClose} className="px-4 py-2 rounded-lg text-sm text-white transition-colors" style={{ background: "#0d9488" }}>
            我知道了
          </button>
        </div>
      </div>
    </div>
  );
};

// ===================== 组件 =====================
export default function Chat({ user }) {
  const currentUser = user?.user || {};
  const userId = String(currentUser.user_id || "captain_park");
  const userRole = Number(currentUser.role || 0);

  const [userSettings, setUserSettings] = useState(() =>
    normalizeSettings(user?.settings || {})
  );

  const [chats, setChats] = useState(() =>
    Array.isArray(user?.chats) ? user.chats : INITIAL_CHATS
  );
  const [currentChatId, setCurrentChatId] = useState(null);
  const [inputValue, setInputValue] = useState("");

  // 修改：连接用户设置的默认状态
  const [graphEnabled, setGraphEnabled] = useState(userSettings?.graphOn ?? true);
  const [offlineMode, setOfflineMode] = useState(userSettings?.offlineOn ?? false); // 离线模式
  const [isAnswering, setIsAnswering] = useState(false); // 新增：是否正在回答
  const [showSetting, setShowSetting] = useState(false); // 新增：设置页面开关
  const [showHelpDocs, setShowHelpDocs] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadingFileName, setUploadingFileName] = useState("");
  const [uploadStatus, setUploadStatus] = useState(null);
  const [recentUploads, setRecentUploads] = useState([]);
  const [systemDark, setSystemDark] = useState(() => detectSystemDark());
  const themePref = userSettings?.theme || "dark";
  const effectiveTheme = themePref === "system" ? (systemDark ? "dark" : "light") : themePref;
  const themeSurface = THEME_SURFACES[effectiveTheme] || THEME_SURFACES.dark;
  const isLightUi = effectiveTheme === "light";

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    const parsedFontSize = Number(userSettings?.fontSize);
    const nextFontSize =
      Number.isFinite(parsedFontSize) && parsedFontSize > 0
        ? parsedFontSize
        : DEFAULT_USER_SETTINGS.fontSize;
    document.documentElement.style.fontSize = `${nextFontSize}px`;
  }, [userSettings?.fontSize]);

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

  const chatPersistReadyRef = useRef(false);

  useEffect(() => {
    setUserSettings(normalizeSettings(user?.settings || {}));
    const nextChats = Array.isArray(user?.chats) ? user.chats : INITIAL_CHATS;
    setChats(nextChats);
    setCurrentChatId(null);
    chatPersistReadyRef.current = false;
  }, [user]);

  useEffect(() => {
    if (!userId) return undefined;
    if (!chatPersistReadyRef.current) {
      chatPersistReadyRef.current = true;
      return undefined;
    }
    if (!userSettings.autoSave) return undefined;
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`${API_CONFIG.baseURL}/users/${encodeURIComponent(userId)}/chats`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chats }),
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          console.warn("chat save failed", response.status, text);
        }
      } catch (error) {
        console.warn("chat save request failed", error?.message || error);
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [chats, userId, userSettings.autoSave]);

  // 流式会话管理
  const streamingRefs = useRef(new Map());

  const chatViewRef = useRef(null);
  const fileInputRef = useRef(null);
  const chatInputRef = useRef(null);

  const currentChat = chats.find((c) => c.id === currentChatId) || null;
  const isWelcome = !currentChatId;

  // 当前正在流式输出的 AI 消息（取最后一条）
  const activeStreamingMsg = [...(currentChat?.messages || [])]
    .reverse()
    .find((m) => m.role === "ai" && m.isStreaming);
  const activeStreamingMsgId = activeStreamingMsg?.id;
  const activeStreamingPaused = !!activeStreamingMsg?.isPaused;

  // 自动滚动
  useEffect(() => {
    if (chatViewRef.current && currentChat?.messages.length > 0) {
      chatViewRef.current.scrollTo({
        top: chatViewRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [currentChat?.messages]);

  // 清理流式会话
  useEffect(() => {
    return () => {
      streamingRefs.current.forEach((session) => {
        session.controller?.abort();
        if (session.timeoutId) clearTimeout(session.timeoutId);
        if (session.typeTimerId) clearInterval(session.typeTimerId);
      });
      streamingRefs.current.clear();
    };
  }, []);

  useEffect(() => {
    if (!uploadStatus) return undefined;
    const timer = setTimeout(() => setUploadStatus(null), 6000);
    return () => clearTimeout(timer);
  }, [uploadStatus]);

  // 完成流式输出（提前定义，避免 TDZ 问题）
  const finishStreaming = useCallback((chatId, msgId, finalText) => {
    setChats((prev) =>
      prev.map((c) =>
        c.id !== chatId
          ? c
          : {
            ...c,
            messages: c.messages.map((m) =>
              m.id === msgId
                ? {
                  ...m,
                  content: finalText || " ",
                  fullText: finalText || " ",
                  isStreaming: false,
                  isPaused: false,
                }
                : m
            ),
          }
      )
    );
  }, []);

  // 1. 发送消息并连接后端（流式）
  const handleSend = useCallback(async (text) => {
    text = (text || "").trim();
    if (!text) return;

    if (isAnswering) {
      alert("请等待当前回答完成后，再发送下一条问题。");
      return;
    }

    setInputValue("");

    let chatId = currentChatId;
    let updatedChats = [...chats];
    let sessionUserId = "";

    if (!chatId) {
      chatId = Date.now().toString();
      const title = text.length > 12 ? text.substring(0, 12) + "..." : text;
      sessionUserId = getSessionUserId(chatId);
      updatedChats = [{ id: chatId, title, sessionUserId, messages: [], createdAt: new Date() }, ...updatedChats];
      setCurrentChatId(chatId);
    } else {
      const current = updatedChats.find((c) => c.id === chatId);
      sessionUserId = current?.sessionUserId || getSessionUserId(chatId);
      if (!current?.sessionUserId) {
        updatedChats = updatedChats.map((c) =>
          c.id === chatId ? { ...c, sessionUserId } : c
        );
      }
    }

    const userMsg = { role: "user", content: text };
    const msgId = `ai-${Date.now()}`;
    const aiMsg = {
      id: msgId,
      role: "ai",
      content: "",
      fullText: "",
      searchProcess: `正在检索，模式: ${graphEnabled ? "知识图谱" : "常规检索"}${offlineMode ? " [离线]" : " [在线]"}`,
      traceSteps: [
        {
          stage: "queued",
          text: `请求已提交（${offlineMode ? "离线" : "在线"}模式）`,
        },
      ],
      citations: [],
      kgTriplets: [],
      isStreaming: true,
      isPaused: false,
      isOffline: offlineMode,
      isGraphEnabled: graphEnabled,
    };

    updatedChats = updatedChats.map((c) =>
      c.id === chatId ? { ...c, messages: [...c.messages, userMsg, aiMsg] } : c
    );
    setChats(updatedChats);
    setIsAnswering(true);

    startStreaming(chatId, msgId, text, sessionUserId);
  }, [chats, currentChatId, graphEnabled, offlineMode, isAnswering]);

  // 流式输出逻辑（打字机）
  const startStreaming = useCallback(async (chatId, msgId, originalText, sessionUserId) => {
    const isOnlineMode = !offlineMode; // 离线关闭 => 在线
    const streamSession = {
      controller: new AbortController(),
      timeoutId: null,
      typeTimerId: null,
      isPaused: false,
      pendingText: "",
      renderedText: "",
      streamDone: false,
      finished: false,
      abortReason: "", // 新增
    };
    streamingRefs.current.set(msgId, streamSession);

    const cleanup = () => {
      if (streamSession.timeoutId) clearTimeout(streamSession.timeoutId);
      if (streamSession.typeTimerId) clearInterval(streamSession.typeTimerId);
      streamingRefs.current.delete(msgId);
    };

    const flushToUI = () => {
      const text = streamSession.renderedText;
      setChats((prev) =>
        prev.map((c) =>
          c.id !== chatId
            ? c
            : {
              ...c,
              messages: c.messages.map((m) =>
                m.id === msgId ? { ...m, content: text, fullText: text, isPaused: streamSession.isPaused } : m
              ),
            }
        )
      );
    };

    const tryFinalize = () => {
      if (streamSession.finished) return;
      if (!streamSession.streamDone) return;
      if (streamSession.pendingText.length > 0) return;

      streamSession.finished = true;
      finishStreaming(chatId, msgId, streamSession.renderedText || " ");
      setIsAnswering(false);
      cleanup();
    };

    // 修改：根据设置中的 streamSpeed 动态计算打字间隔和单次字符量 (区间10~100)
    const speed = userSettings?.streamSpeed || 50;
    const tickRate = Math.max(10, 70 - speed); // 速度越大，间隔越短 (10ms - 60ms)
    const chunkSize = speed > 80 ? 4 : (speed > 50 ? 2 : 1); // 极速时加大块尺寸

    // 打字机：每 tick 输出少量字符
    streamSession.typeTimerId = setInterval(() => {
      if (streamSession.finished || streamSession.isPaused) return;

      if (!streamSession.pendingText.length) {
        tryFinalize();
        return;
      }

      streamSession.renderedText += streamSession.pendingText.slice(0, chunkSize);
      streamSession.pendingText = streamSession.pendingText.slice(chunkSize);
      flushToUI();

      if (streamSession.streamDone && streamSession.pendingText.length === 0) {
        tryFinalize();
      }
    }, tickRate);

    const parseSSEBlock = (block) => {
      let event = "message";
      const dataLines = [];
      for (const raw of block.split("\n")) {
        const line = raw.trimEnd();
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) return null;
      return { event, data: dataLines.join("\n") };
    };

    try {
      const timeoutMs = isOnlineMode ? API_CONFIG.timeoutOnline : API_CONFIG.timeoutOffline;
      streamSession.timeoutId = setTimeout(() => {
        streamSession.abortReason = "timeout";
        streamSession.controller.abort();
      }, timeoutMs);

      const response = await fetch(`${API_CONFIG.baseURL}/rag/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          user_id: sessionUserId || getSessionUserId(chatId),
          question: originalText,
          top_k: 5,
          use_kg: graphEnabled,
          use_llm: !offlineMode,
          use_history: true,
          enable_retrieval_optimization: true,
          neo4j_uri: String(userSettings?.neo4jUri || "").trim(),
          neo4j_user: String(userSettings?.neo4jUser || "").trim(),
          neo4j_password: String(userSettings?.neo4jPassword || "").trim(),
        }),
        signal: streamSession.controller.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!response.body) throw new Error("empty stream body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";

        for (const block of blocks) {
          const evt = parseSSEBlock(block);
          if (!evt) continue;

          let data = {};
          try {
            data = JSON.parse(evt.data);
          } catch {
            continue;
          }

          if (evt.event === "meta") {
            setChats((prev) =>
              prev.map((c) =>
                c.id !== chatId
                  ? c
                  : {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === msgId
                        ? {
                          ...m,
                          searchProcess: `检索中（${isOnlineMode ? "在线模式" : "离线模式"}）：${data.question || originalText}`,
                          traceSteps: [
                            {
                              stage: "meta",
                              text: `请求已发送（${isOnlineMode ? "在线" : "离线"}模式）`,
                            },
                          ],
                        }
                        : m
                    ),
                  }
              )
            );
          } else if (evt.event === "trace") {
            const stage = data.stage || "trace";
            let traceText = data.message || "处理中";
            if (stage === "retrieve") {
              traceText = `已检索知识库（命中 ${data.context_count ?? 0} 条）${typeof data.elapsed_ms === "number" ? `，耗时 ${data.elapsed_ms}ms` : ""}`;
            } else if (stage === "kg") {
              traceText = `已查询知识图谱（命中 ${data.kg_triplet_count ?? 0} 条）${typeof data.elapsed_ms === "number" ? `，耗时 ${data.elapsed_ms}ms` : ""}`;
            } else if (stage === "generate") {
              traceText = `开始生成答案（${data.mode === "online" ? "在线模型" : "离线抽取"}）`;
            }

            setChats((prev) =>
              prev.map((c) =>
                c.id !== chatId
                  ? c
                  : {
                    ...c,
                    messages: c.messages.map((m) => {
                      if (m.id !== msgId) return m;
                      const prevSteps = Array.isArray(m.traceSteps) ? m.traceSteps : [];
                      return {
                        ...m,
                        searchProcess: traceText,
                        traceSteps: [...prevSteps, { stage, text: traceText }],
                      };
                    }),
                  }
              )
            );
          } else if (evt.event === "token" && data.text) {
            streamSession.pendingText += data.text;
          } else if (evt.event === "references") {
            const citations = (data.citations || []).map((cit) => ({
              name: cit.source || cit.doc_id || "unknown",
              source: cit.source || "未知来源",
              docId: cit.doc_id || "",
              score: typeof cit.score === "number" ? cit.score : null,
              text: String(cit.text || "").trim(),
            }));
            setChats((prev) =>
              prev.map((c) =>
                c.id !== chatId
                  ? c
                  : {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === msgId
                        ? { ...m, citations, searchProcess: `检索完成，共 ${citations.length} 篇参考文档` }
                        : m
                    ),
                  }
              )
            );
          } else if (evt.event === "kg") {
            setChats((prev) =>
              prev.map((c) =>
                c.id !== chatId
                  ? c
                  : {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === msgId ? { ...m, kgTriplets: data.triplets || [] } : m
                    ),
                  }
              )
            );
          } else if (evt.event === "error") {
            streamSession.pendingText += `\n\n连接失败：${data.message || "unknown"}`;
            streamSession.streamDone = true;
            tryFinalize();
            return;
          } else if (evt.event === "done") {
            setChats((prev) =>
              prev.map((c) =>
                c.id !== chatId
                  ? c
                  : {
                    ...c,
                    messages: c.messages.map((m) => {
                      if (m.id !== msgId) return m;
                      const prevSteps = Array.isArray(m.traceSteps) ? m.traceSteps : [];
                      const doneText = `回答完成（总耗时 ${data.elapsed_ms ?? "-"}ms）`;
                      return {
                        ...m,
                        searchProcess: doneText,
                        traceSteps: [...prevSteps, { stage: "done", text: doneText }],
                      };
                    }),
                  }
              )
            );
            streamSession.streamDone = true;
            tryFinalize();
            return;
          }
        }
      }

      streamSession.streamDone = true;
      tryFinalize();
    } catch (error) {
      if (!streamSession.finished) {
        if (error?.name === "AbortError") {
          if (streamSession.abortReason === "timeout") {
            streamSession.pendingText += "\n\n请求超时（在线模型响应较慢），请重试。";
          } else if (streamSession.abortReason === "switch_chat") {
            // 切会话中断：不追加错误文案
          } else {
            streamSession.pendingText += "\n\n请求已中断";
          }
        } else {
          streamSession.pendingText += `\n\n连接异常：${error.message}`;
        }
        streamSession.streamDone = true;
        tryFinalize();
      }
    }
  }, [graphEnabled, offlineMode, finishStreaming]);

  // 暂停回答：暂停打字机，不中断请求
  const pauseStreaming = useCallback((msgId) => {
    const session = streamingRefs.current.get(msgId);
    if (!session) return;
    session.isPaused = true;

    setChats((prev) =>
      prev.map((c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === msgId ? { ...m, isPaused: true } : m)),
      }))
    );
  }, []);

  // 继续回答：恢复打字机
  const resumeStreaming = useCallback((msgId) => {
    const session = streamingRefs.current.get(msgId);
    if (!session) return;
    session.isPaused = false;

    setChats((prev) =>
      prev.map((c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === msgId ? { ...m, isPaused: false } : m)),
      }))
    );
  }, []);

  // 新建会话
  const handleNewChat = useCallback(() => {
    streamingRefs.current.forEach((session) => {
      session.abortReason = "switch_chat";
      session.controller?.abort();
    });
    streamingRefs.current.clear();
    setCurrentChatId(null);
    setInputValue("");
    setIsAnswering(false);

    // 修改：新建会话时，恢复用户设定的默认开关状态
    const settings = userSettings;
    setGraphEnabled(settings.graphOn ?? true);
    setOfflineMode(settings.offlineOn ?? false);
  }, [userSettings]);

  // 加载历史会话
  const loadChat = useCallback((id) => {
    streamingRefs.current.forEach((session) => {
      session.abortReason = "switch_chat";
      session.controller?.abort();
    });
    streamingRefs.current.clear();

    setCurrentChatId(id);
    const chat = chats.find((c) => c.id === id);
    if (chat && chat.messages.length === 0) {
      const restoreMsg = {
        id: `restore-${Date.now()}`,
        role: "ai",
        content: `已恢复会话："${chat.title}"。请继续提问。`,
        isStreaming: false,
      };
      setChats((prev) => prev.map((c) => (c.id === id ? { ...c, messages: [restoreMsg] } : c)));
    }
  }, [chats]);

  // 文件上传
  const handleFileChange = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const uploadId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const fileName = file.name || "未命名文件";
    const pushUploadRecord = (record) => {
      setRecentUploads((prev) => [record, ...prev.filter((x) => x.id !== record.id)].slice(0, 3));
    };
    const patchUploadRecord = (patch) => {
      setRecentUploads((prev) => prev.map((x) => (x.id === uploadId ? { ...x, ...patch } : x)).slice(0, 3));
    };

    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!["md", "pdf", "txt"].includes(ext)) {
      setUploadStatus({ type: "error", text: "仅支持 md / pdf / txt 文档" });
      pushUploadRecord({
        id: uploadId,
        name: fileName,
        status: "error",
        detail: "格式不支持（仅 md/pdf/txt）",
      });
      e.target.value = "";
      return;
    }

    try {
      setIsUploading(true);
      setUploadingFileName(fileName);
      setUploadStatus({ type: "info", text: `正在上传并建立索引：${file.name}` });
      pushUploadRecord({
        id: uploadId,
        name: fileName,
        status: "uploading",
        detail: "上传并建立索引中...",
      });

      const formData = new FormData();
      formData.append("user_id", userId);
      formData.append("file", file);

      const res = await fetch(`${API_CONFIG.baseURL}/rag/incremental/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        setUploadStatus({
          type: "error",
          text: `上传失败：${data.detail || data.message || "unknown"}`,
        });
        patchUploadRecord({
          status: "error",
          detail: `失败：${data.detail || data.message || "unknown"}`,
        });
        return;
      }

      setUploadStatus({ type: "success", text: `上传成功：${file.name}（索引已完成）` });
      patchUploadRecord({ status: "success", detail: "上传成功，索引已完成" });

      if (!isAnswering) {
        handleSend(`[文件已上传: ${file.name}] 已完成索引，请基于该增量数据回答。`);
      } else {
        setUploadStatus({
          type: "success",
          text: `上传成功：${file.name}（索引已完成，当前回答结束后可继续提问）`,
        });
        patchUploadRecord({
          status: "success",
          detail: "上传成功，索引已完成（等待当前回答结束）",
        });
      }
    } catch (err) {
      setUploadStatus({ type: "error", text: `上传异常：${err.message}` });
      patchUploadRecord({ status: "error", detail: `异常：${err.message}` });
    } finally {
      setIsUploading(false);
      setUploadingFileName("");
      e.target.value = "";
    }
  }, [handleSend, isAnswering, userId]);

  // 导出 Markdown
  const handleExport = useCallback(() => {
    if (!currentChatId) {
      alert("当前没有可导出的对话。");
      return;
    }
    const chat = chats.find((c) => c.id === currentChatId);
    if (!chat || chat.messages.length === 0) {
      alert("对话为空，无法导出。");
      return;
    }

    const markdown = generateMarkdown(chat);
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `智能船舶问答_${chat.title}_${new Date().toISOString().slice(0, 10)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [chats, currentChatId]);

  useEffect(() => {
    if (!userId) return undefined;
    const timer = setTimeout(async () => {
      try {
        const response = await fetch(`${API_CONFIG.baseURL}/users/${encodeURIComponent(userId)}/settings`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ settings: userSettings, changed_by: userId }),
        });
        if (!response.ok) {
          const text = await response.text().catch(() => "");
          console.warn("settings save failed", response.status, text);
        }
      } catch (error) {
        console.warn("settings save request failed", error?.message || error);
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [userId, userSettings]);

  // ===================== 渲染辅助组件 =====================

  const insertCitationToInput = useCallback((citation, idx) => {
    const source = citation?.source || citation?.name || `资料${idx + 1}`;
    const snippet = formatCitationSnippet(citation?.text || "", 120);
    const quoteLine = snippet
      ? `基于资料${idx + 1}（${source}）继续分析："${snippet}"`
      : `基于资料${idx + 1}（${source}）继续分析：`;

    setInputValue((prev) => (prev ? `${prev}\n${quoteLine}` : quoteLine));
    setTimeout(() => {
      chatInputRef.current?.focus();
    }, 0);
  }, []);

  const CitationCard = ({ citation, idx, onQuote }) => {
    const [showFull, setShowFull] = useState(false);
    const scoreText =
      typeof citation.score === "number"
        ? `${Math.max(0, Math.min(100, Math.round(citation.score * 100)))}%`
        : null;
    const fullText = String(citation.text || "").trim();
    const previewText = fullText ? formatCitationSnippet(fullText, 200) : "";
    const isTruncated = !!fullText && fullText.length > 200;

    return (
      <div className="rounded-lg border border-white/10 bg-black/20 p-3 space-y-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[10px] px-1.5 py-0.5 rounded border border-sky-400/25 bg-sky-500/10 text-sky-300">
            资料{idx + 1}
          </span>
          {scoreText && <span className="text-[10px] text-teal-300">相关度 {scoreText}</span>}
        </div>
        <p className="text-[12px] text-gray-200 break-all">来源：{citation.source || citation.name || "未知来源"}</p>
        {citation.docId && <p className="text-[11px] text-gray-500 break-all">文档ID：{citation.docId}</p>}
        {fullText && (
          <div className="rounded-md border border-white/10 bg-black/25 p-2.5">
            <p className="text-[11px] text-gray-300 leading-relaxed whitespace-pre-wrap break-words">
              资料内容：{showFull ? fullText : previewText}
            </p>
            {isTruncated && (
              <button
                type="button"
                onClick={() => setShowFull((v) => !v)}
                className="mt-2 text-[11px] inline-flex items-center gap-1 px-2 py-1 rounded border border-white/10 bg-white/5 text-gray-300 hover:text-white hover:bg-white/10 transition-colors"
              >
                <i className={`ph ${showFull ? "ph-caret-up" : "ph-caret-down"} text-[10px]`}></i>
                {showFull ? "收起全文" : "查看全文"}
              </button>
            )}
          </div>
        )}
        <div className="pt-1">
          <button
            type="button"
            onClick={() => onQuote?.(citation, idx)}
            className="text-[11px] px-2 py-1 rounded border border-sky-400/25 bg-sky-500/10 text-sky-300 hover:bg-sky-500/20 transition-colors"
          >
            点击引用到输入框
          </button>
        </div>
      </div>
    );
  };

  const SearchProcessPanel = ({ msg }) => {
    const [isOpen, setIsOpen] = useState(false);
    const hasCitations = Array.isArray(msg.citations) && msg.citations.length > 0;
    const hasKgTriplets = Array.isArray(msg.kgTriplets) && msg.kgTriplets.length > 0;
    const isOfflineAnswer = !!msg.isOffline || /离线模式|\[离线\]/.test(msg.searchProcess || "");

    const kgNodes = hasKgTriplets
      ? [...new Set(msg.kgTriplets.flatMap((t) => [t?.head, t?.tail]).filter(Boolean))]
      : [];

    if (!hasCitations && !hasKgTriplets) return null;

    return (
      <div className="border border-white/5 rounded-xl bg-black/20 overflow-hidden mt-3">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between p-3 text-xs text-gray-400 hover:bg-white/5 transition-colors"
        >
          <span className="flex items-center gap-2">
            <i className="ph ph-magnifying-glass text-teal-500"></i>
            检索过程与来源
            {msg.isStreaming && (
              <span className="inline-flex items-center gap-1 text-[10px] text-teal-300">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-300 animate-pulse"></span>
                进行中
              </span>
            )}
          </span>
          <i className={`ph ph-caret-down transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}></i>
        </button>
        <div className={`p-3 border-t border-white/5 text-xs text-gray-300 bg-white/[0.02] space-y-3 ${isOpen ? "" : "hidden"}`}>
          <p className="mb-3 text-gray-400">{msg.searchProcess || "完成知识库检索，耗时 0.82s"}</p>

          {hasCitations && (
            <div>
              <p className="text-[11px] text-gray-500 mb-2">
                {isOfflineAnswer ? "参考文档" : "RAG 检索参考资料"}
              </p>
              <div className="space-y-2">
                {msg.citations.map((c, idx) => (
                  <CitationCard key={idx} citation={c} idx={idx} onQuote={insertCitationToInput} />
                ))}
              </div>
            </div>
          )}

          {hasKgTriplets && (
            <div>
              <p className="text-[11px] text-teal-400 mb-2">知识图谱命中节点</p>
              <div className="flex flex-wrap gap-2 mb-2">
                {kgNodes.map((node, idx) => (
                  <span
                    key={`${node}-${idx}`}
                    className="px-2 py-1 rounded-md text-[11px] border border-teal-500/25 bg-teal-500/10 text-teal-200"
                  >
                    {node}
                  </span>
                ))}
              </div>
              <div className="space-y-1">
                {msg.kgTriplets.slice(0, 8).map((t, idx) => {
                  const head = t?.head || "";
                  const rel = t?.rel || t?.relation || "关联";
                  const tail = t?.tail || "";
                  return (
                    <p key={idx} className="text-[11px] text-gray-400 leading-relaxed">
                      {idx + 1}. {head} <span className="text-teal-300">--{rel}--&gt;</span> {tail}
                    </p>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const GraphPanel = ({ msg }) => {
    const [open, setOpen] = useState(false);
    const chartRef = useRef(null);
    const chartInstanceRef = useRef(null);
    const triplets = Array.isArray(msg.kgTriplets) ? msg.kgTriplets : [];
    if (!triplets.length) return null;

    const nodeSet = new Set();
    const edges = [];
    triplets.forEach((t) => {
      const head = String(t?.head || "").trim();
      const tail = String(t?.tail || "").trim();
      const rel = String(t?.rel || t?.relation || "关联").trim();
      if (!head || !tail) return;
      nodeSet.add(head);
      nodeSet.add(tail);
      edges.push({ head, tail, rel });
    });

    const nodes = Array.from(nodeSet).map((name) => ({
      id: name,
      name,
      value: name,
      symbolSize: Math.max(32, Math.min(58, 28 + name.length * 1.6)),
      itemStyle: {
        color: "#14b8a6",
        borderColor: "#99f6e4",
        borderWidth: 1,
      },
      label: {
        show: true,
        color: "#d1fae5",
        fontSize: 11,
        formatter: (p) => shortenLabel(p.name, 14),
      },
    }));
    const links = edges.map((e, idx) => ({
      id: `${e.head}-${e.tail}-${idx}`,
      source: e.head,
      target: e.tail,
      value: e.rel,
      label: {
        show: true,
        color: "#7dd3fc",
        fontSize: 10,
        formatter: shortenLabel(e.rel, 8),
      },
      lineStyle: {
        color: "rgba(94, 234, 212, 0.65)",
        width: 1.4,
        curveness: 0.16,
      },
    }));

    useEffect(() => {
      if (!open || !chartRef.current) return undefined;

      if (!chartInstanceRef.current) {
        chartInstanceRef.current = echarts.init(chartRef.current, undefined, {
          renderer: "canvas",
        });
      }
      const chart = chartInstanceRef.current;

      chart.setOption(
        {
          backgroundColor: "#051926",
          animationDuration: 500,
          animationEasingUpdate: "cubicOut",
          tooltip: {
            trigger: "item",
            backgroundColor: "rgba(3, 18, 30, 0.95)",
            borderColor: "rgba(45, 212, 191, 0.3)",
            textStyle: { color: "#d1d5db" },
            formatter: (params) => {
              if (params.dataType === "edge") {
                return `${params.data.source} --${params.data.value || "关联"}--> ${params.data.target}`;
              }
              return `节点：${params.name}`;
            },
          },
          series: [
            {
              type: "graph",
              layout: "force",
              roam: true,
              draggable: true,
              force: {
                repulsion: 360,
                edgeLength: [80, 150],
                friction: 0.08,
                gravity: 0.06,
              },
              data: nodes,
              links,
              lineStyle: {
                opacity: 0.9,
              },
              emphasis: {
                focus: "adjacency",
                lineStyle: { width: 2.2 },
              },
            },
          ],
        },
        true
      );

      const onResize = () => chart.resize();
      window.addEventListener("resize", onResize);

      return () => {
        window.removeEventListener("resize", onResize);
      };
    }, [open, msg.id, msg.kgTriplets]);

    useEffect(() => {
      return () => {
        if (chartInstanceRef.current) {
          chartInstanceRef.current.dispose();
          chartInstanceRef.current = null;
        }
      };
    }, []);

    return (
      <div className="mt-3 rounded-xl border border-teal-500/20 bg-teal-500/5 overflow-hidden">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="w-full px-3 py-2.5 flex items-center justify-between text-[12px] text-teal-200 hover:bg-white/5"
        >
          <span className="inline-flex items-center gap-2">
            <i className="ph ph-graph text-sm"></i>
            知识图谱可视化
            <span className="text-[10px] text-teal-300/90">{nodes.length} 节点 / {edges.length} 关系</span>
          </span>
          <i className={`ph ph-caret-down text-xs transition-transform ${open ? "rotate-180" : ""}`}></i>
        </button>

        <div className={`${open ? "" : "hidden"} border-t border-teal-500/15 p-3`}>
          <div ref={chartRef} className="w-full h-72 rounded-lg bg-[#051926] border border-white/5" />
          <p className="mt-2 text-[11px] text-gray-500">
            说明：当前展示为问答命中的子图（来自后端 `kg` 事件返回的三元组），用于快速查看实体关系。
          </p>
        </div>
      </div>
    );
  };

  const AIMessage = ({ msg }) => {
    const isStreaming = msg.isStreaming && !msg.isPaused;
    const isOfflineAnswer = !!msg.isOffline || /离线模式|\[离线\]/.test(msg.searchProcess || "");
    const traceSteps = Array.isArray(msg.traceSteps) ? msg.traceSteps : [];
    const hasTraceSteps = traceSteps.length > 0;
    const [traceOpen, setTraceOpen] = useState(!!msg.isStreaming);
    const isGraphAnswer =
      !!msg.isGraphEnabled ||
      /知识图谱/.test(msg.searchProcess || "") ||
      (Array.isArray(msg.kgTriplets) && msg.kgTriplets.length > 0);
    const renderedHtml = isOfflineAnswer ? formatOfflineMarkdown(msg.content) : formatText(msg.content);

    useEffect(() => {
      if (msg.isStreaming && hasTraceSteps) {
        setTraceOpen(true);
      }
    }, [msg.isStreaming, hasTraceSteps]);

    return (
      <div id={msg.id} className="flex justify-start w-full mb-6 ai-message-block">
        <div className="flex items-start gap-4 w-full">
          <div className="w-9 h-9 rounded-xl bg-[#0a2530] border border-teal-500/30 flex items-center justify-center shrink-0 shadow-sm mt-1">
            <i className="ph ph-steering-wheel text-teal-400 text-lg"></i>
          </div>
          <div className="flex flex-col w-full max-w-[85%]">
            <div className="bg-white/[0.03] border border-white/5 text-gray-200 px-6 py-4 rounded-2xl rounded-tl-sm text-sm leading-relaxed shadow-sm relative">
              <div className="mb-3 flex items-center gap-2 flex-wrap">
                {isOfflineAnswer ? (
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] border border-emerald-400/30 bg-emerald-500/10 text-emerald-300">
                    <i className="ph ph-hard-drives text-xs"></i>
                    离线回答
                  </div>
                ) : (
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] border border-sky-400/30 bg-sky-500/10 text-sky-300">
                    <i className="ph ph-wifi-high text-xs"></i>
                    在线回答
                  </div>
                )}
                {isGraphAnswer && (
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] border border-teal-400/30 bg-teal-500/10 text-teal-300">
                    <i className="ph ph-graph text-xs"></i>
                    知识图谱
                  </div>
                )}
              </div>

              {hasTraceSteps && (
                <div className="mb-3 rounded-lg border border-sky-500/20 bg-sky-500/5 px-3 py-2.5">
                  <button
                    type="button"
                    onClick={() => setTraceOpen((v) => !v)}
                    className="w-full flex items-center justify-between text-[11px] text-sky-300"
                  >
                    <span className="inline-flex items-center gap-1.5">
                      <i className="ph ph-activity text-xs"></i>
                      后端响应过程
                    </span>
                    <i className={`ph ph-caret-down text-xs transition-transform ${traceOpen ? "rotate-180" : ""}`}></i>
                  </button>
                  <div className={`space-y-1 mt-2 ${traceOpen ? "" : "hidden"}`}>
                    {traceSteps.map((step, idx) => (
                      <p key={`${step.stage || "step"}-${idx}`} className="text-[11px] text-gray-300 leading-relaxed">
                        {idx + 1}. {step.text}
                      </p>
                    ))}
                  </div>
                </div>
              )}

              <div
                className={`markdown-body text-gray-300 ${isOfflineAnswer ? "markdown-rich" : ""}`}
                dangerouslySetInnerHTML={{
                  __html: renderedHtml + (isStreaming ? '<span class="inline-block w-1.5 h-4 bg-teal-400 ml-1 align-middle animate-pulse"></span>' : "")
                }}
              />

              <GraphPanel msg={msg} />

              {/* 流式控制按钮：仅暂停时显示“继续回答” */}
              {msg.isStreaming && msg.isPaused && (
                <div className="stream-controls mt-4 flex items-center gap-2 border-t border-white/5 pt-3">
                  <button
                    type="button"
                    onClick={() => resumeStreaming(msg.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-500/10 text-sky-300 hover:bg-sky-500/20 transition-colors text-xs border border-sky-500/25"
                  >
                    <i className="ph ph-play"></i> 继续回答
                  </button>
                  <span className="text-xs text-orange-400 flex items-center gap-1">
                    <i className="ph ph-warning-circle"></i> 用户已暂停
                  </span>
                </div>
              )}

              {/* 暂停提示 */}
              {msg.isPaused && !msg.isStreaming && (
                <div className="mt-3 text-xs text-orange-400 flex items-center gap-1">
                  <i className="ph ph-warning-circle"></i> 回答已暂停，点击继续
                </div>
              )}
            </div>

            {/* 检索过程折叠面板 */}
            <SearchProcessPanel msg={msg} />
          </div>
        </div>
      </div>
    );
  };

  const UserMessage = ({ msg }) => (
    <div className="flex justify-end w-full mb-6">
      <div className="flex items-start gap-4 flex-row-reverse max-w-[85%]">
        <div className="w-9 h-9 rounded-full bg-[#3b82f6] flex items-center justify-center text-white text-sm font-bold shrink-0 shadow-sm mt-1 border-2 border-[#041527] overflow-hidden">
          {userSettings?.avatar ? (
            <img src={userSettings.avatar} alt="avatar" className="w-full h-full object-cover" />
          ) : (
            "CP"
          )}
        </div>
        <div className="bg-teal-600 text-white px-6 py-4 rounded-2xl rounded-tr-sm text-sm shadow-md whitespace-pre-wrap leading-relaxed">
          {msg.content}
        </div>
      </div>
    </div>
  );

  // ===================== 主渲染 =====================

  if (showSetting) {
    return <Setting
      onBack={() => {
        setShowSetting(false);
        const newSettings = normalizeSettings(userSettings);

        // 修改：如果当前在欢迎界面，立刻应用最新设定的开关
        if (!currentChatId) {
          setGraphEnabled(newSettings.graphOn ?? true);
          setOfflineMode(newSettings.offlineOn ?? false);
        }
      }}
      chats={chats}
      userId={userId}
      userRole={userRole}
      initialSettings={userSettings}
      onClearChats={() => {
        setChats([]);
        setCurrentChatId(null);
      }}
      onSettingsChange={(next) => {
        const normalized = normalizeSettings(next || {});
        setUserSettings(normalized);
        if (!currentChatId) {
          setGraphEnabled(normalized.graphOn ?? true);
          setOfflineMode(normalized.offlineOn ?? false);
        }
      }}
      onDeleteChat={(id) => {
        setChats((prev) => prev.filter((c) => c.id !== id));
        if (currentChatId === id) setCurrentChatId(null);
      }}
    />;
  }

  return (
    <div
      className={`text-gray-300 antialiased h-screen w-screen flex ${isLightUi ? "db-theme-light" : ""}`}
      style={{
        margin: 0,
        overflow: "hidden",
        background: themeSurface.shellBg,
      }}
    >
      {/* ========== 左侧边栏 ========== */}
      <aside className="w-64 z-20 flex flex-col h-full shrink-0 glass-sidebar">
        <div className="p-5">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-[#0a2530] border border-teal-500/30 flex items-center justify-center">
              <i className="ph ph-steering-wheel text-teal-400 text-xl"></i>
            </div>
            <div>
              <h2 className="text-sm font-bold text-white tracking-wide">智能船舶问答</h2>
              <p className="text-[10px] text-gray-500 uppercase tracking-wider mt-0.5">DeepBlue AI</p>
            </div>
          </div>

          <button onClick={handleNewChat} className="new-chat-btn w-full py-2.5 px-4 rounded-lg bg-sky-400 hover:bg-sky-500 text-white transition-colors flex items-center justify-center gap-2 text-sm font-medium shadow-lg shadow-sky-500/20">
            <i className="ph ph-plus-circle text-lg"></i>
            新建会话
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-3 flex flex-col custom-scrollbar">
          <p className="history-label px-2 text-xs font-medium text-gray-500 mb-3 mt-2 shrink-0">历史记录</p>
          <div className="space-y-1">
            {chats.map((chat) => {
              const isActive = chat.id === currentChatId;
              return (
                <div
                  key={chat.id}
                  onClick={() => loadChat(chat.id)}
                  className={`history-item group flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${isActive
                    ? isLightUi
                      ? "bg-blue-50 border border-blue-200 text-blue-700"
                      : "bg-sky-500/10 border border-sky-400/25"
                    : isLightUi
                      ? "text-slate-700 hover:text-slate-900 hover:bg-slate-100 border border-transparent"
                      : "hover:bg-white/5 text-gray-400 hover:text-gray-200"
                    }`}
                >
                  <i className={`ph ph-chat-centered-text text-lg ${isActive ? (isLightUi ? "text-blue-600" : "text-sky-300") : ""}`}></i>
                  <span className={`text-sm truncate flex-1 ${isActive ? (isLightUi ? "text-blue-800" : "text-sky-100") : ""}`}>
                    {chat.title}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="p-4 border-t border-white/5 shrink-0">
          <div
            onClick={() => setShowSetting(true)}
            className="flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 cursor-pointer transition-colors"
          >
            <div className="w-9 h-9 rounded-lg bg-[#3b82f6] flex items-center justify-center font-bold text-white text-xs overflow-hidden">
              {userSettings?.avatar ? (
                <img src={userSettings.avatar} alt="avatar" className="w-full h-full object-cover" />
              ) : (
                "CP"
              )}
            </div>
            <div className="flex-1 overflow-hidden">
              <p className="text-sm font-medium text-white truncate">{userSettings?.nickname || "Captain Park"}</p>
              <p className="text-[11px] text-gray-500">一级轮机长</p>
            </div>
            <i className="ph ph-gear text-gray-400 hover:text-white transition-colors text-lg"></i>
          </div>
        </div>
      </aside>

      {/* ========== 主内容区 ========== */}
      <main className="flex-1 flex flex-col relative z-10 overflow-hidden" style={{ background: themeSurface.mainBg }}>
        <div className="glow-bg"></div>
        <svg
          className={`wave-bg ${isLightUi ? "wave-light" : "wave-dark"}`}
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 24 150 28"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <path id="wave" d="M-160 44c30 0 58-18 88-18s 58 18 88 18 58-18 88-18 58 18 88 18 v44h-352z" />
          </defs>
          <g className="wave-parallax">
            <use
              className="wave-layer wave-layer-1"
              xlinkHref="#wave"
              x="48"
              y="0"
              fill={isLightUi ? "rgba(186, 230, 253, 0.14)" : "rgba(13, 148, 136, 0.18)"}
            />
            <use
              className="wave-layer wave-layer-2"
              xlinkHref="#wave"
              x="48"
              y="3"
              fill={isLightUi ? "rgba(147, 197, 253, 0.18)" : "rgba(17, 53, 90, 0.28)"}
            />
            <use
              className="wave-layer wave-layer-3"
              xlinkHref="#wave"
              x="48"
              y="7"
              fill={isLightUi ? "rgba(191, 219, 254, 0.22)" : "rgba(4, 21, 39, 0.48)"}
            />
          </g>
        </svg>

        <header className="h-16 flex items-center justify-between px-6 shrink-0 relative z-20 glass-header">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-teal-500"></div>
            <h1 className="text-sm font-medium text-gray-200">
              {currentChat ? currentChat.title : "当前会话"}
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-white/10 hover:bg-white/5 transition-colors text-xs text-gray-300"
            >
              <i className="ph ph-export text-sm"></i> 导出 MD
            </button>
            <div className="w-px h-4 bg-white/10"></div>
            <button className="w-8 h-8 rounded-full hover:bg-white/5 flex items-center justify-center text-gray-400 hover:text-white transition-colors">
              <i className="ph ph-bell text-lg"></i>
            </button>
          </div>
        </header>

        {/* 欢迎界面 */}
        {isWelcome && (
          <div className="flex-1 overflow-y-auto relative z-10 custom-scrollbar">
            <div className="welcome-wrap">
              <div className="logo-wrap">
                <div className="logo-glow"></div>
                <div className="particles">
                  <div className="particle" style={{ top: "50%", left: "50%" }}></div>
                  <div className="particle" style={{ top: "50%", left: "50%", animationDelay: "2s" }}></div>
                </div>
                <div className="logo-box">
                  <i className="ph ph-steering-wheel"></i>
                </div>
              </div>

              <h2>欢迎使用智能船舶问答系统</h2>
              <p className="welcome-desc">
                基于知识图谱的深度故障诊断引擎已就绪
                <br />
                请告诉我您想了解的船舶设备问题
              </p>

              <div className="quick-grid">
                {QUICK_CARDS.map((card) => (
                  <button
                    key={card.title}
                    onClick={() => handleSend(card.title)}
                    className="action-card"
                  >
                    <i className={`ph ${card.icon}`}></i>
                    <div className="action-title">{card.title}</div>
                    <div className="action-desc">{card.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* 聊天消息区 */}
        {!isWelcome && (
          <div ref={chatViewRef} className="flex-1 overflow-y-auto relative z-10 custom-scrollbar">
            <div className="w-full max-w-4xl mx-auto p-6 pb-36 flex flex-col">
              {currentChat?.messages.map((msg, idx) =>
                msg.role === "user" ? (
                  <UserMessage key={idx} msg={msg} />
                ) : (
                  <AIMessage key={msg.id || idx} msg={msg} />
                )
              )}
            </div>
          </div>
        )}

        {/* 底部输入框 */}
        <div className="absolute bottom-0 left-0 w-full p-6 pt-0 z-20" style={{ background: themeSurface.footerBg }}>
          <div className="max-w-4xl mx-auto">
            {isUploading && (
              <div className="mb-2 flex items-center gap-2 px-3 py-2 rounded-lg border border-sky-400/25 bg-sky-500/10 text-sky-200 text-xs">
                <i className="ph ph-spinner-gap animate-spin"></i>
                <span>正在处理文档：{uploadingFileName || "请稍候"}（切块与索引中）</span>
              </div>
            )}
            <div className="glass-input-chat rounded-xl flex items-center p-2 pr-2 shadow-lg">
              <input
                ref={fileInputRef}
                type="file"
                accept=".md,.pdf,.txt"
                className="hidden"
                disabled={isUploading}
                onChange={handleFileChange}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                title="支持上传 md, pdf, txt"
                disabled={isUploading}
                className="p-3 text-gray-400 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <i className={`ph ${isUploading ? "ph-spinner-gap animate-spin" : "ph-paperclip"} text-lg`}></i>
              </button>
              <input
                ref={chatInputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === "Enter" && !isAnswering) handleSend(inputValue);
                }}
                disabled={isAnswering}
                placeholder={isAnswering ? "正在回答中，请稍候..." : "输入您的问题，如：“对比去年同期的油耗数据”"}
                className="flex-1 bg-transparent border-none focus:ring-0 outline-none text-sm text-gray-200 py-3 px-2 h-full placeholder:text-gray-600 disabled:opacity-50"
              />
              <button
                type="button"
                onClick={() => {
                  if (isAnswering) {
                    if (!activeStreamingPaused && activeStreamingMsgId) {
                      pauseStreaming(activeStreamingMsgId); // 输出中 -> 暂停
                    }
                    return;
                  }
                  handleSend(inputValue); // 非输出中 -> 发送
                }}
                disabled={isAnswering && activeStreamingPaused} // 暂停后只能在消息处点“继续”
                className={`ml-2 p-3 rounded-lg text-white transition-colors flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed ${isAnswering
                  ? "bg-orange-500 hover:bg-orange-400"
                  : "bg-sky-400 hover:bg-sky-500"
                  }`}
                title={
                  isAnswering
                    ? activeStreamingPaused
                      ? "已暂停，请在消息处点击继续"
                      : "暂停回答"
                    : "发送"
                }
              >
                <i className={`ph ${isAnswering ? "ph-pause" : "ph-paper-plane-right"} text-lg`}></i>
              </button>
            </div>
            {uploadStatus && (
              <div
                className={`mt-2 text-xs px-2 ${uploadStatus.type === "error"
                  ? "text-rose-400"
                  : uploadStatus.type === "success"
                    ? "text-emerald-400"
                    : "text-sky-300"
                  }`}
              >
                {uploadStatus.text}
              </div>
            )}
            {recentUploads.length > 0 && (
              <div className="mt-2 px-2 py-2 rounded-lg border border-white/10 bg-white/[0.02]">
                <p className="text-[11px] text-gray-400 mb-2">最近上传文件（最多 3 条）</p>
                <div className="space-y-1.5">
                  {recentUploads.map((item) => (
                    <div key={item.id} className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-xs text-gray-200 truncate">{item.name}</p>
                        <p className="text-[11px] text-gray-500 truncate">{item.detail}</p>
                      </div>
                      <span
                        className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded border ${item.status === "success"
                          ? "text-emerald-300 border-emerald-400/30 bg-emerald-500/10"
                          : item.status === "error"
                            ? "text-rose-300 border-rose-400/30 bg-rose-500/10"
                            : "text-sky-300 border-sky-400/30 bg-sky-500/10"
                          }`}
                      >
                        {item.status === "success" ? "成功" : item.status === "error" ? "失败" : "处理中"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ========== 右侧边栏 ========== */}
      <aside className="w-72 z-20 flex flex-col h-full shrink-0 glass-sidebar-right">
          <div className="h-16 px-6 border-b border-white/5 flex items-center justify-between shrink-0">
            <h3 className="text-sm font-bold text-white tracking-wide">系统状况</h3>
            <div className="system-status-pill flex items-center gap-2 bg-teal-500/10 px-2.5 py-1 rounded-full border border-teal-500/20">
              <span className="status-dot w-1.5 h-1.5 rounded-full bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.8)]"></span>
              <span className="status-text text-[11px] text-teal-400 font-medium">运行中</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
            {/* 知识图谱引擎 */}
            <div className="mb-4">
            <div className="sidebar-card p-5 rounded-xl bg-white/[0.02] border border-white/5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs text-gray-400">知识图谱引擎</span>
                <i className="ph ph-brain text-teal-400 text-lg"></i>
              </div>
              <div className="text-xl font-bold text-white mb-3">{offlineMode ? "离线就绪" : "就绪"}</div>
              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-teal-500 rounded-full" style={{ width: "85%" }}></div>
              </div>
            </div>
          </div>

            {/* 故障知识库 */}
            <div className="mb-8">
            <div className="sidebar-card p-5 rounded-xl bg-white/[0.02] border border-white/5">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs text-gray-400">故障知识库</span>
                <i className="ph ph-database text-teal-400 text-lg"></i>
              </div>
              <div className="text-xl font-bold text-white mb-2">已连接</div>
              <div className="text-xs text-gray-500">
                <span className="text-gray-400">12,847</span> 条历史记录
              </div>
            </div>
          </div>

          <div className="space-y-4">
            {/* 知识图谱分析开关 */}
            <div className="sidebar-card p-5 rounded-xl bg-teal-500/5 border border-teal-500/20">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <i className="ph ph-graph text-teal-400 text-lg"></i>
                  <span className="text-sm font-medium text-gray-200">知识图谱分析</span>
                </div>
                <Switch checked={graphEnabled} onChange={setGraphEnabled} />
              </div>
              <p className="text-xs text-gray-500 leading-relaxed mt-2">
                启用后，AI将结合船舶历史维护记录、设备关联图谱进行多维度故障推理
              </p>
            </div>

            {/* 离线模式开关 */}
            <div className="sidebar-card p-5 rounded-xl bg-white/[0.02] border border-white/5">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <i className="ph ph-wifi-slash text-gray-400 text-lg"></i>
                  <span className="text-sm font-medium text-gray-200">离线模式</span>
                </div>
                <Switch checked={offlineMode} onChange={setOfflineMode} />
              </div>
              <p className="text-xs text-gray-500 leading-relaxed mt-2">
                启用后，将调用本地部署的大模型和知识库，不依赖外部网络连接
              </p>
            </div>
          </div>
        </div>

        <div className="p-6 shrink-0">
          <button
            type="button"
            onClick={() => setShowHelpDocs(true)}
            className="sidebar-help-btn w-full py-2.5 rounded-lg border border-white/10 hover:bg-white/5 transition-colors text-xs text-gray-400 hover:text-gray-200 flex items-center justify-center gap-2"
          >
            <i className="ph ph-info text-sm"></i> 帮助与文档
          </button>
        </div>
      </aside>

      <HelpDocsDialog isOpen={showHelpDocs} onClose={() => setShowHelpDocs(false)} />

      {/* CSS 样式补充 */}
      <style>{`
        .glass-sidebar {
          background: ${themeSurface.panelBg};
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border-right: 1px solid ${themeSurface.panelBorder};
          position: relative;
          overflow: hidden;
        }
        .glass-sidebar::before {
          content: "";
          position: absolute;
          inset: 0;
          background: linear-gradient(180deg, rgba(56, 189, 248, 0.1) 0%, rgba(56, 189, 248, 0.03) 36%, transparent 72%);
          pointer-events: none;
          z-index: 0;
        }
        .glass-sidebar > * {
          position: relative;
          z-index: 1;
        }
        .glass-sidebar-right {
          background: ${themeSurface.panelBg};
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border-left: 1px solid ${themeSurface.panelBorder};
        }
        .glass-header {
          background: ${themeSurface.headerBg};
          backdrop-filter: blur(10px);
          border-bottom: 1px solid ${themeSurface.panelBorder};
        }
        .glass-input-chat {
          background: ${themeSurface.inputBg};
          border: 1px solid ${themeSurface.inputBorder};
          backdrop-filter: blur(10px);
          transition: all 0.3s ease;
        }
        .glass-input-chat:focus-within {
          border-color: rgba(45, 212, 191, 0.4);
          background: rgba(255, 255, 255, 0.05);
          box-shadow: 0 0 20px rgba(45, 212, 191, 0.05);
        }
        .glow-bg {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 600px;
          height: 600px;
          background: ${themeSurface.glowBg};
          pointer-events: none;
          z-index: 0;
        }
        .welcome-wrap {
          min-height: 70vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          color: #fff;
          text-align: center;
          animation: fadeIn 0.6s ease-out;
          padding: 2rem 1.5rem 8rem;
          position: relative;
        }
        .welcome-wrap h2 {
          font-size: 24px;
          font-weight: 700;
          margin-bottom: 0.5rem;
          color: #fff;
        }
        .welcome-desc {
          font-size: 14px;
          color: rgba(255, 255, 255, 0.5);
          margin-bottom: 2rem;
          max-width: 34rem;
          line-height: 1.7;
        }
        .logo-wrap {
          position: relative;
          margin-bottom: 2rem;
        }
        .logo-glow {
          position: absolute;
          inset: -20px;
          background: rgba(45, 212, 191, 0.3);
          filter: blur(40px);
          border-radius: 50%;
          animation: pulse 3s ease-in-out infinite;
        }
        .logo-box {
          position: relative;
          width: 80px;
          height: 80px;
          background: linear-gradient(135deg, rgba(45, 212, 191, 0.2), rgba(37, 99, 235, 0.2));
          border: 1px solid rgba(45, 212, 191, 0.4);
          border-radius: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 40px;
          cursor: pointer;
        }
        .logo-box i {
          color: #2dd4bf;
          animation: float 6s ease-in-out infinite;
        }
        .quick-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
          width: 100%;
          max-width: 480px;
        }
        .action-card {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 12px;
          padding: 16px;
          color: white;
          text-align: left;
          cursor: pointer;
          transition: all 0.3s;
        }
        .action-card:hover {
          background: rgba(56, 189, 248, 0.12);
          border-color: rgba(56, 189, 248, 0.3);
          transform: translateY(-2px);
        }
        .action-card i {
          font-size: 24px;
          color: #7dd3fc;
          margin-bottom: 8px;
          display: block;
        }
        .action-title {
          font-size: 13px;
          font-weight: 500;
          margin-bottom: 4px;
        }
        .action-desc {
          font-size: 11px;
          color: rgba(255, 255, 255, 0.45);
        }
        .particles {
          position: absolute;
          inset: -60px;
          pointer-events: none;
        }
        .particle {
          position: absolute;
          width: 4px;
          height: 4px;
          background: rgba(45, 212, 191, 0.6);
          border-radius: 50%;
          animation: orbit 8s linear infinite;
        }
        .wave-bg {
          position: absolute;
          bottom: 0;
          left: 0;
          width: 100%;
          height: 20vh;
          z-index: 2;
          pointer-events: none;
        }
        .wave-bg.wave-dark {
          opacity: 0.75;
        }
        .wave-bg.wave-light {
          opacity: 0.56;
        }
        .wave-bg .wave-parallax {
          animation: wave-tide 7s ease-in-out infinite;
          transform-origin: center bottom;
          will-change: transform;
        }
        .wave-bg.wave-dark .wave-layer {
          animation-name: move-wave, wave-shimmer-dark;
          animation-timing-function: cubic-bezier(0.55, 0.5, 0.45, 0.5), ease-in-out;
          animation-iteration-count: infinite, infinite;
          animation-direction: alternate, alternate;
          will-change: transform, opacity;
        }
        .wave-bg.wave-light .wave-layer {
          animation-name: move-wave, wave-shimmer-light;
          animation-timing-function: cubic-bezier(0.55, 0.5, 0.45, 0.5), ease-in-out;
          animation-iteration-count: infinite, infinite;
          animation-direction: alternate, alternate;
          will-change: transform, opacity;
        }
        .wave-bg .wave-layer-1 {
          animation-duration: 10s, 6s;
          animation-delay: -2s, -1s;
        }
        .wave-bg .wave-layer-2 {
          animation-duration: 14s, 8.5s;
          animation-delay: -3s, -2s;
        }
        .wave-bg .wave-layer-3 {
          animation-duration: 20s, 11s;
          animation-delay: -4s, -3s;
        }
        @keyframes move-wave {
          0% {
            transform: translate3d(-90px, 0, 0);
          }
          100% {
            transform: translate3d(85px, 0, 0);
          }
        }
        @keyframes wave-shimmer-dark {
          0% {
            opacity: 0.78;
          }
          50% {
            opacity: 1;
          }
          100% {
            opacity: 0.84;
          }
        }
        @keyframes wave-shimmer-light {
          0% {
            opacity: 0.55;
          }
          50% {
            opacity: 0.72;
          }
          100% {
            opacity: 0.6;
          }
        }
        @keyframes wave-tide {
          0% {
            transform: translateY(0) scaleY(1);
          }
          50% {
            transform: translateY(-3px) scaleY(1.02);
          }
          100% {
            transform: translateY(1px) scaleY(0.995);
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .wave-bg .wave-parallax,
          .wave-bg .wave-layer {
            animation: none;
          }
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes pulse {
          0%,
          100% {
            opacity: 0.3;
            transform: scale(1);
          }
          50% {
            opacity: 0.6;
            transform: scale(1.1);
          }
        }
        @keyframes float {
          0%,
          100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-10px);
          }
        }
        @keyframes orbit {
          from {
            transform: rotate(0deg) translateX(50px) rotate(0deg);
          }
          to {
            transform: rotate(360deg) translateX(50px) rotate(-360deg);
          }
        }
        .db-theme-light .text-white {
          color: #0f172a !important;
        }
        .db-theme-light .text-gray-500,
        .db-theme-light .text-gray-400 {
          color: #334155 !important;
        }
        .db-theme-light .text-gray-300,
        .db-theme-light .text-gray-200 {
          color: #1e293b !important;
        }
        .db-theme-light .text-gray-100,
        .db-theme-light .text-teal-100 {
          color: #0f172a !important;
        }
        .db-theme-light .glass-input-chat {
          background: rgba(255, 255, 255, 0.98) !important;
          border-color: rgba(148, 163, 184, 0.45) !important;
          box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .db-theme-light .glass-input-chat input {
          color: #0f172a !important;
        }
        .db-theme-light .glass-input-chat input::placeholder {
          color: #64748b !important;
        }
        .db-theme-light [style*="color: #f3f4f6"],
        .db-theme-light [style*="color:#f3f4f6"],
        .db-theme-light [style*="color: #d1d5db"],
        .db-theme-light [style*="color:#d1d5db"] {
          color: #0f172a !important;
        }
        .db-theme-light [style*="color: #9ca3af"],
        .db-theme-light [style*="color:#9ca3af"],
        .db-theme-light [style*="color: #6b7280"],
        .db-theme-light [style*="color:#6b7280"] {
          color: #475569 !important;
        }
        .db-theme-light .bg-white\/\[0\.02\],
        .db-theme-light .bg-white\/\[0\.03\] {
          background: rgba(255, 255, 255, 0.78) !important;
          border-color: rgba(148, 163, 184, 0.28) !important;
        }
        .db-theme-light .bg-black\/20,
        .db-theme-light .bg-black\/25,
        .db-theme-light .bg-black\/30 {
          background: rgba(241, 245, 249, 0.9) !important;
          border-color: rgba(148, 163, 184, 0.32) !important;
        }
        .db-theme-light .border-white\/5,
        .db-theme-light .border-white\/10 {
          border-color: rgba(148, 163, 184, 0.35) !important;
        }
        .db-theme-light .bg-\[\#0a2530\] {
          background: rgba(191, 219, 254, 0.62) !important;
          border-color: rgba(96, 165, 250, 0.35) !important;
        }
        .db-theme-light .new-chat-btn {
          background: #38bdf8 !important;
          box-shadow: 0 10px 24px rgba(56, 189, 248, 0.28);
        }
        .db-theme-light .new-chat-btn:hover {
          background: #0ea5e9 !important;
        }
        .db-theme-light .history-label {
          color: #334155 !important;
        }
        .db-theme-light .glass-sidebar-right {
          background: rgba(255, 255, 255, 0.94);
          border-left: 1px solid rgba(148, 163, 184, 0.42);
        }
        .db-theme-light .glass-sidebar {
          background: linear-gradient(180deg, rgba(239, 246, 255, 0.98) 0%, rgba(255, 255, 255, 0.95) 100%);
          border-right: 1px solid rgba(148, 163, 184, 0.45);
          box-shadow: 8px 0 24px rgba(148, 163, 184, 0.16);
        }
        .db-theme-light .glass-sidebar::before {
          background: linear-gradient(180deg, rgba(147, 197, 253, 0.36) 0%, rgba(191, 219, 254, 0.22) 30%, transparent 72%);
        }
        .db-theme-light .sidebar-card {
          background: #ffffff !important;
          border-color: rgba(148, 163, 184, 0.38) !important;
          box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }
        .db-theme-light .system-status-pill {
          background: rgba(37, 99, 235, 0.12) !important;
          border-color: rgba(37, 99, 235, 0.35) !important;
        }
        .db-theme-light .system-status-pill .status-dot {
          background: #2563eb !important;
          box-shadow: 0 0 8px rgba(37, 99, 235, 0.45) !important;
        }
        .db-theme-light .system-status-pill .status-text {
          color: #1d4ed8 !important;
        }
        .db-theme-light .sidebar-help-btn {
          color: #1e293b !important;
          border-color: rgba(148, 163, 184, 0.5) !important;
          background: rgba(255, 255, 255, 0.88) !important;
        }
        .db-theme-light .sidebar-help-btn:hover {
          background: rgba(226, 232, 240, 0.88) !important;
        }
        .db-theme-light .welcome-wrap {
          color: #0f172a;
        }
        .db-theme-light .welcome-wrap h2,
        .db-theme-light .action-card {
          color: #0f172a;
        }
        .db-theme-light .welcome-desc,
        .db-theme-light .action-desc {
          color: #475569;
        }
        .db-theme-light .action-card {
          background: rgba(255, 255, 255, 0.75);
          border-color: rgba(148, 163, 184, 0.35);
        }
        .db-theme-light .action-card:hover {
          background: rgba(147, 197, 253, 0.2);
          border-color: rgba(59, 130, 246, 0.35);
        }
        .db-theme-light .logo-box {
          background: linear-gradient(135deg, rgba(125, 211, 252, 0.2), rgba(14, 116, 144, 0.15));
          border-color: rgba(14, 116, 144, 0.35);
        }
        .db-theme-light .logo-glow {
          background: rgba(14, 116, 144, 0.2);
        }
        @media (max-width: 768px) {
          .welcome-wrap {
            min-height: calc(100vh - 12rem);
            padding: 1.5rem 1rem 9rem;
          }
          .quick-grid {
            grid-template-columns: 1fr;
            max-width: 360px;
          }
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .markdown-rich h1,
        .markdown-rich h2,
        .markdown-rich h3 {
          color: #f3f4f6;
          margin: 0.75rem 0 0.5rem;
          line-height: 1.35;
          font-weight: 700;
        }
        .markdown-rich h1 { font-size: 1.1rem; }
        .markdown-rich h2 { font-size: 1.02rem; }
        .markdown-rich h3 { font-size: 0.95rem; }
        .markdown-rich p {
          margin: 0.55rem 0;
          line-height: 1.8;
          color: #d1d5db;
        }
        .markdown-rich ul,
        .markdown-rich ol {
          margin: 0.5rem 0 0.6rem;
          padding-left: 1.1rem;
          color: #d1d5db;
        }
        .markdown-rich ul { list-style: disc; }
        .markdown-rich ol { list-style: decimal; }
        .markdown-rich li { margin: 0.3rem 0; line-height: 1.75; }
        .markdown-rich blockquote {
          margin: 0.7rem 0;
          padding: 0.55rem 0.8rem;
          border-left: 3px solid rgba(45, 212, 191, 0.6);
          background: rgba(45, 212, 191, 0.08);
          color: #c7d2fe;
          border-radius: 0.4rem;
        }
        .markdown-rich a {
          color: #5eead4;
          text-decoration: underline;
          text-underline-offset: 2px;
        }
        .markdown-rich code {
          background: rgba(255, 255, 255, 0.08);
          padding: 0.08rem 0.32rem;
          border-radius: 0.28rem;
          font-size: 0.82em;
          color: #c4f5ef;
        }
        .markdown-rich .md-pre {
          margin: 0.7rem 0;
          padding: 0.7rem 0.85rem;
          border-radius: 0.55rem;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(2, 12, 22, 0.72);
          overflow-x: auto;
        }
        .markdown-rich .md-pre code {
          background: transparent;
          padding: 0;
          border-radius: 0;
          color: #d1d5db;
          font-size: 0.8rem;
        }
        .markdown-rich .md-code-lang {
          display: inline-block;
          margin-bottom: 0.45rem;
          font-size: 0.7rem;
          color: #9ca3af;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        .db-theme-light .markdown-body,
        .db-theme-light .markdown-rich {
          color: #0f172a;
        }
        .db-theme-light .markdown-rich h1,
        .db-theme-light .markdown-rich h2,
        .db-theme-light .markdown-rich h3,
        .db-theme-light .markdown-rich p,
        .db-theme-light .markdown-rich ul,
        .db-theme-light .markdown-rich ol,
        .db-theme-light .markdown-rich li {
          color: #0f172a;
        }
        .db-theme-light .markdown-rich blockquote {
          border-left-color: rgba(14, 116, 144, 0.45);
          background: rgba(14, 116, 144, 0.08);
          color: #0f172a;
        }
        .db-theme-light .markdown-rich a {
          color: #0369a1;
        }
        .db-theme-light .markdown-rich code {
          background: rgba(15, 23, 42, 0.06);
          color: #0f172a;
        }
        .db-theme-light .markdown-rich .md-pre {
          border-color: rgba(148, 163, 184, 0.35);
          background: #f8fafc;
        }
        .db-theme-light .markdown-rich .md-pre code,
        .db-theme-light .markdown-rich .md-code-lang {
          color: #334155;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
        .db-theme-light .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(100, 116, 139, 0.35);
        }
        .db-theme-light .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(100, 116, 139, 0.5);
        }
      `}</style>
    </div>
  );
}

// ===================== 开关组件 =====================
function Switch({ checked, onChange }) {
  return (
    <label style={{ position: "relative", display: "inline-block", width: 36, height: 20, cursor: "pointer" }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ opacity: 0, width: 0, height: 0 }}
      />
      <span
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: checked ? "#0d9488" : "rgba(255,255,255,0.1)",
          transition: ".4s",
          borderRadius: 20,
        }}
      >
        <span
          style={{
            position: "absolute",
            content: '""',
            height: 14,
            width: 14,
            left: checked ? "calc(100% - 17px)" : 3,
            bottom: 3,
            backgroundColor: "white",
            transition: ".4s",
            borderRadius: "50%",
          }}
        />
      </span>
    </label>
  );
}
