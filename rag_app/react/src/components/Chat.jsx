import { useState, useRef, useEffect, useCallback } from "react";
import Setting from "./setting";

// ===================== 模拟后端 API =====================
const API_HOST = window.location.hostname || "127.0.0.1";
const API_CONFIG = {
  baseURL: `http://${API_HOST}:8000`,
  timeoutOnline: 90000,
  timeoutOffline: 30000,
};
const DEFAULT_USER_ID = "captain_park";

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
    // 如果是降级到原文检索模式，替换提示语为更友好的格式
    .replace(/^\s*基于检索到的资料，建议如下[：:]\s*/g, "⚠️ **当前未配置大模型或连接失败，系统已自动降级为原文检索模式。为您找到以下参考片段：**\n\n")

    // [H1] [H2]...
    .replace(/\[(?:H\d+|h\d+)\]\s*/g, "")

    // [IMG ...]（有闭合或无闭合都清理）
    .replace(/\[\s*(?:IMG|img)[^\]\n\r]*(?:\]|$)/g, "")

    // IMG path=...（有无方括号都清理，直到行尾）
    .replace(/\b(?:IMG|img)\s*path\s*=\s*(?:data\/KG\/images|images)\/[^\n\r]*/g, "")

    // 裸图片路径（有扩展名）
    .replace(/\b(?:images|data\/KG\/images)\/\S+\.(?:png|jpe?g|gif|webp|bmp)\b/gi, "")

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

// 新增：读取设置工具
const getSystemSettings = () => {
  const saved = localStorage.getItem("deepblue_settings");
  // 匹配 setting.jsx 中的默认值
  return saved ? JSON.parse(saved) : { notify: true, autoSave: true };
};

// 新增：网页标题/标签页闪烁提示（替代提示音）
let flashInterval = null;
const originalTitle = document.title || "智能船舶问答系统";

const notifyNewMessage = () => {
  const settings = getSystemSettings();
  if (!settings.notify) return; // 未开启提示则返回

  // 清除已存在的定时器
  if (flashInterval) clearInterval(flashInterval);

  let flashFlag = false;
  flashInterval = setInterval(() => {
    document.title = flashFlag ? originalTitle : "【新消息】" + originalTitle;
    flashFlag = !flashFlag;
  }, 500);

  // 用户与页面产生交互时（移动鼠标、点击或重新聚焦），停止闪烁
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

// ===================== 组件 =====================
export default function Chat() {
  // 新增：提取并保存一份当前用户的偏好设置到状态中
  const [userSettings, setUserSettings] = useState(() => getSystemSettings());

  const [chats, setChats] = useState(() => {
    // 自动保存逻辑：如果不允许自动保存，则初始化时不读取历史缓存
    const settings = getSystemSettings();
    if (!settings.autoSave) return INITIAL_CHATS;

    const saved = localStorage.getItem("deepblue_chats");
    return saved ? JSON.parse(saved) : INITIAL_CHATS;
  });
  const [currentChatId, setCurrentChatId] = useState(null);
  const [inputValue, setInputValue] = useState("");

  // 修改：连接用户设置的默认状态
  const [graphEnabled, setGraphEnabled] = useState(userSettings?.graphOn ?? true);
  const [offlineMode, setOfflineMode] = useState(userSettings?.offlineOn ?? false); // 离线模式
  const [isAnswering, setIsAnswering] = useState(false); // 新增：是否正在回答
  const [showSetting, setShowSetting] = useState(false); // 新增：设置页面开关

  // 保存数据到本地存储（受 autoSave 控制）
  useEffect(() => {
    const settings = getSystemSettings();
    if (settings.autoSave) {
      localStorage.setItem("deepblue_chats", JSON.stringify(chats));
    } else {
      // 如果关闭了自动保存，则清空现有的记录
      localStorage.removeItem("deepblue_chats");
    }
  }, [chats]);

  // 流式会话管理
  const streamingRefs = useRef(new Map());

  const chatViewRef = useRef(null);
  const fileInputRef = useRef(null);

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

    if (!chatId) {
      chatId = Date.now().toString();
      const title = text.length > 12 ? text.substring(0, 12) + "..." : text;
      updatedChats = [{ id: chatId, title, messages: [], createdAt: new Date() }, ...updatedChats];
      setCurrentChatId(chatId);
    }

    const userMsg = { role: "user", content: text };
    const msgId = `ai-${Date.now()}`;
    const aiMsg = {
      id: msgId,
      role: "ai",
      content: "",
      fullText: "",
      searchProcess: `正在检索，模式: ${graphEnabled ? "知识图谱" : "常规检索"}${offlineMode ? " [离线]" : " [在线]"}`,
      citations: [],
      kgTriplets: [],
      isStreaming: true,
      isPaused: false,
    };

    updatedChats = updatedChats.map((c) =>
      c.id === chatId ? { ...c, messages: [...c.messages, userMsg, aiMsg] } : c
    );
    setChats(updatedChats);
    setIsAnswering(true);

    startStreaming(chatId, msgId, text);
  }, [chats, currentChatId, graphEnabled, offlineMode, isAnswering]);

  // 流式输出逻辑（打字机）
  const startStreaming = useCallback(async (chatId, msgId, originalText) => {
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

      // 提取配置中设定的模型偏好
      const prefModel = userSettings?.model || "hybrid";
      const resolvedProvider = prefModel === "local" ? "ollama" : (prefModel === "cloud" ? "modelscope" : (isOnlineMode ? "modelscope" : "ollama"));

      const response = await fetch(`${API_CONFIG.baseURL}/rag/query/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          user_id: DEFAULT_USER_ID,
          question: originalText,
          top_k: 5,
          use_kg: graphEnabled,
          use_llm: true,
          llm_provider: resolvedProvider,
          offline_mode: offlineMode,
          use_history: true,
          enable_retrieval_optimization: true,
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
                        }
                        : m
                    ),
                  }
              )
            );
          } else if (evt.event === "token" && data.text) {
            streamSession.pendingText += data.text;
          } else if (evt.event === "references") {
            const citations = (data.citations || []).map((cit) => ({
              name: cit.source || cit.doc_id || "unknown",
              url: "#",
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
    const settings = getSystemSettings();
    setGraphEnabled(settings.graphOn ?? true);
    setOfflineMode(settings.offlineOn ?? false);
  }, []);

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

    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!["md", "pdf", "txt"].includes(ext)) {
      alert("仅支持 md/pdf/txt");
      e.target.value = "";
      return;
    }

    try {
      const formData = new FormData();
      formData.append("user_id", DEFAULT_USER_ID);
      formData.append("file", file);

      const res = await fetch(`${API_CONFIG.baseURL}/rag/incremental/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        alert(`上传失败: ${data.detail || data.message || "unknown"}`);
        return;
      }

      handleSend(`[文件已上传: ${file.name}] 已完成索引，请基于该增量数据回答。`);
    } catch (err) {
      alert(`上传异常: ${err.message}`);
    } finally {
      e.target.value = "";
    }
  }, [handleSend]);

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

  // ===================== 渲染辅助组件 =====================

  const CitationLink = ({ citation }) => (
    <a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-teal-500/10 text-teal-400 hover:bg-teal-500/20 border border-teal-500/20 transition-colors text-xs"
    >
      <i className="ph ph-file-text"></i>
      {citation.name}
    </a>
  );

  const SearchProcessPanel = ({ msg }) => {
    const [isOpen, setIsOpen] = useState(false);

    if (!msg.citations || msg.citations.length === 0) return null;
    if (msg.isStreaming) return null;

    return (
      <div className="border border-white/5 rounded-xl bg-black/20 overflow-hidden mt-3">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between p-3 text-xs text-gray-400 hover:bg-white/5 transition-colors"
        >
          <span className="flex items-center gap-2">
            <i className="ph ph-magnifying-glass text-teal-500"></i>
            检索过程与来源
          </span>
          <i className={`ph ph-caret-down transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}></i>
        </button>
        <div className={`p-3 border-t border-white/5 text-xs text-gray-300 bg-white/[0.02] ${isOpen ? "" : "hidden"}`}>
          <p className="mb-3 text-gray-400">{msg.searchProcess || "完成知识库检索，耗时 0.82s"}</p>
          <div className="flex flex-wrap gap-2">
            {msg.citations.map((c, idx) => (
              <CitationLink key={idx} citation={c} />
            ))}
          </div>
        </div>
      </div>
    );
  };

  const AIMessage = ({ msg }) => {
    const isStreaming = msg.isStreaming && !msg.isPaused;

    return (
      <div id={msg.id} className="flex justify-start w-full mb-6 ai-message-block">
        <div className="flex items-start gap-4 w-full">
          <div className="w-9 h-9 rounded-xl bg-[#0a2530] border border-teal-500/30 flex items-center justify-center shrink-0 shadow-sm mt-1">
            <i className="ph ph-steering-wheel text-teal-400 text-lg"></i>
          </div>
          <div className="flex flex-col w-full max-w-[85%]">
            <div className="bg-white/[0.03] border border-white/5 text-gray-200 px-6 py-4 rounded-2xl rounded-tl-sm text-sm leading-relaxed shadow-sm relative">
              <div
                className="markdown-body text-gray-300"
                dangerouslySetInnerHTML={{
                  __html: formatText(msg.content) + (isStreaming ? '<span class="inline-block w-1.5 h-4 bg-teal-400 ml-1 align-middle animate-pulse"></span>' : "")
                }}
              />

              {/* 流式控制按钮：仅暂停时显示“继续回答” */}
              {msg.isStreaming && msg.isPaused && (
                <div className="stream-controls mt-4 flex items-center gap-2 border-t border-white/5 pt-3">
                  <button
                    type="button"
                    onClick={() => resumeStreaming(msg.id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-500/10 text-teal-400 hover:bg-teal-500/20 transition-colors text-xs border border-teal-500/20"
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
        const newSettings = getSystemSettings();
        setUserSettings(newSettings); // 从设置页返回时，重新获取最新的个人信息和头像

        // 修改：如果当前在欢迎界面，立刻应用最新设定的开关
        if (!currentChatId) {
          setGraphEnabled(newSettings.graphOn ?? true);
          setOfflineMode(newSettings.offlineOn ?? false);
        }
      }}
      chats={chats}
      onClearChats={() => {
        setChats([]);
        setCurrentChatId(null);
      }}
      onDeleteChat={(id) => {
        setChats((prev) => prev.filter((c) => c.id !== id));
        if (currentChatId === id) setCurrentChatId(null);
      }}
    />;
  }

  return (
    <div className="text-gray-300 antialiased h-screen w-screen flex bg-[#030d17]" style={{ margin: 0, overflow: "hidden" }}>
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

          <button onClick={handleNewChat} className="w-full py-2.5 px-4 rounded-lg bg-teal-500 hover:bg-teal-400 text-white transition-colors flex items-center justify-center gap-2 text-sm font-medium shadow-lg shadow-teal-500/20">
            <i className="ph ph-plus-circle text-lg"></i>
            新建会话
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pb-3 flex flex-col custom-scrollbar">
          <p className="px-2 text-xs font-medium text-gray-500 mb-3 mt-2 shrink-0">历史记录</p>
          <div className="space-y-1">
            {chats.map((chat) => {
              const isActive = chat.id === currentChatId;
              return (
                <div
                  key={chat.id}
                  onClick={() => loadChat(chat.id)}
                  className={`group flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-colors ${isActive
                    ? "bg-teal-500/10 border border-teal-500/20"
                    : "hover:bg-white/5 text-gray-400 hover:text-gray-200"
                    }`}
                >
                  <i className={`ph ph-chat-centered-text text-lg ${isActive ? "text-teal-400" : ""}`}></i>
                  <span className={`text-sm truncate flex-1 ${isActive ? "text-teal-100" : ""}`}>
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
      <main className="flex-1 flex flex-col relative z-10 overflow-hidden bg-[#041527]">
        <div className="glow-bg"></div>

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
          <div className="flex-1 overflow-y-auto flex flex-col items-center justify-center text-center p-8 pb-32 relative z-10 custom-scrollbar">
            <div className="relative mb-8">
              <div className="w-20 h-20 rounded-2xl bg-[#0a2530] border border-teal-500/20 flex items-center justify-center relative z-10">
                <i className="ph ph-steering-wheel text-4xl text-teal-400"></i>
                <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-teal-500 flex items-center justify-center border-2 border-[#041527]">
                  <i className="ph ph-sparkle text-white text-[10px]"></i>
                </div>
              </div>
              <div className="absolute -top-4 -left-4 w-1.5 h-1.5 bg-teal-500/50 rounded-full"></div>
              <div className="absolute top-1/2 -right-8 w-1 h-1 bg-gray-500/50 rounded-full"></div>
            </div>

            <h2 className="text-2xl font-bold text-white mb-4">欢迎使用智能船舶问答系统</h2>
            <p className="text-gray-400 text-sm max-w-md mb-10 leading-relaxed">
              基于知识图谱的深度故障诊断引擎已就绪<br />请告诉我您想了解的船舶设备问题
            </p>

            <div className="grid grid-cols-2 gap-4 w-full max-w-2xl">
              {QUICK_CARDS.map((card) => (
                <button
                  key={card.title}
                  onClick={() => handleSend(card.title)}
                  className="p-5 rounded-xl bg-white/[0.02] border border-white/5 text-left flex flex-col items-start gap-3 transition-all duration-300 hover:border-teal-400/30 hover:bg-teal-500/5"
                >
                  <i className={`ph ${card.icon} text-2xl text-teal-400`}></i>
                  <div>
                    <p className="text-sm text-gray-200 font-medium mb-1">{card.title}</p>
                    <p className="text-xs text-gray-500">{card.desc}</p>
                  </div>
                </button>
              ))}
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
        <div className="absolute bottom-0 left-0 w-full p-6 pt-0 z-20" style={{ background: "linear-gradient(to top, #041527 0%, #041527 60%, transparent 100%)" }}>
          <div className="max-w-4xl mx-auto">
            <div className="glass-input-chat rounded-xl flex items-center p-2 pr-2 shadow-lg">
              <input
                ref={fileInputRef}
                type="file"
                accept=".md,.pdf,.txt"
                className="hidden"
                onChange={handleFileChange}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                title="支持上传 md, pdf, txt"
                className="p-3 text-gray-400 hover:text-white transition-colors"
              >
                <i className="ph ph-paperclip text-lg"></i>
              </button>
              <input
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
                  : "bg-teal-500 hover:bg-teal-400"
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
          </div>
        </div>
      </main>

      {/* ========== 右侧边栏 ========== */}
      <aside className="w-72 z-20 flex flex-col h-full shrink-0 glass-sidebar-right">
        <div className="h-16 px-6 border-b border-white/5 flex items-center justify-between shrink-0">
          <h3 className="text-sm font-bold text-white tracking-wide">系统状况</h3>
          <div className="flex items-center gap-2 bg-teal-500/10 px-2.5 py-1 rounded-full border border-teal-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-400 shadow-[0_0_8px_rgba(45,212,191,0.8)]"></span>
            <span className="text-[11px] text-teal-400 font-medium">运行中</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
          {/* 知识图谱引擎 */}
          <div className="mb-4">
            <div className="p-5 rounded-xl bg-white/[0.02] border border-white/5">
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
            <div className="p-5 rounded-xl bg-white/[0.02] border border-white/5">
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
            <div className="p-5 rounded-xl bg-teal-500/5 border border-teal-500/20">
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
            <div className="p-5 rounded-xl bg-white/[0.02] border border-white/5">
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
          <button className="w-full py-2.5 rounded-lg border border-white/10 hover:bg-white/5 transition-colors text-xs text-gray-400 hover:text-gray-200 flex items-center justify-center gap-2">
            <i className="ph ph-info text-sm"></i> 帮助与文档
          </button>
        </div>
      </aside>

      {/* CSS 样式补充 */}
      <style>{`
        .glass-sidebar {
          background: rgba(4, 21, 39, 0.7);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-sidebar-right {
          background: rgba(4, 21, 39, 0.7);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border-left: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-header {
          background: rgba(4, 21, 39, 0.4);
          backdrop-filter: blur(10px);
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .glass-input-chat {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
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
          background: radial-gradient(circle, rgba(20, 184, 166, 0.05) 0%, transparent 70%);
          pointer-events: none;
          z-index: 0;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
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