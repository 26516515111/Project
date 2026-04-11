import streamlit as st
import requests

from rag.schema import QueryRequest

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Ship Fault RAG", layout="wide")

# 移除本地 pipeline 初始化，改为直接通过 API 请求
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit"

st.title("船舶装备故障诊断智能问答 (RAG Skeleton)")

question = st.text_input("请输入故障现象或问题")
col1, col2 = st.columns([1, 1])
with col1:
    use_kg = st.checkbox("启用知识图谱检索", value=True)
with col2:
    top_k = st.slider("检索Top-K", min_value=3, max_value=10, value=5, step=1)

if st.button("查询") and question.strip():
    req = QueryRequest(
        question=question.strip(),
        top_k=top_k,
        use_kg=use_kg,
        session_id=st.session_state.session_id,
    )
    
    # 替换为 API 接口请求
    try:
        response = requests.post(f"{API_URL}/rag/query", json=req.model_dump())
        response.raise_for_status()
        
        # 将返回的 dict 映射回对象，或者直接使用字典（此处根据返回结构做简单兼容）
        class DotDict(dict):
            __getattr__ = dict.get
        
        ans_data = response.json()
        ans = DotDict({
            "question": req.question,
            "answer": ans_data.get("answer", ""),
            "citations": [DotDict(c) for c in ans_data.get("citations", [])],
            "kg_triplets": ans_data.get("kg_triplets", [])
        })
        st.session_state.history.append(ans)
    except Exception as e:
        st.error(f"请求后台接口失败: {e}")

if st.session_state.history:
    st.subheader("回答")
    latest = st.session_state.history[-1]
    st.write(latest.answer)

    st.subheader("依据与引用")
    for p in latest.citations:
        st.write(f"- {p.source} (score={p.score:.3f})")

    if latest.kg_triplets:
        st.subheader("知识图谱关联")
        for t in latest.kg_triplets:
            st.write(f"- {t.get('head', '')} {t.get('rel', '')} {t.get('tail', '')}")

    st.subheader("历史记录")
    for i, h in enumerate(reversed(st.session_state.history), start=1):
        st.write(f"{i}. {h.question}")
