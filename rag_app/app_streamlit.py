import streamlit as st

from rag.pipeline import RagPipeline
from rag.schema import QueryRequest


st.set_page_config(page_title="Ship Fault RAG", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RagPipeline()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("船舶装备故障诊断智能问答 (RAG Skeleton)")

question = st.text_input("请输入故障现象或问题")
col1, col2 = st.columns([1, 1])
with col1:
    use_kg = st.checkbox("启用知识图谱检索", value=True)
with col2:
    top_k = st.slider("检索Top-K", min_value=3, max_value=10, value=5, step=1)

if st.button("查询") and question.strip():
    req = QueryRequest(question=question.strip(), top_k=top_k, use_kg=use_kg)
    ans = st.session_state.pipeline.query(req)
    st.session_state.history.append(ans)

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
