import json
import logging
import re
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from .config import SETTINGS
from .model import build_chat_llm

logger = logging.getLogger(__name__)


DEFAULT_QUERY_PROMPT = PromptTemplate.from_template(
    """
请将用户问题拆解为最多{max_subquestions}个子问题，便于检索。
要求：
1) 子问题要简短可检索；
2) 不能重复；
3) 每行输出一个子问题，不要额外解释。

用户问题：{question}
""".strip()
)


DEFAULT_SUB_QUESTION_PROMPT = PromptTemplate.from_template(
    """
你正在回答一个主问题的子问题。
主问题：{question}
子问题：{sub_question}

已检索到文档片段：
{documents}

请基于文档简洁回答该子问题。
""".strip()
)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _heuristic_decompose(question: str) -> List[str]:
    text = _normalize(question)
    if not text:
        return []

    parts: List[str] = []
    for seg in re.split(r"[。！？?!；;]", text):
        seg = _normalize(seg)
        if not seg:
            continue
        subparts = re.split(r"(?:以及|并且|并|同时|而且|还有|或者|或|及|和)", seg)
        for sub in subparts:
            sub = _normalize(sub)
            if sub:
                parts.append(sub)

    filtered = [
        p for p in parts if len(p) >= SETTINGS.decompose_min_length and p != text
    ]
    filtered = _dedupe(filtered)
    return filtered[: SETTINGS.decompose_max_subquestions]


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        raw = _normalize(text)
        if not raw:
            return []

        if raw.startswith("[") and raw.endswith("]"):
            try:
                items = json.loads(raw)
                if isinstance(items, list):
                    parsed = [_normalize(x) for x in items if isinstance(x, str)]
                    parsed = [x for x in parsed if x]
                    return _dedupe(parsed)
            except Exception:
                pass

        lines = []
        for line in str(text or "").splitlines():
            item = _normalize(re.sub(r"^[-*\d\.\)\s]+", "", line))
            if item:
                lines.append(item)
        return _dedupe(lines)

    @property
    def _type(self) -> str:
        return "line_list"


class DecompositionQueryRetriever(BaseRetriever):
    retriever: BaseRetriever
    llm_chain: Runnable
    sub_llm_chain: Runnable

    @classmethod
    def from_llm(
        cls,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = DEFAULT_QUERY_PROMPT,
        sub_prompt: BasePromptTemplate = DEFAULT_SUB_QUESTION_PROMPT,
    ) -> "DecompositionQueryRetriever":
        output_parser = LineListOutputParser()
        llm_chain = prompt | llm | output_parser
        sub_llm_chain = sub_prompt | llm
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            sub_llm_chain=sub_llm_chain,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        sub_queries = self.generate_queries(query)
        return self.retrieve_documents(query, sub_queries)

    def generate_queries(self, question: str) -> List[str]:
        response = self.llm_chain.invoke(
            {
                "question": question,
                "max_subquestions": SETTINGS.decompose_max_subquestions,
            }
        )
        lines = response if isinstance(response, list) else []
        lines = [_normalize(x) for x in lines if isinstance(x, str)]
        lines = [x for x in lines if x and x != _normalize(question)]
        lines = _dedupe(lines)
        return lines[: SETTINGS.decompose_max_subquestions]

    def retrieve_documents(self, query: str, sub_queries: List[str]) -> List[Document]:
        if not sub_queries:
            return []

        sub_llm_chain = RunnableLambda(
            lambda sub_query: self.sub_llm_chain.invoke(
                {
                    "question": query,
                    "sub_question": sub_query,
                    "documents": [
                        doc.page_content for doc in self.retriever.invoke(sub_query)
                    ],
                }
            )
        )
        responses = sub_llm_chain.batch(sub_queries)

        documents = []
        for sub_query, response in zip(sub_queries, responses):
            content = str(getattr(response, "content", response) or "").strip()
            documents.append(Document(page_content=f"{sub_query}\n{content}"))
        return documents


class _EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return []


def _llm_decompose(question: str) -> List[str]:
    if SETTINGS.llm_provider == "none":
        return []
    try:
        llm = build_chat_llm(
            temperature=SETTINGS.decompose_llm_temperature,
            max_tokens=SETTINGS.decompose_llm_max_tokens,
        )
        decomposer = DecompositionQueryRetriever.from_llm(
            retriever=_EmptyRetriever(),
            llm=llm,
        )
        return decomposer.generate_queries(question)
    except Exception as exc:
        logger.info("LLM decomposition failed: %s", type(exc).__name__)
        return []


def decompose_question(question: str) -> List[str]:
    text = _normalize(question)
    if len(text) < SETTINGS.decompose_min_length:
        return []
    if SETTINGS.decomposer_method == "llm":
        items = _llm_decompose(text)
        if items:
            return items
    return _heuristic_decompose(text)
