import logging
from typing import Optional, Tuple

from langchain_community.embeddings import ModelScopeEmbeddings

from .config import SETTINGS
from .schema import Passage, RetrievedContext

logger = logging.getLogger(__name__)

try:
    from langchain_community.chat_models import ChatOllama

    _OLLAMA_AVAILABLE = True
    _OLLAMA_IMPORT_ERROR = None
except Exception as exc:
    ChatOllama = None
    _OLLAMA_AVAILABLE = False
    _OLLAMA_IMPORT_ERROR = exc

try:
    from langchain_openai import ChatOpenAI

    _OPENAI_CHAT_AVAILABLE = True
    _OPENAI_CHAT_IMPORT_ERROR = None
except Exception as exc:
    ChatOpenAI = None
    _OPENAI_CHAT_AVAILABLE = False
    _OPENAI_CHAT_IMPORT_ERROR = exc

try:
    from langchain_huggingface import HuggingFaceEmbeddings

    _EMBEDDINGS_AVAILABLE = True
    _EMBEDDINGS_IMPORT_ERROR = None
except Exception as exc:
    HuggingFaceEmbeddings = None
    _EMBEDDINGS_AVAILABLE = False
    _EMBEDDINGS_IMPORT_ERROR = exc

TRANSFORMERS_AVAILABLE = False
torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.debug("Transformers library not available.")


def ollama_available() -> bool:
    return _OLLAMA_AVAILABLE


def build_ollama_chat(temperature: float, max_tokens: int):
    return build_chat_llm(
        provider="ollama",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_chat_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    """构建聊天模型实例，统一从配置读取模型与地址。

    通过修改 SETTINGS.llm_model_name 与 SETTINGS.llm_base_url 即可切换模型。
    """
    # if SETTINGS.llm_provider != "ollama":
    #     raise RuntimeError(f"Unsupported llm provider: {SETTINGS.llm_provider}")
    # if not _OLLAMA_AVAILABLE:
    #     logger.warning("Ollama unavailable: %s", type(_OLLAMA_IMPORT_ERROR).__name__)
    #     raise RuntimeError("Ollama is not available")

    resolved_provider = (provider or SETTINGS.llm_provider or "ollama").strip().lower()
    resolved_model = model_name or SETTINGS.llm_model_name
    resolved_temperature = (
        SETTINGS.llm_temperature if temperature is None else float(temperature)
    )
    resolved_max_tokens = SETTINGS.llm_max_tokens if max_tokens is None else max_tokens

    if resolved_provider == "ollama":
        if not _OLLAMA_AVAILABLE:
            logger.warning(
                "Ollama unavailable: %s", type(_OLLAMA_IMPORT_ERROR).__name__
            )
            raise RuntimeError("Ollama is not available")
        resolved_base_url = base_url or SETTINGS.llm_base_url
        return ChatOllama(
            model=resolved_model,
            base_url=resolved_base_url,
            temperature=resolved_temperature,
            model_kwargs={"num_predict": resolved_max_tokens},
        )

    if resolved_provider == "modelscope":
        if model_name is None:
            resolved_model = SETTINGS.llm_modelscope_model
        if not _OPENAI_CHAT_AVAILABLE:
            logger.warning(
                "langchain_openai unavailable: %s", _OPENAI_CHAT_IMPORT_ERROR
            )
            raise RuntimeError(
                "ModelScope requires langchain-openai. "
                "Please install it with: pip install -U langchain-openai"
            )
        if not SETTINGS.llm_api_key:
            raise RuntimeError(
                "Missing LLM API key. Set RAG_LLM_API_KEY or MODELSCOPE_API_KEY."
            )
        if ":" in resolved_model:
            raise RuntimeError(
                "ModelScope model name should not use Ollama format like 'qwen2.5:3b'. "
                "Set RAG_MODELSCOPE_MODEL or RAG_LLM_MODEL to a valid model id, "
                "for example 'Qwen/Qwen2.5-3B-Instruct'."
            )
        resolved_base_url = base_url or SETTINGS.llm_modelscope_base_url
        return ChatOpenAI(
            model=resolved_model,
            base_url=resolved_base_url,
            api_key=SETTINGS.llm_api_key,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
        )

    if resolved_provider == "none":
        raise RuntimeError("LLM provider is disabled by configuration")

    raise RuntimeError(f"Unsupported llm provider: {resolved_provider}")


# def build_chat_llm(
#     model_name: Optional[str] = None,
#     base_url: Optional[str] = None,
#     temperature: Optional[float] = None,
#     max_tokens: Optional[int] = None,
# ):
#     """构建聊天模型实例，统一从配置读取模型与地址。
#
#     通过修改 SETTINGS.llm_model_name 与 SETTINGS.llm_base_url 即可切换模型。
#     """
#     # if SETTINGS.llm_provider != "ollama":
#     #     raise RuntimeError(f"Unsupported llm provider: {SETTINGS.llm_provider}")
#     # if not _OLLAMA_AVAILABLE:
#     #     logger.warning("Ollama unavailable: %s", type(_OLLAMA_IMPORT_ERROR).__name__)
#     #     raise RuntimeError("Ollama is not available")
#
#     resolved_model = model_name or SETTINGS.llm_model_name
#     resolved_base_url = base_url or SETTINGS.llm_base_url
#     resolved_temperature = (
#         SETTINGS.llm_temperature if temperature is None else float(temperature)
#     )
#     resolved_max_tokens = SETTINGS.llm_max_tokens if max_tokens is None else max_tokens
#     return ChatOpenAI(
#         model=resolved_model,
#         base_url=resolved_base_url,
#         api_key=SETTINGS.llm_api_key,
#         temperature=resolved_temperature
#     )


def build_embeddings(model_name: Optional[str] = None):
    if not _EMBEDDINGS_AVAILABLE:
        logger.warning(
            "Embeddings unavailable: %s", type(_EMBEDDINGS_IMPORT_ERROR).__name__
        )
        raise RuntimeError("Embeddings are not available")
    embedding_model = model_name or SETTINGS.embedding_model
    return HuggingFaceEmbeddings(model_name=embedding_model)


def build_reranker_components(
    model_name: Optional[str] = None,
) -> Tuple[Optional[object], Optional[object]]:
    if not TRANSFORMERS_AVAILABLE:
        logger.info("Transformers library not available. Reranker disabled.")
        return None, None

    model_name = model_name or SETTINGS.reranker_model
    try:
        logger.info("Loading model %s...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        logger.info("Model loaded successfully.")
        return model, tokenizer
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_name, exc)
        return None, None
