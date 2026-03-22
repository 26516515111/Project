from __future__ import annotations

"""Configuration for KG pipeline runtime and LLM calls."""

LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"
LLM_REQUIRE_API_KEY = True
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4000
LLM_RETRY_LIMIT = 3
LLM_RETRY_BACKOFF = 2
LLM_MIN_CHUNK_CHARS = 50
LLM_SLEEP_SECONDS = 0.3
LLM_CHECKPOINT_EVERY_CHUNKS = 10
LLM_MOCK_ENV = "KG_PIPELINE_MOCK_LLM"
