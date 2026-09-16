"""Backwards-compatibility shim.

The transport moved to llm_client.py when support for OpenAI-compatible
providers (GLM/Zhipu, OpenAI, Groq, DeepSeek, ...) was added alongside
Ollama. This module re-exports the new interface so older imports keep
working. Prefer importing `llm_client` directly in new code.
"""
from __future__ import annotations

from .llm_client import (  # noqa: F401
    DEFAULT_API_KEY,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    call_json,
    current_config,
    extract_json,
    test_connection,
)
