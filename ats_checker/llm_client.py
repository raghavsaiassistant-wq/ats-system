"""Provider-agnostic LLM transport for the semantic and manager layers.

Two backends, same interface:

  openai   — any OpenAI-compatible /chat/completions endpoint.
             Covers Zhipu/GLM (api.z.ai, open.bigmodel.cn), OpenAI, Groq,
             DeepSeek, OpenRouter, Together, and most hosted providers.
  ollama   — local or cloud Ollama (/api/chat).

Config comes from a .env file in the working directory, then the process
environment, then built-in defaults. The API key is NEVER hardcoded here —
put it in .env (which .gitignore excludes) or export it.

    ATS_LLM_PROVIDER=openai          # openai | ollama
    ATS_LLM_BASE_URL=https://api.z.ai/api/paas/v4
    ATS_LLM_MODEL=glm-4-flash
    ATS_LLM_API_KEY=...

Everything fails soft: callers get (None, error_message) rather than an
exception, so a dead endpoint degrades the report instead of killing the run.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------- .env

def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict:
    """Minimal .env reader — avoids adding a dependency for 12 lines.
    Supports KEY=value, comments, blank lines, and quoted values."""
    values: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return values
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return values


load_dotenv()

# ------------------------------------------------------------- defaults

PROVIDER_PRESETS = {
    # Ollama (native /api/chat)
    "ollama-cloud": "https://ollama.com",
    "ollama-local": "http://localhost:11434",
    # OpenAI-compatible (/chat/completions)
    "glm": "https://api.z.ai/api/paas/v4",
    "glm-cn": "https://open.bigmodel.cn/api/paas/v4",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Presets that speak Ollama's native protocol rather than OpenAI's
OLLAMA_PRESETS = {"ollama-cloud", "ollama-local"}


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def current_config() -> dict:
    """Resolve the active provider config. Auto-detects: if an API key and
    base URL are present, use the OpenAI-compatible path; otherwise Ollama."""
    api_key = _env("ATS_LLM_API_KEY", "OLLAMA_API_KEY", "OPENAI_API_KEY",
                    "ZHIPU_API_KEY", "GLM_API_KEY")
    raw_base = _env("ATS_LLM_BASE_URL", "OLLAMA_HOST")
    provider = _env("ATS_LLM_PROVIDER").lower()

    preset_key = raw_base.lower().strip()
    is_ollama_preset = preset_key in OLLAMA_PRESETS
    base_url = PROVIDER_PRESETS.get(preset_key, raw_base)

    if not provider:
        # Infer: an Ollama preset or an ollama/localhost URL means the native
        # protocol, even when an API key is present (Ollama Cloud needs one).
        if is_ollama_preset or "ollama" in base_url.lower() or "11434" in base_url:
            provider = "ollama"
        else:
            provider = "openai" if api_key else "ollama"

    if provider == "openai":
        return {
            "provider": "openai",
            "base_url": base_url or PROVIDER_PRESETS["glm"],
            "model": _env("ATS_LLM_MODEL", default="glm-4-flash"),
            "api_key": api_key,
        }

    resolved = base_url or PROVIDER_PRESETS["ollama-local"]
    is_cloud = "ollama.com" in resolved
    return {
        "provider": "ollama",
        "base_url": resolved,
        "model": _env("ATS_LLM_MODEL", "OLLAMA_MODEL",
                       default="gpt-oss:120b-cloud" if is_cloud else "llama3.1"),
        "api_key": api_key,
    }


_cfg = current_config()
DEFAULT_PROVIDER = _cfg["provider"]
DEFAULT_HOST = _cfg["base_url"]
DEFAULT_MODEL = _cfg["model"]
DEFAULT_API_KEY = _cfg["api_key"]


# ------------------------------------------------------------- parsing

def extract_json(raw: str) -> dict | None:
    """Pull a JSON object out of a model response. Handles fenced blocks,
    leading prose, and trailing commentary — all of which smaller models do
    even when told not to."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Salvage the outermost balanced {...}
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ------------------------------------------------------------- transport

def call_json(
    system_prompt: str,
    user_content: str,
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
    timeout: int = 180,
    temperature: float = 0.2,
    retries: int = 2,
) -> tuple[dict | None, str | None]:
    """Return (parsed_json, None) on success, or (None, error_message)."""
    cfg = current_config()
    provider = (provider or cfg["provider"]).lower()
    host = host or cfg["base_url"]
    model = model or cfg["model"]
    api_key = api_key if api_key is not None else cfg["api_key"]

    if provider == "ollama":
        url = host.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
    else:
        url = host.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        except requests.exceptions.ConnectionError:
            last_error = (
                f"Could not reach {url}. "
                + ("Is Ollama running (`ollama serve`)?" if provider == "ollama"
                   else "Check ATS_LLM_BASE_URL and your network.")
            )
        except requests.exceptions.Timeout:
            last_error = f"Request to {url} timed out after {timeout}s — try a faster model."
        except requests.exceptions.RequestException as e:
            last_error = f"Request failed: {e}"
        else:
            if resp.status_code == 401:
                return None, "Authentication failed (401) — check ATS_LLM_API_KEY."
            if resp.status_code == 404:
                return None, (
                    f"404 from {url} — the base URL or model name is likely wrong. "
                    f"Model requested: '{model}'."
                )
            if resp.status_code == 429:
                last_error = "Rate limited (429)."
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                return None, last_error
            if resp.status_code >= 400:
                detail = resp.text[:300]
                return None, f"HTTP {resp.status_code} from provider: {detail}"

            try:
                data = resp.json()
            except ValueError:
                last_error = "Provider returned a non-JSON HTTP body."
            else:
                content = _extract_content(data, provider)
                if content is None:
                    return None, f"Unexpected response shape from provider: {str(data)[:300]}"
                parsed = extract_json(content)
                if parsed is None:
                    last_error = (
                        "Model replied but the output wasn't valid JSON. "
                        "Try a stronger model."
                    )
                    if attempt < retries:
                        continue
                    return None, last_error
                return parsed, None

        if attempt < retries:
            time.sleep(1)

    return None, last_error or "Unknown error"


def _extract_content(data: dict, provider: str) -> str | None:
    if provider == "ollama":
        return (data.get("message") or {}).get("content")
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    # Some providers (GLM reasoning models among them) put the answer in
    # reasoning_content when content comes back empty.
    if not content:
        content = message.get("reasoning_content")
    return content


def test_connection(
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> tuple[bool, str]:
    """One cheap round-trip to verify credentials and model name."""
    parsed, error = call_json(
        'You are a connection test. Reply with exactly: {"ok": true}',
        "Reply with the JSON object {\"ok\": true} and nothing else.",
        model=model, host=host, api_key=api_key, provider=provider,
        timeout=60, retries=1,
    )
    if error:
        return False, error
    if not isinstance(parsed, dict):
        return False, f"Connected, but got an unexpected payload: {parsed}"
    return True, f"Connected. Parsed response: {json.dumps(parsed)}"
