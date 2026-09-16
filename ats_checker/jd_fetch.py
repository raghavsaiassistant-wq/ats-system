"""Fetch a job description from a posting URL and reduce it to plain text.

Deliberately crude-but-predictable instead of a scraping stack: posting
pages are messy, and all the extraction layers downstream only need
readable text with line structure preserved. No JS execution, no cookies,
no session handling — some dynamic sites will render thin text and that's
visible immediately rather than silently wrong.
"""
from __future__ import annotations

import html as html_mod
import re


def html_to_text(html_text: str) -> str:
    """Strip script/style, convert block tags to newlines, unescape, and
    collapse whitespace. Keeps enough line structure for the JD section
    detectors (Requirements:, Responsibilities:, ...) to work."""
    text = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html_text)
    text = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr|/section)[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html_mod.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def fetch_jd_url(url: str, timeout: int = 30) -> str:
    """GET the URL and return readable text (HTML stripped when needed)."""
    import requests

    resp = requests.get(
        url, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ATS-Checker/1.0)"},
    )
    resp.raise_for_status()
    text = resp.text or ""
    if "html" in (resp.headers.get("Content-Type") or "").lower() or text.lstrip()[:1] == "<":
        return html_to_text(text)
    return text