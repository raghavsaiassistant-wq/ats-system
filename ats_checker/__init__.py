"""ATS Score Checker — local, hybrid resume-vs-JD scoring tool.

Mirrors how real ATS platforms evaluate candidates:
  1. Parsing risk    — can an ATS even extract your data cleanly?
  2. Keyword match    — weighted Boolean/keyword overlap (what legacy + modern
                         ATS Boolean filters still check: tools, certs, hard skills)
  3. Semantic fit      — LLM (via Ollama) judges contextual/narrative fit,
                         mirroring the embedding-based semantic layer used by
                         modern platforms (Workday, Eightfold, Phenom)

See README.md for methodology notes and sources.
"""

__version__ = "1.0.0"
