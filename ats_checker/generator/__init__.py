"""Resume generator — builds a JD-tailored resume from the evidence bank.

The scoring layers answer "will this resume pass?". The generator answers
"what should the resume be?" — by selecting and rephrasing real evidence so
the existing three-layer scorer, used unchanged as the objective function,
rates it as high as it can honestly go.

Truth is structural here: the assembler renders ONLY evidence-bank atoms,
so nothing can appear in a tailored resume that wasn't real first. LLM
rewording is verified against the original bullet (numbers and dates must
survive, none may appear) before it's allowed into the output.
"""
from .evidence_bank import (
    DEFAULT_MASTER_PATH,
    Bullet,
    Education,
    EvidenceBank,
    Role,
    init_from_resume,
    load_bank,
    write_template,
)
from .selector import Selection, select, canonical, alias_normalize, text_covers
from .assembler import assemble, write_docx
from .rewriter import RewriteResult, plausible_targets, reword_for_terms, verify_rewrite
from .optimizer import TailorResult, tailor

__all__ = [
    "DEFAULT_MASTER_PATH", "Bullet", "Education", "EvidenceBank", "Role",
    "init_from_resume", "load_bank", "write_template",
    "Selection", "select", "canonical", "alias_normalize", "text_covers",
    "assemble", "write_docx",
    "RewriteResult", "plausible_targets", "reword_for_terms", "verify_rewrite",
    "TailorResult", "tailor",
]