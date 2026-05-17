"""
models/schemas.py
Shared data models (dataclasses) used across all agents and tools in ProtAgent.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EvidenceItem:
    source: str              # "uniprot" | "blast"
    accession: str
    description: str
    go_terms: list[str] = field(default_factory=list)
    identity_pct: float = 0.0   # BLAST hits only
    e_value: float = 0.0         # BLAST hits only


@dataclass
class EvidenceBundle:
    sequence_id: str
    raw_sequence: str
    uniprot_hits: list[EvidenceItem] = field(default_factory=list)
    blast_hits: list[EvidenceItem] = field(default_factory=list)
    evidence_quality: str = "low"   # "high" | "medium" | "low"
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AnnotationDraft:
    sequence_id: str
    proposed_function: str
    go_term_candidates: list[str] = field(default_factory=list)
    reasoning: str = ""
    evidence_used: list[str] = field(default_factory=list)
    embedding_shape: tuple = (0,)
    annotated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())