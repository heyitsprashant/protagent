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


@dataclass
class CriticVerdict:
    sequence_id: str
    passed: bool
    challenges: list[str] = field(default_factory=list)
    revision_required: bool = False
    revision_round: int = 0
    critique_reasoning: str = ""
    critiqued_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConfidenceReport:
    sequence_id: str
    final_function: str
    confidence_score: float
    confidence_label: str
    go_term_candidates: list[str] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)
    critic_passed: bool = False
    revision_rounds: int = 0
    warnings: list[str] = field(default_factory=list)
    reasoning_summary: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())