"""
agents/gatherer.py
Agent 1 — Evidence Gatherer.
Accepts a FASTA string, parses it with BioPython, then concurrently
fetches evidence from UniProt and BLAST. Returns a populated EvidenceBundle.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import StringIO

from Bio import SeqIO

from models.schemas import EvidenceBundle, EvidenceItem
from tools import uniprot, blast

# Set up file logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/gatherer.log"),
        logging.StreamHandler(),
    ],
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_fasta(fasta_string: str) -> tuple[str, str]:
    """
    Parse a FASTA string using BioPython.
    Returns (sequence_id, amino_acid_sequence).
    Raises ValueError if the FASTA is invalid or empty.
    """
    records = list(SeqIO.parse(StringIO(fasta_string), "fasta"))
    if not records:
        raise ValueError("No valid FASTA records found in input.")
    if len(records) > 1:
        logger.warning(f"Multiple FASTA records found — using only the first: {records[0].id}")

    record = records[0]
    sequence = str(record.seq).upper().strip()

    if not sequence:
        raise ValueError(f"FASTA record {record.id} has an empty sequence.")

    # Basic amino acid validation — reject if >10% are non-standard characters
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    non_standard = sum(1 for aa in sequence if aa not in standard_aa)
    if non_standard / len(sequence) > 0.1:
        logger.warning(
            f"Sequence {record.id} has {non_standard}/{len(sequence)} non-standard characters. "
            f"Proceeding but results may be unreliable."
        )

    return record.id, sequence


def _fetch_uniprot(sequence: str) -> list[EvidenceItem]:
    """Wrapper for concurrent execution."""
    logger.info("Starting UniProt fetch...")
    try:
        results = uniprot.fetch_evidence(sequence, is_accession=False)
        logger.info(f"UniProt returned {len(results)} hits.")
        return results
    except Exception as e:
        logger.error(f"UniProt fetch failed: {e}")
        return []


def _fetch_blast(sequence: str) -> list[EvidenceItem]:
    """Wrapper for concurrent execution."""
    logger.info("Starting BLAST fetch...")
    try:
        results = blast.run_blast(sequence)
        logger.info(f"BLAST returned {len(results)} hits.")
        return results
    except Exception as e:
        logger.error(f"BLAST fetch failed: {e}")
        return []


def _assess_quality(uniprot_hits: list, blast_hits: list) -> str:
    """
    Determine evidence quality level based on what was returned.
    high   = UniProt hits with GO terms + BLAST hits with >50% identity
    medium = some evidence from either source
    low    = both sources returned nothing useful
    """
    has_go_terms = any(len(h.go_terms) > 0 for h in uniprot_hits)
    has_good_blast = any(h.identity_pct > 50.0 for h in blast_hits)
    has_any_uniprot = len(uniprot_hits) > 0
    has_any_blast = len(blast_hits) > 0

    if has_go_terms and has_good_blast:
        return "high"
    elif has_any_uniprot or has_any_blast:
        return "medium"
    else:
        return "low"


def gather(fasta_input: str) -> EvidenceBundle:
    """
    Main entry point for Agent 1.

    Args:
        fasta_input: Raw FASTA string (with header line starting with '>')

    Returns:
        EvidenceBundle with all fetched evidence and quality rating.

    Raises:
        ValueError if the FASTA cannot be parsed.
    """
    logger.info("=== Agent 1: Evidence Gatherer started ===")

    # Step 1: Parse FASTA
    sequence_id, sequence = _parse_fasta(fasta_input)
    logger.info(f"Parsed sequence: {sequence_id} ({len(sequence)} AA)")

    # Step 2: Concurrent UniProt + BLAST fetch
    uniprot_hits: list[EvidenceItem] = []
    blast_hits: list[EvidenceItem] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_uniprot = executor.submit(_fetch_uniprot, sequence)
        future_blast = executor.submit(_fetch_blast, sequence)

        for future in as_completed([future_uniprot, future_blast]):
            if future == future_uniprot:
                uniprot_hits = future.result()
            else:
                blast_hits = future.result()

    # Step 3: Assess evidence quality
    quality = _assess_quality(uniprot_hits, blast_hits)
    logger.info(
        f"Evidence collected — UniProt: {len(uniprot_hits)}, "
        f"BLAST: {len(blast_hits)}, Quality: {quality}"
    )

    if quality == "low":
        logger.warning(
            f"Low evidence quality for {sequence_id}. "
            f"Annotation may be unreliable — Annotator will flag this."
        )

    bundle = EvidenceBundle(
        sequence_id=sequence_id,
        raw_sequence=sequence,
        uniprot_hits=uniprot_hits,
        blast_hits=blast_hits,
        evidence_quality=quality,
        fetched_at=datetime.utcnow().isoformat(),
    )

    logger.info(f"=== Agent 1 complete. Bundle ID: {bundle.sequence_id} ===")
    return bundle