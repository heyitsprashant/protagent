"""
tools/blast.py
Submits a BLASTP job to the NCBI public API and polls for results.
Returns top BLAST hits as EvidenceItem objects.
"""

import logging
import time
import requests
import xml.etree.ElementTree as ET
from models.schemas import EvidenceItem

logger = logging.getLogger(__name__)

BLAST_BASE = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
POLL_INTERVAL = 5    # seconds between status checks
TIMEOUT = 120        # seconds before giving up
MAX_HITS = 5


def _submit_job(sequence: str) -> str | None:
    """Submit a BLASTP job and return the RID (Request ID)."""
    params = {
        "CMD": "Put",
        "PROGRAM": "blastp",
        "DATABASE": "swissprot",   # Swiss-Prot — curated, faster than nr
        "QUERY": sequence,
        "FORMAT_TYPE": "XML",
        "HITLIST_SIZE": MAX_HITS,
        "EXPECT": "0.001",
    }
    logger.info("Submitting BLASTP job to NCBI...")
    try:
        resp = requests.post(BLAST_BASE, data=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"BLAST submit failed: {e}")
        return None

    # Parse RID from the response HTML
    for line in resp.text.splitlines():
        if "RID = " in line:
            rid = line.strip().split("RID = ")[-1].strip()
            logger.info(f"BLAST job submitted. RID: {rid}")
            return rid

    logger.error("Could not parse RID from BLAST submit response.")
    return None


def _poll_for_results(rid: str) -> str | None:
    """Poll NCBI until the job is done. Returns raw XML string or None on timeout."""
    elapsed = 0
    while elapsed < TIMEOUT:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        params = {
            "CMD": "Get",
            "RID": rid,
            "FORMAT_TYPE": "XML",
        }
        try:
            resp = requests.get(BLAST_BASE, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"BLAST poll error: {e}")
            continue

        if "Status=WAITING" in resp.text:
            logger.info(f"BLAST job still running... ({elapsed}s elapsed)")
            continue
        elif "Status=FAILED" in resp.text:
            logger.error("BLAST job failed on NCBI side.")
            return None
        elif "Status=UNKNOWN" in resp.text:
            logger.error("BLAST RID expired or unknown.")
            return None
        else:
            logger.info(f"BLAST job complete ({elapsed}s)")
            return resp.text

    logger.error(f"BLAST job timed out after {TIMEOUT}s")
    return None


def _parse_hits(xml_text: str) -> list[EvidenceItem]:
    """Parse BLAST XML and extract top hits as EvidenceItem objects."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse BLAST XML: {e}")
        return []

    hits = root.findall(".//Hit")
    for hit in hits[:MAX_HITS]:
        accession = hit.findtext("Hit_accession", default="").strip()
        description = hit.findtext("Hit_def", default="").strip()

        # Get best HSP (highest-scoring alignment)
        hsp = hit.find(".//Hsp")
        if hsp is None:
            continue

        identity = int(hsp.findtext("Hsp_identity", default="0"))
        align_len = int(hsp.findtext("Hsp_align-len", default="1"))
        e_value_str = hsp.findtext("Hsp_evalue", default="1.0")

        try:
            e_value = float(e_value_str)
        except ValueError:
            e_value = 1.0

        identity_pct = round((identity / align_len) * 100, 2) if align_len > 0 else 0.0

        items.append(EvidenceItem(
            source="blast",
            accession=accession,
            description=description,
            go_terms=[],          # BLAST doesn't return GO terms directly
            identity_pct=identity_pct,
            e_value=e_value,
        ))

    logger.info(f"Parsed {len(items)} BLAST hits.")
    return items


def run_blast(sequence: str) -> list[EvidenceItem]:
    """
    Main entry point.
    Submits a BLASTP job for the given amino acid sequence,
    polls for results, and returns parsed EvidenceItem hits.
    Returns an empty list with a logged error on any failure.
    """
    if not sequence or len(sequence) < 10:
        logger.warning("Sequence too short for BLAST (<10 AA). Skipping.")
        return []

    # Truncate very long sequences — NCBI accepts up to ~10,000 AA
    # but long sequences are slow; cap at 1000 for Phase 1
    if len(sequence) > 1000:
        logger.warning(f"Sequence length {len(sequence)} truncated to 1000 AA for BLAST.")
        sequence = sequence[:1000]

    rid = _submit_job(sequence)
    if not rid:
        return []

    xml_text = _poll_for_results(rid)
    if not xml_text:
        return []

    return _parse_hits(xml_text)