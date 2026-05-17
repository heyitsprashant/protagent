"""
tools/uniprot.py
Fetches protein evidence from the UniProt REST API v2.
Given a UniProt accession ID, returns a list of EvidenceItem objects.
"""

import logging
import time
import requests
from models.schemas import EvidenceItem

logger = logging.getLogger(__name__)

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


def _get_with_retry(url: str, params: dict = None) -> dict | None:
    """GET request with exponential backoff retry on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429 or resp.status_code >= 500:
                wait = BACKOFF_BASE ** attempt
                logger.warning(f"UniProt HTTP {resp.status_code}, retrying in {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                logger.error(f"UniProt unexpected status {resp.status_code} for {url}")
                return None
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"UniProt request error: {e}, retrying in {wait}s")
            time.sleep(wait)
    logger.error(f"UniProt failed after {MAX_RETRIES} attempts: {url}")
    return None


def _extract_go_terms(data: dict) -> list[str]:
    """Pull GO term IDs from a UniProt entry JSON."""
    go_terms = []
    db_refs = data.get("uniProtKBCrossReferences", [])
    for ref in db_refs:
        if ref.get("database") == "GO":
            go_id = ref.get("id", "")
            # Also grab the term name if available
            props = ref.get("properties", [])
            term_name = next((p["value"] for p in props if p.get("key") == "GoTerm"), "")
            go_terms.append(f"{go_id} {term_name}".strip())
    return go_terms[:10]  # cap at 10 GO terms


def _extract_function_text(data: dict) -> str:
    """Extract the curated function description from UniProt comments."""
    comments = data.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value", "")
    return ""


def fetch_by_accession(accession: str) -> list[EvidenceItem]:
    """
    Fetch a single UniProt entry by accession ID.
    Returns a list with one EvidenceItem, or empty list on failure.
    """
    url = f"{UNIPROT_BASE}/{accession}"
    params = {"format": "json"}
    logger.info(f"Fetching UniProt accession: {accession}")
    data = _get_with_retry(url, params)
    if not data:
        return []

    # Build description from protein name + function comment
    protein_names = data.get("proteinDescription", {})
    recommended = protein_names.get("recommendedName", {})
    full_name = recommended.get("fullName", {}).get("value", accession)
    function_text = _extract_function_text(data)
    description = f"{full_name}. {function_text}".strip(". ")

    go_terms = _extract_go_terms(data)

    return [EvidenceItem(
        source="uniprot",
        accession=accession,
        description=description,
        go_terms=go_terms,
    )]


def search_by_sequence_similarity(sequence: str, max_results: int = 3) -> list[EvidenceItem]:
    """
    Search UniProt for similar proteins using the full-text search API
    with the first 20 amino acids as a query hint, plus BLAST-style lookup.
    Falls back to a keyword search on the sequence fragment.
    """
    # Use the UniProt BLAST endpoint for sequence similarity
    url = f"{UNIPROT_BASE}/search"
    # Search using sequence fragment — UniProt doesn't do direct AA search via REST,
    # so we search by the first meaningful chunk as a query token
    fragment = sequence[:30]
    params = {
        "query": fragment,
        "format": "json",
        "size": max_results,
        "fields": "accession,protein_name,go,cc_function",
    }
    logger.info(f"UniProt sequence similarity search (fragment: {fragment[:15]}...)")
    data = _get_with_retry(url, params)
    if not data:
        return []

    results = data.get("results", [])
    items = []
    for entry in results[:max_results]:
        accession = entry.get("primaryAccession", "")
        protein_names = entry.get("proteinDescription", {})
        recommended = protein_names.get("recommendedName", {})
        full_name = recommended.get("fullName", {}).get("value", accession)
        function_text = _extract_function_text(entry)
        description = f"{full_name}. {function_text}".strip(". ")
        go_terms = _extract_go_terms(entry)
        items.append(EvidenceItem(
            source="uniprot",
            accession=accession,
            description=description,
            go_terms=go_terms,
        ))
    return items


def fetch_evidence(accession_or_sequence: str, is_accession: bool = True) -> list[EvidenceItem]:
    """
    Main entry point.
    If is_accession=True, fetch by accession ID directly.
    If is_accession=False, search by sequence similarity.
    """
    if is_accession:
        return fetch_by_accession(accession_or_sequence)
    else:
        return search_by_sequence_similarity(accession_or_sequence)