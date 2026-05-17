"""
agents/annotator.py
Agent 2 — Annotator.
Takes an EvidenceBundle from Agent 1, generates an ESM-2 embedding,
constructs a structured prompt for Gemini, and returns an AnnotationDraft.
The LLM is forbidden from proposing a function not grounded in fetched evidence.

Uses Google Gemini API (gemini-1.5-flash — free tier available).
Set GEMINI_API_KEY in your .env file.
Get a free key at: https://aistudio.google.com/app/apikey
"""

import json
import logging
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

from models.schemas import AnnotationDraft, EvidenceBundle
from tools.esm import get_embedding, get_embedding_summary

load_dotenv()
logger = logging.getLogger(__name__)

MAX_RETRIES = 2
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = """You are a computational biology expert specializing in protein functional annotation.

Your task is to propose the most likely biological function for an unknown protein, based ONLY on the evidence provided to you. You must not invent functions that are not supported by the provided UniProt matches or BLAST hits.

RULES:
1. Your proposed_function must be grounded in at least one of the provided evidence items.
2. GO term candidates must come from the GO terms listed in the evidence — do not invent new ones.
3. If evidence quality is "low", your reasoning must clearly state that the annotation has low confidence.
4. Be specific — "oxygen transport" is better than "biological process".

You must respond with ONLY a valid JSON object in this exact format:
{
  "proposed_function": "concise function description (1-2 sentences)",
  "go_term_candidates": ["GO:XXXXXX term name", "GO:XXXXXX term name"],
  "reasoning": "explanation of why you proposed this function based on the evidence",
  "evidence_used": ["accession1 - why it was relevant", "accession2 - why it was relevant"]
}

Do not include any text outside the JSON object. Do not use markdown code fences."""


def _build_user_prompt(bundle: EvidenceBundle, embedding_summary: dict) -> str:
    lines = [
        f"Protein ID: {bundle.sequence_id}",
        f"Sequence length: {len(bundle.raw_sequence)} amino acids",
        f"Evidence quality: {bundle.evidence_quality}",
        "",
        "=== ESM-2 Embedding Summary ===",
        f"Embedding norm: {embedding_summary['norm']:.4f}",
        f"Mean activation: {embedding_summary['mean']:.4f}",
        f"Std deviation: {embedding_summary['std']:.4f}",
        "",
    ]

    if bundle.uniprot_hits:
        lines.append("=== UniProt Matches ===")
        for hit in bundle.uniprot_hits:
            lines.append(f"Accession: {hit.accession}")
            lines.append(f"Description: {hit.description}")
            if hit.go_terms:
                lines.append(f"GO Terms: {', '.join(hit.go_terms[:5])}")
            lines.append("")
    else:
        lines.append("=== UniProt Matches ===\nNo UniProt matches found.\n")

    if bundle.blast_hits:
        lines.append("=== BLAST Hits ===")
        for hit in bundle.blast_hits:
            lines.append(f"Accession: {hit.accession}")
            lines.append(f"Description: {hit.description}")
            lines.append(f"Identity: {hit.identity_pct:.1f}% | E-value: {hit.e_value:.2e}")
            lines.append("")
    else:
        lines.append("=== BLAST Hits ===\nNo BLAST hits found.\n")

    lines.append("Based on the above evidence, propose the protein's function.")
    return "\n".join(lines)


def _call_gemini(prompt: str, api_key: str) -> str | None:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        }
    }

    try:
        resp = requests.post(
            GEMINI_ENDPOINT,
            headers=headers,
            params=params,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.HTTPError as e:
        logger.error(f"Gemini HTTP error: {e} — {resp.text[:300]}")
        return None
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected Gemini response structure: {e}")
        return None
    except requests.RequestException as e:
        logger.error(f"Gemini request failed: {e}")
        return None


def _parse_llm_response(raw_text: str) -> dict | None:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )
    try:
        parsed = json.loads(cleaned)
        required = {"proposed_function", "go_term_candidates", "reasoning", "evidence_used"}
        if not required.issubset(parsed.keys()):
            logger.error(f"LLM response missing fields: {required - parsed.keys()}")
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {e}\nRaw:\n{raw_text[:500]}")
        return None


def annotate(bundle: EvidenceBundle) -> AnnotationDraft:
    logger.info(f"=== Agent 2: Annotator started for {bundle.sequence_id} ===")

    # ESM-2 embedding
    try:
        embedding = get_embedding(bundle.raw_sequence)
        embedding_summary = get_embedding_summary(bundle.raw_sequence)
        embedding_shape = embedding.shape
    except Exception as e:
        logger.error(f"ESM-2 embedding failed: {e}")
        embedding_shape = (0,)
        embedding_summary = {"norm": 0, "mean": 0, "std": 0}

    user_prompt = _build_user_prompt(bundle, embedding_summary)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Add to .env: GEMINI_API_KEY=your_key_here"
        )

    parsed_response = None
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Calling Gemini API (attempt {attempt}/{MAX_RETRIES})...")
        raw_text = _call_gemini(user_prompt, api_key)
        if raw_text:
            parsed_response = _parse_llm_response(raw_text)
            if parsed_response:
                break
            logger.warning(f"Parse failed on attempt {attempt}. Retrying...")

    if not parsed_response:
        return AnnotationDraft(
            sequence_id=bundle.sequence_id,
            proposed_function="ANNOTATION_FAILED — could not parse LLM response after retries.",
            go_term_candidates=[],
            reasoning="LLM returned malformed JSON on all attempts.",
            evidence_used=[],
            embedding_shape=embedding_shape,
            annotated_at=datetime.utcnow().isoformat(),
        )

    # Enforce evidence grounding (FR-5.6)
    all_accessions = (
        {h.accession for h in bundle.uniprot_hits} |
        {h.accession for h in bundle.blast_hits}
    )
    cited = str(parsed_response.get("evidence_used", []))
    if all_accessions and not any(acc in cited for acc in all_accessions):
        logger.warning("LLM cited no matching accessions — flagging as potentially ungrounded.")
        parsed_response["reasoning"] = (
            "[WARNING: annotation may not be grounded in fetched evidence] "
            + parsed_response["reasoning"]
        )

    draft = AnnotationDraft(
        sequence_id=bundle.sequence_id,
        proposed_function=parsed_response["proposed_function"],
        go_term_candidates=parsed_response.get("go_term_candidates", []),
        reasoning=parsed_response["reasoning"],
        evidence_used=parsed_response.get("evidence_used", []),
        embedding_shape=embedding_shape,
        annotated_at=datetime.utcnow().isoformat(),
    )

    logger.info(f"=== Agent 2 complete. Function: {draft.proposed_function[:80]}... ===")
    return draft