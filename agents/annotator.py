"""
agents/annotator.py
Agent 2 — Annotator.
Takes an EvidenceBundle from Agent 1, generates an ESM-2 embedding,
constructs a structured prompt for OpenRouter, and returns an AnnotationDraft.
The LLM is forbidden from proposing a function not grounded in fetched evidence.

Uses OpenRouter chat completions.
Set OPENROUTER_API_KEY in your .env file.
"""

import json
import logging
import re
from datetime import datetime

from dotenv import load_dotenv

from agents.openrouter_client import call_openrouter, get_openrouter_api_key
from models.schemas import AnnotationDraft, EvidenceBundle
from tools.esm import get_embedding, get_embedding_summary

load_dotenv()
logger = logging.getLogger(__name__)

MAX_RETRIES = 2

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


def _build_user_prompt(
    bundle: EvidenceBundle,
    embedding_summary: dict,
    critic_challenges: list[str] | None = None,
) -> str:
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

    if critic_challenges:
        lines.append("=== Critic Challenges (must address) ===")
        for challenge in critic_challenges:
            lines.append(f"- {challenge}")
        lines.append("")

    lines.append("Based on the above evidence, propose the protein's function.")
    return "\n".join(lines)


def _call_openrouter(prompt: str, api_key: str) -> str | None:
    return call_openrouter(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        api_key=api_key,
        logger=logger,
        temperature=0.2,
        max_tokens=1024,
    )


def _parse_llm_response(raw_text: str) -> dict | None:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        parsed = json.loads(cleaned, strict=False)
        required = {"proposed_function", "go_term_candidates", "reasoning", "evidence_used"}
        if not required.issubset(parsed.keys()):
            logger.error(f"LLM response missing fields: {required - parsed.keys()}")
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {e}\nRaw:\n{raw_text[:500]}")
        return None


def _clean_description(description: str) -> str:
    text = description.strip()
    rec_name = re.search(r"RecName:\s*Full=([^;>\[]+)", text)
    if rec_name:
        text = rec_name.group(1)
    text = re.sub(r"\s*\[[^\]]+\]\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;") or "uncharacterized protein"


def _fallback_annotation(bundle: EvidenceBundle, embedding_shape: tuple) -> AnnotationDraft:
    hits = bundle.uniprot_hits + bundle.blast_hits
    if not hits:
        function = "ANNOTATION_FAILED - no evidence available to support a function."
        reasoning = "The LLM response could not be parsed, and no UniProt or BLAST evidence was available."
        evidence_used: list[str] = []
    else:
        best_hit = max(hits, key=lambda h: h.identity_pct)
        description = _clean_description(best_hit.description)
        if best_hit.identity_pct >= 70.0:
            function = f"Likely {description}, based on high-identity homology evidence."
        elif best_hit.identity_pct >= 30.0:
            function = f"Possible {description}, based on moderate homology evidence."
        else:
            function = f"Low-confidence similarity to {description}."
        reasoning = (
            "The LLM response could not be parsed, so ProtAgent generated a conservative "
            f"evidence-grounded fallback from the best hit: {best_hit.accession} "
            f"({best_hit.identity_pct:.1f}% identity, E-value {best_hit.e_value:.2e})."
        )
        evidence_used = [
            f"{best_hit.accession} - fallback from {best_hit.source} hit at "
            f"{best_hit.identity_pct:.1f}% identity"
        ]

    go_terms = []
    for hit in bundle.uniprot_hits:
        go_terms.extend(hit.go_terms)

    return AnnotationDraft(
        sequence_id=bundle.sequence_id,
        proposed_function=function,
        go_term_candidates=go_terms[:5],
        reasoning=reasoning,
        evidence_used=evidence_used,
        embedding_shape=embedding_shape,
        annotated_at=datetime.utcnow().isoformat(),
    )


def annotate(bundle: EvidenceBundle, critic_challenges: list[str] | None = None) -> AnnotationDraft:
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

    user_prompt = _build_user_prompt(bundle, embedding_summary, critic_challenges)

    api_key = get_openrouter_api_key()

    parsed_response = None
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Calling OpenRouter API (attempt {attempt}/{MAX_RETRIES})...")
        raw_text = _call_openrouter(user_prompt, api_key)
        if raw_text:
            parsed_response = _parse_llm_response(raw_text)
            if parsed_response:
                break
            logger.warning(f"Parse failed on attempt {attempt}. Retrying...")

    if not parsed_response:
        logger.warning("LLM annotation failed; returning conservative evidence-grounded fallback.")
        return _fallback_annotation(bundle, embedding_shape)

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
