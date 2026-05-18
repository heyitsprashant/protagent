"""
agents/critic.py
Agent 3 - Critic.
Adversarially challenges annotations and flags weak evidence grounding.
"""

import json
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from agents.openrouter_client import call_openrouter, get_openrouter_api_key
from models.schemas import AnnotationDraft, EvidenceBundle, CriticVerdict

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

CRITIC_SYSTEM_PROMPT = """You are an adversarial peer reviewer for protein functional annotation.

You will be shown a proposed annotation for a protein, along with the evidence
that was used to produce it. Your job is to find weaknesses — not to validate.

Check specifically for these four failure modes:
1. UNSUPPORTED FUNCTION: Is the proposed function actually mentioned or clearly
   implied by the UniProt descriptions or BLAST hit descriptions? If not, flag it.
2. GO TERM MISMATCH: Do the proposed GO terms appear in the provided UniProt
   GO term list? If any are invented, flag them.
3. WEAK HOMOLOGY: If the best BLAST identity is below 30%, flag that the
   annotation is based on distant homology and may not transfer.
4. LOW EVIDENCE: If evidence quality is "low" (no UniProt hits, no BLAST hits),
   flag that the annotation is speculative.

Respond with ONLY a valid JSON object:
{
  "passed": true | false,
  "challenges": ["specific challenge 1", "specific challenge 2"],
  "revision_required": true | false,
  "critique_reasoning": "2-3 sentences explaining your verdict"
}

If the annotation is well-supported and none of the four failure modes apply,
return passed=true, challenges=[], revision_required=false.
Do not include markdown fences. Return JSON only.
"""


def _ensure_logger() -> None:
	if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
		return
	os.makedirs("logs", exist_ok=True)
	handler = logging.FileHandler("logs/critic.log")
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)


def _build_prompt(draft: AnnotationDraft, bundle: EvidenceBundle) -> str:
	lines = [
		f"Protein ID: {bundle.sequence_id}",
		f"Evidence quality: {bundle.evidence_quality}",
		"",
		"=== Proposed Annotation ===",
		f"Proposed function: {draft.proposed_function}",
		f"GO terms: {', '.join(draft.go_term_candidates) if draft.go_term_candidates else 'None'}",
		f"Reasoning: {draft.reasoning}",
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

	lines.append("Provide your critique as JSON.")
	return "\n".join(lines)


def _call_openrouter(prompt: str, api_key: str) -> str | None:
	return call_openrouter(
		system_prompt=CRITIC_SYSTEM_PROMPT,
		user_prompt=prompt,
		api_key=api_key,
		logger=logger,
		temperature=0.2,
		max_tokens=512,
	)


def _parse_critic_response(raw_text: str) -> dict | None:
	cleaned = raw_text.strip()
	if cleaned.startswith("```"):
		lines = cleaned.splitlines()
		cleaned = "\n".join(
			lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
		)
	try:
		parsed = json.loads(cleaned)
		required = {"passed", "challenges", "revision_required", "critique_reasoning"}
		if not required.issubset(parsed.keys()):
			logger.error(f"Critic response missing fields: {required - parsed.keys()}")
			return None
		return parsed
	except json.JSONDecodeError as e:
		logger.error(f"Failed to parse critic JSON: {e}\nRaw:\n{raw_text[:500]}")
		return None


def _deterministic_challenges(draft: AnnotationDraft, bundle: EvidenceBundle) -> list[str]:
	challenges: list[str] = []

	evidence_text = " ".join([h.description for h in bundle.uniprot_hits + bundle.blast_hits]).lower()
	tokens = [
		t for t in "".join(ch if ch.isalnum() else " " for ch in draft.proposed_function.lower()).split()
		if len(t) > 3
	]
	if tokens and not any(t in evidence_text for t in tokens):
		challenges.append("UNSUPPORTED FUNCTION: proposed function not supported by evidence descriptions.")

	if draft.go_term_candidates:
		uniprot_terms = {t for h in bundle.uniprot_hits for t in h.go_terms}
		for term in draft.go_term_candidates:
			if term not in uniprot_terms:
				challenges.append(f"GO TERM MISMATCH: {term} not present in UniProt GO terms.")

	if bundle.blast_hits:
		best_identity = max(h.identity_pct for h in bundle.blast_hits)
		if best_identity < 30.0:
			challenges.append("WEAK HOMOLOGY: best BLAST identity below 30%.")

	if bundle.evidence_quality == "low":
		challenges.append("LOW EVIDENCE: evidence quality is low.")

	return challenges


def critique(
	draft: AnnotationDraft,
	bundle: EvidenceBundle,
	revision_round: int,
) -> CriticVerdict:
	_ensure_logger()

	api_key = get_openrouter_api_key()

	prompt = _build_prompt(draft, bundle)
	parsed_response = None
	for attempt in range(1, MAX_RETRIES + 1):
		raw_text = _call_openrouter(prompt, api_key)
		if raw_text:
			parsed_response = _parse_critic_response(raw_text)
			if parsed_response:
				break
			logger.warning(f"Critic parse failed on attempt {attempt}. Retrying...")

	if not parsed_response:
		verdict = CriticVerdict(
			sequence_id=bundle.sequence_id,
			passed=False,
			challenges=["critic parse failed — treating as uncertain"],
			revision_required=False,
			revision_round=revision_round,
			critique_reasoning="Critic LLM response could not be parsed after retries.",
			critiqued_at=datetime.utcnow().isoformat(),
		)
		logger.info(f"Critic verdict: {verdict}")
		return verdict

	challenges = list(parsed_response.get("challenges", []))
	deterministic = _deterministic_challenges(draft, bundle)
	for item in deterministic:
		if item not in challenges:
			challenges.append(item)

	passed = bool(parsed_response.get("passed")) and not deterministic
	revision_required = bool(parsed_response.get("revision_required")) or bool(deterministic)
	critique_reasoning = parsed_response.get("critique_reasoning", "")
	if deterministic and not critique_reasoning:
		critique_reasoning = "Deterministic checks flagged failure modes in the annotation."

	verdict = CriticVerdict(
		sequence_id=bundle.sequence_id,
		passed=passed,
		challenges=challenges,
		revision_required=revision_required,
		revision_round=revision_round,
		critique_reasoning=critique_reasoning,
		critiqued_at=datetime.utcnow().isoformat(),
	)
	logger.info(f"Critic verdict: {verdict}")
	return verdict
