"""
agents/explainer.py
Agent 4 - Explainer.
Produces confidence reports and human-readable summaries.
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from agents.openrouter_client import call_openrouter, get_openrouter_api_key
from models.schemas import AnnotationDraft, CriticVerdict, EvidenceBundle, ConfidenceReport
from tools.esm import MAX_SEQUENCE_LENGTH

load_dotenv()

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = (
	"You are a scientific assistant. Summarize the annotation and confidence "
	"in 2-3 sentences using clear, concise language for biologists."
)


def _ensure_logger() -> None:
	if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
		return
	os.makedirs("logs", exist_ok=True)
	handler = logging.FileHandler("logs/explainer.log")
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)


def _call_openrouter(prompt: str, api_key: str) -> str | None:
	return call_openrouter(
		system_prompt=SUMMARY_SYSTEM_PROMPT,
		user_prompt=prompt,
		api_key=api_key,
		logger=logger,
		temperature=0.3,
		max_tokens=256,
	)


def _confidence_score(bundle: EvidenceBundle, critic: CriticVerdict, revision_rounds: int) -> float:
	score = 0.5

	if bundle.evidence_quality == "high":
		score += 0.2
	elif bundle.evidence_quality == "medium":
		score += 0.1
	elif bundle.evidence_quality == "low":
		score -= 0.3

	if critic.passed:
		score += 0.15

	if bundle.blast_hits:
		best_identity = max(h.identity_pct for h in bundle.blast_hits)
		if best_identity > 70.0:
			score += 0.1
		if best_identity > 90.0:
			score += 0.05

	if revision_rounds == 2:
		score -= 0.2
	elif revision_rounds == 1:
		score -= 0.1

	return max(0.0, min(1.0, round(score, 3)))


def _confidence_label(score: float) -> str:
	if score >= 0.75:
		return "high"
	if score >= 0.5:
		return "medium"
	if score >= 0.3:
		return "low"
	return "uncertain"


def _evidence_citations(bundle: EvidenceBundle) -> list[str]:
	citations: list[str] = []

	for hit in bundle.uniprot_hits + bundle.blast_hits:
		description = hit.description.strip()
		sentence = description.split(".")[0].strip() if description else "Evidence supports the annotation."
		citations.append(f"{hit.accession} ({hit.source}): {sentence}.")

	return citations


def _warnings(bundle: EvidenceBundle, critic: CriticVerdict, revision_rounds: int) -> list[str]:
	warnings: list[str] = []

	if bundle.evidence_quality == "low":
		warnings.append("low evidence quality")

	if len(bundle.raw_sequence) > MAX_SEQUENCE_LENGTH:
		warnings.append("sequence truncated for embedding")

	if revision_rounds >= 2:
		warnings.append("max revisions used")

	if any("critic parse failed" in c for c in critic.challenges):
		warnings.append("critic parse failed — treating as uncertain")

	return warnings


def explain(
	draft: AnnotationDraft,
	critic: CriticVerdict,
	bundle: EvidenceBundle,
	revision_rounds: int,
) -> ConfidenceReport:
	_ensure_logger()

	score = _confidence_score(bundle, critic, revision_rounds)
	label = _confidence_label(score)
	citations = _evidence_citations(bundle)
	warnings = _warnings(bundle, critic, revision_rounds)

	api_key = get_openrouter_api_key()

	prompt = (
		f"Protein ID: {bundle.sequence_id}\n"
		f"Final function: {draft.proposed_function}\n"
		f"Evidence quality: {bundle.evidence_quality}\n"
		f"Critic passed: {critic.passed}\n"
		f"Revision rounds: {revision_rounds}\n"
		f"Confidence score: {score} ({label})\n"
		f"Warnings: {', '.join(warnings) if warnings else 'none'}\n"
		"Provide a 2-3 sentence explanation for a biologist."
	)

	summary = _call_openrouter(prompt, api_key)
	if not summary:
		summary = (
			f"The annotation proposes: {draft.proposed_function}. "
			f"Evidence quality is {bundle.evidence_quality}, with confidence labeled {label}."
		)

	report = ConfidenceReport(
		sequence_id=bundle.sequence_id,
		final_function=draft.proposed_function,
		confidence_score=score,
		confidence_label=label,
		go_term_candidates=draft.go_term_candidates,
		evidence_citations=citations,
		critic_passed=critic.passed,
		revision_rounds=revision_rounds,
		warnings=warnings,
		reasoning_summary=summary.strip(),
		generated_at=datetime.utcnow().isoformat(),
	)

	logger.info(f"Confidence report: {report}")
	return report
