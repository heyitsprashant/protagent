"""
pipeline.py
Orchestrates Agent 1 (gatherer) → Agent 2 (annotator) → Agent 3 (critic) → Agent 4 (explainer).
Accepts a FASTA file path or raw FASTA string.
Saves output JSON to outputs/ and prints a readable summary.

Usage:
    python pipeline.py --input data/test_sequences/hemoglobin.fasta
    python pipeline.py --fasta ">seq1\nMVHLTPEEKSAVTALWGKVNVDEVGGEALGR"
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import asdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from agents.gatherer import gather
from agents.annotator import annotate
from agents.critic import critique
from agents.explainer import explain

os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def run(fasta_input: str) -> dict:
    """
    Run the full Phase 2 pipeline on a FASTA string.
    Returns the output JSON as a dict.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("ProtAgent Phase 2 Pipeline — START")
    logger.info("=" * 60)

    # Agent 1
    bundle = gather(fasta_input)

    # Agent 2
    draft = annotate(bundle)

    # Agent 3 + revision loop
    revision_rounds = 0
    verdict = critique(draft, bundle, revision_round=1)
    while verdict.revision_required and revision_rounds < 2:
        revision_rounds += 1
        logger.info(f"Revision round {revision_rounds} triggered by critic.")
        draft = annotate(bundle, critic_challenges=verdict.challenges)
        verdict = critique(draft, bundle, revision_round=revision_rounds + 1)

    if revision_rounds >= 2 and verdict.revision_required and not verdict.passed:
        logger.warning("Max revision rounds used; proceeding with warning.")

    # Agent 4
    report = explain(draft, verdict, bundle, revision_rounds)

    elapsed = round(time.time() - start, 2)

    # Save output
    output_data = {
        "pipeline_version": "phase2",
        "run_at": datetime.utcnow().isoformat(),
        "elapsed_seconds": elapsed,
        "evidence_bundle": asdict(bundle),
        "annotation_draft": asdict(draft),
        "critic_verdict": asdict(verdict),
        "confidence_report": asdict(report),
    }

    # Sanitize sequence_id for use as a filename — strip characters
    # that are invalid on Windows (|, :, /, \, *, ?, ", <, >)
    safe_id = re.sub(r'[|:\/\\*?"<>]', '_', bundle.sequence_id)
    output_path = f"outputs/{safe_id}_phase2.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Output saved to {output_path}")

    # Print readable summary
    print("\n" + "=" * 60)
    print("ProtAgent — Phase 2 Summary")
    print("=" * 60)
    print(f"Protein ID      : {draft.sequence_id}")
    print(f"Evidence Quality: {bundle.evidence_quality.upper()}")
    print(f"Critic Passed   : {verdict.passed}")
    print(f"Confidence Score: {report.confidence_score} ({report.confidence_label})")
    if report.warnings:
        print(f"Warnings        : {', '.join(report.warnings)}")
    print(f"UniProt Hits    : {len(bundle.uniprot_hits)}")
    print(f"BLAST Hits      : {len(bundle.blast_hits)}")
    print(f"Proposed Function:")
    print(f"  {draft.proposed_function}")
    print(f"GO Term Candidates:")
    for go in draft.go_term_candidates:
        print(f"  • {go}")
    print(f"Reasoning:")
    print(f"  {draft.reasoning[:300]}{'...' if len(draft.reasoning) > 300 else ''}")
    print(f"Evidence Used:")
    for ev in draft.evidence_used:
        print(f"  • {ev}")
    print(f"Embedding Shape : {draft.embedding_shape}")
    print(f"Runtime         : {elapsed}s")
    print(f"Output saved to : {output_path}")
    print("=" * 60 + "\n")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="ProtAgent Phase 1 Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to a FASTA file")
    group.add_argument("--fasta", type=str, help="Raw FASTA string")
    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"[ERROR] File not found: {args.input}")
            exit(1)
        with open(args.input, "r") as f:
            fasta_input = f.read()
    else:
        fasta_input = args.fasta

    run(fasta_input)


if __name__ == "__main__":
    main()