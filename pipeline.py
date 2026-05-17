"""
pipeline.py
Orchestrates Agent 1 (gatherer) → Agent 2 (annotator).
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
    Run the full Phase 1 pipeline on a FASTA string.
    Returns the AnnotationDraft as a dict.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("ProtAgent Phase 1 Pipeline — START")
    logger.info("=" * 60)

    # Agent 1
    bundle = gather(fasta_input)

    # Agent 2
    draft = annotate(bundle)

    elapsed = round(time.time() - start, 2)

    # Save output
    output_data = {
        "pipeline_version": "phase1",
        "run_at": datetime.utcnow().isoformat(),
        "elapsed_seconds": elapsed,
        "evidence_bundle": {
            "sequence_id": bundle.sequence_id,
            "sequence_length": len(bundle.raw_sequence),
            "evidence_quality": bundle.evidence_quality,
            "uniprot_hits": len(bundle.uniprot_hits),
            "blast_hits": len(bundle.blast_hits),
        },
        "annotation_draft": asdict(draft),
    }

    # Sanitize sequence_id for use as a filename — strip characters
    # that are invalid on Windows (|, :, /, \, *, ?, ", <, >)
    safe_id = re.sub(r'[|:\/\\*?"<>]', '_', bundle.sequence_id)
    output_path = f"outputs/{safe_id}_phase1.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Output saved to {output_path}")

    # Print readable summary
    print("\n" + "=" * 60)
    print("ProtAgent — Annotation Summary")
    print("=" * 60)
    print(f"Protein ID      : {draft.sequence_id}")
    print(f"Evidence Quality: {bundle.evidence_quality.upper()}")
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