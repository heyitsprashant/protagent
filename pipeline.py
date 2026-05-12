"""Phase 1 orchestration entry point for the ProtAgent pipeline.

This module will eventually coordinate sequence retrieval, annotation,
critique, and explanation once the agent stack is introduced.
"""

from dotenv import load_dotenv


load_dotenv()


def main() -> int:
    """Entry point reserved for the future multi-agent pipeline."""

    print("ProtAgent pipeline scaffold ready for Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())