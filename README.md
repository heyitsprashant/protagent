# ProtAgent

ProtAgent is a local research scaffold for a protein functional annotation system. Phase 0 exists to make the project reproducible from day one: a clean repo layout, a pinned Python environment, static test sequences, environment-variable loading, and a smoke test that proves the machine is ready before any real model logic is added. That matters because future phases depend on a stable baseline, and the fastest way to lose time in research software is to start from a broken setup.

## Quickstart

```bash
git clone https://github.com/<your-username>/protagent.git
cd protagent
python3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python smoke_test.py
```

If you are on PowerShell and `cp` is unavailable, use `Copy-Item .env.example .env` instead. After cloning on Windows, activate the venv with `venv\Scripts\activate`.

Configure OpenRouter in `.env` before running the annotation pipeline:

```text
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=deepseek/deepseek-v4-flash:free,meta-llama/llama-3.3-70b-instruct:free,meta-llama/llama-3.2-3b-instruct:free,openai/gpt-oss-20b:free
```

`OPENROUTER_MODEL` accepts a comma-separated fallback list. If one model rate limits or has a transient server error, ProtAgent tries the next model.

## Project structure

```text
protagent/
├── agents/              future agent roles for gathering, annotating, criticizing, and explaining
├── tools/               shared utilities for UniProt, BLAST, and ESM access
├── models/              data schemas and structured outputs
├── tests/               placeholder test modules for future phases
├── data/test_sequences/ static FASTA fixtures used by the smoke test
├── outputs/             generated results, gitignored except for .gitkeep
├── logs/                runtime logs, gitignored except for .gitkeep
├── pipeline.py          future pipeline entry point
├── smoke_test.py        Phase 0 environment verification script
├── requirements.txt     pinned Python dependencies
├── .env.example         committed environment template
├── .gitignore           repository hygiene rules
└── README.md            project overview and setup instructions
```

## Phases

Phase 0 establishes the local foundation: repository layout, dependencies, environment variables, and a passing smoke test.

Phase 1 adds the first agentic pipeline for retrieval and annotation.

Phase 2 exposes the system through a FastAPI service.

Phase 3 improves orchestration, reliability, and explainability across the agent stack.

Phase 4 packages the research system for evaluation, cleanup, and release-quality use.

## Stack

Python 3.11, Biopython, Requests, HTTPX, OpenRouter-compatible chat completions, Transformers, PyTorch CPU, NumPy, FastAPI, Uvicorn, python-dotenv, Pytest, Black, and Ruff.

## Research

The paper target is a biorxiv preprint first, followed by submission to ISMB or RECOMB.

## What Phase 0 guarantees

After setup, `python smoke_test.py` verifies the interpreter, imports, API-key loading, required directories, and the three protein FASTA fixtures so the next phase starts from a known-good baseline.
