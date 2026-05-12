"""Environment smoke test for ProtAgent Phase 0."""

from __future__ import annotations

from pathlib import Path
import os
import sys

from Bio import SeqIO
from dotenv import load_dotenv


load_dotenv()


ROOT = Path(__file__).resolve().parent
SEQUENCE_DIR = ROOT / "data" / "test_sequences"
REQUIRED_FASTA = {
    "hemoglobin.fasta": "hemoglobin",
    "beta_galactosidase.fasta": "beta_galactosidase",
    "gfp.fasta": "gfp",
}
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*-")


def check(condition: bool, message: str) -> tuple[bool, str]:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return condition, message


def check_import(module_name: str, import_target: str | None = None) -> bool:
    try:
        __import__(import_target or module_name)
    except Exception as exc:  # pragma: no cover - reported inline
        print(f"[FAIL] {module_name} import failed: {exc}")
        return False

    print(f"[PASS] {module_name} imported")
    return True


def check_python_version() -> bool:
    version = sys.version_info
    ok = version >= (3, 11)
    label = f"Python {version.major}.{version.minor}.{version.micro}"
    if ok:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label} (3.11+ required)")
    return ok


def check_api_key() -> bool:
    value = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if value:
        print("[PASS] ANTHROPIC_API_KEY present")
        return True

    print("[FAIL] ANTHROPIC_API_KEY not found or empty")
    return False


def check_directories() -> bool:
    outputs_exists = (ROOT / "outputs").is_dir()
    logs_exists = (ROOT / "logs").is_dir()
    print(f"[{'PASS' if outputs_exists else 'FAIL'}] outputs/ directory exists")
    print(f"[{'PASS' if logs_exists else 'FAIL'}] logs/ directory exists")
    return outputs_exists and logs_exists


def validate_fasta(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        print(f"[FAIL] {path.name} missing or empty")
        return False

    try:
        record = next(SeqIO.parse(str(path), "fasta"))
    except StopIteration:
        print(f"[FAIL] {path.name} could not be parsed as FASTA")
        return False
    except Exception as exc:  # pragma: no cover - reported inline
        print(f"[FAIL] {path.name} parse failed: {exc}")
        return False

    sequence = str(record.seq).upper().replace(" ", "")
    if not sequence:
        print(f"[FAIL] {path.name} has an empty sequence")
        return False

    invalid = {char for char in sequence if char not in AMINO_ACIDS}
    if invalid:
        invalid_text = "".join(sorted(invalid))
        print(f"[FAIL] {path.name} contains invalid amino acid characters: {invalid_text}")
        return False

    print(f"[PASS] {path.name} exists and is valid")
    return True


def main() -> int:
    print("ProtAgent Phase 0 — Smoke Test")
    print("================================")

    checks = [
        check_python_version(),
        check_import("biopython", "Bio"),
        check_import("requests"),
        check_import("httpx"),
        check_import("anthropic"),
        check_import("langchain"),
        check_import("transformers"),
        check_import("torch"),
        check_import("numpy"),
        check_import("fastapi"),
        check_import("python-dotenv", "dotenv"),
        check_api_key(),
        check_directories(),
    ]

    for filename in REQUIRED_FASTA:
        checks.append(validate_fasta(SEQUENCE_DIR / filename))

    passed = sum(1 for result in checks if result)
    failed = len(checks) - passed

    print("================================")
    print(f"{passed}/{len(checks)} checks passed. {'Phase 0 complete.' if failed == 0 else 'Phase 0 incomplete.'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())