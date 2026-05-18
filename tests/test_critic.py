import json
from unittest.mock import patch

from models.schemas import AnnotationDraft, EvidenceBundle, EvidenceItem
from agents.critic import critique


def _bundle(evidence_quality="high", blast_identities=None, go_terms=None):
    blast_identities = blast_identities or [95.0]
    go_terms = go_terms or ["GO:0005344 oxygen carrier activity"]

    uniprot_hits = [
        EvidenceItem(
            source="uniprot",
            accession="P69905",
            description="Hemoglobin subunit alpha involved in oxygen transport.",
            go_terms=go_terms,
        )
    ] if evidence_quality != "low" else []

    blast_hits = [
        EvidenceItem(
            source="blast",
            accession="P69905",
            description="Hemoglobin subunit alpha [Homo sapiens]",
            identity_pct=blast_identities[0],
            e_value=1e-120,
        )
    ] if evidence_quality != "low" else []

    return EvidenceBundle(
        sequence_id="sp|P69905|HBA_HUMAN",
        raw_sequence="MVLSPADKTNVKAAWGKVG",
        uniprot_hits=uniprot_hits,
        blast_hits=blast_hits,
        evidence_quality=evidence_quality,
    )


def _draft(function="oxygen transport", go_terms=None):
    return AnnotationDraft(
        sequence_id="sp|P69905|HBA_HUMAN",
        proposed_function=function,
        go_term_candidates=go_terms or ["GO:0005344 oxygen carrier activity"],
        reasoning="Test reasoning",
        evidence_used=["P69905"],
        embedding_shape=(320,),
    )


def _critic_json(passed=True, challenges=None, revision_required=False):
    return json.dumps({
        "passed": passed,
        "challenges": challenges or [],
        "revision_required": revision_required,
        "critique_reasoning": "Test critique",
    })


@patch("agents.critic._call_openrouter", return_value=_critic_json())
def test_critic_passes_when_supported(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        verdict = critique(_draft(), _bundle(), revision_round=1)
    assert verdict.passed is True
    assert verdict.revision_required is False


@patch("agents.critic._call_openrouter", return_value=_critic_json())
def test_unsupported_function_flags(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        verdict = critique(_draft(function="kinase activity"), _bundle(), revision_round=1)
    assert verdict.passed is False
    assert verdict.revision_required is True
    assert any("UNSUPPORTED FUNCTION" in c for c in verdict.challenges)


@patch("agents.critic._call_openrouter", return_value=_critic_json())
def test_go_term_mismatch_flags(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        verdict = critique(
            _draft(go_terms=["GO:999999 fake term"]),
            _bundle(go_terms=["GO:0005344 oxygen carrier activity"]),
            revision_round=1,
        )
    assert verdict.passed is False
    assert verdict.revision_required is True
    assert any("GO TERM MISMATCH" in c for c in verdict.challenges)


@patch("agents.critic._call_openrouter", return_value=_critic_json())
def test_weak_homology_flags(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        verdict = critique(_draft(), _bundle(blast_identities=[12.0]), revision_round=1)
    assert verdict.passed is False
    assert verdict.revision_required is True
    assert any("WEAK HOMOLOGY" in c for c in verdict.challenges)


@patch("agents.critic._call_openrouter", return_value=_critic_json())
def test_low_evidence_flags(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        verdict = critique(_draft(), _bundle(evidence_quality="low"), revision_round=1)
    assert verdict.passed is False
    assert verdict.revision_required is True
    assert any("LOW EVIDENCE" in c for c in verdict.challenges)


@patch("agents.critic._call_openrouter", side_effect=["not json", "still not json"])
def test_parse_failure_returns_conservative_verdict(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        verdict = critique(_draft(), _bundle(), revision_round=1)
    assert verdict.passed is False
    assert verdict.revision_required is False
    assert "critic parse failed — treating as uncertain" in verdict.challenges
