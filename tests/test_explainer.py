from unittest.mock import patch

from models.schemas import AnnotationDraft, CriticVerdict, EvidenceBundle, EvidenceItem
from agents.explainer import explain


def _bundle(evidence_quality="high", blast_identities=None, seq_len=100):
    blast_identities = blast_identities or [95.0]
    sequence = "A" * seq_len

    uniprot_hits = []
    blast_hits = []
    if evidence_quality != "low":
        uniprot_hits = [
            EvidenceItem(
                source="uniprot",
                accession="P69905",
                description="Hemoglobin subunit alpha involved in oxygen transport.",
                go_terms=["GO:0005344 oxygen carrier activity"],
            )
        ]
        blast_hits = [
            EvidenceItem(
                source="blast",
                accession="P69905",
                description="Hemoglobin subunit alpha [Homo sapiens]",
                identity_pct=blast_identities[0],
                e_value=1e-120,
            )
        ]

    return EvidenceBundle(
        sequence_id="sp|P69905|HBA_HUMAN",
        raw_sequence=sequence,
        uniprot_hits=uniprot_hits,
        blast_hits=blast_hits,
        evidence_quality=evidence_quality,
    )


def _draft():
    return AnnotationDraft(
        sequence_id="sp|P69905|HBA_HUMAN",
        proposed_function="oxygen transport",
        go_term_candidates=["GO:0005344 oxygen carrier activity"],
        reasoning="Test reasoning",
        evidence_used=["P69905"],
        embedding_shape=(320,),
    )


def _critic(passed=True, challenges=None):
    return CriticVerdict(
        sequence_id="sp|P69905|HBA_HUMAN",
        passed=passed,
        challenges=challenges or [],
        revision_required=not passed,
        revision_round=1,
        critique_reasoning="Test critique",
    )


@patch("agents.explainer._call_openrouter", return_value="Summary.")
def test_confidence_score_high_passes(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        report = explain(_draft(), _critic(True), _bundle("high", [95.0]), revision_rounds=0)
    assert report.confidence_score == 1.0
    assert report.confidence_label == "high"


@patch("agents.explainer._call_openrouter", return_value="Summary.")
def test_confidence_score_medium_high_boundary(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        report = explain(_draft(), _critic(True), _bundle("medium", [80.0]), revision_rounds=1)
    assert report.confidence_score == 0.75
    assert report.confidence_label == "high"


@patch("agents.explainer._call_openrouter", return_value="Summary.")
def test_confidence_score_low_uncertain(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        report = explain(_draft(), _critic(False), _bundle("low", [0.0]), revision_rounds=0)
    assert report.confidence_score == 0.2
    assert report.confidence_label == "uncertain"


@patch("agents.explainer._call_openrouter", return_value="Summary.")
def test_confidence_score_medium_with_revisions(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        report = explain(_draft(), _critic(False), _bundle("high", [50.0]), revision_rounds=2)
    assert report.confidence_score == 0.5
    assert report.confidence_label == "medium"


@patch("agents.explainer._call_openrouter", return_value="Summary.")
def test_confidence_score_medium_high_combo(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        report = explain(_draft(), _critic(True), _bundle("medium", [95.0]), revision_rounds=2)
    assert report.confidence_score == 0.7
    assert report.confidence_label == "medium"


@patch("agents.explainer._call_openrouter", return_value="Summary.")
def test_confidence_score_high_with_revision(mock_openrouter):
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        report = explain(_draft(), _critic(True), _bundle("high", [75.0]), revision_rounds=1)
    assert report.confidence_score == 0.85
    assert report.confidence_label == "high"


@patch("agents.explainer._call_openrouter", return_value="Summary.")
def test_warning_collection(mock_openrouter):
    critic = _critic(False, challenges=["critic parse failed — treating as uncertain"])
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        report = explain(_draft(), critic, _bundle("low", [0.0], seq_len=600), revision_rounds=2)
    assert "low evidence quality" in report.warnings
    assert "sequence truncated for embedding" in report.warnings
    assert "max revisions used" in report.warnings
    assert "critic parse failed — treating as uncertain" in report.warnings
