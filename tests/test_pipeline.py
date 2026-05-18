"""
tests/test_pipeline.py
Integration and unit tests for ProtAgent Phase 1.
Mocks external API calls so tests run fast and offline.
Real API calls are tested in separate live_test_*.py files (not run in CI).
"""

import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from models.schemas import EvidenceItem, EvidenceBundle, AnnotationDraft


# ─── Fixtures ────────────────────────────────────────────────────────────────

HEMOGLOBIN_FASTA = """>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG
KKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTP
AVHASLDKFLASVSTVLTSKYR"""

MOCK_UNIPROT_HITS = [
    EvidenceItem(
        source="uniprot",
        accession="P69905",
        description="Hemoglobin subunit alpha. Involved in oxygen transport from the lung to the various peripheral tissues.",
        go_terms=["GO:0005344 oxygen carrier activity", "GO:0020037 heme binding"],
        identity_pct=0.0,
        e_value=0.0,
    )
]

MOCK_BLAST_HITS = [
    EvidenceItem(
        source="blast",
        accession="P69905",
        description="Hemoglobin subunit alpha [Homo sapiens]",
        go_terms=[],
        identity_pct=99.3,
        e_value=1e-120,
    ),
    EvidenceItem(
        source="blast",
        accession="P01942",
        description="Hemoglobin subunit alpha [Mus musculus]",
        go_terms=[],
        identity_pct=87.2,
        e_value=2e-98,
    ),
]

MOCK_LLM_RESPONSE = json.dumps({
    "proposed_function": "This protein functions as an oxygen carrier in the blood, responsible for transporting oxygen from the lungs to peripheral tissues as part of the hemoglobin tetramer.",
    "go_term_candidates": ["GO:0005344 oxygen carrier activity", "GO:0020037 heme binding"],
    "reasoning": "UniProt match P69905 (identity 99.3%) describes this protein as hemoglobin subunit alpha with explicit oxygen transport function. BLAST hit confirms near-identical match to human hemoglobin alpha chain.",
    "evidence_used": ["P69905 - 99.3% identity BLAST hit, UniProt curated oxygen transport function", "P01942 - 87.2% identity to mouse hemoglobin alpha, confirming conserved function"],
})


# ─── Schema Tests ─────────────────────────────────────────────────────────────

class TestSchemas:
    def test_evidence_item_defaults(self):
        item = EvidenceItem(source="uniprot", accession="P12345", description="Test protein")
        assert item.go_terms == []
        assert item.identity_pct == 0.0
        assert item.e_value == 0.0

    def test_evidence_bundle_quality_default(self):
        bundle = EvidenceBundle(sequence_id="test", raw_sequence="MVLS")
        assert bundle.evidence_quality == "low"
        assert bundle.uniprot_hits == []
        assert bundle.blast_hits == []

    def test_annotation_draft_serializable(self):
        draft = AnnotationDraft(
            sequence_id="test",
            proposed_function="oxygen transport",
            go_term_candidates=["GO:0005344"],
            reasoning="test reasoning",
            evidence_used=["P69905"],
            embedding_shape=(320,),
        )
        d = asdict(draft)
        assert d["proposed_function"] == "oxygen transport"
        assert d["embedding_shape"] == (320,)


# ─── UniProt Tool Tests ────────────────────────────────────────────────────────

class TestUniprotTool:
    @patch("tools.uniprot.requests.get")
    def test_fetch_by_accession_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "primaryAccession": "P69905",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Hemoglobin subunit alpha"}}
            },
            "comments": [
                {
                    "commentType": "FUNCTION",
                    "texts": [{"value": "Involved in oxygen transport."}],
                }
            ],
            "uniProtKBCrossReferences": [
                {
                    "database": "GO",
                    "id": "GO:0005344",
                    "properties": [{"key": "GoTerm", "value": "oxygen carrier activity"}],
                }
            ],
        }
        mock_get.return_value = mock_resp

        from tools.uniprot import fetch_by_accession
        results = fetch_by_accession("P69905")
        assert len(results) == 1
        assert results[0].accession == "P69905"
        assert "GO:0005344" in results[0].go_terms[0]
        assert "oxygen transport" in results[0].description.lower()

    @patch("tools.uniprot.requests.get")
    def test_fetch_handles_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        from tools.uniprot import fetch_by_accession
        results = fetch_by_accession("NOTREAL")
        assert results == []

    @patch("tools.uniprot.requests.get")
    def test_retry_on_429(self, mock_get):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {
            "primaryAccession": "P00001",
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Test"}}},
            "comments": [],
            "uniProtKBCrossReferences": [],
        }
        mock_get.side_effect = [rate_limited, rate_limited, success]

        from tools.uniprot import fetch_by_accession
        with patch("tools.uniprot.time.sleep"):  # skip actual sleep in tests
            results = fetch_by_accession("P00001")
        assert len(results) == 1


# ─── BLAST Tool Tests ──────────────────────────────────────────────────────────

class TestBlastTool:
    def test_short_sequence_skipped(self):
        from tools.blast import run_blast
        results = run_blast("MVLS")   # < 10 AA
        assert results == []

    def test_empty_sequence_skipped(self):
        from tools.blast import run_blast
        results = run_blast("")
        assert results == []

    @patch("tools.blast.requests.post")
    @patch("tools.blast.requests.get")
    def test_blast_submit_and_parse(self, mock_get, mock_post):
        # Mock submit
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.text = "    RID = ABC12345XYZ\n    RTOE = 20\n"
        mock_post.return_value = submit_resp

        # Mock poll — return completed XML
        xml_response = """<?xml version="1.0"?>
<BlastOutput>
  <BlastOutput_iterations>
    <Iteration>
      <Iteration_hits>
        <Hit>
          <Hit_accession>P69905</Hit_accession>
          <Hit_def>Hemoglobin subunit alpha [Homo sapiens]</Hit_def>
          <Hit_hsps>
            <Hsp>
              <Hsp_identity>141</Hsp_identity>
              <Hsp_align-len>142</Hsp_align-len>
              <Hsp_evalue>1e-120</Hsp_evalue>
            </Hsp>
          </Hit_hsps>
        </Hit>
      </Iteration_hits>
    </Iteration>
  </BlastOutput_iterations>
</BlastOutput>"""
        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.text = xml_response
        mock_get.return_value = poll_resp

        with patch("tools.blast.time.sleep"):
            from tools.blast import run_blast
            results = run_blast("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFP")

        assert len(results) == 1
        assert results[0].accession == "P69905"
        assert results[0].identity_pct > 90.0


# ─── ESM Tool Tests ────────────────────────────────────────────────────────────

class TestEsmTool:
    @patch("tools.esm._load_model")
    def test_embedding_shape(self, mock_load):
        """Mock the model load and verify output shape contract."""
        import torch
        from unittest.mock import patch as inner_patch

        # Fake model that returns a dummy hidden state
        fake_output = MagicMock()
        fake_output.last_hidden_state = torch.zeros(1, 12, 320)  # (batch, seq, hidden)

        fake_model = MagicMock()
        fake_model.return_value = fake_output

        fake_tokenizer = MagicMock()
        fake_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 12, dtype=torch.long),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
        }

        import tools.esm as esm_module
        esm_module._tokenizer = fake_tokenizer
        esm_module._model = fake_model

        embedding = esm_module.get_embedding("MVLSPADKTNVK")
        assert embedding.shape == (320,)

    def test_empty_sequence_raises(self):
        from tools.esm import get_embedding
        with pytest.raises(ValueError):
            get_embedding("")


# ─── Gatherer Agent Tests ──────────────────────────────────────────────────────

class TestGathererAgent:
    def test_invalid_fasta_raises(self):
        from agents.gatherer import gather
        with pytest.raises(ValueError):
            gather("this is not a fasta string")

    @patch("agents.gatherer.uniprot.fetch_evidence", return_value=MOCK_UNIPROT_HITS)
    @patch("agents.gatherer.blast.run_blast", return_value=MOCK_BLAST_HITS)
    def test_gather_returns_bundle(self, mock_blast, mock_uniprot):
        from agents.gatherer import gather
        bundle = gather(HEMOGLOBIN_FASTA)
        assert bundle.sequence_id is not None
        assert len(bundle.raw_sequence) > 0
        assert len(bundle.uniprot_hits) == 1
        assert len(bundle.blast_hits) == 2
        assert bundle.evidence_quality in ("high", "medium", "low")

    @patch("agents.gatherer.uniprot.fetch_evidence", return_value=MOCK_UNIPROT_HITS)
    @patch("agents.gatherer.blast.run_blast", return_value=MOCK_BLAST_HITS)
    def test_quality_high_with_go_and_blast(self, mock_blast, mock_uniprot):
        from agents.gatherer import gather
        bundle = gather(HEMOGLOBIN_FASTA)
        # MOCK_UNIPROT_HITS has GO terms + MOCK_BLAST_HITS has >50% identity
        assert bundle.evidence_quality == "high"

    @patch("agents.gatherer.uniprot.fetch_evidence", return_value=[])
    @patch("agents.gatherer.blast.run_blast", return_value=[])
    def test_quality_low_when_no_evidence(self, mock_blast, mock_uniprot):
        from agents.gatherer import gather
        bundle = gather(HEMOGLOBIN_FASTA)
        assert bundle.evidence_quality == "low"


# ─── Annotator Agent Tests ─────────────────────────────────────────────────────

class TestAnnotatorAgent:
    def _make_bundle(self, quality="high"):
        return EvidenceBundle(
            sequence_id="sp|P69905|HBA_HUMAN",
            raw_sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            uniprot_hits=MOCK_UNIPROT_HITS,
            blast_hits=MOCK_BLAST_HITS,
            evidence_quality=quality,
        )

    @patch("agents.annotator.get_embedding", return_value=np.zeros(320))
    @patch("agents.annotator.get_embedding_summary", return_value={"norm": 1.0, "mean": 0.0, "std": 0.1})
    @patch("agents.annotator._call_openrouter")
    def test_annotate_returns_draft(self, mock_openrouter, mock_summary, mock_embed):
        mock_openrouter.return_value = MOCK_LLM_RESPONSE

        import os
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from agents.annotator import annotate
            draft = annotate(self._make_bundle())

        assert draft.proposed_function != ""
        assert "ANNOTATION_FAILED" not in draft.proposed_function
        assert len(draft.go_term_candidates) > 0
        assert draft.reasoning != ""

    @patch("agents.annotator.get_embedding", return_value=np.zeros(320))
    @patch("agents.annotator.get_embedding_summary", return_value={"norm": 1.0, "mean": 0.0, "std": 0.1})
    @patch("agents.annotator._call_openrouter")
    def test_malformed_json_returns_error_draft(self, mock_openrouter, mock_summary, mock_embed):
        mock_openrouter.return_value = "this is not json at all"

        import os
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from agents.annotator import annotate
            draft = annotate(self._make_bundle())

        assert "ANNOTATION_FAILED" not in draft.proposed_function
        assert "fallback" in draft.reasoning.lower()
        assert draft.evidence_used


# ─── Integration Test ──────────────────────────────────────────────────────────

class TestPipelineIntegration:
    @patch("agents.gatherer.uniprot.fetch_evidence", return_value=MOCK_UNIPROT_HITS)
    @patch("agents.gatherer.blast.run_blast", return_value=MOCK_BLAST_HITS)
    @patch("agents.annotator.get_embedding", return_value=np.zeros(320))
    @patch("agents.annotator.get_embedding_summary", return_value={"norm": 1.0, "mean": 0.0, "std": 0.1})
    @patch("agents.annotator._call_openrouter", return_value=MOCK_LLM_RESPONSE)
    @patch("agents.critic._call_openrouter", return_value=json.dumps({
        "passed": True,
        "challenges": [],
        "revision_required": False,
        "critique_reasoning": "Annotation is supported by evidence."
    }))
    @patch("agents.explainer._call_openrouter", return_value="High confidence based on strong evidence.")
    def test_full_pipeline_end_to_end(
        self,
        mock_explainer,
        mock_critic,
        mock_openrouter,
        mock_summary,
        mock_embed,
        mock_blast,
        mock_uniprot,
    ):
        import os
        import re

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from pipeline import run
            result = run(HEMOGLOBIN_FASTA)

        assert result["annotation_draft"]["proposed_function"] != ""
        assert result["evidence_bundle"]["evidence_quality"] == "high"
        assert result["elapsed_seconds"] >= 0
        assert result["critic_verdict"]["passed"] is True
        assert 0.0 <= result["confidence_report"]["confidence_score"] <= 1.0
        assert result["confidence_report"]["confidence_label"] in ("high", "medium", "low", "uncertain")

        # Verify output JSON was saved — filename uses sanitized sequence_id
        raw_id = result["evidence_bundle"]["sequence_id"]
        safe_id = re.sub(r'[|:\/\\*?"<>]', '_', raw_id)
        output_path = f"outputs/{safe_id}_phase2.json"
        assert os.path.exists(output_path)
