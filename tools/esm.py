"""
tools/esm.py
Loads ESM-2 (facebook/esm2_t6_8M_UR50D) from HuggingFace and generates
fixed-size protein sequence embeddings. CPU-friendly (8M parameter model).
Model is cached in memory after first load.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTH = 500   # truncate longer sequences with a warning

# Module-level cache — model loads once per process
_tokenizer = None
_model = None


def _load_model():
    """Load ESM-2 tokenizer and model from HuggingFace (once)."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return  # already loaded

    try:
        from transformers import AutoTokenizer, EsmModel
        import torch
        logger.info("Loading ESM-2 model (facebook/esm2_t6_8M_UR50D)... this may take ~30s on first run.")
        _tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        _model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        _model.eval()  # inference mode
        logger.info("ESM-2 model loaded and cached.")
    except Exception as e:
        logger.error(f"Failed to load ESM-2 model: {e}")
        raise


def get_embedding(sequence: str) -> np.ndarray:
    """
    Generate a mean-pooled ESM-2 embedding for an amino acid sequence.

    Args:
        sequence: Raw amino acid string (single-letter codes, no FASTA header)

    Returns:
        numpy array of shape (320,) — the mean-pooled hidden states

    Raises:
        RuntimeError if the model cannot be loaded
    """
    import torch

    if not sequence:
        raise ValueError("Cannot embed an empty sequence.")

    # Truncate with warning if needed
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        logger.warning(
            f"Sequence length {len(sequence)} exceeds max {MAX_SEQUENCE_LENGTH}. "
            f"Truncating to first {MAX_SEQUENCE_LENGTH} amino acids."
        )
        sequence = sequence[:MAX_SEQUENCE_LENGTH]

    # Ensure model is loaded
    _load_model()

    try:
        inputs = _tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH + 2,  # +2 for [CLS] and [EOS]
        )

        with torch.no_grad():
            outputs = _model(**inputs)

        # Mean pool over sequence length dimension (exclude special tokens)
        # outputs.last_hidden_state shape: (1, seq_len, 320)
        hidden_states = outputs.last_hidden_state.squeeze(0)  # (seq_len, 320)
        # Exclude first ([CLS]) and last ([EOS]) tokens
        token_embeddings = hidden_states[1:-1]
        mean_embedding = token_embeddings.mean(dim=0).numpy()  # (320,)

        logger.info(f"ESM-2 embedding generated. Shape: {mean_embedding.shape}")
        return mean_embedding

    except Exception as e:
        logger.error(f"ESM-2 embedding failed: {e}")
        raise


def get_embedding_summary(sequence: str) -> dict:
    """
    Returns the embedding plus a summary dict for use in the annotator prompt.
    The LLM doesn't receive raw vectors — it receives statistical summaries.
    """
    embedding = get_embedding(sequence)
    return {
        "shape": embedding.shape,
        "mean": float(np.mean(embedding)),
        "std": float(np.std(embedding)),
        "min": float(np.min(embedding)),
        "max": float(np.max(embedding)),
        "norm": float(np.linalg.norm(embedding)),
        "top_dimensions": embedding.argsort()[-5:][::-1].tolist(),  # indices of 5 highest activations
    }