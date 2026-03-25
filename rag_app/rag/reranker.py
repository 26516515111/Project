from typing import List, Optional
import logging
from .config import SETTINGS
from .schema import Passage, RetrievedContext

logger = logging.getLogger(__name__)

TRANSFORMERS_AVAILABLE = False
torch = None
AutoModelForSequenceClassification = None
AutoTokenizer = None

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.debug("Transformers library not available.")


class CrossEncoderReranker:
    def __init__(self, model_name: str = None, top_k: int = None):
        self.model_name = model_name or SETTINGS.reranker_model
        self.top_k = top_k or SETTINGS.reranker_top_k

        self.tokenizer = None
        self.model = None

        if not TRANSFORMERS_AVAILABLE:
            logger.info("Transformers library not available. Reranker disabled.")
            return

        # Load model and tokenizer
        try:
            logger.info("Loading model %s...", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to("cuda")
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e)
            self.model = None
            self.tokenizer = None

    def rerank(self, query: str, passages: List[Passage]) -> RetrievedContext:
        """Rerank passages using cross-encoder model."""
        if self.model is None or self.tokenizer is None:
            # Fallback to original order if model loading failed
            logger.warning("Model not loaded, returning original passages.")
            return RetrievedContext(passages=passages[: self.top_k])

        if len(passages) == 0:
            return RetrievedContext(passages=[])

        # Create query-passage pairs
        pairs = [(query, p.text) for p in passages]

        # Tokenize
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Predict scores
        with torch.no_grad():
            scores = self.model(**inputs).logits
            scores = scores[:, 1] if scores.shape[1] > 1 else scores.squeeze(-1)
            scores = scores.cpu().numpy()

        # Sort by score
        scored_passages = [(score, passage) for score, passage in zip(scores, passages)]
        scored_passages.sort(key=lambda x: x[0], reverse=True)

        # Get top_k
        top_passages = []
        for score, passage in scored_passages[: self.top_k]:
            # Update passage score with reranker score (normalized to 0-1 for consistency)
            # The BGE reranker outputs logits, which can be negative; we normalize sigmoid-style.
            import math

            normalized_score = 1.0 / (1.0 + math.exp(-float(score)))
            passage.score = normalized_score
            top_passages.append(passage)

        return RetrievedContext(passages=top_passages)


def get_reranker() -> Optional[CrossEncoderReranker]:
    """Get reranker instance if enabled in config."""
    if not SETTINGS.use_reranker:
        return None
    if not TRANSFORMERS_AVAILABLE:
        logger.info("Transformers not installed, skipping reranker.")
        return None
    try:
        return CrossEncoderReranker()
    except Exception as e:
        logger.error("Reranker initialization failed: %s", e)
        return None
