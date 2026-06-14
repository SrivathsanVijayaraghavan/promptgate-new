"""
detector/semantic.py
--------------------
Semantic similarity detector for paraphrased and rephrased prompt injection attacks.

Catches attacks that bypass keyword matching by comparing the meaning of the
input against a library of known attack sentences using sentence embeddings.

Phase 2 addition: sliding window chunking prevents attack phrases embedded
mid-sentence from being diluted by surrounding conversational context.

Requires the optional [semantic] extras:
    pip install promptgate[semantic]

If sentence-transformers is not installed, the detector degrades gracefully:
is_available() returns False and detect() returns an empty list, so the
pipeline continues with rule-based detection only.
"""

import json
import warnings
from pathlib import Path
from typing import Any

_IMPORT_ERROR: Exception | None = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError as exc:
    _IMPORT_ERROR = exc
    SentenceTransformer = None  # type: ignore[assignment, misc]
    cosine_similarity = None    # type: ignore[assignment]
    np = None                   # type: ignore[assignment]

_ATTACKS_PATH = Path(__file__).resolve().parents[1] / "data" / "embeddings" / "known_attacks.json"
_MODEL_NAME   = "all-MiniLM-L6-v2"

_CHUNK_SIZE    = 12   # words per chunk
_CHUNK_OVERLAP = 4    # words of overlap between chunks


class SemanticDetector:
    """Detect semantically similar prompt injection attacks using sentence embeddings.

    Loads the ``all-MiniLM-L6-v2`` model and pre-computes embeddings for all
    known attack sentences at instantiation. Each call to ``detect()`` encodes
    the input (and overlapping chunks of it) and computes cosine similarity
    against the cached embeddings — no model reload per call.

    Chunking ensures that attack phrases embedded mid-sentence are not diluted
    by surrounding conversational context. Each chunk is scored independently;
    the highest-scoring chunk determines whether a signal is emitted.
    """

    def __init__(self, threshold: float = 0.65) -> None:
        """Load model and pre-compute known attack embeddings.

        Args:
            threshold: Cosine similarity score [0.0, 1.0] above which a
                       match is considered a signal. Default 0.65.
                       0.75 proved too strict for catching paraphrases
                       with all-MiniLM-L6-v2 in practice.
        """
        self.threshold = threshold
        self._available = False
        self._model = None
        self._attacks: list[dict[str, str]] = []
        self._embeddings = None

        if _IMPORT_ERROR is not None:
            return

        try:
            self._model = SentenceTransformer(_MODEL_NAME)
        except Exception as exc:
            warnings.warn(
                f"SemanticDetector: failed to load model '{_MODEL_NAME}': {exc}. "
                "Semantic detection disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        try:
            self._attacks = self._load_attacks()
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"SemanticDetector: failed to read known_attacks.json: {exc}. "
                "Semantic detection disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        if self._attacks:
            texts = [a["text"] for a in self._attacks]
            self._embeddings = self._model.encode(texts, convert_to_numpy=True)
        self._available = True

    def _load_attacks(self) -> list[dict[str, str]]:
        """Load known attack entries from known_attacks.json.

        Returns:
            List of attack dicts with keys: text, category, source.
            Returns empty list if the file cannot be read.
        """
        if not _ATTACKS_PATH.is_file():
            return []
        with _ATTACKS_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping word-window chunks.

        Splits the input into overlapping windows of _CHUNK_SIZE words
        with _CHUNK_OVERLAP words of overlap. This ensures attack phrases
        embedded mid-sentence are not diluted by surrounding context.

        Always includes the full text as the first chunk so short inputs
        are still handled correctly without chunking. Window chunks that
        would be identical to the full text (possible when the input is
        only one word longer than _CHUNK_SIZE) are skipped to avoid
        encoding the same text twice.

        Args:
            text: Cleaned lowercase input text.

        Returns:
            List of text chunks. Always contains at least one entry.
        """
        words = text.split()
        if len(words) <= _CHUNK_SIZE:
            return [text]

        chunks = [text]  # always include full text
        step = _CHUNK_SIZE - _CHUNK_OVERLAP
        for i in range(0, len(words) - _CHUNK_SIZE + 1, step):
            chunk = " ".join(words[i: i + _CHUNK_SIZE])
            if chunk != text:          # skip if identical to full-text chunk
                chunks.append(chunk)

        return chunks

    def detect(self, cleaned_text: str) -> list[dict[str, Any]]:
        """Compare cleaned_text against known attack embeddings.

        Splits input into overlapping chunks before encoding. This prevents
        attack phrases embedded mid-sentence from being diluted by surrounding
        context. Each chunk is compared independently — if any chunk exceeds
        the threshold for a given attack category, a signal is emitted.

        Groups known attacks by category. For each category, finds the
        highest-similarity match across ALL chunks. If that similarity exceeds
        threshold, emits one signal for that category.

        Args:
            cleaned_text: Normalised lowercase text from the parser.

        Returns:
            List of signal dicts. Empty list if unavailable or no match found.
            Each signal has: signal, severity, matched, category.
        """
        if not self._available or self._embeddings is None or not cleaned_text.strip():
            return []

        chunks = self._chunk_text(cleaned_text)

        # Encode all chunks in one batch call — efficient
        chunk_embeddings = self._model.encode(chunks, convert_to_numpy=True)

        # For each category, find best similarity across ALL chunks
        best_per_category: dict[str, tuple[float, str]] = {}

        for chunk_emb in chunk_embeddings:
            sims = cosine_similarity(chunk_emb.reshape(1, -1), self._embeddings)[0]
            for idx, attack in enumerate(self._attacks):
                category = attack["category"]
                sim = float(sims[idx])
                if category not in best_per_category or sim > best_per_category[category][0]:
                    best_per_category[category] = (sim, attack["text"])

        # Emit one signal per category that exceeds threshold
        signals: list[dict[str, Any]] = []
        for category, (sim, matched_text) in sorted(best_per_category.items()):
            if sim >= self.threshold:
                signals.append({
                    "signal":   "semantic_similarity",
                    "severity": 0.60,
                    "matched":  f"similar to: {matched_text} ({sim:.2f}) [{category}]",
                    "category": "semantic",
                })

        return signals

    def detect_batch(self, cleaned_texts: list[str]) -> list[list[dict[str, Any]]]:
        """Encode all texts in one batch and return signals per text.

        More efficient than calling detect() in a loop because all input
        chunks across all texts are encoded in a single model.encode() call,
        avoiding repeated GPU/CPU round-trips.

        Args:
            cleaned_texts: List of cleaned text strings from the parser.

        Returns:
            List of signal lists, one per input text, in the same order
            as cleaned_texts. Each inner list has the same structure as
            detect(). Returns [] (empty list, not list of empty lists)
            if cleaned_texts is empty.
        """
        if not self._available or self._embeddings is None or not cleaned_texts:
            return []

        # Build flat list of (text_index, chunk) pairs so we can encode all
        # chunks from all inputs in one batch call.
        index_chunk_pairs: list[tuple[int, str]] = []
        for text_idx, text in enumerate(cleaned_texts):
            if text.strip():
                for chunk in self._chunk_text(text):
                    index_chunk_pairs.append((text_idx, chunk))

        if not index_chunk_pairs:
            return [[] for _ in cleaned_texts]

        all_chunks = [chunk for _, chunk in index_chunk_pairs]
        all_embeddings = self._model.encode(all_chunks, convert_to_numpy=True)

        # Per-text, per-category best similarity accumulator
        # best[text_idx][category] = (best_sim, matched_text)
        best_per_text: list[dict[str, tuple[float, str]]] = [
            {} for _ in cleaned_texts
        ]

        for (text_idx, _), chunk_emb in zip(index_chunk_pairs, all_embeddings):
            sims = cosine_similarity(chunk_emb.reshape(1, -1), self._embeddings)[0]
            for attack_idx, attack in enumerate(self._attacks):
                category = attack["category"]
                sim = float(sims[attack_idx])
                current_best = best_per_text[text_idx].get(category)
                if current_best is None or sim > current_best[0]:
                    best_per_text[text_idx][category] = (sim, attack["text"])

        # Build signal lists — one per input text
        results: list[list[dict[str, Any]]] = []
        for best_per_category in best_per_text:
            signals: list[dict[str, Any]] = []
            for category, (sim, matched_text) in sorted(best_per_category.items()):
                if sim >= self.threshold:
                    signals.append({
                        "signal":   "semantic_similarity",
                        "severity": 0.60,
                        "matched":  f"similar to: {matched_text} ({sim:.2f}) [{category}]",
                        "category": "semantic",
                    })
            results.append(signals)

        return results

    def is_available(self) -> bool:
        """Return True if the model loaded successfully and detection is active.

        Returns False when sentence-transformers is not installed or the model
        failed to load. In both cases detect() safely returns an empty list.

        Returns:
            bool: True if semantic detection is operational.
        """
        return self._available