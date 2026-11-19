"""
Sludge Filter for AI Research Assistant.

Filters out low-quality chunks (table dumps, OCR noise, mostly-numeric blocks)
before they reach the answer generator. Uses cheap heuristics only.
"""
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default thresholds (can be overridden via config or constants)
MIN_LETTER_RATIO = 0.30  # letters / max(len(text), 1)
MAX_DIGIT_RATIO = 0.50   # digits / max(len(text), 1)
MAX_PUNCT_RATIO = 0.40   # punctuation / max(len(text), 1)
MIN_WORDS = 5            # minimum words to be considered valid prose
MIN_ALPHA_WORDS = 2      # minimum alphabetic words

# Pattern for glued tokens (e.g., "ThisIsAGluedToken" or "word1word2word3")
GLUED_TOKEN_PATTERN = re.compile(r'[a-z][A-Z]|[a-zA-Z]\d|\d[a-zA-Z]')


def _count_letters(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def _count_digits(text: str) -> int:
    return sum(1 for c in text if c.isdigit())


def _count_punct(text: str) -> int:
    return sum(1 for c in text if not c.isalnum() and not c.isspace())


def _count_words(text: str) -> int:
    return len(text.split())


def _count_alpha_words(text: str) -> int:
    return sum(1 for w in text.split() if any(c.isalpha() for c in w))


def _glued_token_ratio(text: str) -> float:
    """Ratio of glued-token-like transitions to total chars."""
    if not text:
        return 0.0
    matches = len(GLUED_TOKEN_PATTERN.findall(text))
    return matches / max(len(text), 1)


def is_sludge(text: str,
              min_letter_ratio: float = MIN_LETTER_RATIO,
              max_digit_ratio: float = MAX_DIGIT_RATIO,
              max_punct_ratio: float = MAX_PUNCT_RATIO,
              min_words: int = MIN_WORDS,
              min_alpha_words: int = MIN_ALPHA_WORDS,
              max_glued_ratio: float = 0.05) -> bool:
    """
    Determine if a text chunk is 'sludge' (low quality, table dump, OCR noise).

    Args:
        text: Chunk text to evaluate
        min_letter_ratio: Minimum ratio of alphabetic characters
        max_digit_ratio: Maximum ratio of digits
        max_punct_ratio: Maximum ratio of punctuation
        min_words: Minimum word count
        min_alpha_words: Minimum words containing letters
        max_glued_ratio: Maximum ratio of glued-token transitions

    Returns:
        True if chunk should be filtered out
    """
    if not text or not text.strip():
        return True

    text_len = max(len(text), 1)
    letter_ratio = _count_letters(text) / text_len
    digit_ratio = _count_digits(text) / text_len
    punct_ratio = _count_punct(text) / text_len
    word_count = _count_words(text)
    alpha_word_count = _count_alpha_words(text)
    glued_ratio = _glued_token_ratio(text)

    # Check all sludge indicators
    if letter_ratio < min_letter_ratio:
        logger.debug(f"Sludge: low letter ratio {letter_ratio:.2f} < {min_letter_ratio}")
        return True

    if digit_ratio > max_digit_ratio:
        logger.debug(f"Sludge: high digit ratio {digit_ratio:.2f} > {max_digit_ratio}")
        return True

    if punct_ratio > max_punct_ratio:
        logger.debug(f"Sludge: high punct ratio {punct_ratio:.2f} > {max_punct_ratio}")
        return True

    if word_count < min_words:
        logger.debug(f"Sludge: too few words {word_count} < {min_words}")
        return True

    if alpha_word_count < min_alpha_words:
        logger.debug(f"Sludge: too few alpha words {alpha_word_count} < {min_alpha_words}")
        return True

    if glued_ratio > max_glued_ratio:
        logger.debug(f"Sludge: high glued ratio {glued_ratio:.2f} > {max_glued_ratio}")
        return True

    return False


def filter_sludge(chunks: List[Dict[str, Any]],
                  text_key: str = "text",
                  **kwargs) -> List[Dict[str, Any]]:
    """
    Filter out sludge chunks from a list of chunk dicts.

    Args:
        chunks: List of chunk dictionaries
        text_key: Key in chunk dict containing the text content
        **kwargs: Additional arguments passed to is_sludge()

    Returns:
        Filtered list of chunks (sludge removed)
    """
    if not chunks:
        return []

    filtered = []
    dropped = 0

    for chunk in chunks:
        text = chunk.get(text_key, "")
        if is_sludge(text, **kwargs):
            dropped += 1
            logger.debug(f"Dropped sludge chunk: {chunk.get('chunk_id', 'unknown')} "
                         f"({len(text)} chars, preview: {text[:50]}...)")
        else:
            filtered.append(chunk)

    if dropped:
        logger.info(f"Sludge filter: dropped {dropped}/{len(chunks)} chunks, "
                    f"{len(filtered)} remaining")

    return filtered