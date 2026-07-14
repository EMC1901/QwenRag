"""Deterministic, conservative token estimates used before an LLM request."""

from __future__ import annotations

from math import ceil
from typing import Protocol


TRUNCATION_MARKER = "……（该资料片段因上下文长度限制已截断）"
_NATURAL_BOUNDARIES = ("\n\n", "\n", "。", "；", "！", "？", ".", ";", "!", "?")


class TokenCounter(Protocol):
    """Count and safely shorten text under one deterministic token budget."""

    def count(self, text: str) -> int:
        """Return a non-negative conservative token estimate."""

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return the longest safe, optionally marked prefix within the budget."""


class ConservativeTokenEstimator:
    """Estimate tokens from UTF-8 bytes without model files or network access.

    The estimate intentionally over-allocates for ordinary English and Chinese
    text.  It is a request-budget guard rather than a replacement for an exact
    model tokenizer, so callers must keep an additional context safety margin.
    """

    def count(self, text: str) -> int:
        """Return zero for empty text and a deterministic estimate otherwise."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return 0
        return ceil(len(text.encode("utf-8")) / 2)

    def truncate(self, text: str, max_tokens: int) -> str:
        """Shorten text on a natural boundary without exceeding ``max_tokens``."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")
        if not text or max_tokens == 0:
            return ""
        if self.count(text) <= max_tokens:
            return text

        marker_tokens = self.count(TRUNCATION_MARKER)
        if marker_tokens > max_tokens:
            return self._longest_prefix(TRUNCATION_MARKER, max_tokens)

        prefix = self._longest_prefix(text, max_tokens - marker_tokens)
        prefix = self._prefer_natural_boundary(prefix)
        result = f"{prefix.rstrip()}{TRUNCATION_MARKER}"
        while result and self.count(result) > max_tokens:
            prefix = self._longest_prefix(prefix, max(0, self.count(prefix) - 1))
            result = f"{prefix.rstrip()}{TRUNCATION_MARKER}"
        return result

    def _longest_prefix(self, text: str, max_tokens: int) -> str:
        """Find the longest code-point prefix within a token estimate budget."""
        if max_tokens <= 0:
            return ""
        low, high = 0, len(text)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle]
            if self.count(candidate) <= max_tokens:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _prefer_natural_boundary(prefix: str) -> str:
        """Avoid cutting a useful long text in the middle when space permits."""
        if len(prefix) < 10:
            return prefix
        search_start = int(len(prefix) * 0.8)
        boundary_end = -1
        for boundary in _NATURAL_BOUNDARIES:
            position = prefix.rfind(boundary, search_start)
            if position >= 0:
                boundary_end = max(boundary_end, position + len(boundary))
        return prefix[:boundary_end] if boundary_end > 0 else prefix
