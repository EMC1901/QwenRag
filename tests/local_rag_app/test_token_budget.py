"""Tests for deterministic conservative token estimates and truncation."""

import pytest

from local_rag_app.token_budget import ConservativeTokenEstimator, TRUNCATION_MARKER


def test_count_is_deterministic_for_empty_ascii_and_chinese_text() -> None:
    """The estimator has no dependency on a model file or machine locale."""
    counter = ConservativeTokenEstimator()

    assert counter.count("") == 0
    assert counter.count("abcd") == 2
    assert counter.count("合同") == 3
    assert counter.count("合同") == counter.count("合同")


def test_truncate_preserves_short_text() -> None:
    """Texts already within budget must not be normalized or modified."""
    counter = ConservativeTokenEstimator()
    text = "第一段。\n\n第二段。"

    assert counter.truncate(text, counter.count(text)) == text


def test_truncate_adds_marker_and_never_exceeds_budget() -> None:
    """Long mixed-language input is shortened safely with an explicit marker."""
    counter = ConservativeTokenEstimator()
    text = "第一句内容。第二句内容。第三句内容。" * 30
    budget = 80

    result = counter.truncate(text, budget)

    assert result.endswith(TRUNCATION_MARKER)
    assert counter.count(result) <= budget


def test_truncate_uses_a_shorter_prefix_when_marker_cannot_fit() -> None:
    """Tiny budgets still terminate without an infinite loop or overflow."""
    counter = ConservativeTokenEstimator()

    result = counter.truncate("这是一个很长的资料文本", 1)

    assert counter.count(result) <= 1


@pytest.mark.parametrize("budget", [0, 1, 2, 10])
def test_truncate_respects_every_non_negative_budget(budget: int) -> None:
    """Every public budget boundary returns a valid bounded string."""
    counter = ConservativeTokenEstimator()
    result = counter.truncate("emoji 😀 and quotes \\\" with 换行\n文本", budget)

    assert counter.count(result) <= budget


@pytest.mark.parametrize(
    "text, budget, error",
    [
        (None, 1, TypeError),
        ("text", -1, ValueError),
        ("text", True, TypeError),
    ],
)
def test_truncate_rejects_invalid_arguments(
    text: object,
    budget: object,
    error: type[Exception],
) -> None:
    """Input-validation failures are explicit instead of silently changing text."""
    with pytest.raises(error):
        ConservativeTokenEstimator().truncate(text, budget)  # type: ignore[arg-type]
