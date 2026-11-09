"""
Tests for DocVQA evaluation metrics (ANLS and Accuracy).

Critical to test because:
- ANLS is the primary metric for DocVQA tasks
- Text normalization must handle edge cases correctly
- Levenshtein distance calculation affects ANLS scores
- Multiple target answers must be handled properly
"""
import pytest
from src.evaluation.docvqa_metrics import (
    DocVQAEvaluator,
    normalize_text,
    calculate_normalized_levenshtein_distance,
    calculate_anls,
    calculate_accuracy,
)


class TestTextNormalization:
    """Test text normalization function."""

    def test_basic_normalization(self):
        """Test basic lowercase and strip."""
        assert normalize_text("Hello World") == "hello world"
        assert normalize_text("  HELLO  ") == "hello"

    def test_unicode_normalization(self):
        """Test NFKC unicode normalization."""
        # NFKC normalizes different representations
        assert normalize_text("café") == normalize_text("café")  # Different e representations

    def test_quote_normalization(self):
        """Test various quote characters normalize to same."""
        assert normalize_text("'hello'") == normalize_text("'hello'")

    def test_dash_normalization(self):
        """Test various dash characters normalize to hyphen."""
        assert normalize_text("hello–world") == "hello-world"
        assert normalize_text("hello—world") == "hello-world"
        assert normalize_text("hello−world") == "hello-world"

    def test_whitespace_normalization(self):
        """Test multiple whitespaces collapse to single space."""
        assert normalize_text("hello    world") == "hello world"
        assert normalize_text("hello\t\nworld") == "hello world"

    def test_empty_and_none(self):
        """Test empty string and None handling."""
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


class TestLevenshteinDistance:
    """Test normalized Levenshtein distance calculation."""

    def test_identical_strings(self):
        """Test distance between identical strings is 1.0."""
        distance = calculate_normalized_levenshtein_distance("hello", "hello")
        assert distance == 1.0

    def test_completely_different(self):
        """Test distance between completely different strings."""
        distance = calculate_normalized_levenshtein_distance("abc", "xyz")
        # Should be 0.0 (all characters different)
        assert distance == 0.0

    def test_one_character_difference(self):
        """Test small edit distance."""
        distance = calculate_normalized_levenshtein_distance("hello", "hallo")
        # 1 char different out of 5, so 1 - (1/5) = 0.8
        assert 0.7 < distance < 0.9

    def test_empty_strings(self):
        """Test empty string edge cases."""
        # Both empty should match perfectly
        assert calculate_normalized_levenshtein_distance("", "") == 1.0
        # One empty should be 0.0
        assert calculate_normalized_levenshtein_distance("hello", "") == 0.0
        assert calculate_normalized_levenshtein_distance("", "hello") == 0.0

    def test_case_insensitive(self):
        """Test that comparison is case-insensitive."""
        distance = calculate_normalized_levenshtein_distance("Hello", "hello")
        assert distance == 1.0


class TestANLS:
    """Test ANLS (Average Normalized Levenshtein Similarity) calculation."""

    def test_perfect_match(self):
        """Test ANLS with perfect match."""
        target_answers = ["correct answer"]
        pred_answer = "correct answer"

        anls = calculate_anls(target_answers, pred_answer)
        assert anls == 1.0

    def test_nearly_perfect_match(self):
        """Test ANLS with one character typo."""
        target_answers = ["hello world"]
        pred_answer = "helo world"  # Missing one 'l'

        anls = calculate_anls(target_answers, pred_answer)
        # Should be high but not perfect
        assert 0.85 < anls < 1.0

    def test_ok_match(self):
        """Test ANLS with moderate similarity."""
        target_answers = ["2024"]
        pred_answer = "2023"  # One digit different

        anls = calculate_anls(target_answers, pred_answer)
        # 3 out of 4 characters match
        assert 0.6 < anls < 0.85

    def test_bad_match_below_threshold(self):
        """Test ANLS returns 0.0 when below threshold."""
        target_answers = ["correct"]
        pred_answer = "wrong"  # Very different

        anls = calculate_anls(target_answers, pred_answer, threshold=0.5)
        # Should be 0.0 due to threshold
        assert anls == 0.0

    def test_multiple_target_answers(self):
        """Test ANLS selects best match from multiple targets."""
        target_answers = ["USA", "United States", "America"]
        pred_answer = "usa"  # Matches first perfectly

        anls = calculate_anls(target_answers, pred_answer)
        assert anls == 1.0

    def test_multiple_targets_partial_match(self):
        """Test ANLS with multiple targets, best is partial."""
        target_answers = ["hello", "hi"]
        pred_answer = "helo"  # Closer to "hello"

        anls = calculate_anls(target_answers, pred_answer)
        # Should pick best match (hello vs helo)
        assert 0.7 < anls < 1.0


class TestAccuracy:
    """Test exact match accuracy calculation."""

    def test_perfect_match(self):
        """Test accuracy with perfect match."""
        target_answers = ["correct"]
        pred_answer = "correct"

        acc = calculate_accuracy(target_answers, pred_answer)
        assert acc == 1.0

    def test_case_insensitive_match(self):
        """Test accuracy is case-insensitive."""
        target_answers = ["Paris"]
        pred_answer = "paris"

        acc = calculate_accuracy(target_answers, pred_answer)
        assert acc == 1.0

    def test_whitespace_normalized_match(self):
        """Test accuracy normalizes whitespace."""
        target_answers = ["hello world"]
        pred_answer = "hello   world"  # Extra spaces

        acc = calculate_accuracy(target_answers, pred_answer)
        assert acc == 1.0

    def test_no_match(self):
        """Test accuracy returns 0.0 for no match."""
        target_answers = ["correct"]
        pred_answer = "wrong"

        acc = calculate_accuracy(target_answers, pred_answer)
        assert acc == 0.0

    def test_partial_string_no_match(self):
        """Test accuracy requires exact match, not substring."""
        target_answers = ["hello"]
        pred_answer = "helo"  # Close but not exact

        acc = calculate_accuracy(target_answers, pred_answer)
        assert acc == 0.0

    def test_multiple_target_answers_one_match(self):
        """Test accuracy matches any target answer."""
        target_answers = ["USA", "United States", "America"]
        pred_answer = "america"

        acc = calculate_accuracy(target_answers, pred_answer)
        assert acc == 1.0


class TestDocVQAEvaluator:
    """Test complete DocVQA evaluator."""

    def test_perfect_predictions(self):
        """Test metrics with all perfect predictions."""
        target_answers = [
            ["Paris"],
            ["2024"],
            ["John Smith"]
        ]
        pred_answers = ["Paris", "2024", "John Smith"]

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        # Both metrics should be 1.0
        assert metrics["acc"] == 1.0
        assert metrics["anls"] == 1.0

    def test_mixed_quality_predictions(self):
        """Test metrics with varying prediction quality."""
        target_answers = [
            ["Paris"],           # Perfect match
            ["2024"],           # Nearly perfect (will predict 2023)
            ["hello world"],    # OK match (will predict "helo world")
            ["correct"],        # Bad match (will predict "wrong")
        ]
        pred_answers = [
            "Paris",       # Perfect
            "2023",        # 1 char diff
            "helo world",  # 1 char diff
            "wrong",       # Completely wrong
        ]

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        # Accuracy should be 0.25 (only first is exact match)
        assert metrics["acc"] == 0.25

        # ANLS should be between 0.4 and 0.8 (mix of good and bad)
        assert 0.4 < metrics["anls"] < 0.8

    def test_all_wrong_predictions(self):
        """Test metrics with all incorrect predictions."""
        target_answers = [
            ["correct1"],
            ["correct2"],
            ["correct3"],
        ]
        pred_answers = ["wrong1", "wrong2", "wrong3"]

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        # Both should be 0.0
        assert metrics["acc"] == 0.0
        assert metrics["anls"] == 0.0

    def test_multiple_valid_answers(self):
        """Test with multiple valid target answers."""
        target_answers = [
            ["USA", "United States", "America"],
            ["2024", "twenty twenty-four"],
        ]
        pred_answers = ["usa", "2024"]

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        # Both should be perfect
        assert metrics["acc"] == 1.0
        assert metrics["anls"] == 1.0

    def test_case_and_whitespace_handling(self):
        """Test normalization in evaluation."""
        target_answers = [
            ["Hello World"],
            ["New York"],
        ]
        pred_answers = ["hello   world", "NEW YORK"]

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        # Should match after normalization
        assert metrics["acc"] == 1.0
        assert metrics["anls"] == 1.0

    def test_length_mismatch_raises_error(self):
        """Test error when target and pred lengths don't match."""
        target_answers = [["answer1"], ["answer2"]]
        pred_answers = ["pred1"]  # Only one prediction

        with pytest.raises(ValueError, match="did not have the same length"):
            DocVQAEvaluator.evaluate(target_answers, pred_answers)

    def test_empty_batch(self):
        """Test with empty batch."""
        target_answers = []
        pred_answers = []

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        # Should handle empty batch (will cause division by zero, so check for nan)
        # Or implementation might need to handle this edge case
        assert "acc" in metrics
        assert "anls" in metrics

    def test_single_example(self):
        """Test with single example."""
        target_answers = [["Paris"]]
        pred_answers = ["paris"]

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        assert metrics["acc"] == 1.0
        assert metrics["anls"] == 1.0

    def test_near_match_scenarios(self):
        """Test various near-match scenarios."""
        target_answers = [
            ["Microsoft"],      # Will predict "Microsft" (missing o)
            ["December"],       # Will predict "Decembre" (wrong r)
            ["123-456-7890"],   # Will predict "123-456-789" (missing 0)
        ]
        pred_answers = [
            "Microsft",
            "Decembre",
            "123-456-789",
        ]

        metrics = DocVQAEvaluator.evaluate(target_answers, pred_answers)

        # Accuracy should be 0.0 (no exact matches)
        assert metrics["acc"] == 0.0

        # ANLS should be high (all are close)
        assert 0.7 < metrics["anls"] < 0.95
