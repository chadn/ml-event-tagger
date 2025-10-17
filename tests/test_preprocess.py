"""Unit tests for preprocessing module."""

import pytest
import numpy as np
from ml_event_tagger.preprocess import (
    clean_text,
    combine_text_fields,
    prepare_dataset,
    split_dataset
)


class TestCleanText:
    """Tests for clean_text function."""

    def test_lowercase(self):
        """Should convert to lowercase."""
        assert clean_text("HELLO WORLD") == "hello world"

    def test_remove_urls(self):
        """Should remove URLs."""
        text = "Check https://example.com and www.test.com"
        result = clean_text(text)
        assert "https" not in result
        assert "www" not in result

    def test_remove_html(self):
        """Should remove HTML tags."""
        text = "<p>Hello</p> <a href='test'>world</a>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "<a" not in result
        assert "hello world" in result

    def test_normalize_whitespace(self):
        """Should normalize multiple spaces."""
        text = "hello    world   test"
        result = clean_text(text)
        assert result == "hello world test"

    def test_remove_emails(self):
        """Should remove email addresses."""
        text = "Contact us at test@example.com for info"
        result = clean_text(text)
        assert "@" not in result

    def test_empty_string(self):
        """Should handle empty strings."""
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestCombineTextFields:
    """Tests for combine_text_fields function."""

    def test_combines_all_fields(self):
        """Should combine name, description, and location."""
        event = {
            "name": "Test Event",
            "description": "This is a test",
            "location": "San Francisco"
        }
        result = combine_text_fields(event)
        assert "test event" in result
        assert "this is a test" in result
        assert "san francisco" in result

    def test_handles_missing_fields(self):
        """Should handle missing fields gracefully."""
        event = {"name": "Test Event"}
        result = combine_text_fields(event)
        assert "test event" in result
        assert len(result) > 0

    def test_cleans_combined_text(self):
        """Should clean the combined text."""
        event = {
            "name": "TEST",
            "description": "<p>Description</p>",
            "location": "https://example.com"
        }
        result = combine_text_fields(event)
        assert "<p>" not in result
        assert "https" not in result
        assert result.islower()


class TestPrepareDataset:
    """Tests for prepare_dataset function."""

    def test_returns_correct_format(self):
        """Should return texts, labels, and tag names."""
        events = [
            {
                "name": "Test Event",
                "description": "Test description",
                "location": "Test location",
                "tags": ["music", "dance"]
            }
        ]
        texts, labels, tag_names = prepare_dataset(events)

        assert len(texts) == 1
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (1, 21)  # 21 tags
        assert isinstance(tag_names, list)
        assert len(tag_names) == 21

    def test_binary_labels(self):
        """Should create binary label vectors."""
        events = [
            {
                "name": "Test",
                "description": "Test",
                "location": "Test",
                "tags": ["music", "dance"]
            }
        ]
        texts, labels, tag_names = prepare_dataset(events)

        # Should only have 0s and 1s
        assert set(labels[0]).issubset({0, 1})

        # Should have exactly 2 tags marked
        assert np.sum(labels[0]) == 2

        # Check specific tags
        music_idx = tag_names.index("music")
        dance_idx = tag_names.index("dance")
        assert labels[0][music_idx] == 1
        assert labels[0][dance_idx] == 1


class TestSplitDataset:
    """Tests for split_dataset function."""

    def test_split_ratios(self):
        """Should split according to specified ratios."""
        texts = [f"text_{i}" for i in range(100)]
        labels = np.random.randint(0, 2, (100, 21))

        result = split_dataset(texts, labels, 0.7, 0.15, 0.15, random_seed=42)
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = result

        assert len(train_texts) == 70
        assert len(val_texts) == 15
        assert len(test_texts) == 15

        assert train_labels.shape == (70, 21)
        assert val_labels.shape == (15, 21)
        assert test_labels.shape == (15, 21)

    def test_reproducibility(self):
        """Should produce same split with same seed."""
        texts = [f"text_{i}" for i in range(100)]
        labels = np.random.randint(0, 2, (100, 21))

        result1 = split_dataset(texts, labels, random_seed=42)
        result2 = split_dataset(texts, labels, random_seed=42)

        assert result1[0] == result2[0]  # train_texts
        np.testing.assert_array_equal(result1[1], result2[1])  # train_labels

    def test_no_overlap(self):
        """Train/val/test sets should not overlap."""
        texts = [f"text_{i}" for i in range(100)]
        labels = np.random.randint(0, 2, (100, 21))

        result = split_dataset(texts, labels, random_seed=42)
        train_texts, _, val_texts, _, test_texts, _ = result

        train_set = set(train_texts)
        val_set = set(val_texts)
        test_set = set(test_texts)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_invalid_splits(self):
        """Should raise error if splits don't sum to 1."""
        texts = ["text"]
        labels = np.array([[1, 0, 0]])

        with pytest.raises(AssertionError):
            split_dataset(texts, labels, 0.5, 0.3, 0.1)  # Sums to 0.9

