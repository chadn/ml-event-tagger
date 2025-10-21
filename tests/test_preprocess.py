"""Unit tests for preprocessing module."""

import pytest
import numpy as np
import os
import json
import tempfile
from ml_event_tagger.preprocess import (
    clean_text,
    combine_text_fields,
    prepare_dataset,
    split_dataset,
    save_preprocessed_data,
    load_labeled_events
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


class TestPrepareDatasetEdgeCases:
    """Edge case tests for prepare_dataset function."""

    def test_prepare_dataset_with_empty_list(self):
        """Should handle empty event list gracefully."""
        events = []
        texts, labels, tag_names = prepare_dataset(events)

        assert len(texts) == 0
        assert labels.shape == (0, 21)
        assert len(tag_names) == 21

    def test_prepare_dataset_with_missing_tags(self):
        """Should handle events without tags field."""
        events = [
            {
                "name": "Test Event",
                "description": "Test description",
                "location": "Test location"
                # No tags field
            }
        ]
        texts, labels, tag_names = prepare_dataset(events)

        assert len(texts) == 1
        assert np.sum(labels[0]) == 0  # All zeros (no tags)

    def test_prepare_dataset_with_empty_tags(self):
        """Should handle events with empty tags list."""
        events = [
            {
                "name": "Test Event",
                "description": "Test description",
                "location": "Test location",
                "tags": []
            }
        ]
        texts, labels, tag_names = prepare_dataset(events)

        assert len(texts) == 1
        assert np.sum(labels[0]) == 0  # All zeros

    def test_prepare_dataset_with_invalid_tags(self):
        """Should ignore invalid tags not in taxonomy."""
        events = [
            {
                "name": "Test Event",
                "description": "Test description",
                "location": "Test location",
                "tags": ["music", "invalid_tag", "dance", "another_invalid"]
            }
        ]
        texts, labels, tag_names = prepare_dataset(events)

        # Should only count valid tags (music, dance)
        assert np.sum(labels[0]) == 2

    def test_prepare_dataset_with_none_values(self):
        """Should handle None values in event fields."""
        events = [
            {
                "name": "Test Event",
                "description": None,
                "location": None,
                "tags": ["music"]
            }
        ]
        texts, labels, tag_names = prepare_dataset(events)

        assert len(texts) == 1
        assert len(texts[0]) > 0  # Should still have name
        assert np.sum(labels[0]) == 1  # Should have music tag


class TestSavePreprocessedData:
    """Tests for save_preprocessed_data function."""

    def test_save_creates_files(self):
        """Should create all required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            train_texts = ["text1", "text2"]
            train_labels = np.array([[1, 0, 0], [0, 1, 0]])
            val_texts = ["text3"]
            val_labels = np.array([[0, 0, 1]])
            test_texts = ["text4"]
            test_labels = np.array([[1, 1, 0]])
            tag_names = ["tag1", "tag2", "tag3"]

            # Save data
            save_preprocessed_data(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                tag_names,
                output_dir=tmpdir
            )

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "train_texts.npy"))
            assert os.path.exists(os.path.join(tmpdir, "train_labels.npy"))
            assert os.path.exists(os.path.join(tmpdir, "val_texts.npy"))
            assert os.path.exists(os.path.join(tmpdir, "val_labels.npy"))
            assert os.path.exists(os.path.join(tmpdir, "test_texts.npy"))
            assert os.path.exists(os.path.join(tmpdir, "test_labels.npy"))
            assert os.path.exists(os.path.join(tmpdir, "tag_names.json"))

    def test_save_data_can_be_loaded(self):
        """Saved data can be loaded back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data
            train_texts = ["text1", "text2", "text3"]
            train_labels = np.array([[1, 0], [0, 1], [1, 1]])
            val_texts = ["text4"]
            val_labels = np.array([[0, 1]])
            test_texts = ["text5"]
            test_labels = np.array([[1, 0]])
            tag_names = ["tag1", "tag2"]

            # Save data
            save_preprocessed_data(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                tag_names,
                output_dir=tmpdir
            )

            # Load data back
            loaded_train_texts = np.load(os.path.join(tmpdir, "train_texts.npy"), allow_pickle=True)
            loaded_train_labels = np.load(os.path.join(tmpdir, "train_labels.npy"))

            with open(os.path.join(tmpdir, "tag_names.json")) as f:
                loaded_tag_names = json.load(f)

            # Verify loaded data matches original
            assert list(loaded_train_texts) == train_texts
            np.testing.assert_array_equal(loaded_train_labels, train_labels)
            assert loaded_tag_names == tag_names

    def test_save_creates_directory_if_not_exists(self):
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a nested directory that doesn't exist
            output_dir = os.path.join(tmpdir, "nested", "output")

            # Create minimal data
            train_texts = ["text"]
            train_labels = np.array([[1]])
            val_texts = ["text"]
            val_labels = np.array([[1]])
            test_texts = ["text"]
            test_labels = np.array([[1]])
            tag_names = ["tag"]

            # Save should create the directory
            save_preprocessed_data(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                tag_names,
                output_dir=output_dir
            )

            # Verify directory was created
            assert os.path.exists(output_dir)
            assert os.path.isdir(output_dir)


class TestLoadLabeledEvents:
    """Tests for load_labeled_events function."""

    def test_load_labeled_events_returns_list(self):
        """Should return a list of events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON file
            test_file = os.path.join(tmpdir, "test_events.json")
            test_data = [
                {"name": "Event 1", "tags": ["music"]},
                {"name": "Event 2", "tags": ["dance"]}
            ]

            with open(test_file, 'w') as f:
                json.dump(test_data, f)

            # Load events
            events = load_labeled_events(test_file)

            assert isinstance(events, list)
            assert len(events) == 2
            assert events[0]["name"] == "Event 1"

    def test_load_labeled_events_handles_empty_file(self):
        """Should handle empty JSON array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "empty_events.json")

            with open(test_file, 'w') as f:
                json.dump([], f)

            events = load_labeled_events(test_file)

            assert isinstance(events, list)
            assert len(events) == 0

