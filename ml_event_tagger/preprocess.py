"""Text preprocessing for event classification.

This module handles:
- Text cleaning and normalization
- Combining text fields (name + description + location)
- Data splitting (train/val/test)
"""

import re
import json
from typing import List, Dict, Tuple
import numpy as np


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s\.,!?-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def combine_text_fields(event: Dict) -> str:
    """Combine event text fields into single string.

    Combines: name + description + location

    Args:
        event: Event dictionary with 'name', 'description', 'location'

    Returns:
        Combined text string
    """
    name = event.get('name', '')
    description = event.get('description', '')
    location = event.get('location', '')

    # Clean each field
    name_clean = clean_text(name)
    desc_clean = clean_text(description)
    loc_clean = clean_text(location)

    # Combine with spaces
    combined = f"{name_clean} {desc_clean} {loc_clean}"

    return combined.strip()


def load_labeled_events(filepath: str = "data/labeled_events.json") -> List[Dict]:
    """Load labeled events from JSON file.

    Args:
        filepath: Path to labeled events JSON file

    Returns:
        List of event dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        events = json.load(f)
    return events


def prepare_dataset(events: List[Dict]) -> Tuple[List[str], np.ndarray, List[str]]:
    """Prepare dataset for training.

    Args:
        events: List of event dictionaries

    Returns:
        Tuple of (texts, labels, tag_names)
        - texts: List of combined text strings
        - labels: Binary label matrix (n_samples, n_tags)
        - tag_names: List of tag names (ordered)
    """
    from ml_event_tagger.config import TAGS

    texts = []
    labels = []

    for event in events:
        # Combine text fields
        text = combine_text_fields(event)
        texts.append(text)

        # Create binary label vector
        event_tags = set(event.get('tags', []))
        label_vector = [1 if tag in event_tags else 0 for tag in TAGS]
        labels.append(label_vector)

    # Ensure labels array has correct shape even when empty
    if len(labels) == 0:
        labels_array = np.empty((0, len(TAGS)), dtype=int)
    else:
        labels_array = np.array(labels)

    return texts, labels_array, TAGS


def split_dataset(
    texts: List[str],
    labels: np.ndarray,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple:
    """Split dataset into train/val/test sets.

    Args:
        texts: List of text strings
        labels: Label matrix
        train_split: Fraction for training (default 0.70)
        val_split: Fraction for validation (default 0.15)
        test_split: Fraction for test (default 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Splits must sum to 1.0"

    # Set random seed
    np.random.seed(random_seed)

    # Create indices and shuffle
    n_samples = len(texts)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Calculate split points
    train_end = int(n_samples * train_split)
    val_end = train_end + int(n_samples * val_split)

    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Split data
    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_labels = labels[val_idx]

    test_texts = [texts[i] for i in test_idx]
    test_labels = labels[test_idx]

    return (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)


def save_preprocessed_data(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    tag_names: List[str],
    output_dir: str = "data"
):
    """Save preprocessed data to files.

    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        test_texts: Test texts
        test_labels: Test labels
        tag_names: List of tag names
        output_dir: Output directory
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save as numpy arrays
    np.save(f"{output_dir}/train_texts.npy", train_texts)
    np.save(f"{output_dir}/train_labels.npy", train_labels)
    np.save(f"{output_dir}/val_texts.npy", val_texts)
    np.save(f"{output_dir}/val_labels.npy", val_labels)
    np.save(f"{output_dir}/test_texts.npy", test_texts)
    np.save(f"{output_dir}/test_labels.npy", test_labels)

    # Save tag names
    with open(f"{output_dir}/tag_names.json", 'w') as f:
        json.dump(tag_names, f, indent=2)

    print(f"✅ Saved preprocessed data to {output_dir}/")
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Val:   {len(val_texts)} samples")
    print(f"   Test:  {len(test_texts)} samples")


if __name__ == "__main__":
    """Test preprocessing pipeline."""

    print("=" * 80)
    print("Testing Preprocessing Pipeline")
    print("=" * 80)
    print()

    # Load data
    print("Loading labeled events...")
    events = load_labeled_events()
    print(f"✅ Loaded {len(events)} events")
    print()

    # Prepare dataset
    print("Preparing dataset...")
    texts, labels, tag_names = prepare_dataset(events)
    print(f"✅ Prepared {len(texts)} text samples")
    print(f"✅ Label shape: {labels.shape}")
    print(f"✅ Tags: {len(tag_names)}")
    print()

    # Show sample
    print("Sample preprocessed text:")
    print(f"  Original name: {events[0]['name']}")
    print(f"  Processed: {texts[0][:100]}...")
    print()

    # Split dataset
    print("Splitting dataset (70/15/15)...")
    split_data = split_dataset(texts, labels)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = split_data

    print(f"✅ Train: {len(train_texts)} samples ({len(train_texts)/len(texts)*100:.1f}%)")
    print(f"✅ Val:   {len(val_texts)} samples ({len(val_texts)/len(texts)*100:.1f}%)")
    print(f"✅ Test:  {len(test_texts)} samples ({len(test_texts)/len(texts)*100:.1f}%)")
    print()

    # Save
    print("Saving preprocessed data...")
    save_preprocessed_data(
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
        tag_names
    )

    print()
    print("=" * 80)
    print("✅ Preprocessing complete!")
    print("=" * 80)

