"""Training script for event classification model.

Handles:
- Text tokenization
- Model training with validation
- Early stopping
- Model and tokenizer saving
- Training history visualization
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ml_event_tagger.model import create_model
from ml_event_tagger.config import TAGS


# Training configuration
MAX_VOCAB_SIZE = 10000
MAX_LENGTH = 200
EMBEDDING_DIM = 64
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 10


def load_preprocessed_data(data_dir: str = "data"):
    """Load preprocessed train/val/test data.

    Args:
        data_dir: Directory containing preprocessed .npy files

    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tag_names)
    """
    train_texts = np.load(f"{data_dir}/train_texts.npy", allow_pickle=True)
    train_labels = np.load(f"{data_dir}/train_labels.npy")

    val_texts = np.load(f"{data_dir}/val_texts.npy", allow_pickle=True)
    val_labels = np.load(f"{data_dir}/val_labels.npy")

    test_texts = np.load(f"{data_dir}/test_texts.npy", allow_pickle=True)
    test_labels = np.load(f"{data_dir}/test_labels.npy")

    with open(f"{data_dir}/tag_names.json") as f:
        tag_names = json.load(f)

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tag_names


def create_tokenizer(texts, vocab_size=MAX_VOCAB_SIZE):
    """Create and fit tokenizer on texts.

    Args:
        texts: List of text strings
        vocab_size: Maximum vocabulary size

    Returns:
        Fitted Tokenizer
    """
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>",
        lower=True
    )
    tokenizer.fit_on_texts(texts)

    return tokenizer


def tokenize_texts(tokenizer, texts, max_length=MAX_LENGTH):
    """Convert texts to padded sequences.

    Args:
        tokenizer: Fitted Tokenizer
        texts: List of text strings
        max_length: Maximum sequence length

    Returns:
        Padded sequences array
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded


def plot_training_history(history, save_path="models/training_history.png"):
    """Plot training history (loss, precision, recall).

    Args:
        history: Keras History object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training History', fontsize=16)

    # Loss
    ax = axes[0, 0]
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Binary Crossentropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(history.history['binary_accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_binary_accuracy'], label='Val Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Binary Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision
    ax = axes[1, 0]
    ax.plot(history.history['precision'], label='Train Precision')
    ax.plot(history.history['val_precision'], label='Val Precision')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall
    ax = axes[1, 1]
    ax.plot(history.history['recall'], label='Train Recall')
    ax.plot(history.history['val_recall'], label='Val Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training history plot saved to {save_path}")

    plt.close()


def save_model_and_artifacts(model, tokenizer, history, models_dir="models"):
    """Save trained model, tokenizer, and training history.

    Args:
        model: Trained Keras model
        tokenizer: Fitted Tokenizer
        history: Training History object
        models_dir: Directory to save artifacts
    """
    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = f"{models_dir}/event_tagger_model.h5"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

    # Save tokenizer
    tokenizer_config = {
        'word_index': tokenizer.word_index,
        'num_words': tokenizer.num_words,
        'oov_token': tokenizer.oov_token
    }
    tokenizer_path = f"{models_dir}/tokenizer.json"
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✅ Tokenizer saved to {tokenizer_path}")

    # Save training history
    history_path = f"{models_dir}/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"✅ Training history saved to {history_path}")

    # Save model config
    config = {
        'max_vocab_size': MAX_VOCAB_SIZE,
        'max_length': MAX_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'num_tags': len(TAGS),
        'tags': TAGS
    }
    config_path = f"{models_dir}/model_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Model config saved to {config_path}")


def train_model():
    """Main training function."""

    print("=" * 80)
    print("Training Event Classification Model")
    print("=" * 80)
    print()

    # Load data
    print("Loading preprocessed data...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tag_names = load_preprocessed_data()

    print(f"✅ Loaded data:")
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Val:   {len(val_texts)} samples")
    print(f"   Test:  {len(test_texts)} samples")
    print(f"   Tags:  {len(tag_names)}")
    print()

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(train_texts, vocab_size=MAX_VOCAB_SIZE)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"✅ Tokenizer created with vocabulary size: {vocab_size}")
    print()

    # Tokenize texts
    print("Tokenizing texts...")
    X_train = tokenize_texts(tokenizer, train_texts, max_length=MAX_LENGTH)
    X_val = tokenize_texts(tokenizer, val_texts, max_length=MAX_LENGTH)
    X_test = tokenize_texts(tokenizer, test_texts, max_length=MAX_LENGTH)

    print(f"✅ Tokenized texts:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_val shape:   {X_val.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print()

    # Create model
    print("Creating model...")
    model = create_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_LENGTH,
        num_tags=len(tag_names)
    )

    print("✅ Model created:")
    model.summary()
    print()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # Train model
    print("Training model...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Early stopping patience: {PATIENCE}")
    print()

    history = model.fit(
        X_train, train_labels,
        validation_data=(X_val, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = model.evaluate(X_test, test_labels, verbose=0)

    print(f"✅ Test Results:")
    print(f"   Loss: {test_results[0]:.4f}")
    print(f"   Binary Accuracy: {test_results[1]:.4f}")
    print(f"   Precision: {test_results[2]:.4f}")
    print(f"   Recall: {test_results[3]:.4f}")
    print()

    # Calculate F1 score
    precision = test_results[2]
    recall = test_results[3]
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"   F1 Score: {f1_score:.4f}")
    print()

    # Plot training history
    print("Creating training visualizations...")
    plot_training_history(history)
    print()

    # Save artifacts
    print("Saving model and artifacts...")
    save_model_and_artifacts(model, tokenizer, history)
    print()

    print("=" * 80)
    print("✅ Training pipeline completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  • Review training plots: models/training_history.png")
    print("  • Test predictions with the model")
    print("  • Create API service (Phase 5)")

    return model, tokenizer, history


if __name__ == "__main__":
    train_model()

