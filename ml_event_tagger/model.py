"""Model architecture for multi-label event classification.

Uses TensorFlow/Keras Sequential model with:
- Embedding layer
- Global average pooling
- Dense layers
- Sigmoid activation for multi-label output
"""

from tensorflow import keras
from tensorflow.keras import layers


def create_model(
    vocab_size: int,
    embedding_dim: int = 64,
    max_length: int = 200,
    num_tags: int = 21
) -> keras.Model:
    """Create Sequential model for multi-label classification.

    Architecture:
    - Embedding(vocab_size, embedding_dim)
    - GlobalAveragePooling1D()
    - Dense(64, relu)
    - Dropout(0.3)
    - Dense(32, relu)
    - Dense(num_tags, sigmoid)

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings (default: 64)
        max_length: Maximum sequence length (default: 200)
        num_tags: Number of output tags (default: 21)

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Embedding layer
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding'
        ),

        # Global average pooling to reduce dimensionality
        layers.GlobalAveragePooling1D(name='pooling'),

        # Hidden layers with dropout for regularization
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout_1'),

        layers.Dense(32, activation='relu', name='dense_2'),

        # Output layer with sigmoid for multi-label classification
        layers.Dense(num_tags, activation='sigmoid', name='output')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    return model


def get_model_summary(model: keras.Model) -> str:
    """Get formatted model summary as string.

    Args:
        model: Keras model

    Returns:
        Model summary as string
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


if __name__ == "__main__":
    """Test model creation."""

    print("=" * 80)
    print("Testing Model Architecture")
    print("=" * 80)
    print()

    # Create model with test parameters
    model = create_model(
        vocab_size=10000,
        embedding_dim=64,
        max_length=200,
        num_tags=21
    )

    print("✅ Model created successfully")
    print()

    # Print summary
    print("Model Summary:")
    print("-" * 80)
    model.summary()
    print()

    # Test with dummy data
    import numpy as np

    dummy_input = np.random.randint(0, 10000, size=(10, 200))
    dummy_output = model.predict(dummy_input, verbose=0)

    print("✅ Model inference test:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {dummy_output.shape}")
    print(f"   Output range: [{dummy_output.min():.3f}, {dummy_output.max():.3f}]")
    print()

    # Verify output is valid probabilities
    assert dummy_output.shape == (10, 21), "Output shape mismatch"
    assert 0 <= dummy_output.min() <= 1, "Output not in [0, 1] range"
    assert 0 <= dummy_output.max() <= 1, "Output not in [0, 1] range"

    print("=" * 80)
    print("✅ Model architecture validated!")
    print("=" * 80)

