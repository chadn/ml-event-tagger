"""Unit tests for model architecture."""

import pytest
import numpy as np
from tensorflow import keras

from ml_event_tagger.model import create_model, get_model_summary
from ml_event_tagger.config import TAGS


class TestCreateModel:
    """Tests for create_model function."""

    def test_create_model_returns_keras_model(self):
        """Model creation returns a Keras Sequential model."""
        model = create_model(vocab_size=1000, embedding_dim=32, max_length=100, num_tags=10)

        assert isinstance(model, keras.Model)
        assert isinstance(model, keras.Sequential)

    def test_create_model_correct_architecture(self):
        """Model has expected layers (Embedding, Pooling, Dense, etc.)."""
        model = create_model(vocab_size=1000, embedding_dim=32, max_length=100, num_tags=10)

        layer_names = [layer.name for layer in model.layers]

        # Check expected layers exist
        assert 'embedding' in layer_names
        assert 'pooling' in layer_names
        assert 'dense_1' in layer_names
        assert 'dropout_1' in layer_names
        assert 'dense_2' in layer_names
        assert 'output' in layer_names

        # Check layer order
        assert layer_names[0] == 'embedding'
        assert layer_names[1] == 'pooling'
        assert layer_names[-1] == 'output'

    def test_create_model_correct_output_shape(self):
        """Model output shape matches number of tags."""
        num_tags = 15
        vocab_size = 1000
        max_length = 100
        model = create_model(vocab_size=vocab_size, embedding_dim=32, max_length=max_length, num_tags=num_tags)

        # Build the model by running inference once
        dummy_input = np.random.randint(0, vocab_size, size=(1, max_length))
        _ = model.predict(dummy_input, verbose=0)

        # Get output layer
        output_layer = model.layers[-1]

        # Check output shape using model's output_shape (now that it's built)
        assert model.output_shape[-1] == num_tags

        # Check activation is sigmoid (for multi-label)
        assert output_layer.activation.__name__ == 'sigmoid'

    def test_create_model_is_compiled(self):
        """Model is compiled with optimizer, loss, and metrics."""
        model = create_model(vocab_size=1000, embedding_dim=32, max_length=100, num_tags=10)

        # Check optimizer
        assert model.optimizer is not None
        assert 'adam' in model.optimizer.__class__.__name__.lower()

        # Check loss function
        assert model.loss == 'binary_crossentropy'

        # Check metrics are configured - access compiled metrics
        metric_config = model.get_compile_config()
        assert metric_config is not None

        # Verify model can perform training step (compiled properly)
        dummy_input = np.random.randint(0, 1000, size=(2, 100))
        dummy_labels = np.random.randint(0, 2, size=(2, 10))

        # This will fail if model is not compiled correctly
        # train_on_batch returns a list: [loss, metric1, metric2, ...]
        result = model.train_on_batch(dummy_input, dummy_labels)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0  # Should have at least loss value

    def test_create_model_inference(self):
        """Model can perform inference on dummy data."""
        vocab_size = 1000
        max_length = 100
        num_tags = 10
        batch_size = 5

        model = create_model(
            vocab_size=vocab_size,
            embedding_dim=32,
            max_length=max_length,
            num_tags=num_tags
        )

        # Create dummy input (random integers representing token IDs)
        dummy_input = np.random.randint(0, vocab_size, size=(batch_size, max_length))

        # Run inference
        predictions = model.predict(dummy_input, verbose=0)

        # Check output shape
        assert predictions.shape == (batch_size, num_tags)

        # Check output is valid probabilities (between 0 and 1)
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0

        # Check each sample has predictions for all tags
        for i in range(batch_size):
            assert len(predictions[i]) == num_tags

    def test_create_model_with_default_params(self):
        """Model can be created with default parameters."""
        # Should work with just vocab_size
        model = create_model(vocab_size=5000)

        assert model is not None
        assert isinstance(model, keras.Model)

        # Build model to check output shape
        dummy_input = np.random.randint(0, 5000, size=(1, 200))  # default max_length=200
        output = model.predict(dummy_input, verbose=0)

        # Check default parameters are applied
        assert output.shape[-1] == 21  # Default num_tags

    def test_create_model_different_vocab_sizes(self):
        """Model works with different vocabulary sizes."""
        vocab_sizes = [100, 1000, 10000, 50000]

        for vocab_size in vocab_sizes:
            model = create_model(vocab_size=vocab_size, num_tags=5)

            # Get embedding layer
            embedding_layer = model.layers[0]

            # Check vocab size is correct
            assert embedding_layer.input_dim == vocab_size

    def test_create_model_different_embedding_dims(self):
        """Model works with different embedding dimensions."""
        embedding_dims = [16, 32, 64, 128]

        for embedding_dim in embedding_dims:
            model = create_model(
                vocab_size=1000,
                embedding_dim=embedding_dim,
                num_tags=5
            )

            # Get embedding layer
            embedding_layer = model.layers[0]

            # Check embedding dimension is correct
            assert embedding_layer.output_dim == embedding_dim

    def test_create_model_trainable(self):
        """Model is trainable (not frozen)."""
        model = create_model(vocab_size=1000, num_tags=5)

        # Check model is trainable
        assert model.trainable is True

        # Check all layers are trainable
        for layer in model.layers:
            if hasattr(layer, 'trainable'):
                assert layer.trainable is True


class TestGetModelSummary:
    """Tests for get_model_summary function."""

    def test_get_model_summary_returns_string(self):
        """get_model_summary returns a string."""
        model = create_model(vocab_size=1000, num_tags=5)
        summary = get_model_summary(model)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_model_summary_contains_layer_info(self):
        """Model summary contains layer information."""
        model = create_model(vocab_size=1000, num_tags=5)
        summary = get_model_summary(model)

        # Check for expected layer names
        assert 'embedding' in summary.lower()
        assert 'pooling' in summary.lower()
        assert 'dense' in summary.lower()
        assert 'dropout' in summary.lower()

    def test_get_model_summary_contains_param_count(self):
        """Model summary contains parameter counts."""
        model = create_model(vocab_size=1000, num_tags=5)
        summary = get_model_summary(model)

        # Check for parameter information
        assert 'param' in summary.lower() or 'parameters' in summary.lower()
        assert 'total' in summary.lower()


class TestModelIntegration:
    """Integration tests for model with real config."""

    def test_model_with_actual_config(self):
        """Model works with actual project configuration."""
        from ml_event_tagger.config import MAX_VOCAB_SIZE, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH

        model = create_model(
            vocab_size=MAX_VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            max_length=MAX_SEQUENCE_LENGTH,
            num_tags=len(TAGS)
        )

        assert model is not None

        # Test inference with realistic input
        batch_size = 3
        dummy_input = np.random.randint(0, MAX_VOCAB_SIZE, size=(batch_size, MAX_SEQUENCE_LENGTH))
        predictions = model.predict(dummy_input, verbose=0)

        assert predictions.shape == (batch_size, len(TAGS))

    def test_model_can_be_saved_and_loaded(self):
        """Model can be saved and loaded (for deployment)."""
        import tempfile
        import os

        # Create model
        model = create_model(vocab_size=1000, num_tags=5)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.h5")
            model.save(model_path)

            # Check file exists
            assert os.path.exists(model_path)

            # Load model
            loaded_model = keras.models.load_model(model_path)

            # Check loaded model works
            dummy_input = np.random.randint(0, 1000, size=(2, 100))
            predictions = loaded_model.predict(dummy_input, verbose=0)

            assert predictions.shape == (2, 5)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

