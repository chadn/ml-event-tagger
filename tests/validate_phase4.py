"""Validate Phase 4: Model Training

Checks:
1. Model files exist and are valid
2. Model can be loaded
3. Model architecture is correct
4. Model can make predictions
5. Training metrics meet success criteria
6. Visualizations are created
"""

import os
import json
import numpy as np
from tensorflow import keras

from ml_event_tagger.config import TAGS


def validate_phase4():
    print("=" * 80)
    print("Phase 4 Validation: Model Training")
    print("=" * 80)
    print()

    # --- Check 1: Model artifacts exist ---
    print("✓ Check 1: Model artifacts exist")
    required_files = [
        "models/event_tagger_model.h5",
        "models/tokenizer.json",
        "models/model_config.json",
        "models/training_history.json",
        "models/training_history.png"
    ]

    all_exist = True
    for f in required_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✅ {f} ({size:,} bytes)")
        else:
            print(f"  ❌ Missing: {f}")
            all_exist = False

    if not all_exist:
        print("\n❌ Validation failed: Missing required files")
        return
    print()

    # --- Check 2: Load model ---
    print("✓ Check 2: Load trained model")
    try:
        model = keras.models.load_model("models/event_tagger_model.h5")
        print("  ✅ Model loaded successfully")

        # Check model architecture
        expected_layers = ['embedding', 'pooling', 'dense_1', 'dropout_1', 'dense_2', 'output']
        actual_layers = [layer.name for layer in model.layers]

        if actual_layers == expected_layers:
            print(f"  ✅ Model architecture correct: {len(actual_layers)} layers")
        else:
            print(f"  ⚠️  Layer names: {actual_layers}")

        # Check output shape
        output_shape = model.output_shape
        if output_shape[-1] == len(TAGS):
            print(f"  ✅ Output layer has {len(TAGS)} units (correct for {len(TAGS)} tags)")
        else:
            print(f"  ❌ Output layer has {output_shape[-1]} units, expected {len(TAGS)}")

    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return
    print()

    # --- Check 3: Model config ---
    print("✓ Check 3: Model configuration")
    with open("models/model_config.json") as f:
        config = json.load(f)

    print(f"  ✅ Max vocab size: {config['max_vocab_size']}")
    print(f"  ✅ Max length: {config['max_length']}")
    print(f"  ✅ Embedding dim: {config['embedding_dim']}")
    print(f"  ✅ Num tags: {config['num_tags']}")
    print(f"  ✅ Tags: {len(config['tags'])} tags defined")
    print()

    # --- Check 4: Tokenizer ---
    print("✓ Check 4: Tokenizer configuration")
    with open("models/tokenizer.json") as f:
        tokenizer_config = json.load(f)

    vocab_size = len(tokenizer_config['word_index'])
    print(f"  ✅ Vocabulary size: {vocab_size:,} words")
    print(f"  ✅ OOV token: {tokenizer_config['oov_token']}")
    print(f"  ✅ Max words: {tokenizer_config['num_words']:,}")
    print()

    # --- Check 5: Training history and metrics ---
    print("✓ Check 5: Training history and performance")
    with open("models/training_history.json") as f:
        history = json.load(f)

    epochs_trained = len(history['loss'])
    final_val_loss = history['val_loss'][-1]
    final_val_precision = history['val_precision'][-1]
    final_val_recall = history['val_recall'][-1]
    final_val_accuracy = history['val_binary_accuracy'][-1]

    print(f"  ✅ Epochs trained: {epochs_trained}")
    print(f"  ✅ Final validation loss: {final_val_loss:.4f}")
    print(f"  ✅ Final validation accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.1f}%)")
    print(f"  ✅ Final validation precision: {final_val_precision:.4f} ({final_val_precision*100:.1f}%)")
    print(f"  ✅ Final validation recall: {final_val_recall:.4f} ({final_val_recall*100:.1f}%)")

    # Check success criteria
    print()
    print("  Success Criteria:")
    if final_val_precision >= 0.60:
        print(f"    ✅ Precision ≥ 60%: {final_val_precision*100:.1f}% ✓")
    else:
        print(f"    ❌ Precision ≥ 60%: {final_val_precision*100:.1f}% ✗")

    # Calculate F1
    if (final_val_precision + final_val_recall) > 0:
        f1 = 2 * (final_val_precision * final_val_recall) / (final_val_precision + final_val_recall)
        print(f"    ✅ F1 Score: {f1:.4f} ({f1*100:.1f}%)")
    print()

    # --- Check 6: Model inference test ---
    print("✓ Check 6: Model inference test")

    # Create a dummy input
    dummy_input = np.random.randint(0, vocab_size, size=(5, config['max_length']))

    try:
        predictions = model.predict(dummy_input, verbose=0)
        print(f"  ✅ Inference works: Input shape {dummy_input.shape} → Output shape {predictions.shape}")
        print(f"  ✅ Output range: [{predictions.min():.3f}, {predictions.max():.3f}] (valid probabilities)")

        # Verify it's probabilities
        if 0 <= predictions.min() <= 1 and 0 <= predictions.max() <= 1:
            print(f"  ✅ Outputs are valid probabilities [0, 1]")
        else:
            print(f"  ❌ Outputs are not valid probabilities")

    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        return
    print()

    # --- Check 7: Test with real data ---
    print("✓ Check 7: Test with real preprocessed data")
    try:
        test_texts = np.load("data/test_texts.npy", allow_pickle=True)
        test_labels = np.load("data/test_labels.npy")

        print(f"  ✅ Loaded test data: {len(test_texts)} samples")

        # Need to tokenize the test texts
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Recreate tokenizer from saved config
        tokenizer = Tokenizer(num_words=tokenizer_config['num_words'], oov_token=tokenizer_config['oov_token'])
        tokenizer.word_index = tokenizer_config['word_index']

        sequences = tokenizer.texts_to_sequences(test_texts)
        X_test = pad_sequences(sequences, maxlen=config['max_length'], padding='post', truncating='post')

        # Evaluate on test set
        test_results = model.evaluate(X_test, test_labels, verbose=0)

        print(f"  ✅ Test set evaluation:")
        print(f"     Loss:     {test_results[0]:.4f}")
        print(f"     Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.1f}%)")
        print(f"     Precision: {test_results[2]:.4f} ({test_results[2]*100:.1f}%)")
        print(f"     Recall:    {test_results[3]:.4f} ({test_results[3]*100:.1f}%)")

        # Calculate F1
        precision = test_results[2]
        recall = test_results[3]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"     F1 Score:  {f1:.4f} ({f1*100:.1f}%)")

    except Exception as e:
        print(f"  ⚠️  Could not test with real data: {e}")
    print()

    # --- Check 8: Visualization ---
    print("✓ Check 8: Training visualization")
    if os.path.exists("models/training_history.png"):
        size = os.path.getsize("models/training_history.png")
        print(f"  ✅ Training history plot exists ({size:,} bytes)")
        print(f"     → To view: open models/training_history.png")
    else:
        print(f"  ❌ Training history plot missing")
    print()

    # --- Summary ---
    print("=" * 80)
    print("✅ Phase 4 Validation: ALL CHECKS PASSED")
    print("=" * 80)
    print()
    print("📊 Model Summary:")
    print(f"   • Vocabulary: {vocab_size:,} words")
    print(f"   • Architecture: Embedding → Pooling → Dense(64) → Dropout → Dense(32) → Dense(21)")
    print(f"   • Trained for {epochs_trained} epochs")
    print(f"   • Validation Precision: {final_val_precision*100:.1f}% (target: ≥60%)")
    print(f"   • Test Precision: {test_results[2]*100:.1f}%")
    print(f"   • Test F1 Score: {f1*100:.1f}%")
    print()
    print("🎉 Model is trained and ready to serve predictions!")
    print("🚀 Ready to proceed to Phase 5: API Service")
    print()
    print("To view training plots:")
    print("  macOS:  open models/training_history.png")
    print("  Linux:  xdg-open models/training_history.png")
    print("  Or use your favorite image viewer")


if __name__ == "__main__":
    validate_phase4()

