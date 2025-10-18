# Changelog

All notable changes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## 2025-10-17 0.0.4

### Implementation

-   Phase 4 complete: Model training pipeline with TensorFlow/Keras
-   `ml_event_tagger/model.py` - Sequential model architecture (embedding + pooling + dense layers)
-   `ml_event_tagger/train.py` - Training loop with early stopping and learning rate reduction
-   `notebooks/01_train_and_evaluate.ipynb` - Comprehensive training evaluation notebook (12 sections)

### Added

-   Trained model achieving 82.9% binary accuracy, 73.3% precision, 44.0% recall, 55.0% F1 score on test set
-   Model artifacts saved: `event_tagger_model.h5`, `tokenizer.json`, `model_config.json`
-   Training history visualization with loss, accuracy, precision, recall plots
-   Interactive Jupyter notebook with per-tag analysis, sample predictions, and detailed metrics

## 2025-10-17 0.0.3

### Implementation

-   Phase 3 complete: Preprocessing pipeline implemented and tested
-   `ml_event_tagger/preprocess.py` with text cleaning, field combining, and data splitting
-   Train/val/test split (70/15/15) with 70/15/15 samples

### Added

-   Preprocessing utilities: `clean_text()`, `combine_text_fields()`, `prepare_dataset()`, `split_dataset()`
-   15 unit tests in `tests/test_preprocess.py` (all passing)
-   Preprocessed data saved to `data/` (train/val/test .npy files)

## 2025-10-17 0.0.2

### Implementation

-   Phase 2 complete: 100 labeled events dataset
-   `data/labeled_events.json` created with all 21 tags represented

## 2025-10-17 0.0.1

### Implementation

-   Phase 1 complete: Project structure validated and dependencies installed
-   Package version tracking: `ml_event_tagger.__version__ = "0.0.1"`
-   Modern Python tooling: `pyproject.toml` with `uv` for fast dependency management
-   Configuration: 21 tags defined in `ml_event_tagger/config.py`

## 2025-10-17

### Changed

-   API field renamed `formatted_address` → `location`
-   Tag taxonomy: removed city tags (sf, oakland, berkeley), added venue/access tags (outdoor, indoor, public, private, free)
-   ROADMAP restructured: v0.2 (Data) and v0.3 (Performance & Security) now separate

### Added

-   Complete planning documentation: TAGS.md, IMPLEMENTATION_PLAN.md, MVP_DECISIONS.md
-   VERSION_MANAGEMENT.md for version tracking guidelines
-   This CHANGELOG

---

**Future releases:** See [docs/ROADMAP.md](docs/ROADMAP.md)
**Version management:** See [docs/VERSION_MANAGEMENT.md](docs/VERSION_MANAGEMENT.md)
