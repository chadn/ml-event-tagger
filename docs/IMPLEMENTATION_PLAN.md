# 🛠️ Implementation Plan - v0.1 MVP

Step-by-step guide for implementing the ml-event-tagger MVP.

**Target:** Working end-to-end ML service demonstrating TensorFlow/Keras and FastAPI integration.

---

## 📋 Overview

| Phase     | Focus                  | Estimated Time  | Status                 |
| --------- | ---------------------- | --------------- | ---------------------- |
| 1         | Project Setup          | 1-2 hours       | ✅ Complete (v0.0.1)   |
| 2         | Data Preparation       | 3-5 hours       | ✅ Complete (v0.0.2)   |
| 3         | Preprocessing Pipeline | 2-3 hours       | ✅ Complete (v0.0.3)   |
| 4         | Model Training         | 3-4 hours       | ✅ Complete (v0.0.4)   |
| 5         | API Service            | 2-3 hours       | ✅ Complete (v0.0.5)   |
| 6         | Testing & Validation   | 1-2 hours       | ⬜ Not Started         |
| 7         | Docker & Deployment    | 1-2 hours       | ⬜ Not Started         |
| 8         | Documentation Polish   | 2-3 hours       | ⬜ Not Started         |
| **Total** | **End-to-End**         | **15-25 hours** | **Phase 5/8 complete** |

---

## Phase 1: Project Setup

**Goal:** Create clean repository structure with all necessary files.

### Tasks

-   [x] Create directory structure:

    ```
    ml-event-tagger/
    ├── ml_event_tagger/
    │   ├── __init__.py
    │   ├── train.py
    │   ├── serve.py
    │   ├── preprocess.py
    │   ├── model.py
    │   └── config.py
    ├── notebooks/
    ├── models/
    ├── data/
    ├── tests/
    ├── docs/ (already exists)
    └── (root files)
    ```

-   [x] Create `requirements.txt`:

    **Actual:** Created `pyproject.toml` (modern Python packaging standard) as primary source, plus `requirements.txt` for legacy compatibility.

    -   `pyproject.toml` includes dependencies, dev dependencies, build config, and tool settings
    -   `requirements.txt` generated for backwards compatibility
    -   Using `uv` for fast dependency management

-   [x] Create `.gitignore`:

    ```
    # Python
    __pycache__/
    *.py[cod]
    *$py.class
    .venv/
    venv/
    ENV/

    # ML artifacts
    models/*.h5
    models/*.pkl
    models/*.json

    # Data
    data/events-raw-fb.json

    # IDE
    .vscode/
    .idea/
    .DS_Store

    # Jupyter
    .ipynb_checkpoints/
    *.ipynb_checkpoints

    # Logs
    *.log
    ```

-   [x] Create basic `ml_event_tagger/__init__.py`:

    **Actual:** Created with dynamic version reading from `pyproject.toml`:

    ```python
    """ML Event Tagger - Multi-label event classification service."""

    from importlib.metadata import version
    __version__ = version("ml-event-tagger")
    ```

    **Note:** Version is tracked in `pyproject.toml` (single source of truth) and read dynamically by `__init__.py`. Update `pyproject.toml` version when releasing.

-   [x] Create `ml_event_tagger/config.py` with tag list:

    **Actual:** Created with 21 tags (removed city-specific tags, added venue/performer/access tags):

    ```python
    """Configuration and constants."""

    # Tag taxonomy (21 tags)
    TAGS = [
        "music", "house", "techno", "breaks", "jazz", "rock", "punk", "hiphop", "dj", "band",
        "dance", "yoga", "art", "food",
        "outdoor", "indoor", "public", "private", "free",
        "weekly", "community"
    ]

    # Model hyperparameters (streamlined for MVP)
    MAX_VOCAB_SIZE = 10000
    EMBEDDING_DIM = 64
    MAX_SEQUENCE_LENGTH = 200
    BATCH_SIZE = 16
    EPOCHS = 50
    ```

-   [x] Set up virtual environment:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    ```

**Success Criteria:**

-   ✅ Directory structure matches plan
-   ✅ All dependencies install without errors
-   ✅ Can import `ml_event_tagger` package
-   ✅ Git ignores models and virtual environment

---

## Phase 2: Data Preparation ✅ Complete (v0.0.2)

**Goal:** Create labeled dataset with 20 events (validation), scale to 100.

### Tasks

-   [x] Review existing CMF events in `data/events-raw-fb.json`

-   [x] Finalize tag taxonomy (see TAGS.md)

-   [x] Label initial 20 events:

    -   Create `data/labeled_events.json` with structure:
        ```json
        [
            {
                "id": "event_id",
                "name": "Event name",
                "description": "Event description",
                "location": "Location address",
                "tags": ["tag1", "tag2", "tag3"]
            }
        ]
        ```
    -   Focus on diverse event types
    -   Apply 2-5 tags per event
    -   Ensure each tag appears at least 3-4 times

-   [x] Test preprocessing on 20 events

-   [x] Validate pipeline works with small dataset

-   [x] Continue labeling to 50 events

-   [x] Continue labeling to 100 events

-   [x] Analyze tag distribution:
    -   Count events per tag
    -   Identify rare tags (<5 occurrences)
    -   Check for balance

**Success Criteria:**

-   ✅ 100 labeled events in JSON format
-   ✅ Each tag appears at least 5 times
-   ✅ Average 2-5 tags per event
-   ✅ Tag distribution is reasonable (not 90% one tag)

**Time Estimate:**

-   20 events: ~1 hour
-   50 events: ~2.5 hours
-   100 events: ~5 hours total

---

## Phase 3: Preprocessing Pipeline ✅ Complete (v0.0.3)

**Goal:** Build text preprocessing functions.

### Tasks

-   [x] Create `ml_event_tagger/preprocess.py`:

    ```python
    """Text preprocessing utilities."""
    import re
    from typing import List

    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def combine_event_fields(event: dict) -> str:
        """Combine event fields into single text."""
        parts = [
            event.get('name', ''),
            event.get('description', ''),
            event.get('location', '')
        ]
        return ' '.join(filter(None, parts))

    def preprocess_events(events: List[dict]) -> List[str]:
        """Preprocess all events."""
        texts = []
        for event in events:
            combined = combine_event_fields(event)
            cleaned = clean_text(combined)
            texts.append(cleaned)
        return texts
    ```

-   [x] Test preprocessing:

    ```python
    from ml_event_tagger.preprocess import preprocess_events
    import json

    with open('data/labeled_events.json') as f:
        events = json.load(f)

    texts = preprocess_events(events)
    print(texts[0])  # Should be cleaned text
    ```

-   [x] Add unit tests in `tests/test_preprocess.py`:

    ```python
    from ml_event_tagger.preprocess import clean_text

    def test_clean_text_lowercase():
        assert clean_text("HELLO") == "hello"

    def test_clean_text_removes_urls():
        text = "Check https://example.com for info"
        assert "https" not in clean_text(text)
    ```

**Success Criteria:**

-   ✅ Preprocessing functions work on sample data
-   ✅ URLs and HTML tags are removed
-   ✅ Text is properly normalized
-   ✅ Unit tests pass

---

## Phase 4: Model Training ✅ Complete (v0.0.4)

**Goal:** Train multi-label classifier and evaluate performance.

### Tasks

-   [x] Create `ml_event_tagger/model.py`:

    ```python
    """Model architecture definition."""
    from tensorflow import keras
    from tensorflow.keras import layers

    def create_model(vocab_size: int, embedding_dim: int,
                     num_tags: int) -> keras.Model:
        """Create Sequential model for multi-label classification."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_tags, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['precision', 'recall']
        )

        return model
    ```

-   [x] Create `ml_event_tagger/train.py`:

    -   Load labeled events
    -   Preprocess text
    -   Create train/val/test splits (70/15/15)
    -   Fit tokenizer on training data
    -   Build model
    -   Train with early stopping
    -   Evaluate on test set
    -   Save model, tokenizer, metrics

-   [x] Create training notebook `notebooks/01_train_and_evaluate.ipynb`:

    **Actual:** Created comprehensive 12-section notebook (20 KB):

    -   Section 1-6: Load data, tokenize, load model, evaluate, make predictions
    -   Section 7: Training history visualization (4 plots: loss, accuracy, precision, recall)
    -   Section 8: Per-tag performance analysis (precision/recall/F1 table)
    -   Section 9: Per-tag precision/recall bar chart
    -   Section 10: Tag frequency distribution
    -   Section 11: Sample predictions with confidence scores
    -   Section 12: Summary and conclusions

    **Setup:** Added `jupyter` and `ipykernel` to dev dependencies. Created dedicated Jupyter kernel for project.

    **Usage:** In Jupyter, select "Python (ml-event-tagger)" kernel to use project dependencies.

    **Note:** Goes beyond `train.py` with interactive exploration, per-tag breakdown, and sample predictions.

-   [x] Run training:

    ```bash
    python -m ml_event_tagger.train
    ```

-   [x] Review outputs:
    -   `models/tagger_v1_YYYYMMDD.h5` exists
    -   `models/tokenizer_v1.pkl` exists
    -   `models/metrics_v1.json` contains precision/recall
    -   Model achieves ≥60% precision

**Success Criteria:**

-   ✅ Training runs without errors
-   ✅ Model achieves ≥60% macro-averaged precision
-   ✅ Evaluation plots are generated
-   ✅ Model and tokenizer are saved
-   ✅ Metrics are documented

---

## Phase 5: API Service ✅ Complete (v0.0.5)

**Goal:** Build FastAPI service for inference.

### Tasks

-   [x] Create `ml_event_tagger/serve.py`:

    **Actual:** Created comprehensive FastAPI service (330 lines):

    -   Pydantic models for request/response validation
    -   Model and tokenizer loading on startup
    -   `/health` endpoint with version and model status
    -   `/predict` endpoint with top-5 predictions and confidence scores
    -   Auto-generated API docs at `/docs` and `/redoc`
    -   Comprehensive error handling

    ```python
    """FastAPI service for event tag prediction."""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List
    import tensorflow as keras
    import pickle
    from ml_event_tagger import __version__

    app = FastAPI(title="ML Event Tagger", version=__version__)

    # Load model and tokenizer at startup
    model = None
    tokenizer = None

    @app.on_event("startup")
    async def load_model():
        global model, tokenizer
        model = keras.models.load_model("models/tagger_v1.h5")
        with open("models/tokenizer_v1.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    class Event(BaseModel):
        name: str
        description: str = ""
        location: str = ""

    class PredictRequest(BaseModel):
        events: List[Event]

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "version": __version__
        }

    @app.post("/predict")
    async def predict(request: PredictRequest):
        # Preprocess events
        # Tokenize and pad
        # Predict
        # Return top-5 tags per event
        pass  # Implement
    ```

-   [x] Implement predict endpoint logic:

    **Actual:** Fully implemented with:

    -   Text preprocessing using existing `clean_text` and `combine_text_fields` functions
    -   Custom tokenization using saved tokenizer config
    -   Sequence padding to model's expected length
    -   Predictions sorted by confidence
    -   Top-5 tags returned per event

-   [x] Test locally - Created `test_api.py` with automated tests

    ```bash
    # view details from test demonstrating its working
    pytest tests/test_api.py -v -s
    ```

-   [x] Test endpoints - All tests passing:

    -   ✅ Health check returns status, model_loaded, and version
    -   ✅ Root endpoint provides API information
    -   ✅ Prediction endpoint handles single and multiple events
    -   ✅ Response format matches API contract

-   [x] Verify response format matches spec - Confirmed

**Success Criteria:**

-   ✅ API starts without errors
-   ✅ /health returns status
-   ✅ /predict accepts requests and returns predictions
-   ✅ Response format matches API contract
-   ✅ Inference time <300ms

---

## Phase 6: Testing & Validation

**Goal:** Verify end-to-end functionality.

### Tasks

-   [x] Create `tests/test_serve.py`:

    ```python
    from fastapi.testclient import TestClient
    from ml_event_tagger.serve import app

    client = TestClient(app)

    def test_health():
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict():
        request = {
            "events": [{
            "name": "House Music Night",
            "description": "Dance party",
            "location": "Oakland, CA"
            }]
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 200
        assert "predictions" in response.json()
    ```

-   [x] Run tests: `pytest tests/test_serve.py -v`

-   [x] 25 comprehensive unit tests covering:

    -   Health endpoint (4 tests)
    -   Root endpoint (3 tests)
    -   Predict endpoint (6 tests)
    -   Input validation (5 tests)
    -   Performance benchmarks (2 tests)
    -   Confidence score validation (3 tests)
    -   API documentation (3 tests)

-   [x] Manual testing:

    -   Test with various event types
    -   Verify predictions make sense
    -   Check edge cases (empty description, etc.)

-   [x] Measure inference latency:
    -   Single event: <300ms ✅
    -   Batch (10 events): <1000ms ✅

**Success Criteria:**

-   ✅ All 25 tests pass
-   ✅ Manual predictions are reasonable
-   ✅ Inference latency meets target (<300ms for single event)
-   ✅ No crashes or errors

**Actual:**

-   `tests/test_serve.py` - 25 tests, 350+ lines
-   Comprehensive test coverage: health checks, predictions, validation, performance, serialization
-   FastAPI TestClient with lifespan context manager for proper model loading
-   All tests passing with <5s total runtime

---

## Phase 7: Docker & Deployment

**Goal:** Create deployable container.

### Tasks

-   [x] Create `Dockerfile`

-   [x] Test Docker build:

    ```bash
    docker build -t ml-event-tagger:0.0.7 .
    # Or use the test script
    ./docker-test.sh
    ```

-   [x] Test Docker run:

    ```bash
    docker run -p 8000:8000 ml-event-tagger:0.0.7
    ```

-   [x] Test containerized API:

    ```bash
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"events": [{"name": "Test Event", "description": "Testing", "location": "SF"}]}'
    ```

-   [x] Choose deployment platform (Render, Fly.io, or Hugging Face) - Render chosen

-   [ ] Deploy to chosen platform

-   [ ] Test live endpoint

-   [ ] Update README with live demo link

**Success Criteria:**

-   ✅ Docker builds successfully
-   ✅ Container runs locally
-   ✅ API works in container
-   ✅ Health check passes
-   ✅ Deployment succeeds

**Actual:**

-   [Dockerfile](../Dockerfile) - Multi-stage build (builder + runtime), 300MB final image size
-   `.dockerignore` - Excludes unnecessary files (tests, docs, raw data)
-   `docker-test.sh` - Automated testing script for Docker build and validation
-   Features: Non-root user, health check, optimized caching, uv for fast installs

---

## Phase 8: Documentation Polish

**Goal:** Final documentation review and cleanup.

### Tasks

-   [ ] Update README with:

    -   Actual model metrics achieved
    -   Confirmed setup time
    -   Working curl examples

-   [ ] Add actual metrics to ARCHITECTURE.md

-   [ ] Review all docs for consistency

-   [ ] Add screenshots or plots (optional):

    -   Confusion matrix
    -   Precision/recall charts
    -   Tag distribution

-   [ ] Create CHANGELOG.md:

    ```markdown
    # Changelog

    ## v0.1.0 - MVP Release (YYYY-MM-DD)

    Initial release demonstrating TensorFlow/Keras + FastAPI integration.

    ### Features

    -   Multi-label event classification with 15-20 tags
    -   Trained on 100 labeled CMF events
    -   FastAPI service with /predict and /health endpoints
    -   Macro-averaged precision: X.XX%
    -   Docker deployment ready
    ```

-   [ ] Final review checklist:
    -   [ ] All code has docstrings
    -   [ ] No hardcoded paths
    -   [ ] requirements.txt is complete
    -   [ ] .gitignore covers artifacts
    -   [ ] README quick start works
    -   [ ] All links in docs are valid

**Success Criteria:**

-   ✅ Documentation is complete and accurate
-   ✅ All examples work as documented
-   ✅ Project is ready to share

---

## 🎯 Definition of Done

The MVP is complete when:

-   ✅ All 8 phases are marked complete
-   ✅ Model achieves ≥60% precision on test set
-   ✅ API responds <300ms per event
-   ✅ Docker container builds and runs
-   ✅ README quick start works (verified fresh clone)
-   ✅ All tests pass
-   ✅ Documentation is accurate and complete

---

## 🐛 Troubleshooting

Common issues and solutions:

### Training fails with OOM error

-   Reduce batch size to 8
-   Reduce vocabulary size to 3000
-   Reduce max sequence length to 80

### Model underfits (<50% precision)

-   Label more events (aim for 150)
-   Increase training epochs
-   Check data quality (are labels correct?)
-   Ensure preprocessing isn't removing too much

### API is slow (>500ms)

-   Check if model is loaded at startup (not per request)
-   Profile preprocessing step
-   Consider reducing model complexity

### Docker build fails

-   Check all files are copied in Dockerfile
-   Verify requirements.txt has all dependencies
-   Ensure models/ directory exists and has artifacts

---

## 📚 Next Steps After MVP

Once v0.1 is complete:

1. Deploy to live URL
2. Share with potential employers/collaborators
3. Gather feedback
4. Plan v0.2 improvements (see ROADMAP.md)
5. Consider blog post explaining the project

---

## 🧩 Related Docs

-   [MVP_DECISIONS.md](./MVP_DECISIONS.md) - Architectural decisions
-   [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical design
-   [ROADMAP.md](./ROADMAP.md) - Future plans
-   [TAGS.md](./TAGS.md) - Tag taxonomy
