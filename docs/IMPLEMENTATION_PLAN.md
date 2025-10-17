# 🛠️ Implementation Plan - v0.1 MVP

Step-by-step guide for implementing the ml-event-tagger MVP.

**Target:** Working end-to-end ML service demonstrating TensorFlow/Keras and FastAPI integration.

---

## 📋 Overview

| Phase     | Focus                  | Estimated Time  | Status         |
| --------- | ---------------------- | --------------- | -------------- |
| 1         | Project Setup          | 1-2 hours       | ⬜ Not Started |
| 2         | Data Preparation       | 3-5 hours       | ⬜ Not Started |
| 3         | Preprocessing Pipeline | 2-3 hours       | ⬜ Not Started |
| 4         | Model Training         | 3-4 hours       | ⬜ Not Started |
| 5         | API Service            | 2-3 hours       | ⬜ Not Started |
| 6         | Testing & Validation   | 1-2 hours       | ⬜ Not Started |
| 7         | Docker & Deployment    | 1-2 hours       | ⬜ Not Started |
| 8         | Documentation Polish   | 2-3 hours       | ⬜ Not Started |
| **Total** | **End-to-End**         | **15-25 hours** |                |

---

## Phase 1: Project Setup

**Goal:** Create clean repository structure with all necessary files.

### Tasks

-   [ ] Create directory structure:

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

-   [ ] Create `requirements.txt`:

    ```
    tensorflow>=2.13.0,<2.16.0
    fastapi>=0.104.0
    uvicorn[standard]>=0.24.0
    pandas>=2.0.0
    scikit-learn>=1.3.0
    numpy>=1.24.0
    matplotlib>=3.7.0
    seaborn>=0.12.0
    python-dotenv>=1.0.0
    pydantic>=2.0.0
    ```

-   [ ] Create `.gitignore`:

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

-   [ ] Create basic `ml_event_tagger/__init__.py`:

    ```python
    """ML Event Tagger - Multi-label event classification service."""

    __version__ = "0.1.0"  # Version tracked here (source of truth)
    ```

    **Note:** Version is tracked in `__init__.py` following Python convention. Update both here and CHANGELOG.md when releasing.

-   [ ] Create `ml_event_tagger/config.py` with tag list:

    ```python
    """Configuration and constants."""

    # Tag taxonomy (19 tags)
    TAGS = [
        "music", "house", "techno", "breaks", "jazz", "rock", "punk", "hiphop", "dj", "band",
        "dance", "yoga", "art", "food",
        "outdoor", "indoor", "public", "private", "free",
        "weekly", "community"
    ]

    # Model hyperparameters
    MAX_VOCAB_SIZE = 5000
    EMBEDDING_DIM = 64
    MAX_SEQUENCE_LENGTH = 100
    DENSE_UNITS = 32
    BATCH_SIZE = 16
    EPOCHS = 30

    # Training parameters
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    ```

-   [ ] Set up virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

**Success Criteria:**

-   ✅ Directory structure matches plan
-   ✅ All dependencies install without errors
-   ✅ Can import `ml_event_tagger` package
-   ✅ Git ignores models and virtual environment

---

## Phase 2: Data Preparation

**Goal:** Create labeled dataset with 20 events (validation), scale to 100.

### Tasks

-   [ ] Review existing CMF events in `data/events-raw-fb.json`

-   [ ] Finalize tag taxonomy (see TAGS.md)

-   [ ] Label initial 20 events:

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

-   [ ] Test preprocessing on 20 events:

    ```bash
    python -c "
    import json
    with open('data/labeled_events.json') as f:
        events = json.load(f)
    print(f'Loaded {len(events)} events')
    "
    ```

-   [ ] Validate pipeline works with small dataset

-   [ ] Continue labeling to 50 events

-   [ ] Continue labeling to 100 events

-   [ ] Analyze tag distribution:
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

## Phase 3: Preprocessing Pipeline

**Goal:** Build text preprocessing functions.

### Tasks

-   [ ] Create `ml_event_tagger/preprocess.py`:

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

-   [ ] Test preprocessing:

    ```python
    from ml_event_tagger.preprocess import preprocess_events
    import json

    with open('data/labeled_events.json') as f:
        events = json.load(f)

    texts = preprocess_events(events)
    print(texts[0])  # Should be cleaned text
    ```

-   [ ] Add unit tests in `tests/test_preprocess.py`:

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

## Phase 4: Model Training

**Goal:** Train multi-label classifier and evaluate performance.

### Tasks

-   [ ] Create `ml_event_tagger/model.py`:

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

-   [ ] Create `ml_event_tagger/train.py`:

    -   Load labeled events
    -   Preprocess text
    -   Create train/val/test splits (70/15/15)
    -   Fit tokenizer on training data
    -   Build model
    -   Train with early stopping
    -   Evaluate on test set
    -   Save model, tokenizer, metrics

-   [ ] Create training notebook `notebooks/01_train_and_evaluate.ipynb`:

    -   Import and run training
    -   Visualize results:
        -   Training/validation loss curves
        -   Confusion matrix heatmap
        -   Per-tag precision/recall bar charts
        -   Tag frequency distribution
    -   Print summary metrics
    -   Save plots as images

-   [ ] Run training:

    ```bash
    python -m ml_event_tagger.train
    ```

-   [ ] Review outputs:
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

## Phase 5: API Service

**Goal:** Build FastAPI service for inference.

### Tasks

-   [ ] Create `ml_event_tagger/serve.py`:

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

-   [ ] Implement predict endpoint logic:

    -   Preprocess input events
    -   Tokenize and pad sequences
    -   Get model predictions
    -   Sort by confidence
    -   Return top-5 tags per event

-   [ ] Test locally:

    ```bash
    uvicorn ml_event_tagger.serve:app --reload
    ```

-   [ ] Test endpoints:

    ```bash
    # Health check
    curl http://localhost:8000/health

    # Prediction
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"events":[{"name":"Test Event","description":"Test","location":"Oakland"}]}'
    ```

-   [ ] Verify response format matches spec

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

-   [ ] Create `tests/test_api.py`:

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

-   [ ] Run tests:

    ```bash
    pytest tests/
    ```

-   [ ] Manual testing:

    -   Test with various event types
    -   Verify predictions make sense
    -   Check edge cases (empty description, etc.)

-   [ ] Measure inference latency:

    ```python
    import time
    # Test 100 predictions, calculate p95 latency
    ```

-   [ ] Document actual metrics in README

**Success Criteria:**

-   ✅ All tests pass
-   ✅ Manual predictions are reasonable
-   ✅ Inference latency meets target (<300ms)
-   ✅ No crashes or errors

---

## Phase 7: Docker & Deployment

**Goal:** Create deployable container.

### Tasks

-   [ ] Create `Dockerfile`:

    ```dockerfile
    FROM python:3.11-slim

    WORKDIR /app

    # Install dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application
    COPY ml_event_tagger/ ./ml_event_tagger/
    COPY models/ ./models/

    # Expose port
    EXPOSE 8080

    # Run server
    CMD ["uvicorn", "ml_event_tagger.serve:app", "--host", "0.0.0.0", "--port", "8080"]
    ```

-   [ ] Test Docker build:

    ```bash
    docker build -t ml-event-tagger .
    ```

-   [ ] Test Docker run:

    ```bash
    docker run -p 8080:8080 ml-event-tagger
    ```

-   [ ] Test containerized API:

    ```bash
    curl http://localhost:8080/health
    ```

-   [ ] Choose deployment platform (Render, Fly.io, or Hugging Face)

-   [ ] Deploy to chosen platform

-   [ ] Test live endpoint

-   [ ] Update README with live demo link (optional)

**Success Criteria:**

-   ✅ Docker builds successfully
-   ✅ Container runs locally
-   ✅ API works in container
-   ✅ Deployment succeeds (optional for MVP)

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
