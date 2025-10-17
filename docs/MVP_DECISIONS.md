# 🎯 MVP Decisions & Rationale

This document captures the key architectural and scope decisions made for the v0.1 MVP of **ml-event-tagger**.

**Last updated:** October 17, 2025

---

## 🧭 Core Philosophy

**Primary Goal:** Demonstrate TensorFlow/Keras and full-stack ML integration skills through a working, reproducible project.

**Key Principles:**

-   **Clarity over complexity** - Simple, well-documented solutions that showcase understanding
-   **Working over perfect** - End-to-end functionality matters more than state-of-the-art performance
-   **Reproducible over elaborate** - Anyone should be able to clone and run in ~10 minutes
-   **Demonstrable over theoretical** - Include visualizations and metrics to prove understanding

---

## 📊 Dataset Decisions

### Size & Scope

-   **Target:** 100 labeled events
-   **MVP strategy:** Start with 20 events to validate pipeline, scale to 100
-   **Rationale:**
    -   Large enough to train a meaningful model
    -   Small enough to label manually in a few hours
    -   Demonstrates data preparation workflow without overengineering

### Data Source

-   **Primary:** CMF event data (Google Calendar events for Oakland/Bay Area)
-   **Format:** JSON with fields: name, description, formatted_address
-   **Rationale:** Real-world data from actual application, demonstrating practical use case

### Fields Used for Training

-   **Selected:** `name + description + formatted_address`
-   **Excluded:** Google Maps types, URLs, timestamps
-   **Rationale:**
    -   These three fields contain the most semantic information
    -   Location helps disambiguate event types (e.g., "Oakland" vs "SF")
    -   Keeps preprocessing simple and focused

---

## 🏷️ Tag Taxonomy

### Size

-   **Count:** 15-20 tags
-   **Rationale:** Manageable to label consistently, diverse enough to show multi-label classification

### Categories

1. **Music genres:** music, house, techno, jazz, classical, rock
2. **Activities:** dance, yoga, art, food, market
3. **Locations:** oakland, sf, berkeley
4. **Characteristics:** outdoor, weekly, community, family

### Labeling Guidelines

-   **Tags per event:** 2-5 tags typical
-   **Approach:** Label primary characteristics, not exhaustive
-   **Quality:** Consistency matters more than coverage
-   **See:** [TAGS.md](./TAGS.md) for complete taxonomy

---

## 🤖 Model Architecture

### Type

-   **Choice:** Keras Sequential model
-   **Architecture:** Embedding → GlobalAveragePooling1D → Dense(32, relu) → Dense(n_tags, sigmoid)
-   **Rationale:**
    -   Simple, well-understood architecture
    -   Fast to train on CPU
    -   Easy to explain and reproduce
    -   Sufficient for demonstrating multi-label classification

### Hyperparameters

-   **Vocabulary size:** 5000 tokens
-   **Embedding dimension:** 64
-   **Max sequence length:** 100 tokens
-   **Dense layer size:** 32 neurons
-   **Batch size:** 16
-   **Epochs:** 20-30 with early stopping
-   **Optimizer:** Adam (lr=0.001)
-   **Loss:** Binary crossentropy

**Rationale:** Standard hyperparameters that work well for small text classification tasks. Easy to tune later if needed.

---

## 📏 Training Strategy

### Data Split

-   **Train:** 70%
-   **Validation:** 15%
-   **Test:** 15%
-   **Method:** Stratified split to maintain tag distribution

**Rationale:** Industry-standard split showing understanding of validation practices. Proper train/val/test separation prevents overfitting.

### Evaluation Metrics

**Primary:**

-   Macro-averaged precision (≥60% target)
-   Macro-averaged recall (≥40% target)

**Secondary:**

-   Per-tag precision and recall
-   F1-score
-   Confusion matrix

**Rationale:** Macro-averaging treats all tags equally, important for imbalanced multi-label scenarios.

---

## ✅ Success Criteria

### 1. Model Performance

-   **Metric:** ≥60% macro-averaged precision on test set
-   **Why:** Demonstrates meaningful learning above baseline (random ~5-10%, most-common ~20-30%)
-   **Acceptable range:** 50-70% is realistic for 100 events

### 2. System Performance

-   **Metric:** <300ms inference time (p95 latency)
-   **Why:** Shows the model is lightweight and practical for real-time use
-   **Expected:** 50-100ms on CPU with this architecture

### 3. End-to-End Functionality

-   **Requirements:**
    -   Training script runs without errors
    -   Model saves and loads successfully
    -   API accepts requests and returns predictions
    -   Predictions are in correct format
    -   Docker container builds and runs
-   **Why:** Proves engineering maturity, not just ML experimentation

### 4. Reproducibility

-   **Metric:** Clone to first prediction in ~10 minutes
-   **Requirements:**
    -   Clear README with setup steps
    -   requirements.txt with pinned versions
    -   No manual configuration needed
    -   Example curl command works
-   **Why:** Demonstrates professional-level documentation and project structure

### 5. Code Quality

-   **Requirements:**
    -   Clean, readable code with docstrings
    -   Logical file organization
    -   No hardcoded paths or credentials
    -   Basic error handling
-   **Why:** Shows software engineering best practices

---

## 🚫 What We're Skipping for MVP

### Deferred to v0.2

-   ✨ API authentication (x-api-key header)
-   ✨ CORS configuration
-   ✨ Rate limiting
-   ✨ Monitoring and logging infrastructure
-   ✨ Caching layer (Redis)
-   ✨ Larger dataset (200-300 events)
-   ✨ Integration tests with pytest

### Deferred to v0.3+

-   ✨ TF-Hub sentence encoders (better embeddings)
-   ✨ Human feedback loop
-   ✨ Automated retraining pipeline
-   ✨ Model explainability (SHAP)
-   ✨ Multiple data source adapters
-   ✨ Tag taxonomy versioning

**Rationale:** Focus on core ML demonstration first. These features are valuable but not essential for demonstrating TensorFlow/Keras skills.

---

## 🧱 Repository Structure

### Package Organization

-   **Choice:** Organized Python package (`ml_event_tagger/`)
-   **Alternative considered:** Flat structure (train.py, serve.py in root)
-   **Rationale:**
    -   Shows proper Python project structure
    -   Makes imports clean and testable
    -   Scales better as project grows
    -   Demonstrates software engineering maturity

### Key Files

```
ml_event_tagger/
├── __init__.py          # Package marker
├── train.py             # Training logic
├── serve.py             # FastAPI app
├── preprocess.py        # Text preprocessing
├── model.py             # Model architecture
└── config.py            # Configuration & tags
```

**Rationale:** Clear separation of concerns, easy to test and maintain.

---

## 🔧 Technology Choices

### Core Stack

-   **Python 3.11+** - Modern Python with good type hints support
-   **TensorFlow/Keras** - Industry-standard ML framework, requirement for demonstrating skills
-   **FastAPI** - Modern, fast web framework with automatic OpenAPI docs
-   **scikit-learn** - For train/test splits and metrics
-   **pandas** - For data manipulation

### Deployment

-   **Docker** - Containerization for consistent deployment
-   **Target platforms:** Render, Fly.io, or Hugging Face Spaces
-   **Rationale:** Free tiers available, easy to deploy, professional practice

### Not Using (for MVP)

-   ❌ TensorFlow Serving - Overkill for demo
-   ❌ PostgreSQL - JSON file sufficient for 100 events
-   ❌ Redis - No caching needed in MVP
-   ❌ Kubernetes - Not needed for single-service deployment

---

## 📈 Expected Results

### Model Performance

-   **Realistic expectation:** 55-65% precision, 40-50% recall
-   **Best case:** 70%+ precision on well-represented tags
-   **Worst case:** 45-55% precision (still demonstrates learning)

### Development Time

-   **Phase 1 (Setup):** 1-2 hours
-   **Phase 2 (Data labeling):** 3-5 hours
-   **Phase 3 (Preprocessing):** 2-3 hours
-   **Phase 4 (Training):** 3-4 hours
-   **Phase 5 (API):** 2-3 hours
-   **Phase 6 (Testing):** 1-2 hours
-   **Phase 7 (Docker):** 1-2 hours
-   **Phase 8 (Documentation):** 2-3 hours
-   **Total:** ~15-25 hours

---

## 🎓 Learning Demonstrations

This MVP explicitly demonstrates understanding of:

1. **Data preparation:** Labeling, cleaning, preprocessing text
2. **Feature engineering:** Concatenating fields, tokenization
3. **Model selection:** Appropriate architecture for task
4. **Training:** Proper splits, validation, early stopping
5. **Evaluation:** Multiple metrics, visualizations, interpretation
6. **Deployment:** API design, model serving, containerization
7. **Engineering:** Code organization, documentation, reproducibility

**Result:** A portfolio piece that proves ML skills in a practical context.

---

## 🔄 Iteration Plan

1. **Build MVP** - Get everything working end-to-end
2. **Evaluate** - Measure actual performance vs targets
3. **Document results** - Add actual metrics to docs
4. **Share** - Deploy and add live demo link
5. **Iterate** - v0.2+ improvements based on learnings

---

## 📚 Related Docs

-   [ARCHITECTURE.md](./ARCHITECTURE.md) - Technical design
-   [ROADMAP.md](./ROADMAP.md) - Product evolution
-   [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) - Step-by-step guide
-   [TAGS.md](./TAGS.md) - Tag taxonomy
