# 🗺️ ML Event Tagger — Roadmap

This roadmap outlines the planned evolution of the **ml-event-tagger** project — from initial MVP to advanced AI/ML service.

---

## 🎯 Overall Vision

Create a robust, general-purpose event tagging microservice that:

-   Demonstrates practical TensorFlow/Keras and FastAPI integration
-   Starts with CMF event data but supports multiple data sources
-   Evolves toward automated, intelligent, and explainable tagging

---

## 🚀 v0.1 — MVP (Current Phase)

**Goal:** Ship a working demo that clearly demonstrates AI/ML, TensorFlow/Keras, and full-stack integration skills.

**Deliverables:**

-   [x] Separate `ml-event-tagger` repo with clean structure
-   [x] TensorFlow/Keras Sequential model for tag prediction
-   [x] Labeled dataset (start with 20 events for pipeline validation, scale to 100)
-   [x] Training pipeline (`train.py` and notebook with evaluation plots)
-   [x] FastAPI service with `/predict` and `/health` endpoints
-   [x] Basic unit tests using `uv run pytest`
-   [x] Documentation (README, ARCHITECTURE, implementation guides)
-   [x] Docker deployment setup

**Success Criteria:**

-   Model achieves ≥60% macro-averaged precision on validation data
-   API responds under 300ms per event (p95 latency)
-   Working end-to-end pipeline (train → save → serve → predict)
-   Clone to first prediction: ~10 minutes

**Evaluation Outputs:**

Training notebook demonstrates metrics understanding:

-   Confusion matrix heatmap
-   Tag frequency bar chart
-   Precision/Recall per tag
-   Model performance summary

Completed Successfully. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details on the 8 phases.

---

## ⚙️ v0.2 — Data Expansion & Automation

**Goal:** Increase dataset quality, tagging precision, and operational polish.

**Planned Enhancements:**

-   [ ] Expand labeled dataset to 200-300 events
-   [ ] Auto-label bootstrapper using regex or keyword heuristics
-   [ ] Tag taxonomy versioning
-   [ ] Adapter to fetch live CMF events (`adapters/cmf/`)

**Target Outcome:** Improved model accuracy through larger, higher-quality dataset.

### Documentation & Testing

-   [ ] FastAPI `/docs` route with Swagger UI
-   [ ] Integration tests with pytest
-   [ ] Performance benchmarking results

---

## 🔒 v0.3 — Performance & Security

**Goal:** Add production-ready infrastructure features.

**Planned Enhancements:**

-   [ ] API key authentication with `ML_API_KEY` and `x-api-key` header
-   [ ] CORS configuration for CMF domain
-   [ ] Rate limiting (per API key)
-   [ ] Caching layer (Redis or in-memory cache)
-   [ ] Basic monitoring and logging (request/error tracking)
-   [ ] Error tracking and alerting

**Target Outcome:** Production-ready API with proper security and performance optimizations.

**Note:** These features are infrastructure/operations focused, separate from core ML improvements.

---

## 🧠 v0.4 — Smarter Models & Retraining

**Goal:** Improve semantic understanding and add retraining automation.

**Planned Enhancements:**

-   [ ] Replace basic embedding with TF-Hub sentence encoder (`nnlm-en-dim128`)
-   [ ] Add human feedback loop for tag corrections
-   [ ] Implement model versioning (`models/tagger.v1.h5`, etc.)
-   [ ] Automate retraining pipeline triggered by dataset changes
-   [ ] Store evaluation metrics in JSON for comparison over time

**Target Outcome:** Continuous improvement pipeline with measurable progress.

---

## 🌐 v0.5 — Broader Data Sources & Open Release

**Goal:** Broaden applicability beyond CMF, with clear extensibility hooks.

**Planned Enhancements:**

-   [ ] Add adapters for other event sources (Meetup, Eventbrite)
-   [ ] Add configuration-driven preprocessing per adapter
-   [ ] Optional open dataset export with tag labels
-   [ ] Publish minimal web demo or Hugging Face Space

**Target Outcome:** Publicly accessible event tagging API demonstrating professional ML engineering design.

---

## 📊 v1.0 — Production-Ready & Insight Features

**Goal:** Reach a stable, production-grade release suitable for integration in real apps.

**Planned Enhancements:**

-   [ ] Retraining via CI/CD pipeline
-   [ ] Tag co-occurrence and clustering visualization dashboard
-   [ ] Model explainability using SHAP or integrated gradients
-   [ ] TensorFlow.js or ONNX export for edge inference
-   [ ] Comprehensive unit + integration tests

**Target Outcome:** Reliable, well-instrumented ML microservice with human-interpretable outputs.

---

## 🧭 Long-Term Vision

-   Fully automated tagging service deployable as a managed API
-   Configurable tag vocabularies per client (CMF, etc.)
-   Unified tagging + recommendation system for event discovery
-   Dataset and model shared as open research artifact

---

## 📅 Timeline Snapshot

| Version | Focus                               | Status     |
| ------- | ----------------------------------- | ---------- |
| v0.1    | MVP — Model + API + Docs            | ✅ Current |
| v0.2    | Data Expansion                      | ⏳ Planned |
| v0.3    | Performance & Security              | ⏳ Planned |
| v0.4    | Smart Models + Feedback Loop        | 🔜 Next    |
| v0.5    | Multi-Source Adapters               | 🧭 Future  |
| v1.0    | Production Quality + Explainability | 🧭 Future  |

---

## 🧩 Related Docs

-   [README.md](../README.md)
-   [ARCHITECTURE.md](./ARCHITECTURE.md)
-   [MVP_DECISIONS.md](./MVP_DECISIONS.md)
-   [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
-   [TAGS.md](./TAGS.md)
