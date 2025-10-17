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

-   [x] Separate `ml-event-tagger` repo
-   [x] TensorFlow/Keras Sequential model for tag prediction
-   [x] Static labeled dataset (`data/labeled_events.json`)
-   [x] Training pipeline (`train.py` and notebook)
-   [x] FastAPI service with `/predict` endpoint
-   [x] Client-side CMF integration (background tag fetch)
-   [x] Architecture and documentation (README + ARCHITECTURE.md)

**Success Criteria:**

-   Model achieves ≥70% precision on validation data
-   API responds under 300ms per event
-   Tags are contextually relevant

### Suggestions

A small notebooks/01_train_event_tagger.ipynb to demonstrate metrics understanding that outputs:
-   Confusion matrix heatmap
-   Tag frequency bar chart
-   Precision@3 / Recall@3 printed nicely

---

## ⚙️ v0.2 — Data Expansion & Automation

**Goal:** Increase dataset quality, tagging precision, and operational polish.

**Planned Enhancements:**

-   [ ] Adapter to fetch live CMF events (`adapters/cmf/`)
-   [ ] Auto-label bootstrapper using regex or keyword heuristics
-   [ ] Caching layer (Redis or Supabase table)
-   [ ] Tag taxonomy versioning
-   [ ] Metrics dashboard (FastAPI `/metrics` or notebook charts)
-   [ ] Add evaluation notebook with precision/recall plots

**Target Outcome:** Sustainable dataset + visible model improvement process.

### Suggestions

FastAPI Testing & Demo UX. After deployment:

- Add a /docs route (FastAPI auto-generates Swagger UI — free win).
- Include a short curl or Postman example in the README.
- Optionally embed a link like:
  Try it live → ml-event-tagger.fly.dev/docs

---

## 🧠 v0.3 — Smarter Models & Retraining

**Goal:** Improve semantic understanding and add retraining automation.

**Planned Enhancements:**

-   [ ] Replace basic embedding with TF-Hub sentence encoder (`nnlm-en-dim128`)
-   [ ] Add human feedback loop for tag corrections
-   [ ] Implement model versioning (`models/tagger.v1.h5`, etc.)
-   [ ] Automate retraining pipeline triggered by dataset changes
-   [ ] Store evaluation metrics in JSON for comparison over time

**Target Outcome:** Continuous improvement pipeline with measurable progress.

---

## 🌐 v0.4 — Broader Data Sources & Open Release

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
| v0.2    | Data + Metrics + Caching            | ⏳ Planned |
| v0.3    | Smart Models + Feedback Loop        | 🔜 Next    |
| v0.4    | Multi-Source Adapters               | 🧭 Future  |
| v1.0    | Production Quality + Explainability | 🧭 Future  |

---

## 🧩 Related Docs

-   [README.md](../README.md)
-   [ARCHITECTURE.md](../docs/ARCHITECTURE.md)
