# 🧠 ML Event Tagger

**Machine-learning microservice for tagging event metadata using TensorFlow/Keras.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-success)](https://ml-event-tagger.onrender.com)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Originally designed for [CMF](https://github.com/chadn/cmf) event data but structured for general event tagging tasks.

**Goal:** Demonstrate TensorFlow/Keras and full-stack ML integration skills with clarity and reproducibility.

**Time to first prediction:** ~10 minutes from clone to working API.

**Live Demo:** https://ml-event-tagger.onrender.com

---

## 🎯 Skills Demonstrated

This project showcases practical AI/ML engineering skills:

| Skill Area          | What's Demonstrated                                                                  |
| ------------------- | ------------------------------------------------------------------------------------ |
| **Data Pipeline**   | Text preprocessing, feature extraction, labeled dataset creation                     |
| **Model Training**  | TensorFlow/Keras Sequential model, multi-label classification, train/val/test splits |
| **Evaluation**      | Precision/Recall metrics, confusion matrix, model performance visualization          |
| **API Development** | FastAPI service, model serving, REST endpoint design                                 |
| **Integration**     | Client-ready API for real-world applications (CMF event map)                         |
| **Reproducibility** | Clean code, documentation, Docker deployment, version control                        |

**Built with:** Python, TensorFlow/Keras, FastAPI, scikit-learn, pandas
**Test Coverage:** 75 tests, 46% overall (80%+ production code)

---

## 🧩 Key Features

-   ✅ **TensorFlow/Keras** Sequential model for multi-label text classification
-   ✅ **FastAPI** inference service with `/predict` and `/health` endpoints
-   ✅ Clean separation between **training** and **serving** pipelines
-   ✅ **Evaluation notebook** with precision/recall plots and confusion matrix
-   ✅ **Docker deployment** ready for Render, Fly.io, or Hugging Face Spaces
-   ✅ Designed to integrate with **Next.js** or any REST client

---

## 🚀 Quick Start

### 1️⃣ Setup Environment

```bash
uv venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"
```

### 2️⃣ Train Model

```bash
python -m ml_event_tagger.train
```

Outputs:

-   `models/tagger_v1_YYYYMMDD.h5` (model weights)
-   `models/tokenizer_v1.pkl` (tokenizer)
-   `models/metrics_v1.json` (evaluation results)

### 3️⃣ Serve API

```bash
uvicorn ml_event_tagger.serve:app --reload
```

API available at `http://localhost:8000`

### 4️⃣ Test Inference

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "name": "Days Like This - House Music",
      "description": "Weekly house music gathering",
      "location": "The Pergola at Lake Merritt, 599 El Embarcadero, Oakland, CA 94610"
    }]
  }'
```

### 🧠 Example Response

```json
{
    "predictions": [
        {
            "tags": [
                { "name": "music", "confidence": 0.92 },
                { "name": "house", "confidence": 0.87 },
                { "name": "oakland", "confidence": 0.86 },
                { "name": "dance", "confidence": 0.78 },
                { "name": "weekly", "confidence": 0.65 }
            ]
        }
    ]
}
```

### 5️⃣ Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=ml_event_tagger --cov-report=term-missing

# 75 tests total
# - 14 model tests
# - 25 preprocessing tests
# - 36 API/serve tests
# Coverage: 46% overall, 80%+ production code
```

---

## 🏷️ Data & Labeling

**Tag Taxonomy:** ~20 tags covering event categories:

-   **Music & performers:** music, house, techno, breaks, jazz, rock, punk, hiphop, dj, band
-   **Activities:** dance, yoga, art, food
-   **Access & venue:** outdoor, indoor, public, private, free
-   **Other:** weekly, community

See [docs/TAGS.md](docs/TAGS.md) for complete list and definitions.

**Labeled Dataset:** ~100 events manually tagged with 2-5 tags each.

-   Started with 20 events for pipeline validation
-   Scaled to 100 for robust model training
-   Fields used: `name + description + location`

---

## 🧩 Integration with CMF

The CMF client (Next.js on Vercel) can:

-   Fetch events from `/api/events`
-   Then asynchronously call this service’s `/predict` endpoint
-   Merge returned tags for display or filtering

This allows events to appear immediately on the CMF map, while tags load in the background.

---

## 🧰 Deployment

### 🌐 Live Demo

The API is deployed on Render (free tier):

**URL:** https://ml-event-tagger.onrender.com

```bash
# Test health endpoint
curl https://ml-event-tagger.onrender.com/health

# Test prediction
curl -X POST https://ml-event-tagger.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "name": "House Music Night",
      "description": "DJ performance with dancing",
      "location": "Oakland, CA"
    }]
  }'
```

⚠️ **Note:** Free tier spins down after 15 min of inactivity. First request may take ~30-60 seconds (cold start).

### 🐳 Docker

Build and run locally:

```bash
docker build -t ml-event-tagger:latest .
docker run -p 8000:8000 ml-event-tagger:latest
```

### 🚀 Deploy to Render

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for complete deployment guide.

Quick deploy:

1. Push to GitHub
2. Connect to Render (auto-detects `render.yaml`)
3. Deploy (builds in ~5-10 minutes)

---

## 📊 Model Performance

**Achieved Metrics (Test Set):**

-   ✅ **Binary Accuracy:** 82.9%
-   ✅ **Precision:** 73.3% (macro-averaged)
-   ✅ **Recall:** 44.0% (macro-averaged)
-   ✅ **F1 Score:** 55.0%
-   ✅ **Inference Latency:** <300ms per event (p95)

**Trained on:**

-   100 labeled events
-   70/15/15 train/val/test split
-   21 event tags (music genres, activities, venue types)

See [`notebooks/01_train_and_evaluate.ipynb`](notebooks/01_train_and_evaluate.ipynb) for detailed evaluation with:

-   Confusion matrix heatmap
-   Per-tag precision/recall charts
-   Training/validation loss curves
-   Tag frequency distribution
-   Sample predictions with confidence scores

---

## 📚 Documentation

-   [**LEARNINGS.md**](docs/LEARNINGS.md) - 📚 **Summary of demonstrated ML & engineering skills** ⭐
-   [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) - Technical design, data pipeline, model training
-   [**TAGS.md**](docs/TAGS.md) - Tag taxonomy and labeling guideliness
-   [ROADMAP.md](docs/ROADMAP.md) - Project evolution and future plans
-   [TEST_COVERAGE_PLAN.md](docs/TEST_COVERAGE_PLAN.md) - Testing strategy & rationale
-   [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Docker & Render deployment guide
-   [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) - 8-phase MVP implementation (completed)
-   [MVP_DECISIONS.md](docs/MVP_DECISIONS.md) - Architectural decisions and rationale

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🧭 Author

**Chad Norwood**
[chadnorwood.com](https://chadnorwood.com) | [github.com/chadn](https://github.com/chadn)
