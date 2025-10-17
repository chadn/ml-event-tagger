# 🧠 ML Event Tagger

**Machine-learning microservice for tagging event metadata.**
Originally designed for [CMF](https://cmf.chadnorwood.com) event data but structured for general event tagging tasks.

> **Goal:** Demonstrate TensorFlow/Keras and full-stack ML integration skills with clarity and reproducibility.
> **Time to first prediction:** ~10 minutes from clone to working API.

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

---

## 🧩 Key Features

-   ✅ **TensorFlow/Keras** Sequential model for multi-label text classification
-   ✅ **FastAPI** inference service with `/predict` and `/health` endpoints
-   ✅ Clean separation between **training** and **serving** pipelines
-   ✅ **Evaluation notebook** with precision/recall plots and confusion matrix
-   ✅ **Docker deployment** ready for Render, Fly.io, or Hugging Face Spaces
-   ✅ Designed to integrate with **Next.js** or any REST client

---

## 🧱 Repository Structure

```
ml-event-tagger/
├── ml_event_tagger/           # Main Python package
│   ├── __init__.py
│   ├── train.py               # Training script
│   ├── serve.py               # FastAPI app
│   ├── preprocess.py          # Text preprocessing utilities
│   ├── model.py               # Model architecture definition
│   └── config.py              # Configuration & tag taxonomy
├── notebooks/
│   └── 01_train_and_evaluate.ipynb  # Training + evaluation plots
├── models/                     # Saved models (gitignored)
├── data/
│   └── labeled_events.json    # Labeled training data (~100 events)
├── tests/
│   └── test_api.py            # API tests
├── docs/
│   ├── ARCHITECTURE.md        # Technical design
│   ├── ROADMAP.md             # Product evolution
│   ├── MVP_DECISIONS.md       # Architectural decisions
│   ├── IMPLEMENTATION_PLAN.md # Step-by-step guide
│   └── TAGS.md                # Tag taxonomy
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Setup Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
      "formatted_address": "599 El Embarcadero, Oakland, CA 94610"
    }]
  }'
```

---

## 🧠 Example Response

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

---

## 🏷️ Data & Labeling

**Tag Taxonomy:** 15-20 predefined tags covering event categories:

-   **Music genres:** music, house, techno, jazz, classical
-   **Activities:** dance, yoga, art, food, market
-   **Locations:** oakland, sf, berkeley
-   **Characteristics:** outdoor, weekly, community, family

See [docs/TAGS.md](docs/TAGS.md) for complete list and definitions.

**Labeled Dataset:** ~100 events manually tagged with 2-5 tags each.

-   Started with 20 events for pipeline validation
-   Scaled to 100 for robust model training
-   Fields used: `name + description + formatted_address`

---

## 🧩 Integration with CMF

The CMF client (Next.js on Vercel) can:

-   Fetch events from `/api/events`
-   Then asynchronously call this service’s `/predict` endpoint
-   Merge returned tags for display or filtering

This allows events to appear immediately on the CMF map, while tags load in the background.

---

## 🧰 Deployment

Deploy on **Render**, **Fly.io**, or **Hugging Face Spaces** using Docker.

```bash
docker build -t ml-event-tagger .
docker run -p 8080:8080 ml-event-tagger
```

The included `Dockerfile` uses Python 3.11-slim and exposes port 8080.

---

## 📊 Model Performance

**MVP Target Metrics:**

-   Macro-averaged precision: ≥60%
-   Macro-averaged recall: ≥40%
-   Inference latency (p95): <300ms per event

See training notebook for detailed evaluation with:

-   Confusion matrix heatmap
-   Per-tag precision/recall charts
-   Training/validation loss curves
-   Tag frequency distribution

---

## 📚 Documentation

-   [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical design and API contract
-   [ROADMAP.md](docs/ROADMAP.md) - Project evolution and future plans
-   [MVP_DECISIONS.md](docs/MVP_DECISIONS.md) - Architectural decisions and rationale
-   [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) - Step-by-step implementation guide
-   [TAGS.md](docs/TAGS.md) - Tag taxonomy and labeling guidelines

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🧭 Author

**Chad Norwood**
[chadnorwood.com](https://chadnorwood.com) | [github.com/chadn](https://github.com/chadn)
