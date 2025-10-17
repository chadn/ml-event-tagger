# 🧠 Event Tagging AI Blueprint

## 🎯 Goal

Build a lightweight AI service that suggests relevant tags for CMF events using TensorFlow/Keras.
Demonstrates full-stack + ML integration, including data preparation, model training, and API deployment.

---

## 🏗️ Architecture Overview

-   **Frontend:** Next.js (existing CMF UI)
-   **Backend:** Node.js API routes for event CRUD
-   **ML Service:** Python (FastAPI + TensorFlow/Keras)
-   **Storage:** JSON or PostgreSQL for labeled data
-   **Deployment:** Vercel (Next.js) + Render/DigitalOcean (FastAPI)

---

## 📚 Data Pipeline

1. Fetch event data from `/api/events`
2. Extract and normalize text (`name`, `description`, `location`)
3. Preprocess text (lowercase, remove punctuation, URLs)
4. Encode using Tokenizer
5. Train multi-label classifier on hand-labeled data

---

## 🧩 Model Design

-   Framework: TensorFlow / Keras
-   Architecture: Embedding → GlobalAveragePooling → Dense(32, relu) → Dense(n_tags, sigmoid)
-   Loss: Binary Crossentropy
-   Metrics: Precision, Recall, AUC
-   Output: Top-N tag predictions with confidence scores

---

## 🧪 MVP Scope

-   Manually labeled 100–300 events
-   10–20 tag vocabulary
-   Simple training notebook + saved `.h5` model
-   REST endpoint `/api/tags/suggest`
-   JSON response: `{ "tags": [ { "name": "music", "confidence": 0.87 }, ... ] }`

---

## 🚀 Future Improvements

-   Automated labeling heuristics
-   Larger training data
-   Model retraining pipeline
-   Frontend tag visualization dashboard
-   TensorFlow.js or ONNX deployment
-   Continuous feedback loop from admin UI

---

## 🧰 Folder Structure

```
ml/
├── data/
│  ├── labeled_events.json
├── notebooks/
│  └── 01_train_event_tagger.ipynb
├── models/
│  └── tagger.h5
├── app.py
└── README.md
```

---

## 🔗 Integration Points

-   `/api/tags/suggest` in CMF calls external FastAPI endpoint
-   Cache suggestions in DB for repeated events
-   Optionally display suggested tags in CMF admin view

---

## ✅ Success Criteria

-   Model achieves ≥70% precision on validation data
-   Tags are contextually relevant and non-redundant
-   API responds within <300ms for single event
-   Clean, documented, reproducible pipeline

---

## 🧭 Next Steps

1. Finalize tag list and labeling guide
2. Build `ml/data/labeled_events.json`
3. Create training notebook + save model
4. Implement `/api/tags/suggest` integration
5. Evaluate + iterate on tag accuracy

# Purpose

## **🧠 1\. What Demonstrates Your AI/ML \+ TensorFlow/Keras Skills**

You don’t need a huge dataset or fancy model — the _demonstration of understanding and engineering integration_ is what counts.

### **Key Skills to Show:**

| Area                              | What to Demonstrate                                                                                                | Why It Matters                                                                         |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| **Data Pipeline Design**          | Ingest event JSON, normalize fields (`name`, `description`, `location`), tokenize, and prepare for model training. | Shows you understand feature extraction and data preparation — the _foundation_ of ML. |
| **Model Architecture & Training** | Define and train a small Keras model (Embedding → Dense → Sigmoid output).                                         | Demonstrates TensorFlow/Keras fluency, not just library usage.                         |
| **Evaluation**                    | Display metrics like accuracy, precision, recall, ROC curve, confusion matrix in notebook.                         | Shows you know how to assess model quality, not just run code.                         |
| **Integration**                   | Serve model predictions through an endpoint (or internal function) in your Next.js app.                            | Shows full-stack \+ ML synergy (data → model → app → user).                            |
| **Reproducibility**               | Store data, notebook, and trained `.h5` model under version control.                                               | Demonstrates engineering maturity and reproducible research practices.                 |

If you can hit all five of those points — even with a tiny dataset — you’ve shown solid ML-engineering skills.

---

## **🧩 2\. MVP vs Later Improvements**

### **MVP (Week 1–2)**

-   ✅ Collect events via your CMF API

-   ✅ Manually label \~100 events with 5–10 tags (e.g. “music”, “DJ”, “Oakland”, “art”, “food”)

-   ✅ Preprocess text (tokenize, lowercase, remove URLs)

-   ✅ Train small multi-label classifier (Keras Sequential model)

-   ✅ Save model \+ vocab to `/ml/models/tagger.h5`

-   ✅ Expose endpoint `/api/tags/suggest` returning top-N tags

You now have a working ML-backed feature and a résumé-ready artifact.

---

### **Phase 2: Improvements**

| Goal                          | Description                                                                            |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| **Auto-label bootstrapping**  | Use regex or embedding similarity to expand training set.                              |
| **Better model**              | Swap simple embedding for TF-Hub text encoder (e.g., `tensorflow_hub/nnlm-en-dim128`). |
| **Feedback loop**             | Store suggested tags and allow human correction, feeding back into training data.      |
| **Tag scoring**               | Weight tags by confidence or event popularity.                                         |
| **Dashboard / visualization** | Add a “Tag Insights” page summarizing tag frequency, co-occurrence, clusters.          |
| **Semantic grouping**         | Use embeddings to cluster similar tags (“house music”, “deep house”, “techno”).        |

---

## **🏷️ 3\. Tag Granularity — Single Event vs All Events**

| Strategy                 | Pros                                                | Cons                                    |
| ------------------------ | --------------------------------------------------- | --------------------------------------- |
| **Single-event tagging** | Fine-grained; helps for search and personalization. | May produce noisy or redundant tags.    |
| **Global tagging**       | Clear, stable taxonomy (e.g., 20–50 tags total).    | Coarse; can miss event-specific nuance. |

**Best compromise for MVP:**
Use a **small controlled tag set** (10–20 tags) applied per event.
This shows classification logic clearly, keeps labeling manageable, and avoids overfitting or noise.

Example tag vocab:
`["music", "house", "dj", "dance", "art", "yoga", "oakland", "community", "market", "festival"]`
