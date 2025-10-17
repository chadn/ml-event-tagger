# 🧠 ML Event Tagger

**Machine-learning microservice for tagging event metadata.**  
Originally designed for [CMF](https://cmf.chadnorwood.com) event data but structured for general event tagging tasks.

---

## 🎯 Purpose

This project demonstrates end-to-end AI/ML integration skills using **Python**, **TensorFlow/Keras**, and **FastAPI**.  
It ingests event metadata (title, description, location) and suggests relevant tags such as “music”, “oakland”, or “community”.

It was created to demonstrate practical use of TensorFlow/Keras within a full-stack integration context, bridging ML model design, API deployment, and client integration.

---

## 🧩 Key Features

- ✅ **TensorFlow/Keras** model for text-based multi-label classification  
- ✅ **FastAPI** inference service for REST-based tag suggestions  
- ✅ Clean separation between **training** and **serving** pipelines  
- ✅ Designed to integrate easily with **Next.js** or any REST client  
- ✅ Extensible architecture for additional data sources or retraining

---

## 🧱 Repository Structure

```
ml-event-tagger/
├── adapters/
│   └── cmf/                # Data adapter for CMF event API
├── core/
│   ├── train.py            # Model training script
│   ├── serve.py            # FastAPI app for inference
│   ├── tokenizer.pkl       # Tokenizer (after training)
│   └── tagger.h5           # Saved Keras model
├── data/
│   └── labeled_events.json # Manually labeled event dataset
├── notebooks/
│   └── 01_train_event_tagger.ipynb
├── models/                 # Saved model versions
├── docs/
│   └── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣  Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣  Train Model
```bash
python core/train.py
```
Outputs `models/tagger.h5` and `models/tokenizer.pkl`.

### 3️⃣  Serve API
```bash
uvicorn core.serve:app --reload
```

### 4️⃣  Test Inference
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"events":[{"name":"Days Like This - House Music","description":"Oakland house music","location":"Oakland"}]}'
```

---

## 🧠 Example Response

```json
{
  "tags": [
    {
      "event_index": 0,
      "tags": [
        {"name": "music", "confidence": 0.92},
        {"name": "oakland", "confidence": 0.86}
      ]
    }
  ]
}
```

---

## 🧩 Integration with CMF

The CMF client (Next.js on Vercel) can:
- Fetch events from `/api/events`
- Then asynchronously call this service’s `/predict` endpoint
- Merge returned tags for display or filtering

This allows events to appear immediately on the CMF map, while tags load in the background.

---

## 🧰 Deployment

Deploy on **Render**, **Fly.io**, or **Hugging Face Spaces** using Docker.

Example `Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "core.serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 🧭 Author

**Chad Norwood**  
[chadnorwood.com](https://chadnorwood.com) | [github.com/chadn](https://github.com/chadn)

