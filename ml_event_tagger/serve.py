"""FastAPI service for event tag prediction.

Provides REST API endpoints for:
- /health: Health check
- /predict: Event tag prediction
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ml_event_tagger import __version__
from ml_event_tagger.config import TAGS
from ml_event_tagger.preprocess import clean_text, combine_text_fields


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class Event(BaseModel):
    """Event input model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Days Like This - House Music",
            "description": "Weekly house music gathering with local DJs",
            "location": "The Pergola at Lake Merritt, 599 El Embarcadero, Oakland, CA 94610, USA"
        }
    })

    name: str = Field(..., description="Event name", min_length=1)
    description: str = Field(default="", description="Event description")
    location: str = Field(default="", description="Event location (venue + address)")


class PredictRequest(BaseModel):
    """Prediction request model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "events": [
                {
                    "name": "Days Like This - House Music",
                    "description": "Weekly house music gathering",
                    "location": "The Pergola at Lake Merritt, Oakland, CA"
                }
            ]
        }
    })

    events: List[Event] = Field(..., description="List of events to predict tags for", min_length=1)


class TagPrediction(BaseModel):
    """Single tag prediction."""
    tag: str = Field(..., description="Tag name")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)

    @field_serializer('confidence')
    def serialize_confidence(self, value: float) -> float:
        """Round confidence to 2 decimal places for JSON output."""
        return round(value, 2)


class EventPrediction(BaseModel):
    """Prediction result for a single event."""
    tags: List[TagPrediction] = Field(..., description="Predicted tags with confidence scores")


class PredictResponse(BaseModel):
    """Prediction response model."""
    predictions: List[EventPrediction] = Field(..., description="Predictions for each event")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


# ============================================================================
# Global State (Model and Tokenizer)
# ============================================================================

model: keras.Model = None
tokenizer_config: Dict[str, Any] = None
model_config: Dict[str, Any] = None


# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading/unloading model."""
    global model, tokenizer_config, model_config

    # Startup: Load model
    try:
        print("=" * 80)
        print("Starting ML Event Tagger API")
        print("=" * 80)

        # Load model
        model_path = Path("models/event_tagger_model.h5")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = keras.models.load_model(str(model_path))
        print(f"✅ Model loaded from {model_path}")

        # Load tokenizer config
        tokenizer_path = Path("models/tokenizer.json")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        with open(tokenizer_path) as f:
            tokenizer_config = json.load(f)
        print(f"✅ Tokenizer loaded from {tokenizer_path}")

        # Load model config
        config_path = Path("models/model_config.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")

        with open(config_path) as f:
            model_config = json.load(f)
        print(f"✅ Model config loaded from {config_path}")

        print(f"✅ Service ready with model vocabulary: {len(tokenizer_config['word_index'])} words")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        tokenizer_config = None
        model_config = None

    yield  # Application runs

    # Shutdown: Clean up (nothing to do for now)
    print("Shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="ML Event Tagger",
    description="Multi-label event classification service using TensorFlow/Keras",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ============================================================================
# Prediction Helper Functions
# ============================================================================

def preprocess_event(event: Event) -> str:
    """Preprocess a single event into cleaned text.

    Args:
        event: Event to preprocess

    Returns:
        Cleaned and combined text
    """
    # Convert to dict format expected by combine_event_fields
    event_dict = {
        'name': event.name,
        'description': event.description,
        'location': event.location
    }

    # Combine fields and clean
    combined = combine_text_fields(event_dict)
    cleaned = clean_text(combined)

    return cleaned


def texts_to_sequences(texts: List[str]) -> List[List[int]]:
    """Convert texts to sequences using loaded tokenizer.

    Args:
        texts: List of text strings

    Returns:
        List of token sequences
    """
    word_index = tokenizer_config['word_index']
    oov_token_index = word_index.get(tokenizer_config['oov_token'], 1)

    sequences = []
    for text in texts:
        words = text.lower().split()
        sequence = [word_index.get(word, oov_token_index) for word in words]
        sequences.append(sequence)

    return sequences


def predict_events(events: List[Event], top_k: int = 5) -> List[EventPrediction]:
    """Predict tags for a list of events.

    Args:
        events: List of events to predict
        top_k: Number of top predictions to return per event

    Returns:
        List of predictions
    """
    if model is None or tokenizer_config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Preprocess events
    texts = [preprocess_event(event) for event in events]

    # Convert to sequences
    sequences = texts_to_sequences(texts)

    # Pad sequences
    max_length = model_config['max_length']
    X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    # Get predictions
    predictions = model.predict(X, verbose=0)

    # Format results
    results = []
    for pred in predictions:
        # Get top-k predictions
        top_indices = pred.argsort()[-top_k:][::-1]

        tags = []
        for idx in top_indices:
            # Round to 2 decimal places
            confidence_rounded = float(round(pred[idx], 2))
            tags.append(TagPrediction(
                tag=TAGS[idx],
                confidence=confidence_rounded
            ))

        results.append(EventPrediction(tags=tags))

    return results


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "ML Event Tagger API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint.

    Returns service status, model load state, and version.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version=__version__
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """Predict tags for events.

    Takes a list of events and returns predicted tags with confidence scores.
    Returns top 5 tags per event by default.

    Args:
        request: Prediction request with events

    Returns:
        Predictions for each event

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    try:
        predictions = predict_events(request.events, top_k=5)
        return PredictResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# Main (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting development server...")
    print("API docs available at: http://localhost:8000/docs")

    uvicorn.run(
        "ml_event_tagger.serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

