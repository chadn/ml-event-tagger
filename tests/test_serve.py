"""Unit tests for FastAPI service using TestClient.

Tests API endpoints without needing to start a real server.
"""

import pytest
from fastapi.testclient import TestClient
from ml_event_tagger.serve import app
from ml_event_tagger import __version__


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health endpoint returns correct response structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_healthy(self, client):
        """Health endpoint reports healthy status."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_version_correct(self, client):
        """Health endpoint returns correct version."""
        response = client.get("/health")
        data = response.json()

        assert data["version"] == __version__


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_200(self, client):
        """Root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response_structure(self, client):
        """Root endpoint returns API information."""
        response = client.get("/")
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data

    def test_root_links_correct(self, client):
        """Root endpoint provides correct documentation links."""
        response = client.get("/")
        data = response.json()

        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_single_event(self, client):
        """Predict endpoint handles single event."""
        request_data = {
            "events": [
                {
                    "name": "Test Event",
                    "description": "Test description",
                    "location": "Test location"
                }
            ]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1
        assert "tags" in data["predictions"][0]

    def test_predict_returns_top_5_tags(self, client):
        """Predict endpoint returns top 5 tags per event."""
        request_data = {
            "events": [
                {
                    "name": "House Music Party",
                    "description": "DJ performance",
                    "location": "Oakland"
                }
            ]
        }

        response = client.post("/predict", json=request_data)
        data = response.json()

        tags = data["predictions"][0]["tags"]
        assert len(tags) == 5

        # Check tag structure
        for tag in tags:
            assert "tag" in tag
            assert "confidence" in tag
            assert 0 <= tag["confidence"] <= 1

    def test_predict_multiple_events(self, client):
        """Predict endpoint handles multiple events."""
        request_data = {
            "events": [
                {"name": "Event 1", "description": "Desc 1", "location": "Loc 1"},
                {"name": "Event 2", "description": "Desc 2", "location": "Loc 2"},
                {"name": "Event 3", "description": "Desc 3", "location": "Loc 3"}
            ]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 3

    def test_predict_with_empty_description(self, client):
        """Predict endpoint handles empty description."""
        request_data = {
            "events": [
                {
                    "name": "Test Event",
                    "description": "",
                    "location": ""
                }
            ]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

    def test_predict_with_minimal_data(self, client):
        """Predict endpoint handles minimal event data (name only)."""
        request_data = {
            "events": [
                {"name": "Minimal Event"}
            ]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200


class TestPredictValidation:
    """Tests for /predict endpoint input validation."""

    def test_predict_requires_events(self, client):
        """Predict endpoint requires events field."""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

    def test_predict_requires_name(self, client):
        """Predict endpoint requires name field in event."""
        request_data = {
            "events": [
                {"description": "No name provided"}
            ]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_rejects_empty_name(self, client):
        """Predict endpoint rejects empty name."""
        request_data = {
            "events": [
                {"name": "", "description": "Empty name"}
            ]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_rejects_empty_events_list(self, client):
        """Predict endpoint rejects empty events list."""
        request_data = {"events": []}

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422

    def test_predict_rejects_invalid_json(self, client):
        """Predict endpoint rejects invalid JSON."""
        response = client.post(
            "/predict",
            content=b"not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestPredictPerformance:
    """Performance tests for /predict endpoint."""

    def test_predict_response_time_single_event(self, client):
        """Predict endpoint responds quickly for single event."""
        import time

        request_data = {
            "events": [
                {
                    "name": "Performance Test Event",
                    "description": "Testing response time",
                    "location": "Test Location"
                }
            ]
        }

        start_time = time.time()
        response = client.post("/predict", json=request_data)
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 0.3  # Should be under 300ms
        print(f"\nSingle event prediction time: {elapsed*1000:.2f}ms")

    def test_predict_response_time_batch(self, client):
        """Predict endpoint responds quickly for batch of events."""
        import time

        # Create 10 events
        events = [
            {
                "name": f"Event {i}",
                "description": f"Description {i}",
                "location": f"Location {i}"
            }
            for i in range(10)
        ]

        request_data = {"events": events}

        start_time = time.time()
        response = client.post("/predict", json=request_data)
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 1.0  # Should be under 1 second for 10 events
        print(f"\nBatch (10 events) prediction time: {elapsed*1000:.2f}ms")


class TestConfidenceScores:
    """Tests for prediction confidence scores."""

    def test_confidence_scores_sorted_descending(self, client):
        """Confidence scores are sorted from highest to lowest."""
        request_data = {
            "events": [
                {
                    "name": "House Music Event",
                    "description": "DJ performance with dancing",
                    "location": "Oakland nightclub"
                }
            ]
        }

        response = client.post("/predict", json=request_data)
        data = response.json()

        tags = data["predictions"][0]["tags"]
        confidences = [tag["confidence"] for tag in tags]

        # Check that confidences are in descending order
        assert confidences == sorted(confidences, reverse=True)

    def test_confidence_scores_in_valid_range(self, client):
        """All confidence scores are between 0 and 1."""
        request_data = {
            "events": [
                {"name": "Test", "description": "Test", "location": "Test"}
            ]
        }

        response = client.post("/predict", json=request_data)
        data = response.json()

        tags = data["predictions"][0]["tags"]
        for tag in tags:
            assert 0 <= tag["confidence"] <= 1

    def test_confidence_scores_rounded(self, client):
        """Confidence scores are rounded to 2 decimal places."""
        request_data = {
            "events": [
                {"name": "Music Event", "description": "Concert", "location": "SF"}
            ]
        }

        response = client.post("/predict", json=request_data)
        data = response.json()

        tags = data["predictions"][0]["tags"]
        for tag in tags:
            # Check that confidence has at most 2 decimal places
            confidence_str = str(tag["confidence"])
            if '.' in confidence_str:
                decimal_places = len(confidence_str.split('.')[1])
                assert decimal_places <= 2


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_docs_endpoint_exists(self, client):
        """Swagger UI docs endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_exists(self, client):
        """ReDoc documentation endpoint is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema_exists(self, client):
        """OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        # Check it's valid JSON
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

