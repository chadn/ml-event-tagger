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


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_predict_with_extremely_long_text(self, client):
        """Predict endpoint handles very long input text."""
        # Create a very long description (10,000 characters)
        long_text = "word " * 2000

        request_data = {
            "events": [{
                "name": "Test Event",
                "description": long_text,
                "location": "Test Location"
            }]
        }

        response = client.post("/predict", json=request_data)

        # Should still work (text will be truncated/padded)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data

    def test_predict_with_special_characters(self, client):
        """Predict endpoint handles special characters in text."""
        request_data = {
            "events": [{
                "name": "Test 🎵 Event™",
                "description": "Special chars: €£¥ © ® @#$%^&*()",
                "location": "Test 📍 Location"
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

    def test_predict_with_unicode(self, client):
        """Predict endpoint handles Unicode characters."""
        request_data = {
            "events": [{
                "name": "日本語イベント",
                "description": "Événement français avec accents",
                "location": "Москва"
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

    def test_predict_with_html_in_text(self, client):
        """Predict endpoint handles HTML in text (should be cleaned)."""
        request_data = {
            "events": [{
                "name": "<script>alert('test')</script>Music Event",
                "description": "<p>This is <b>bold</b> text</p>",
                "location": "<a href='test'>Link</a>"
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        # HTML should be cleaned before prediction

    def test_predict_with_urls_in_text(self, client):
        """Predict endpoint handles URLs in text (should be cleaned)."""
        request_data = {
            "events": [{
                "name": "Event https://example.com",
                "description": "Check www.test.com for details",
                "location": "Visit http://location.com"
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

    def test_predict_with_only_whitespace(self, client):
        """Predict endpoint handles events with only whitespace."""
        request_data = {
            "events": [{
                "name": "   ",
                "description": "\n\n\n",
                "location": "\t\t"
            }]
        }

        # This might fail validation due to empty name after stripping
        response = client.post("/predict", json=request_data)
        # Could be 200 (handled) or 422 (validation error) - both acceptable
        assert response.status_code in [200, 422]

    def test_predict_handles_model_output_edge_cases(self, client):
        """Predict endpoint handles edge cases in model output."""
        # Test with minimal input
        request_data = {
            "events": [{
                "name": "a"  # Single character
            }]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Should still return top 5 predictions
        assert len(data["predictions"][0]["tags"]) == 5

    def test_predict_batch_with_mixed_quality(self, client):
        """Predict endpoint handles batch with varying input quality."""
        request_data = {
            "events": [
                {"name": "Good Event", "description": "Detailed description", "location": "SF"},
                {"name": "Minimal"},
                {"name": "Long" * 1000, "description": "x"},
                {"name": "Special ™©®", "description": "Symbols"}
            ]
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        # Should have predictions for all 4 events
        assert len(data["predictions"]) == 4


class TestRobustness:
    """Tests for API robustness and stability."""

    def test_multiple_concurrent_predictions(self, client):
        """API handles multiple predictions in sequence."""
        request_data = {
            "events": [{"name": "Test Event", "description": "Test"}]
        }

        # Make 10 requests in sequence
        for _ in range(10):
            response = client.post("/predict", json=request_data)
            assert response.status_code == 200

    def test_large_batch_prediction(self, client):
        """API handles large batch of events."""
        # Create 50 events
        events = [
            {
                "name": f"Event {i}",
                "description": f"Description for event {i}",
                "location": f"Location {i}"
            }
            for i in range(50)
        ]

        request_data = {"events": events}

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 50

    def test_prediction_consistency(self, client):
        """Same input produces consistent predictions."""
        request_data = {
            "events": [{"name": "House Music Party", "description": "DJ set", "location": "Oakland"}]
        }

        # Make same request twice
        response1 = client.post("/predict", json=request_data)
        response2 = client.post("/predict", json=request_data)

        # Should get same predictions
        data1 = response1.json()
        data2 = response2.json()

        tags1 = data1["predictions"][0]["tags"]
        tags2 = data2["predictions"][0]["tags"]

        # Top tags should be the same
        assert tags1[0]["tag"] == tags2[0]["tag"]
        assert tags1[0]["confidence"] == tags2[0]["confidence"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

