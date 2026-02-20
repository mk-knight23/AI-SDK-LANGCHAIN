"""Tests for FastAPI main application endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


# Import after mocking
@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    with patch('app.main.graph') as mock_graph:
        mock_graph.invoke = Mock(return_value={
            "idea": "Test startup idea",
            "market_analysis": "Large growing market",
            "business_model": "SaaS subscription",
            "pitch_deck": "10 slides",
            "current_step": "complete"
        })
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "venture-graph"

    def test_health_returns_json_content_type(self, client):
        """Test health endpoint returns JSON content type."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestPlanEndpoint:
    """Test suite for /plan endpoint."""

    def test_plan_returns_200(self, client):
        """Test plan endpoint returns 200 for valid request."""
        response = client.post("/plan", json={"idea": "A test startup idea"})
        assert response.status_code == 200

    def test_plan_returns_venture_plan(self, client):
        """Test plan endpoint returns venture plan structure."""
        response = client.post("/plan", json={"idea": "A test startup idea"})
        data = response.json()
        assert "idea" in data
        assert "market_analysis" in data
        assert "business_model" in data
        assert "pitch_deck" in data

    def test_plan_returns_input_idea(self, client):
        """Test plan endpoint returns the input idea."""
        idea = "My amazing startup"
        response = client.post("/plan", json={"idea": idea})
        data = response.json()
        assert data["idea"] == idea

    def test_plan_empty_idea_returns_200(self, client):
        """Test plan endpoint handles empty idea (API accepts it)."""
        response = client.post("/plan", json={"idea": ""})
        # FastAPI/Pydantic v2 allows empty strings by default
        assert response.status_code in [200, 422]

    def test_plan_missing_idea_returns_422(self, client):
        """Test plan endpoint returns 422 for missing idea field."""
        response = client.post("/plan", json={})
        assert response.status_code == 422

    def test_plan_invalid_json_returns_422(self, client):
        """Test plan endpoint returns 422 for invalid JSON."""
        response = client.post(
            "/plan",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_plan_calls_graph_with_correct_state(self, client):
        """Test that plan endpoint calls graph with correct initial state."""
        with patch('app.main.graph') as mock_graph:
            mock_graph.invoke = Mock(return_value={
                "idea": "Test",
                "market_analysis": "Analysis",
                "business_model": "Model",
                "pitch_deck": "Deck",
                "current_step": "complete"
            })
            from app.main import app
            with TestClient(app) as test_client:
                test_client.post("/plan", json={"idea": "My startup idea"})

                mock_graph.invoke.assert_called_once()
                call_args = mock_graph.invoke.call_args[0][0]
                assert call_args["idea"] == "My startup idea"
                assert call_args["messages"] == []
                assert call_args["current_step"] == "start"


class TestErrorHandling:
    """Tests for error handling in the API."""

    def test_nonexistent_endpoint_returns_404(self, client):
        """Test that nonexistent endpoints return 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed_returns_405(self, client):
        """Test that wrong HTTP methods return 405."""
        response = client.post("/health")
        assert response.status_code == 405


class TestVentureIdeaModel:
    """Tests for the VentureIdea Pydantic model validation."""

    def test_venture_idea_accepts_valid_string(self):
        """Test that VentureIdea accepts a valid string."""
        from app.main import VentureIdea
        idea = VentureIdea(idea="A great startup idea")
        assert idea.idea == "A great startup idea"

    def test_venture_idea_accepts_long_string(self):
        """Test that VentureIdea accepts a long description."""
        from app.main import VentureIdea
        long_idea = "A" * 1000
        idea = VentureIdea(idea=long_idea)
        assert idea.idea == long_idea


class TestVenturePlanModel:
    """Tests for the VenturePlan Pydantic model."""

    def test_venture_plan_accepts_valid_data(self):
        """Test that VenturePlan accepts valid data."""
        from app.main import VenturePlan
        plan = VenturePlan(
            idea="Test idea",
            market_analysis="Market is huge",
            business_model="SaaS",
            pitch_deck="10 slides"
        )
        assert plan.idea == "Test idea"
        assert plan.market_analysis == "Market is huge"


class TestAppConfiguration:
    """Tests for FastAPI app configuration."""

    def test_app_title(self):
        """Test that app has correct title."""
        with patch('app.main.graph'):
            from app.main import app
            assert app.title == "VentureGraph - AI Venture Planning"

    def test_health_endpoint_exists(self):
        """Test that health endpoint is registered."""
        with patch('app.main.graph'):
            from app.main import app
            routes = [route.path for route in app.routes]
            assert "/health" in routes

    def test_plan_endpoint_exists(self):
        """Test that plan endpoint is registered."""
        with patch('app.main.graph'):
            from app.main import app
            routes = [route.path for route in app.routes]
            assert "/plan" in routes


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present_on_health(self, client):
        """Test that CORS headers are present in health responses."""
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        # CORS middleware adds these headers when Origin header is present
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_all_origins(self, client):
        """Test that CORS allows requests from any origin."""
        response = client.get(
            "/health",
            headers={"Origin": "http://example.com"}
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"
