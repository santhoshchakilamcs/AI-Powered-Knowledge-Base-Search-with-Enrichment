"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "vector_store_status" in data
        assert "documents_count" in data
    
    def test_list_documents_endpoint(self):
        """Test document listing endpoint."""
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_count" in data
        assert isinstance(data["documents"], list)
    
    def test_enrichment_capabilities_endpoint(self):
        """Test enrichment capabilities endpoint."""
        response = client.get("/api/enrichment/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "wikipedia" in data
        assert "web_search" in data
        assert "auto_enrichment_enabled" in data
    
    def test_upload_without_file_fails(self):
        """Test that upload without file returns error."""
        response = client.post("/api/documents/upload")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_search_without_query_fails(self):
        """Test that search without query returns error."""
        response = client.post("/api/search", json={})
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_search_with_valid_query(self):
        """Test search with valid query structure."""
        response = client.post("/api/search", json={
            "query": "What is artificial intelligence?",
            "top_k": 5,
            "enable_auto_enrichment": False
        })
        # May return 200 or 500 depending on whether documents exist
        # Just check that it doesn't return 422 (validation error)
        assert response.status_code != 422
    
    def test_rating_endpoint(self):
        """Test rating submission endpoint."""
        response = client.post("/api/rate", json={
            "query": "Test query",
            "answer": "Test answer",
            "rating": 5,
            "feedback": "Great answer!"
        })
        assert response.status_code == 200
        data = response.json()
        assert "rating_id" in data
        assert "message" in data
    
    def test_rating_statistics_endpoint(self):
        """Test rating statistics endpoint."""
        response = client.get("/api/ratings/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "total_ratings" in data
        assert "average_rating" in data

