"""Pytest configuration and fixtures."""
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_upload_dir():
    """Create a temporary upload directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_text_file(temp_upload_dir):
    """Create a sample text file for testing."""
    file_path = Path(temp_upload_dir) / "sample.txt"
    content = """
    This is a sample document for testing.
    It contains information about artificial intelligence and machine learning.
    AI systems can process natural language and generate responses.
    Machine learning models learn from data to make predictions.
    """
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4 sample content"


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "answer": "This is a test answer based on the provided documents.",
        "confidence": 0.85,
        "is_complete": True,
        "missing_info": [],
        "reasoning": "The documents contain sufficient information.",
        "relevant_sources": [0, 1]
    }

