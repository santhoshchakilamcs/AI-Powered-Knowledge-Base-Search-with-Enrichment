"""Tests for document processor service."""
import pytest
from pathlib import Path
from app.services.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def test_is_supported_format(self):
        """Test file format validation."""
        processor = DocumentProcessor()
        
        assert processor.is_supported_format("test.pdf") is True
        assert processor.is_supported_format("test.txt") is True
        assert processor.is_supported_format("test.docx") is True
        assert processor.is_supported_format("test.jpg") is False
        assert processor.is_supported_format("test.xlsx") is False
    
    def test_extract_from_txt(self, sample_text_file):
        """Test text extraction from TXT file."""
        processor = DocumentProcessor()
        text = processor._extract_from_txt(sample_text_file)
        
        assert "artificial intelligence" in text.lower()
        assert "machine learning" in text.lower()
        assert len(text) > 0
    
    def test_chunk_document(self):
        """Test document chunking."""
        processor = DocumentProcessor()
        
        text = "This is a test. " * 200  # Create long text
        metadata = {
            'document_id': 'test-123',
            'filename': 'test.txt'
        }
        
        chunks = processor._chunk_document(text, metadata)
        
        assert len(chunks) > 0
        assert all(hasattr(chunk, 'page_content') for chunk in chunks)
        assert all(hasattr(chunk, 'metadata') for chunk in chunks)
        assert all('chunk_id' in chunk.metadata for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_process_upload(self, temp_upload_dir):
        """Test complete upload processing."""
        processor = DocumentProcessor()
        processor.upload_dir = Path(temp_upload_dir)
        
        content = b"This is test content for upload processing."
        filename = "test_upload.txt"
        
        doc_id, chunks = await processor.process_upload(content, filename)
        
        assert doc_id is not None
        assert len(chunks) > 0
        assert (Path(temp_upload_dir) / f"{doc_id}_{filename}").exists()
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported formats raise ValueError."""
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError):
            processor._extract_text(Path("test.unsupported"))

