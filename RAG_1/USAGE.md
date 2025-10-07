# Usage Guide

## Getting Started

### 1. Initial Setup

```bash
# Run the setup script
chmod +x setup.sh
./setup.sh

# Configure your OpenAI API key
nano .env  # or use your preferred editor
# Set: OPENAI_API_KEY=sk-your-actual-key-here
```

### 2. Start the Application

```bash
# Make run script executable
chmod +x run.sh

# Start the server
./run.sh
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Using the Web Interface

### Upload Documents

1. Click on the upload area or drag & drop files
2. Supported formats: PDF, TXT, DOCX
3. Wait for processing (you'll see a success message)
4. Documents appear in the "Documents" list

### Ask Questions

1. Type your question in the search box
2. (Optional) Enable auto-enrichment for external sources
3. Click "Search" or press Enter
4. View the answer with:
   - Confidence score
   - Completeness indicator
   - Source references
   - Missing information (if incomplete)
   - Enrichment suggestions

### Rate Answers

1. After receiving an answer, click the stars to rate (1-5)
2. Your feedback helps improve the system

## Using the API

### Upload a Document

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@/path/to/your/document.pdf"
```

Response:
```json
{
  "document_id": "uuid-here",
  "filename": "document.pdf",
  "file_size": 12345,
  "file_type": "pdf",
  "chunks_created": 15,
  "upload_timestamp": "2024-01-01T12:00:00",
  "message": "Document uploaded and processed successfully"
}
```

### Search and Get Answer

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the company policy on remote work?",
    "top_k": 5,
    "enable_auto_enrichment": false
  }'
```

Response:
```json
{
  "query": "What is the company policy on remote work?",
  "answer": "According to the employee handbook...",
  "confidence": 0.85,
  "is_complete": true,
  "sources": [
    {
      "document_id": "uuid",
      "document_name": "handbook.pdf",
      "chunk_id": "uuid_chunk_0",
      "content": "Relevant excerpt...",
      "relevance_score": 0.92,
      "metadata": {}
    }
  ],
  "missing_info": [],
  "enrichment_suggestions": [],
  "auto_enrichment_applied": false,
  "auto_enrichment_sources": [],
  "timestamp": "2024-01-01T12:00:00"
}
```

### List Documents

```bash
curl -X GET "http://localhost:8000/api/documents"
```

### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/api/documents/{document_id}"
```

### Rate an Answer

```bash
curl -X POST "http://localhost:8000/api/rate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "answer": "AI is...",
    "rating": 5,
    "feedback": "Very helpful!"
  }'
```

### Get Rating Statistics

```bash
curl -X GET "http://localhost:8000/api/ratings/statistics"
```

## Understanding the Response

### Confidence Score
- **0.8 - 1.0**: High confidence - answer is well-supported by documents
- **0.5 - 0.8**: Medium confidence - answer is partially supported
- **0.0 - 0.5**: Low confidence - limited information available

### Completeness
- **Complete**: All necessary information is available in the knowledge base
- **Incomplete**: Some information is missing; check `missing_info` field

### Enrichment Suggestions
When an answer is incomplete, the system suggests:
- **Document**: Upload specific documents
- **External Source**: Information available from Wikipedia/web
- **Clarification**: Rephrase or provide more context
- **Related Topic**: Explore related areas

## Auto-Enrichment Feature

Enable auto-enrichment to automatically fetch missing information from:
- Wikipedia
- Web search (DuckDuckGo)

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is quantum computing?",
    "enable_auto_enrichment": true
  }'
```

When enabled:
1. System searches your documents first
2. If incomplete, fetches from external sources
3. Adds external content to knowledge base
4. Re-runs search with enriched data
5. Returns improved answer

## Best Practices

### Document Upload
- Upload related documents together
- Use descriptive filenames
- Break large documents into logical sections
- Include metadata-rich documents (with headers, structure)

### Asking Questions
- Be specific and clear
- Use natural language
- Include context when needed
- Try different phrasings if results aren't satisfactory

### Managing the Knowledge Base
- Regularly review low-rated answers
- Update documents when information changes
- Remove outdated documents
- Use enrichment suggestions to identify gaps

## Troubleshooting

### "No documents found"
- Upload relevant documents first
- Check that documents were processed successfully
- Verify documents contain text (not just images)

### Low confidence scores
- Upload more relevant documents
- Ensure documents contain the specific information
- Try enabling auto-enrichment
- Rephrase your question

### API errors
- Check that .env is configured correctly
- Verify OpenAI API key is valid
- Ensure sufficient API credits
- Check logs for detailed error messages

## Advanced Configuration

Edit `.env` to customize:

```bash
# Use different LLM model
LLM_MODEL=gpt-4-turbo-preview

# Adjust chunk size for documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Change number of retrieved documents
TOP_K_RESULTS=5

# Adjust confidence threshold
CONFIDENCE_THRESHOLD=0.7
```

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_api.py -v
```

## Monitoring

Check application health:
```bash
curl http://localhost:8000/api/health
```

View rating statistics:
```bash
curl http://localhost:8000/api/ratings/statistics
```

Check enrichment capabilities:
```bash
curl http://localhost:8000/api/enrichment/capabilities
```

