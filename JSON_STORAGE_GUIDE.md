# JSON Storage Guide - AI-Powered Knowledge Base

## Overview
The application stores various types of data in JSON format across different directories. Here's a complete guide to where everything is stored.

---

## 📁 Directory Structure

```

├── data/
│   ├── uploads/                    # Original uploaded documents
│   ├── extracted_text/             # Extracted text from documents
│   ├── metadata/                   # Document metadata
│   ├── chroma_db/                  # Vector store (ChromaDB)
│   └── ratings.jsonl               # User ratings (JSONL format)
```

---

## 🔍 Detailed Storage Locations

### 1. **Search Answers (In-Memory Response)**
- **Location**: Returned directly via API (not persisted to disk by default)
- **Format**: JSON
- **Structure**:
```json
{
  "query": "what is pyspark?",
  "answer": "PySpark is a Python API for Apache Spark...",
  "confidence": 0.85,
  "is_complete": true,
  "sources": [
    {
      "document_id": "doc_123",
      "document_name": "STAR_answers.txt",
      "chunk_id": "chunk_1",
      "content": "...",
      "relevance_score": 0.92,
      "metadata": {...}
    }
  ],
  "missing_info": [],
  "enrichment_suggestions": [],
  "auto_enrichment_applied": false,
  "auto_enrichment_sources": [],
  "timestamp": "2025-10-19T16:40:13.026Z"
}
```

### 2. **User Ratings (Persisted)**
- **Location**: `./data/ratings.jsonl`
- **Format**: JSONL (JSON Lines - one JSON object per line)
- **Structure**:
```json
{
  "rating_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "what is pyspark?",
  "answer": "PySpark is a Python API for Apache Spark...",
  "rating": 5,
  "feedback": "Very helpful and accurate!",
  "timestamp": "2025-10-19T16:40:15.123Z"
}
```

### 3. **Document Metadata**
- **Location**: `./data/metadata/{filename}.json`
- **Format**: JSON
- **Structure**:
```json
{
  "filename": "STAR_answers.txt",
  "file_size": 5432,
  "upload_timestamp": "2025-10-19T16:38:05.470Z",
  "text_length": 5000,
  "word_count": 850
}
```

### 4. **Extracted Text**
- **Location**: `./data/extracted_text/{filename}.txt`
- **Format**: Plain text (not JSON)
- **Content**: Raw text extracted from uploaded documents

### 5. **Vector Store (ChromaDB)**
- **Location**: `./data/chroma_db/`
- **Format**: Binary database (not JSON)
- **Contains**: Document embeddings and metadata for semantic search

---

## 📊 How to Access the Data

### View Ratings
```bash
cat ./data/ratings.jsonl
```

### View Specific Rating
```bash
grep "rating_id" ./data/ratings.jsonl
```

### View Document Metadata
```bash
cat ./data/metadata/STAR_answers.txt.json
```

### Get Rating Statistics via API
```bash
curl http://localhost:8000/api/ratings/statistics
```

---

## 🔄 Data Flow

1. **Upload Document** → Saved to `./data/uploads/`
2. **Extract Text** → Saved to `./data/extracted_text/`
3. **Save Metadata** → Saved to `./data/metadata/`
4. **Create Embeddings** → Stored in `./data/chroma_db/`
5. **Search Query** → Returns JSON response (in-memory)
6. **Rate Answer** → Appended to `./data/ratings.jsonl`

---

## 💾 Persistence Strategy

| Data Type | Storage | Persistence | Format |
|-----------|---------|-------------|--------|
| Search Answers | API Response | In-Memory | JSON |
| User Ratings | `ratings.jsonl` | Persistent | JSONL |
| Document Metadata | `metadata/` | Persistent | JSON |
| Extracted Text | `extracted_text/` | Persistent | TXT |
| Embeddings | `chroma_db/` | Persistent | Binary |

---

## 🛠️ Configuration

All storage paths are configurable in `.env`:

```bash
# Document Storage
UPLOAD_DIRECTORY=./data/uploads

# Vector Store
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
```

---

## 📝 Example: Accessing Search Results Programmatically

```python
import requests
import json

# Make a search request
response = requests.post(
    "http://localhost:8000/api/search",
    json={"query": "what is pyspark?"}
)

# Get JSON response
result = response.json()

# Access structured data
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {result['sources']}")

# Save to file if needed
with open("search_result.json", "w") as f:
    json.dump(result, f, indent=2)
```

---

## 🔐 Data Privacy

- Ratings are stored locally in `./data/ratings.jsonl`
- No data is sent to external servers except OpenAI API calls
- All data remains on your machine

---

## 📌 Important Notes

1. **Search responses are NOT automatically saved** - they're returned via API
2. **Ratings ARE automatically saved** to `ratings.jsonl` when you rate an answer
3. **Vector embeddings are stored in ChromaDB** - not in JSON format
4. **All paths are relative to the project root** (``)

