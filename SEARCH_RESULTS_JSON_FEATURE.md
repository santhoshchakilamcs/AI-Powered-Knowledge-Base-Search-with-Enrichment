# Search Results JSON Feature - Complete Guide

## ✅ Feature Implemented

Every search query now **automatically saves a JSON file** with the complete search result including:
- ✅ Query
- ✅ Answer
- ✅ Confidence score (0.0-1.0)
- ✅ Is Complete flag
- ✅ Sources with relevance scores
- ✅ Missing information
- ✅ Enrichment suggestions
- ✅ Auto-enrichment status
- ✅ Timestamp
- ✅ Metadata

---

## 📁 Storage Location

**Path**: `./data/search_results/`

**Filename Format**: `search_YYYYMMDD_HHMMSS_mmm_{query}.json`

**Example**: `search_20251019_164655_630_Tell me about chromaDB_.json`

---

## 📊 JSON Structure

```json
{
  "query": "Tell me about chromaDB?",
  "answer": "ChromaDB is a vector database...",
  "confidence": 0.85,
  "is_complete": true,
  "sources": [
    {
      "document_id": "doc_123",
      "document_name": "Naware.pdf",
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
  "timestamp": "2025-10-19T16:46:55.630454",
  "metadata": {
    "num_sources": 1,
    "num_missing_info": 0,
    "has_enrichment": false
  }
}
```

---

## 🔍 Key Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The search query |
| `answer` | string | AI-generated answer |
| `confidence` | float | 0.0-1.0 confidence score |
| `is_complete` | boolean | Whether answer is complete |
| `sources` | array | List of relevant documents |
| `missing_info` | array | What information is missing |
| `auto_enrichment_applied` | boolean | Whether external sources were used |
| `timestamp` | string | ISO format timestamp |
| `metadata` | object | Statistics about the result |

---

## 🛠️ New API Endpoints

### 1. Get Recent Search Results
```bash
GET /api/search-results?limit=50
```
Returns the 50 most recent search results.

### 2. Get Results for Specific Query
```bash
GET /api/search-results/query/{query}
```
Returns all results for a specific query.

### 3. Get Search Statistics
```bash
GET /api/search-results/statistics
```
Returns statistics about all searches:
```json
{
  "total_searches": 5,
  "average_confidence": 0.85,
  "complete_answers": 4,
  "incomplete_answers": 1,
  "completion_rate": 80.0
}
```

---

## 📝 Example Usage

### Via cURL
```bash
# Get recent results
curl http://localhost:8000/api/search-results?limit=10

# Get results for specific query
curl "http://localhost:8000/api/search-results/query/what%20is%20pyspark"

# Get statistics
curl http://localhost:8000/api/search-results/statistics
```

### Via Python
```python
import requests
import json

# Make a search
response = requests.post(
    "http://localhost:8000/api/search",
    json={"query": "what is pyspark?"}
)

# Get recent results
results = requests.get(
    "http://localhost:8000/api/search-results?limit=10"
).json()

print(f"Total searches: {results['total_results']}")
for result in results['results']:
    print(f"Query: {result['query']}")
    print(f"Confidence: {result['confidence']}")
```

---

## 📂 File Organization

```

├── data/
│   ├── search_results/          # ← NEW: Search results JSON files
│   │   ├── search_20251019_164655_630_Tell me about chromaDB_.json
│   │   ├── search_20251019_164640_123_where am I traveling_.json
│   │   └── ...
│   ├── uploads/                 # Original documents
│   ├── extracted_text/          # Extracted text
│   ├── metadata/                # Document metadata
│   ├── chroma_db/               # Vector store
│   └── ratings.jsonl            # User ratings
```

---

## 🔄 How It Works

1. **User searches** → POST `/api/search`
2. **RAG pipeline processes** → Generates answer
3. **Result is saved** → `./data/search_results/search_*.json`
4. **Response returned** → JSON sent to user
5. **User can retrieve** → GET `/api/search-results`

---

## 💾 Implementation Details

### New Service: `SearchResultsService`
- **File**: `app/services/search_results_service.py`
- **Methods**:
  - `save_search_result()` - Saves result to JSON file
  - `get_search_results()` - Retrieves recent results
  - `get_search_result_by_query()` - Finds results by query
  - `get_statistics()` - Calculates search statistics

### Updated Endpoint: `/api/search`
- **File**: `app/api/routes.py` (lines 542-608)
- **Change**: Now calls `search_results_service.save_search_result()`
- **Behavior**: Saves result even if API response fails

### New Endpoints
- **File**: `app/api/routes.py` (lines 790-839)
- **Endpoints**:
  - `GET /api/search-results`
  - `GET /api/search-results/query/{query}`
  - `GET /api/search-results/statistics`

---

## ✨ Features

✅ **Automatic Saving** - Every search is saved automatically  
✅ **Structured Format** - Consistent JSON structure  
✅ **Timestamped** - Each result has ISO timestamp  
✅ **Queryable** - Retrieve results by query or date  
✅ **Statistics** - Track search quality metrics  
✅ **Graceful Handling** - Works even if saving fails  
✅ **Metadata** - Includes relevance scores and sources  

---

## 🚀 Testing

### Test 1: Search and Verify File
```bash
# Make a search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is pyspark?"}'

# Check if file was created
ls -la data/search_results/
```

### Test 2: Retrieve Results
```bash
# Get all recent results
curl http://localhost:8000/api/search-results

# Get results for specific query
curl "http://localhost:8000/api/search-results/query/what%20is%20pyspark"
```

### Test 3: Check Statistics
```bash
curl http://localhost:8000/api/search-results/statistics
```

---

## 📌 Notes

- Results are stored **locally** on your machine
- Files are **never deleted** automatically
- You can manually delete old results from `./data/search_results/`
- Each search creates a **new file** (no overwrites)
- Filenames are **sanitized** for filesystem compatibility

