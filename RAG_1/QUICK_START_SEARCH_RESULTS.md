# Quick Start - Search Results JSON Feature

## 🎯 What's New?

Every search query now **automatically saves a JSON file** with:
- Query
- Answer
- Confidence score
- Sources with relevance scores
- Missing information
- Timestamp

---

## 📍 Where Are Results Stored?

**Location**: `./data/search_results/`

**Example file**: `search_20251019_164655_630_Tell me about chromaDB_.json`

---

## 🚀 Quick Test

### 1. Make a Search
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is pyspark?"}'
```

### 2. Check the JSON File
```bash
ls -la data/search_results/
cat data/search_results/search_*.json
```

### 3. Get All Results
```bash
curl http://localhost:8000/api/search-results
```

### 4. Get Results for Specific Query
```bash
curl "http://localhost:8000/api/search-results/query/what%20is%20pyspark"
```

### 5. Get Statistics
```bash
curl http://localhost:8000/api/search-results/statistics
```

---

## 📋 JSON File Example

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

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Search and auto-save result |
| `/api/search-results` | GET | Get recent results (limit=50) |
| `/api/search-results/query/{query}` | GET | Get results for specific query |
| `/api/search-results/statistics` | GET | Get search statistics |

---

## 💡 Use Cases

### 1. Track All Searches
```bash
curl http://localhost:8000/api/search-results?limit=100
```

### 2. Find Previous Searches
```bash
curl "http://localhost:8000/api/search-results/query/pyspark"
```

### 3. Monitor Quality
```bash
curl http://localhost:8000/api/search-results/statistics
```

### 4. Export Results
```bash
curl http://localhost:8000/api/search-results > all_results.json
```

---

## 📂 Directory Structure

```
RAG_1/
├── data/
│   └── search_results/          ← All search results saved here
│       ├── search_20251019_164655_630_Tell me about chromaDB_.json
│       ├── search_20251019_164640_123_where am I traveling_.json
│       └── ...
```

---

## ✨ Features

✅ **Automatic** - Every search is saved  
✅ **Structured** - Consistent JSON format  
✅ **Timestamped** - ISO format timestamps  
✅ **Queryable** - Retrieve by query or date  
✅ **Statistics** - Track search quality  
✅ **Graceful** - Works even if saving fails  

---

## 🔍 View Results in Terminal

### List all results
```bash
ls -la data/search_results/
```

### View latest result
```bash
cat data/search_results/search_*.json | tail -1
```

### Pretty print JSON
```bash
cat data/search_results/search_*.json | python -m json.tool
```

### Count total searches
```bash
ls data/search_results/ | wc -l
```

---

## 🛠️ Implementation Files

- **Service**: `app/services/search_results_service.py`
- **Endpoints**: `app/api/routes.py` (lines 542-839)
- **Storage**: `./data/search_results/`

---

## 📌 Important Notes

- Results are stored **locally** on your machine
- Each search creates a **new file** (no overwrites)
- Files are **never deleted** automatically
- You can manually delete old results if needed
- Filenames are **sanitized** for filesystem compatibility

---

## 🎓 Next Steps

1. ✅ Make a search query
2. ✅ Check the JSON file in `./data/search_results/`
3. ✅ Use the API endpoints to retrieve results
4. ✅ Monitor search quality with statistics
5. ✅ Export results for analysis

**Happy searching! 🚀**

