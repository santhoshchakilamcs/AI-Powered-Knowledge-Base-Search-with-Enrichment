# 🏗️ Technical Architecture Guide - AI-Powered Knowledge Base with RAG

**Document Purpose**: Comprehensive technical explanation for engineering leadership and technical teams.

---

## 📋 Executive Summary

This is a **Retrieval-Augmented Generation (RAG)** system that combines:
- **Semantic Search** using vector embeddings
- **LLM-based Answer Generation** with confidence scoring
- **Automatic Enrichment** from trusted external sources
- **Structured Output** with completeness detection
- **Persistent Storage** of all search results and ratings

**Key Innovation**: Automatic enrichment from external sources (Wikipedia, arXiv, PubMed) when the knowledge base has incomplete information.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                        │
│                    (app/main.py)                             │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌──────────┐
    │ Upload │  │ Search │  │  Rating  │
    │ Routes │  │ Routes │  │  Routes  │
    └────┬───┘  └────┬───┘  └────┬─────┘
         │           │            │
         └───────────┼────────────┘
                     │
        ┌────────────▼────────────┐
        │   RAG Pipeline Service  │
        │  (rag_pipeline.py)      │
        └────────────┬────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │  Vector  │ │   LLM    │ │ Enrichment   │
    │  Store   │ │ (GPT-4)  │ │  Engine      │
    │(ChromaDB)│ │          │ │(Wikipedia,   │
    └──────────┘ └──────────┘ │ arXiv, etc)  │
                               └──────────────┘
```

---

## 🔑 Core Components

### 1. **Vector Store Service** (`app/services/vector_store.py`)

**Purpose**: Manages semantic search using embeddings.

**Technology Stack**:
- **ChromaDB**: Vector database for storing embeddings
- **OpenAI Embeddings**: text-embedding-3-small model
- **LangChain**: Abstraction layer for vector operations

**Key Methods**:
```python
async def add_documents(documents: List[Document]) -> List[str]:
    """Add documents to vector store with embeddings."""
    ids = self.vector_store.add_documents(documents)
    return ids

async def similarity_search_with_score(query: str, k: int) -> List[tuple]:
    """Retrieve top-k most similar documents with relevance scores."""
    results = self.vector_store.similarity_search_with_score(query, k=k)
    return results
```

**How It Works**:
1. Documents are split into chunks (1000 chars, 200 char overlap)
2. Each chunk is converted to a 1536-dimensional embedding
3. Embeddings are stored in ChromaDB with metadata
4. User queries are embedded and compared using cosine similarity
5. Top-k most similar documents are returned

---

### 2. **RAG Pipeline** (`app/services/rag_pipeline.py`)

**Purpose**: Core intelligence - retrieves documents and generates answers.

**Workflow**:
```
User Query
    ↓
[1] Semantic Search → Retrieve top-5 documents
    ↓
[2] Prepare Context → Format documents for LLM
    ↓
[3] Generate Answer → LLM processes query + context
    ↓
[4] Assess Completeness → Check confidence & missing info
    ↓
[5] Auto-Enrichment? → If incomplete, fetch from external sources
    ↓
[6] Return Structured Response → JSON with confidence, sources, etc.
```

**Key Code**:
```python
async def search_and_answer(
    query: str,
    top_k: int = 5,
    enable_auto_enrichment: bool = True
) -> SearchResponse:
    # Step 1: Retrieve documents
    retrieved_docs = await vector_store_service.similarity_search_with_score(
        query=query, k=top_k
    )
    
    # Step 2: Prepare context
    context = self._prepare_context(retrieved_docs)
    
    # Step 3: Generate answer with LLM
    llm_response = await self._generate_answer(query, context, retrieved_docs)
    
    # Step 4: Check if answer is complete
    if not llm_response.is_complete and enable_auto_enrichment:
        # Step 5: Auto-enrich from external sources
        enrichment_result = await enrichment_engine.auto_enrich(
            query=query,
            missing_info=llm_response.missing_info
        )
        # Re-run search with enriched documents
        llm_response = await self._generate_answer(...)
    
    return llm_response
```

**LLM Prompt Strategy**:
- System prompt instructs LLM to assess completeness
- LLM returns JSON with: answer, confidence (0-1), is_complete, missing_info
- Confidence < 0.7 or is_complete=False triggers auto-enrichment

---

### 3. **Enrichment Engine** (`app/services/enrichment_engine.py`)

**Purpose**: Automatically fetch information from trusted external sources.

**Trusted Sources**:
| Source | Type | Use Case |
|--------|------|----------|
| Wikipedia | General Knowledge | Definitions, concepts |
| arXiv | Academic Papers | Research, technical topics |
| PubMed | Medical Research | Health, medical topics |
| Web Search | General Web | Fallback for other topics |

**Auto-Enrichment Flow**:
```python
async def auto_enrich(
    query: str,
    missing_info: List[str],
    max_sources: int = 3
) -> Dict[str, Any]:
    enriched_docs = []
    
    # Try Wikipedia first
    wiki_docs = await self._enrich_from_wikipedia(query, missing_info)
    enriched_docs.extend(wiki_docs)
    
    # Try arXiv for academic content
    arxiv_docs = await self._enrich_from_arxiv(query, missing_info)
    enriched_docs.extend(arxiv_docs)
    
    # Try PubMed for medical content
    pubmed_docs = await self._enrich_from_pubmed(query, missing_info)
    enriched_docs.extend(pubmed_docs)
    
    # Add enriched documents to vector store
    await vector_store_service.add_documents(enriched_docs)
    
    return {
        'enriched_count': len(enriched_docs),
        'sources': [doc.metadata['source'] for doc in enriched_docs]
    }
```

---

### 4. **Search Results Service** (`app/services/search_results_service.py`)

**Purpose**: Persist all search results to JSON files for audit trail and analytics.

**Storage Structure**:
```
data/search_results/
├── search_20251019_165903_469_what is machine learning.json
├── search_20251019_165834_052_what is xgboost.json
└── ...
```

**JSON Schema**:
```json
{
  "query": "what is machine learning",
  "answer": "Machine learning is...",
  "confidence": 1.0,
  "is_complete": true,
  "sources": [
    {
      "document_id": "enriched_wikipedia_...",
      "document_name": "wikipedia: Machine learning",
      "relevance_score": 0.803,
      "metadata": {"source": "wikipedia", "url": "..."}
    }
  ],
  "missing_info": [],
  "enrichment_suggestions": [
    {
      "type": "external_source",
      "suggestion": "✅ Fetched from Wikipedia: Machine learning",
      "priority": "high",
      "external_source_url": "https://en.wikipedia.org/wiki/Machine_learning"
    }
  ],
  "auto_enrichment_applied": true,
  "auto_enrichment_sources": ["✅ Fetched from Wikipedia: Machine learning"],
  "timestamp": "2025-10-19T16:59:03.469000"
}
```

---

## 🔄 Data Flow: Complete Example

**User Query**: "What is XGBoost?"

**Step 1: Semantic Search**
```
Query embedding: [0.123, -0.456, 0.789, ...]  (1536 dimensions)
↓
ChromaDB similarity search
↓
Retrieved: 5 documents with scores [0.85, 0.78, 0.72, 0.65, 0.58]
```

**Step 2: LLM Processing**
```
System Prompt: "Answer based on provided documents..."
Context: [5 retrieved documents]
User Query: "What is XGBoost?"
↓
GPT-4 Turbo processes
↓
Response: {
  "answer": "XGBoost is a gradient boosting library...",
  "confidence": 0.95,
  "is_complete": true,
  "missing_info": [],
  "relevant_sources": [0, 1, 2]
}
```

**Step 3: Response Formatting**
```
Convert to SearchResponse:
- Query: "What is XGBoost?"
- Answer: "XGBoost is..."
- Confidence: 0.95
- Sources: [SourceReference objects]
- Enrichment: [EnrichmentSuggestion objects]
```

**Step 4: Persistence**
```
Save to: data/search_results/search_20251019_165550_508_what is xgboost.json
```

---

## 📊 Data Models (Pydantic Schemas)

**SearchRequest**:
```python
class SearchRequest(BaseModel):
    query: str  # User's question
    top_k: int = 5  # Number of documents to retrieve
    enable_auto_enrichment: bool = True  # Auto-fetch from external sources
```

**SearchResponse**:
```python
class SearchResponse(BaseModel):
    query: str
    answer: str
    confidence: float  # 0.0 to 1.0
    is_complete: bool  # True if all info available
    sources: List[SourceReference]  # Retrieved documents
    missing_info: List[str]  # What's missing
    enrichment_suggestions: List[EnrichmentSuggestion]
    auto_enrichment_applied: bool
    auto_enrichment_sources: List[str]
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | FastAPI | 0.104.1 | REST API |
| LLM | GPT-4 Turbo | Latest | Answer generation |
| Embeddings | OpenAI | text-embedding-3-small | Semantic search |
| Vector DB | ChromaDB | 0.4.24 | Embedding storage |
| LangChain | LangChain | 1.0.0 | LLM/Vector abstractions |
| Validation | Pydantic | 2.12.3 | Request/response validation |
| Server | Uvicorn | 0.24.0 | ASGI server |

---

## 🚀 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/search` | Search knowledge base |
| POST | `/api/documents/upload` | Upload documents |
| GET | `/api/documents` | List documents |
| DELETE | `/api/documents/{name}` | Delete document |
| POST | `/api/rate` | Rate an answer |
| GET | `/api/search-results` | Get recent searches |
| GET | `/api/health` | Health check |

---

## ✅ Quality Assurance

**Confidence Scoring**:
- 0.0-0.3: Low confidence (incomplete answer)
- 0.3-0.7: Medium confidence (partial answer)
- 0.7-1.0: High confidence (complete answer)

**Auto-Enrichment Triggers**:
- Confidence < 0.7 OR
- is_complete = False OR
- missing_info list is not empty

**Error Handling**:
- Graceful fallback if external sources unavailable
- Pydantic validation for all requests/responses
- Comprehensive logging for debugging

---

## 📈 Performance Metrics

- **Search Latency**: ~3-5 seconds (including LLM inference)
- **Enrichment Latency**: ~5-10 seconds (fetching from external sources)
- **Vector Store Size**: Scales with document count
- **Embedding Dimension**: 1536 (OpenAI standard)

---

## 🔐 Security Considerations

- OpenAI API key stored in environment variables
- CORS enabled for frontend access
- Input validation via Pydantic
- No sensitive data in logs
- File uploads validated by extension

---

## 📝 Next Steps for Production

1. **Database**: Replace JSON files with PostgreSQL for search results
2. **Caching**: Add Redis for frequently asked questions
3. **Monitoring**: Implement Prometheus metrics
4. **Authentication**: Add API key authentication
5. **Rate Limiting**: Implement per-user rate limits
6. **Deployment**: Docker containerization and Kubernetes orchestration


