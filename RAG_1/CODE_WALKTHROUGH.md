# 💻 Code Walkthrough - Deep Dive into Implementation

---

## 1️⃣ Application Entry Point (`app/main.py`)

**Purpose**: Initialize FastAPI application with middleware and routes.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Create FastAPI app with metadata
app = FastAPI(
    title="AI-Powered Knowledge Base",
    description="RAG system with completeness detection",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include all API routes
app.include_router(router, prefix="/api", tags=["API"])

# Startup event - initialize services
@app.on_event("startup")
async def startup_event():
    logger.info(f"Vector store: {settings.chroma_persist_directory}")
    logger.info(f"LLM model: {settings.llm_model}")
```

**Key Points**:
- CORS enabled for frontend communication
- Automatic API documentation at `/docs`
- Startup logging for debugging

---

## 2️⃣ Search Endpoint (`app/api/routes.py` - Lines 542-629)

**Purpose**: Main API endpoint that orchestrates the entire RAG pipeline.

```python
@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
    """
    Search the knowledge base and generate an answer using RAG pipeline.
    
    Flow:
    1. Validate request
    2. Initialize RAG pipeline
    3. Execute semantic search + LLM generation
    4. Convert objects to dicts for Pydantic validation
    5. Save results to JSON
    6. Return structured response
    """
    try:
        logger.info(f"Processing search query: {request.query}")
        
        # Import services
        from app.services.rag_pipeline import RAGPipeline
        from app.services.search_results_service import search_results_service
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Execute RAG pipeline with auto-enrichment enabled
        response = await rag_pipeline.search_and_answer(
            query=request.query,
            top_k=request.top_k,
            enable_auto_enrichment=True  # Always enabled
        )
        
        # CRITICAL FIX: Convert Pydantic objects to dicts
        # Pydantic v2 requires dicts, not objects
        sources_dict = [
            source.dict() if hasattr(source, 'dict') else source
            for source in response.sources
        ]
        
        # Convert enrichment suggestions to dicts
        enrichment_suggestions_dicts = [
            s.dict() if hasattr(s, 'dict') else s
            for s in response.enrichment_suggestions
        ]
        
        # Determine if auto-enrichment was applied
        auto_enrichment_applied = any(
            s.type.value == 'external_source' 
            for s in response.enrichment_suggestions
        )
        
        # Extract external source names
        auto_enrichment_sources = [
            s.suggestion for s in response.enrichment_suggestions
            if s.type.value == 'external_source'
        ]
        
        # Save to JSON file
        result_file = await search_results_service.save_search_result(
            query=request.query,
            answer=response.answer,
            confidence=response.confidence,
            is_complete=response.is_complete,
            sources=sources_dict,
            missing_info=response.missing_info,
            enrichment_suggestions=enrichment_suggestions_dicts,
            auto_enrichment_applied=auto_enrichment_applied,
            auto_enrichment_sources=auto_enrichment_sources
        )
        
        # Return response with properly formatted dicts
        return SearchResponse(
            query=response.query,
            answer=response.answer,
            confidence=response.confidence,
            is_complete=response.is_complete,
            sources=sources_dict,
            missing_info=response.missing_info,
            enrichment_suggestions=enrichment_suggestions_dicts,
            auto_enrichment_applied=auto_enrichment_applied,
            auto_enrichment_sources=auto_enrichment_sources
        )
        
    except Exception as e:
        logger.error(f"Error processing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Critical Implementation Details**:

1. **Object-to-Dict Conversion**: 
   - Pydantic v2 requires dictionaries, not objects
   - Use `.dict()` method for conversion
   - This was the bug causing 500 errors

2. **Auto-Enrichment Always Enabled**:
   - `enable_auto_enrichment=True` ensures external sources are fetched
   - Triggered when confidence < 0.7 or is_complete = False

3. **Error Handling**:
   - Try-catch wraps entire operation
   - Logs errors for debugging
   - Returns HTTP 500 with error message

---

## 3️⃣ RAG Pipeline (`app/services/rag_pipeline.py`)

**Purpose**: Core intelligence - orchestrates retrieval and generation.

```python
class RAGPipeline:
    """Core RAG pipeline with completeness detection."""
    
    SYSTEM_PROMPT = """You are an AI assistant that answers questions based on documents.
    
Your task is to:
1. Answer using ONLY provided context
2. Assess completeness and confidence
3. Identify missing information
4. Suggest enrichment opportunities

IMPORTANT: Respond with JSON:
{
    "answer": "Your answer here",
    "confidence": 0.85,
    "is_complete": true,
    "missing_info": ["List of missing info"],
    "reasoning": "Why this confidence level",
    "relevant_sources": [0, 1, 2]
}

Guidelines:
- confidence: 0.0 to 1.0 (1.0 = completely confident)
- is_complete: true if all info available
- missing_info: specific gaps in knowledge
- Be honest about uncertainty
"""
    
    async def search_and_answer(
        self,
        query: str,
        top_k: int = 5,
        enable_auto_enrichment: bool = False
    ) -> SearchResponse:
        """Main RAG pipeline."""
        
        # Step 1: Retrieve documents
        retrieved_docs = await vector_store_service.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        if not retrieved_docs:
            return self._create_no_documents_response(query)
        
        # Step 2: Prepare context
        context = self._prepare_context(retrieved_docs)
        
        # Step 3: Generate answer with LLM
        llm_response = await self._generate_answer(
            query, context, retrieved_docs
        )
        
        # Step 4: Auto-enrichment if needed
        if enable_auto_enrichment and (
            llm_response.confidence < 0.7 or 
            not llm_response.is_complete
        ):
            logger.info(
                f"Answer incomplete (confidence: {llm_response.confidence}). "
                f"Auto-fetching from external sources..."
            )
            
            # Fetch from external sources
            enrichment_result = await enrichment_engine.auto_enrich(
                query=query,
                missing_info=llm_response.missing_info
            )
            
            # Re-run search with enriched documents
            llm_response = await self._generate_answer(
                query, context, retrieved_docs
            )
        
        return llm_response
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
        retrieved_docs: List[tuple]
    ) -> SearchResponse:
        """Generate answer using LLM."""
        
        # Prepare messages for LLM
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        
        # Call LLM
        response = await self.llm.ainvoke(messages)
        
        # Parse JSON response
        response_json = json.loads(response.content)
        
        # Build SearchResponse
        return SearchResponse(
            query=query,
            answer=response_json['answer'],
            confidence=response_json['confidence'],
            is_complete=response_json['is_complete'],
            sources=self._build_source_references(retrieved_docs),
            missing_info=response_json.get('missing_info', []),
            enrichment_suggestions=self._generate_enrichment_suggestions(
                response_json
            )
        )
```

**Key Concepts**:

1. **System Prompt Strategy**:
   - Instructs LLM to return JSON
   - Defines confidence and completeness criteria
   - Encourages honesty about gaps

2. **Completeness Detection**:
   - LLM assesses if answer is complete
   - Confidence < 0.7 triggers enrichment
   - Missing info explicitly listed

3. **Auto-Enrichment Trigger**:
   - Automatic, no user intervention needed
   - Fetches from Wikipedia, arXiv, PubMed
   - Re-runs search with enriched documents

---

## 4️⃣ Vector Store (`app/services/vector_store.py`)

**Purpose**: Manages embeddings and semantic search.

```python
class VectorStoreService:
    """Manages vector embeddings using ChromaDB."""
    
    def __init__(self):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize ChromaDB
        self.vector_store = Chroma(
            persist_directory="./data/chroma_db",
            embedding_function=self.embeddings,
            collection_name="knowledge_base"
        )
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store."""
        ids = self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents")
        return ids
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        Semantic search using embeddings.
        
        Process:
        1. Embed query: "What is XGBoost?" → [0.123, -0.456, ...]
        2. Compare with all document embeddings using cosine similarity
        3. Return top-k most similar documents with scores
        """
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )
        
        logger.info(f"Found {len(results)} similar documents")
        return results
```

**How Semantic Search Works**:

```
Query: "What is XGBoost?"
    ↓
OpenAI Embedding Model
    ↓
Query Vector: [0.123, -0.456, 0.789, ...] (1536 dimensions)
    ↓
ChromaDB Similarity Search (Cosine Similarity)
    ↓
Results:
- Doc 1: "XGBoost is a gradient boosting library..." (score: 0.95)
- Doc 2: "Gradient boosting explained..." (score: 0.87)
- Doc 3: "Machine learning algorithms..." (score: 0.72)
```

---

## 5️⃣ Enrichment Engine (`app/services/enrichment_engine.py`)

**Purpose**: Fetch information from trusted external sources.

```python
class EnrichmentEngine:
    """Auto-enrichment from trusted sources."""
    
    TRUSTED_SOURCES = {
        'wikipedia': {'enabled': True, 'priority': 1},
        'arxiv': {'enabled': True, 'priority': 2},
        'pubmed': {'enabled': True, 'priority': 3},
        'web_search': {'enabled': True, 'priority': 4}
    }
    
    async def auto_enrich(
        self,
        query: str,
        missing_info: List[str],
        max_sources: int = 3
    ) -> Dict[str, Any]:
        """Automatically enrich knowledge base."""
        
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
        
        # Add to vector store
        if enriched_docs:
            await vector_store_service.add_documents(enriched_docs)
            logger.info(f"Added {len(enriched_docs)} enriched documents")
        
        return {
            'enriched_count': len(enriched_docs),
            'sources': [doc.metadata['source'] for doc in enriched_docs]
        }
    
    async def _enrich_from_wikipedia(
        self,
        query: str,
        missing_info: List[str]
    ) -> List[Document]:
        """Fetch from Wikipedia."""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query)
            
            # Get page content
            page = wikipedia.page(search_results[0])
            
            # Create Document with metadata
            doc = Document(
                page_content=page.content,
                metadata={
                    'source': 'wikipedia',
                    'title': page.title,
                    'url': page.url,
                    'enriched': True
                }
            )
            
            return [doc]
        except Exception as e:
            logger.error(f"Wikipedia enrichment failed: {e}")
            return []
```

**Enrichment Flow**:

```
Query: "What is XGBoost?"
Missing Info: ["Description of XGBoost", "Applications"]
    ↓
Try Wikipedia
    ↓
Found: "XGBoost - Wikipedia"
    ↓
Extract content and create Document
    ↓
Add to vector store
    ↓
Re-run search with enriched documents
    ↓
Generate new answer with higher confidence
```

---

## 6️⃣ Data Persistence (`app/services/search_results_service.py`)

**Purpose**: Save all search results to JSON files.

```python
class SearchResultsService:
    """Persist search results to JSON files."""
    
    async def save_search_result(
        self,
        query: str,
        answer: str,
        confidence: float,
        is_complete: bool,
        sources: List[dict],
        missing_info: List[str],
        enrichment_suggestions: List[dict],
        auto_enrichment_applied: bool,
        auto_enrichment_sources: List[str]
    ) -> str:
        """Save search result to JSON file."""
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"search_{timestamp}_{query[:50]}.json"
        
        # Build result object
        result = {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "is_complete": is_complete,
            "sources": sources,
            "missing_info": missing_info,
            "enrichment_suggestions": enrichment_suggestions,
            "auto_enrichment_applied": auto_enrichment_applied,
            "auto_enrichment_sources": auto_enrichment_sources,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save to file
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved search result to: {filepath}")
        return str(filepath)
```

**JSON Output Example**:
```json
{
  "query": "what is machine learning",
  "answer": "Machine learning is a field of study...",
  "confidence": 1.0,
  "is_complete": true,
  "sources": [
    {
      "document_id": "enriched_wikipedia_...",
      "document_name": "wikipedia: Machine learning",
      "relevance_score": 0.803,
      "metadata": {"source": "wikipedia"}
    }
  ],
  "missing_info": [],
  "enrichment_suggestions": [...],
  "auto_enrichment_applied": true,
  "auto_enrichment_sources": ["✅ Fetched from Wikipedia"],
  "timestamp": "2025-10-19T16:59:03.469000"
}
```

---

## 🔍 Error Handling & Fixes

**Bug #1: Pydantic Validation Error (500 Internal Server Error)**

**Problem**:
```
6 validation errors for SearchResponse
enrichment_suggestions.0
  Input should be a valid dictionary [type=dict_type, input_value=EnrichmentSuggestion(...)]
```

**Root Cause**: Passing Pydantic objects instead of dicts to response model.

**Solution**:
```python
# Before (WRONG):
enrichment_suggestions = response.enrichment_suggestions  # Objects!
return SearchResponse(..., enrichment_suggestions=enrichment_suggestions)

# After (CORRECT):
enrichment_suggestions_dicts = [
    s.dict() if hasattr(s, 'dict') else s
    for s in response.enrichment_suggestions
]
return SearchResponse(..., enrichment_suggestions=enrichment_suggestions_dicts)
```

---

## 📊 Performance Optimization

**Caching Strategy**:
- Vector embeddings cached in ChromaDB
- No re-embedding of same documents
- LLM responses not cached (always fresh)

**Async/Await**:
- All I/O operations are async
- Parallel external source fetching
- Non-blocking API responses

**Batch Processing**:
- Documents chunked before embedding
- Chunk size: 1000 characters
- Overlap: 200 characters


