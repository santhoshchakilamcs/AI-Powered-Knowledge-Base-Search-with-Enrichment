# Scaled Retrieval (10M documents, answers in under 2 seconds)

This is the design from the interview screenshot, implemented in
`app/services/scaled_retrieval.py` and wired into the API. The original
ChromaDB-only search still exists at `POST /api/search`; the scaled
pipeline is at `POST /api/search/scaled`.

## The 7 steps, as implemented

1. **Chunk** — every document is split into ~500-token pieces with a
   50-token overlap, so an answer straddling a boundary is not cut in
   half. Token counting is approximate without a tokenizer installed
   (1 word ≈ 1.33 tokens); `tiktoken` is used when available.
2. **Embed** — each chunk becomes one 1,536-dim vector.
   `text-embedding-3-small` when `OPENAI_API_KEY` is set, otherwise a
   deterministic hashed bag-of-words embedding in 1,536 dims for offline
   testing. The hashed embedding is NOT a semantic model.
3. **Store** — binary quantization: the sign of each dimension is packed
   to 1 bit, so 100M chunks x 1,536 dims is ~20 GB in memory instead of
   ~600 GB of float32. The fast binary pass picks 200 candidates, which
   are rescored with the full float vectors kept on disk (here, in the
   local store file). At true scale, replace the local store with
   Qdrant: set `QDRANT_URL` and install `qdrant-client`, and run the
   same binary-quantized query there. The local store is NOT Qdrant,
   and this doc does not claim otherwise.
4. **Filter** — every chunk carries `allowed_teams`. Search runs only
   over chunks the caller may see: chunks with no teams are public,
   others require an intersection with the caller's teams. Filtering
   happens BEFORE ranking, never after, so a private chunk cannot leak
   through a high score.
5. **Hybrid search** — top 50 by meaning (vector) plus top 50 by exact
   words (BM25 keyword index, the Elasticsearch role), merged with
   reciprocal rank fusion (k=60). Ranks are merged, not raw scores,
   because cosine similarity and BM25 are not on the same scale. At
   true scale, replace local BM25 with an Elasticsearch query by
   setting `ELASTICSEARCH_URL`.
6. **Rerank** — the top 100 fused candidates are reranked. Cohere
   Rerank (`rerank-english-v3.0`) is used when `COHERE_API_KEY` is set;
   otherwise a local scorer (cosine + BM25 + query-term overlap) runs
   as an honest fallback, not a real cross-encoder. A BGE reranker can
   be swapped in here with the same interface.
7. **Generate** — only the top 5 reranked chunks go to the LLM, with an
   instruction to answer ONLY from them and cite the source. Without
   an OpenAI key, generation is extractive (top chunks + citations),
   so the pipeline runs offline and never invents unretrieved text.

## Latency budget (why under 2 seconds is plausible)

- Embed the query: tens of milliseconds
- Vector + keyword search in parallel: ~50-150 ms
- Rerank 100 candidates in one batch: ~100-300 ms
- Generate a short answer from 5 chunks, streaming: first tokens fast

The expensive steps (rerank, LLM) run on 100 and then 5 items, never on
100M. That funnel is the whole design.

## API

```bash
# Upload with a permission tag (comma-separated, empty = public)
curl -F "file=@doc.pdf" -F "allowed_teams=engineering,product" \
  http://localhost:8000/api/documents/upload

# Scaled search, only chunks the caller may see
curl -X POST http://localhost:8000/api/search/scaled \
  -H "Content-Type: application/json" \
  -d '{"query": "How does enrichment work?", "allowed_teams": ["engineering"]}'
```

## Environment

```
CHUNK_TOKENS=500
CHUNK_OVERLAP_TOKENS=50
VECTOR_TOP_K=50
KEYWORD_TOP_K=50
RERANK_TOP_K=100
FINAL_TOP_K=5
SCALED_STORE_PATH=./data/scaled_store.json
QDRANT_URL=            # optional, for true-scale deployment
ELASTICSEARCH_URL=     # optional, for true-scale deployment
COHERE_API_KEY=        # optional, for Cohere Rerank
```

## Tests

`tests/test_scaled_retrieval.py` covers, offline with no API keys:
chunk size and overlap, 1,536-dim embeddings, binary packing to 192
bytes, permission filtering (a team-restricted chunk is invisible to
other teams), hybrid search finding both a paraphrased and an
exact-term match, RRF fusion, reranking, and top-5-only generation
with citations.
