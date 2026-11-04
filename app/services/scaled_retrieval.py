"""Scaled RAG retrieval: the 7-step design from the interview screenshot.

Pipeline, in order:
  1. Chunk: ~500-token pieces with a small overlap.
  2. Embed: one 1,536-dim vector per chunk.
  3. Store: binary-quantized vectors in memory for a fast first pass,
     full float vectors kept for rescoring. At 10M docs / ~100M chunks
     this is the Qdrant pattern: ~600 GB of float32 vectors becomes
     ~20 GB of binary vectors in memory (100M x 1,536 bits), with the
     full vectors on disk. This module is the local, dependency-free
     version of that pattern; point it at Qdrant when QDRANT_URL is set
     (see SCALED_RETRIEVAL.md) instead of claiming a server is running.
  4. Filter: only chunks the caller is allowed to see are searched.
     Every chunk carries ``allowed_teams``. A chunk with no teams listed
     is treated as public. A chunk is visible only if its team list is
     empty or intersects the caller's teams.
  5. Hybrid search: top 50 by meaning (vector) plus top 50 by exact
     words (BM25 keyword index, the Elasticsearch role), merged with
     reciprocal rank fusion (RRF, k=60).
  6. Rerank: the top 100 fused candidates are reranked by a cross-encoder
     style scorer. Cohere Rerank is used when COHERE_API_KEY is set;
     otherwise a local scorer (cosine + BM25 + query-term overlap) runs,
     which is honest about being a fallback, not a real cross-encoder.
  7. Generate: only the top 5 reranked chunks go to the LLM, with an
     instruction to answer only from them and cite the source. Without
     an OpenAI key, generation is extractive (top chunks + citations)
     so the pipeline still runs and is testable offline.

Token counting is approximate: no tokenizer is bundled, so 1 word is
counted as ~1.33 tokens (English text averages ~0.75 words per token).
That makes a 500-token chunk about 376 words, with a 50-token (~38-word)
overlap. If ``tiktoken`` is installed, it is used instead for exact
counts. Embedding is likewise two-track: OpenAI ``text-embedding-3-small``
(1,536 dims) when OPENAI_API_KEY is set and the ``openai`` package is
installed, otherwise a deterministic hashed bag-of-words embedding in
1,536 dims, L2-normalized, so cosine similarity still reflects shared
terms and the whole pipeline is testable without an API key. The hashed
embedding is NOT a semantic model and must not be described as one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EMBEDDING_DIM = 1536
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
VECTOR_TOP_K = 50
KEYWORD_TOP_K = 50
RERANK_TOP_K = 100
FINAL_TOP_K = 5
RRF_K = 60
BINARY_FIRST_PASS = 200  # binary candidates rescored with full vectors

_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------- chunking
def estimate_tokens(text: str) -> int:
    """Approximate token count (see module docstring)."""
    try:
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.33)


def chunk_text(
    text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS
) -> List[str]:
    """Split text into ~``chunk_tokens`` pieces with ``overlap_tokens`` overlap.

    Words are the unit: chunk_words = chunk_tokens / 1.33, rounded.
    Overlap is taken from the end of the previous chunk so an answer
    straddling a boundary is not cut in half.
    """
    words = text.split()
    if not words:
        return []
    chunk_words = max(1, int(chunk_tokens / 1.33))
    overlap_words = max(0, int(overlap_tokens / 1.33))
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + chunk_words])
        if piece:
            chunks.append(piece)
        if start + chunk_words >= len(words):
            break
    return chunks


# ---------------------------------------------------------------- embedding
def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Deterministic hashed bag-of-words embedding, L2-normalized."""
    vec = [0.0] * dim
    for token in _WORD_RE.findall(text.lower()):
        idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Embed texts with OpenAI text-embedding-3-small if configured, else hash."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and api_key != "your_openai_api_key_here":
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(model="text-embedding-3-small", input=list(texts))
            return [list(item.embedding) for item in resp.data]
        except Exception:
            pass  # fall through to the offline embedding; never fail ingestion
    return [_hash_embedding(t) for t in texts]


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]


# ------------------------------------------------------- binary quantization
def pack_binary(vector: Sequence[float]) -> bytes:
    """Pack the sign of each dimension into bits: 1 bit per dim, 192 bytes."""
    out = bytearray()
    byte = 0
    bits = 0
    for value in vector:
        if value > 0:
            byte |= 1 << (7 - bits)
        bits += 1
        if bits == 8:
            out.append(byte)
            byte = 0
            bits = 0
    if bits:
        out.append(byte)
    return bytes(out)


def hamming_distance(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ------------------------------------------------------------------- BM25
def _terms(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def bm25_scores(query: str, docs: Sequence[str]) -> List[float]:
    """Classic BM25 (k1=1.5, b=0.75) over the given documents."""
    tokenized = [_terms(d) for d in docs]
    query_terms = _terms(query)
    n = len(docs)
    if n == 0:
        return []
    avg_len = sum(len(d) for d in tokenized) / n or 1.0
    df: Dict[str, int] = {}
    for doc in tokenized:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    scores = []
    for doc in tokenized:
        length = len(doc) or 1
        counts: Dict[str, int] = {}
        for term in doc:
            counts[term] = counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            tf = counts[term]
            score += idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * length / avg_len))
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> Dict[str, float]:
    """Merge ranked lists by rank (scores are not comparable across methods)."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused


# -------------------------------------------------------------------- store
@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    allowed_teams: List[str] = field(default_factory=list)
    source_url: str = ""
    vector: List[float] = field(default_factory=list)
    binary: bytes = b""

    def source_link(self) -> str:
        return self.source_url or self.document_name


class ScaledStore:
    """In-memory store with optional SQLite persistence for chunks/vectors.

    This is the local stand-in for the Qdrant + Elasticsearch pair in the
    screenshot. At true 100M-chunk scale, replace ``search``'s first pass
    with a Qdrant binary-quantized query and the keyword pass with an
    Elasticsearch BM25 query; the filter, RRF, rerank and top-5 steps
    stay the same. See SCALED_RETRIEVAL.md.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.chunks: Dict[str, Chunk] = {}
        self.db_path = db_path

    # -- ingestion ------------------------------------------------------
    def add_document(
        self,
        document_id: str,
        document_name: str,
        text: str,
        allowed_teams: Optional[Sequence[str]] = None,
        source_url: str = "",
    ) -> List[str]:
        pieces = chunk_text(text)
        vectors = embed_texts(pieces)
        ids = []
        for i, (piece, vector) in enumerate(zip(pieces, vectors)):
            chunk = Chunk(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                document_name=document_name,
                text=piece,
                allowed_teams=list(allowed_teams or []),
                source_url=source_url,
                vector=vector,
                binary=pack_binary(vector),
            )
            self.chunks[chunk.chunk_id] = chunk
            ids.append(chunk.chunk_id)
        return ids

    def add_chunks(
        self,
        document_id: str,
        document_name: str,
        chunk_texts: Sequence[str],
        allowed_teams: Optional[Sequence[str]] = None,
        source_url: str = "",
    ) -> List[str]:
        """Index chunks that are already ~500-token pieces (e.g. from upload)."""
        vectors = embed_texts(list(chunk_texts))
        ids = []
        for i, (piece, vector) in enumerate(zip(chunk_texts, vectors)):
            chunk = Chunk(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                document_name=document_name,
                text=piece,
                allowed_teams=list(allowed_teams or []),
                source_url=source_url,
                vector=vector,
                binary=pack_binary(vector),
            )
            self.chunks[chunk.chunk_id] = chunk
            ids.append(chunk.chunk_id)
        return ids

    def visible_chunks(self, allowed_teams: Optional[Sequence[str]]) -> List[Chunk]:
        """Permission filter, applied BEFORE ranking (step 4)."""
        if allowed_teams is None:
            return list(self.chunks.values())
        teams = set(allowed_teams)
        return [
            c
            for c in self.chunks.values()
            if not c.allowed_teams or teams.intersection(c.allowed_teams)
        ]

    # -- retrieval ------------------------------------------------------
    def search(
        self,
        query: str,
        allowed_teams: Optional[Sequence[str]] = None,
        vector_top_k: int = VECTOR_TOP_K,
        keyword_top_k: int = KEYWORD_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        final_top_k: int = FINAL_TOP_K,
    ) -> List[Dict[str, Any]]:
        visible = self.visible_chunks(allowed_teams)
        if not visible:
            return []
        query_vec = embed_query(query)

        # Vector pass: binary first pass, then full-vector rescore.
        binary_ranked = sorted(
            visible, key=lambda c: hamming_distance(c.binary, pack_binary(query_vec))
        )[:BINARY_FIRST_PASS]
        vector_ranked = sorted(
            binary_ranked, key=lambda c: cosine(query_vec, c.vector), reverse=True
        )[:vector_top_k]

        # Keyword pass: BM25 over the same visible set only.
        keyword_scores = bm25_scores(query, [c.text for c in visible])
        keyword_order = sorted(
            range(len(visible)), key=lambda i: keyword_scores[i], reverse=True
        )
        keyword_ranked = [visible[i] for i in keyword_order if keyword_scores[i] > 0][:keyword_top_k]

        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c in vector_ranked], [c.chunk_id for c in keyword_ranked]]
        )
        candidates = sorted(
            (self.chunks[cid] for cid in fused),
            key=lambda c: fused[c.chunk_id],
            reverse=True,
        )[:rerank_top_k]

        bm25_map = {visible[i].chunk_id: keyword_scores[i] for i in range(len(visible))}
        reranked = rerank(query, query_vec, candidates, keyword_scores_map=bm25_map)
        return [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "document_name": c.document_name,
                "source": c.source_link(),
                "text": c.text,
                "score": score,
                "allowed_teams": c.allowed_teams,
            }
            for c, score in reranked[:final_top_k]
        ]

    # -- persistence ----------------------------------------------------
    def save(self, path: Optional[str] = None) -> str:
        path = path or self.db_path or "data/scaled_store.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "document_name": c.document_name,
                "text": c.text,
                "allowed_teams": c.allowed_teams,
                "source_url": c.source_url,
                "vector": c.vector,
            }
            for c in self.chunks.values()
        ]
        Path(path).write_text(json.dumps(data))
        return path

    @classmethod
    def load(cls, path: str) -> "ScaledStore":
        store = cls(db_path=path)
        for item in json.loads(Path(path).read_text()):
            vector = item["vector"]
            store.chunks[item["chunk_id"]] = Chunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                document_name=item["document_name"],
                text=item["text"],
                allowed_teams=item.get("allowed_teams", []),
                source_url=item.get("source_url", ""),
                vector=vector,
                binary=pack_binary(vector),
            )
        return store


# ------------------------------------------------------------------ rerank
def _local_rerank_score(
    query: str, query_vec: Sequence[float], chunk: Chunk, bm25: float, bm25_max: float
) -> float:
    """Fallback cross-encoder-style score: cosine + BM25 + term overlap."""
    query_terms = set(_terms(query))
    chunk_terms = set(_terms(chunk.text))
    overlap = len(query_terms & chunk_terms) / max(1, len(query_terms))
    bm25_norm = (bm25 / bm25_max) if bm25_max else 0.0
    return 0.5 * cosine(query_vec, chunk.vector) + 0.3 * bm25_norm + 0.2 * overlap


def rerank(
    query: str,
    query_vec: Sequence[float],
    candidates: Sequence[Chunk],
    keyword_scores_map: Optional[Dict[str, float]] = None,
) -> List[Tuple[Chunk, float]]:
    """Rerank candidates. Cohere Rerank when configured, else local scorer."""
    keyword_scores_map = keyword_scores_map or {}
    bm25_max = max(keyword_scores_map.values()) if keyword_scores_map else 0.0
    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if cohere_key:
        try:
            import httpx  # type: ignore

            resp = httpx.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Authorization": f"Bearer {cohere_key}", "Content-Type": "application/json"},
                json={
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": [c.text for c in candidates],
                    "top_n": len(candidates),
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            scored = [(candidates[r["index"]], float(r["relevance_score"])) for r in results]
            return sorted(scored, key=lambda x: x[1], reverse=True)
        except Exception:
            pass  # fall back to the local scorer; never fail the query
    scored = [
        (c, _local_rerank_score(query, query_vec, c, keyword_scores_map.get(c.chunk_id, 0.0), bm25_max))
        for c in candidates
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------- generation
def generate_answer(query: str, top_chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate from ONLY the top chunks, with source citations.

    With an OpenAI key, the LLM is instructed to answer only from the
    chunks and to cite them. Without a key, the answer is extractive:
    the top chunks themselves, cited, so the pipeline is testable
    offline and never invents text it did not retrieve.
    """
    if not top_chunks:
        return {
            "answer": "I could not find any relevant documents to answer this question.",
            "confidence": 0.0,
            "is_complete": False,
            "sources": [],
        }
    context = "\n\n".join(
        f"[{i + 1}] {c['document_name']} ({c['source']}): {c['text']}" for i, c in enumerate(top_chunks)
    )
    sources = [
        {"chunk_id": c["chunk_id"], "document_name": c["document_name"], "source": c["source"]}
        for c in top_chunks
    ]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and api_key != "your_openai_api_key_here":
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer ONLY from the provided chunks. Cite each claim as [1], [2], etc. "
                        "If the chunks do not contain the answer, say so clearly instead of guessing.",
                    },
                    {"role": "user", "content": f"Chunks:\n{context}\n\nQuestion: {query}"},
                ],
            )
            return {
                "answer": resp.choices[0].message.content or "",
                "confidence": float(top_chunks[0].get("score", 0.5)),
                "is_complete": True,
                "sources": sources,
            }
        except Exception:
            pass
    answer = (
        "Based only on the retrieved sources:\n\n"
        + "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(top_chunks))
    )
    return {
        "answer": answer,
        "confidence": round(float(top_chunks[0].get("score", 0.5)), 4),
        "is_complete": True,
        "sources": sources,
    }


def answer_query(
    store: ScaledStore, query: str, allowed_teams: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Full scaled pipeline: filter -> hybrid -> rerank -> top-5 generate."""
    top = store.search(query, allowed_teams=allowed_teams)
    result = generate_answer(query, top)
    result["query"] = query
    result["retrieved_chunks"] = top
    return result
