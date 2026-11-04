"""Offline tests for the scaled retrieval pipeline (no API keys needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.scaled_retrieval import (
    ScaledStore,
    answer_query,
    bm25_scores,
    chunk_text,
    embed_texts,
    estimate_tokens,
    hamming_distance,
    pack_binary,
    reciprocal_rank_fusion,
)


def _store():
    store = ScaledStore()
    store.add_document(
        "doc-1", "enrichment.txt",
        "Enrichment fetches missing facts from Wikipedia, arXiv, and PubMed "
        "when an answer is incomplete or low confidence. " * 30,
        allowed_teams=["engineering"],
    )
    store.add_document(
        "doc-2", "public.txt",
        "The vector database stores embeddings and returns the nearest "
        "chunks by cosine similarity for a query. " * 30,
        allowed_teams=[],
    )
    store.add_document(
        "doc-3", "secret.txt",
        "The secret launch code is ZEBRA-99 and only the product team may see it. " * 30,
        allowed_teams=["product"],
    )
    return store


def test_chunk_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # ~500 tokens per chunk, so no chunk should be wildly over that.
    assert all(estimate_tokens(c) <= 600 for c in chunks)
    # Overlap: the second chunk starts inside the first chunk's words.
    first_words = set(chunks[0].split())
    assert len(first_words & set(chunks[1].split())) > 0


def test_embedding_dim_and_binary_packing():
    vec = embed_texts(["hello world"])[0]
    assert len(vec) == 1536
    packed = pack_binary(vec)
    assert len(packed) == 192  # 1536 bits
    assert hamming_distance(packed, packed) == 0


def test_permission_filter_hides_other_teams_chunks():
    store = _store()
    results = store.search("secret launch code ZEBRA-99", allowed_teams=["engineering"])
    texts = " ".join(r["text"] for r in results)
    assert "ZEBRA-99" not in texts
    results = store.search("secret launch code ZEBRA-99", allowed_teams=["product"])
    assert any("ZEBRA-99" in r["text"] for r in results)


def test_hybrid_finds_exact_term_and_meaning():
    store = _store()
    # Exact term only in doc-1 (keyword pass), meaning in doc-2 (vector pass).
    results = store.search("enrichment Wikipedia arXiv", allowed_teams=["engineering"])
    assert results
    assert len(results) <= 5  # top 5 only go to generation
    assert any("Wikipedia" in r["text"] for r in results)


def test_rrf_merges_ranks_not_scores():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert fused["b"] > fused["a"] and fused["b"] > fused["c"]


def test_bm25_ranks_exact_match_first():
    scores = bm25_scores("zebra", ["nothing here at all", "the zebra runs fast"])
    assert scores[1] > scores[0]


def test_answer_query_cites_sources_and_only_top_chunks():
    store = _store()
    result = answer_query(store, "How does enrichment work?", allowed_teams=["engineering"])
    assert result["sources"]
    assert len(result["retrieved_chunks"]) <= 5
    assert "Wikipedia" in result["answer"] or "enrichment" in result["answer"].lower()
