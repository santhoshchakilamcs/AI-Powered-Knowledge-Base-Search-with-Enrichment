"""Configuration management for the RAG application."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = "your_openai_api_key_here"
    
    # Vector Store Configuration
    chroma_persist_directory: str = "./data/chroma_db"
    
    # Document Storage
    upload_directory: str = "./data/uploads"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # RAG Configuration (legacy char-based splitter, kept for compatibility)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    confidence_threshold: float = 0.7

    # Scaled retrieval (screenshot design, app/services/scaled_retrieval.py)
    chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50
    embedding_dim: int = 1536
    vector_top_k: int = 50
    keyword_top_k: int = 50
    rerank_top_k: int = 100
    final_top_k: int = 5
    scaled_store_path: str = "./data/scaled_store.json"
    qdrant_url: Optional[str] = None  # set to use Qdrant instead of the local store
    elasticsearch_url: Optional[str] = None  # set to use Elasticsearch BM25 instead of local BM25
    cohere_api_key: Optional[str] = None  # set to use Cohere Rerank instead of the local reranker
    
    # LLM Configuration
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    max_tokens: int = 2000
    
    # Embedding Model
    embedding_model: str = "text-embedding-3-small"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.upload_directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

