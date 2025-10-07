"""Vector store service for managing embeddings and similarity search."""
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages vector embeddings and similarity search using ChromaDB."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store."""
        try:
            self.vector_store = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=self.embeddings,
                collection_name="knowledge_base"
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects with content and metadata
            
        Returns:
            List of document IDs added to the vector store
        """
        try:
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of most similar documents
        """
        if k is None:
            k = settings.top_k_results
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of tuples (document, relevance_score)
        """
        if k is None:
            k = settings.top_k_results
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {e}")
            raise
    
    async def delete_documents(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all chunks for this document
            results = self.vector_store.get(
                where={"document_id": document_id}
            )
            
            if results and results['ids']:
                self.vector_store.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'total_chunks': count,
                'collection_name': 'knowledge_base',
                'status': 'healthy' if count >= 0 else 'error'
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'total_chunks': 0,
                'collection_name': 'knowledge_base',
                'status': 'error'
            }
    
    def get_unique_documents_count(self) -> int:
        """Get count of unique documents in the vector store."""
        try:
            # Get all metadata
            results = self.vector_store.get()
            if results and results['metadatas']:
                unique_docs = set()
                for metadata in results['metadatas']:
                    if 'document_id' in metadata:
                        unique_docs.add(metadata['document_id'])
                return len(unique_docs)
            return 0
        except Exception as e:
            logger.error(f"Error getting unique documents count: {e}")
            return 0
    
    async def get_document_chunks(self, document_id: str) -> List[Document]:
        """Get all chunks for a specific document."""
        try:
            results = self.vector_store.get(
                where={"document_id": document_id}
            )
            
            if results and results['documents']:
                documents = []
                for i, doc_content in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    documents.append(Document(
                        page_content=doc_content,
                        metadata=metadata
                    ))
                return documents
            return []
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []


# Global instance
vector_store_service = VectorStoreService()

