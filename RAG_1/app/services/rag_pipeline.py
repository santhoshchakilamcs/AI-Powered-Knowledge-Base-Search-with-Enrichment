"""RAG pipeline with completeness detection and structured output."""
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage

from app.config import settings
from app.models.schemas import (
    SearchResponse,
    SourceReference,
    EnrichmentSuggestion,
    EnrichmentType
)
from app.services.vector_store import vector_store_service
from app.services.enrichment_engine import enrichment_engine

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Core RAG pipeline with completeness detection."""
    
    SYSTEM_PROMPT = """You are an AI assistant that answers questions based on provided documents.

Your task is to:
1. Answer the user's question using ONLY the information from the provided context documents
2. Assess the completeness and confidence of your answer
3. Identify any missing information that would improve the answer
4. Suggest ways to enrich the knowledge base

IMPORTANT: You must respond with a valid JSON object with the following structure:
{
    "answer": "Your detailed answer here",
    "confidence": 0.85,
    "is_complete": true,
    "missing_info": ["List of missing information"],
    "reasoning": "Explanation of your confidence and completeness assessment",
    "relevant_sources": [0, 1, 2]
}

Guidelines:
- confidence: 0.0 to 1.0, where 1.0 is completely confident
- is_complete: true if you have all information needed, false if information is missing or uncertain
- missing_info: List specific pieces of information that are missing or would improve the answer
- relevant_sources: List indices of the source documents you used (0-indexed)
- If the context doesn't contain relevant information, say so clearly and set confidence low
- Be honest about uncertainty - it's better to admit gaps than to make up information
"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key
        )
    
    async def search_and_answer(
        self,
        query: str,
        top_k: int = None,
        enable_auto_enrichment: bool = False
    ) -> SearchResponse:
        """
        Main RAG pipeline: retrieve relevant documents and generate answer.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            enable_auto_enrichment: Whether to attempt auto-enrichment

        Returns:
            SearchResponse with structured output
        """
        if top_k is None:
            top_k = settings.top_k_results

        logger.info(f"Processing query: {query}")

        # Step 1: Retrieve relevant documents
        retrieved_docs = await vector_store_service.similarity_search_with_score(
            query=query,
            k=top_k
        )

        if not retrieved_docs:
            return self._create_no_documents_response(query)

        # Step 2: Prepare context for LLM
        context = self._prepare_context(retrieved_docs)

        # Step 3: Generate answer with completeness detection
        llm_response = await self._generate_answer(query, context, retrieved_docs)

        # Step 4: Automatic enrichment - fetch from external sources ONLY if answer is incomplete
        # This happens automatically without user intervention
        enrichment_performed = False

        if not llm_response.is_complete and llm_response.missing_info:
            # Answer is incomplete - automatically fetch from trusted external sources
            logger.info(f"Answer incomplete (confidence: {llm_response.confidence}). Auto-fetching from external sources for missing info: {llm_response.missing_info}")

            llm_response.enrichment_suggestions = await self._generate_enrichment_suggestions(
                query=query,
                answer=llm_response.answer,
                missing_info=llm_response.missing_info,
                enable_auto_enrichment=True  # Always enable when answer is incomplete
            )

            # Check if enrichment was actually performed
            enrichment_performed = any(
                s.type == EnrichmentType.EXTERNAL_SOURCE
                for s in llm_response.enrichment_suggestions
            )

        # Step 5: If enrichment was performed, re-run the search with new content
        if enrichment_performed:
            logger.info("Re-running search after auto-enrichment")

            # Retrieve documents again (now including enriched content)
            retrieved_docs_enriched = await vector_store_service.similarity_search_with_score(
                query=query,
                k=top_k
            )

            # Prepare new context
            context_enriched = self._prepare_context(retrieved_docs_enriched)

            # Generate new answer with enriched content
            llm_response_enriched = await self._generate_answer(query, context_enriched, retrieved_docs_enriched)

            # Keep the enrichment suggestions from the first pass
            llm_response_enriched.enrichment_suggestions = llm_response.enrichment_suggestions

            logger.info(f"Re-generated answer with enriched content. New confidence: {llm_response_enriched.confidence}")

            return llm_response_enriched

        logger.info(f"Generated answer with confidence: {llm_response.confidence}")

        return llm_response
    
    def _prepare_context(self, retrieved_docs: List[tuple[Document, float]]) -> str:
        """Prepare context string from retrieved documents."""
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs):
            context_parts.append(
                f"[Source {i}] (Relevance: {score:.2f})\n"
                f"Document: {doc.metadata.get('filename', 'Unknown')}\n"
                f"Content: {doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    async def _generate_answer(
        self,
        query: str,
        context: str,
        retrieved_docs: List[tuple[Document, float]]
    ) -> SearchResponse:
        """Generate answer using LLM with structured output."""
        
        user_prompt = f"""Context Documents:
{context}

User Question: {query}

Please provide your response as a JSON object following the specified format."""
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Extract JSON if wrapped in markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])
            
            llm_output = json.loads(response_text)
            
            # Create source references
            sources = self._create_source_references(
                retrieved_docs,
                llm_output.get('relevant_sources', [])
            )
            
            # Build response
            return SearchResponse(
                query=query,
                answer=llm_output.get('answer', ''),
                confidence=float(llm_output.get('confidence', 0.5)),
                is_complete=llm_output.get('is_complete', False),
                sources=sources,
                missing_info=llm_output.get('missing_info', []),
                enrichment_suggestions=[]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            # Fallback to basic response
            return self._create_fallback_response(query, retrieved_docs, response.content)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _create_source_references(
        self,
        retrieved_docs: List[tuple[Document, float]],
        relevant_indices: List[int]
    ) -> List[SourceReference]:
        """Create source reference objects from retrieved documents."""
        sources = []

        for idx in relevant_indices:
            if 0 <= idx < len(retrieved_docs):
                doc, score = retrieved_docs[idx]
                # Normalize score to be between 0 and 1
                # ChromaDB can return scores > 1 depending on distance metric
                normalized_score = min(1.0, max(0.0, float(score)))

                sources.append(SourceReference(
                    document_id=doc.metadata.get('document_id', 'unknown'),
                    document_name=doc.metadata.get('filename', 'Unknown'),
                    chunk_id=doc.metadata.get('chunk_id', 'unknown'),
                    content=doc.page_content[:500],  # Truncate for response
                    relevance_score=normalized_score,
                    metadata=doc.metadata
                ))

        return sources
    
    async def _generate_enrichment_suggestions(
        self,
        query: str,
        answer: str,
        missing_info: List[str],
        enable_auto_enrichment: bool
    ) -> List[EnrichmentSuggestion]:
        """Generate suggestions for enriching the knowledge base."""
        suggestions = []

        # Automatically fetch from external sources when information is missing
        if enable_auto_enrichment and missing_info:
            logger.info(f"Fetching from trusted external sources to fill knowledge gaps: {missing_info}")

            try:
                enrichment_results = await enrichment_engine.auto_enrich(
                    query=query,
                    missing_info=missing_info,
                    max_sources=3  # Fetch from up to 3 trusted sources
                )

                if enrichment_results['success']:
                    # Add suggestions showing what was enriched with URLs
                    for i, source_name in enumerate(enrichment_results['sources_added']):
                        # Get the corresponding content item to extract URL
                        content_item = enrichment_results['content_added'][i] if i < len(enrichment_results['content_added']) else None
                        source_url = content_item.get('url', '') if content_item else ''

                        # Extract just the title from source_name (e.g., "Wikipedia: Article" -> "Article")
                        title = source_name.split(': ', 1)[1] if ': ' in source_name else source_name
                        source_type = source_name.split(': ', 1)[0] if ': ' in source_name else 'External'

                        suggestions.append(EnrichmentSuggestion(
                            type=EnrichmentType.EXTERNAL_SOURCE,
                            suggestion=f"✅ Fetched from {source_type}: {title}",
                            priority="high",
                            reasoning=f"Automatically retrieved from trusted source ({source_type}) to fill knowledge gaps",
                            auto_enrichment_available=True,
                            external_source_url=source_url if source_url else None
                        ))

                    logger.info(f"Successfully auto-enriched with {len(enrichment_results['sources_added'])} sources")
                else:
                    logger.warning("Auto-enrichment attempted but no content was added")

            except Exception as e:
                logger.error(f"Error during auto-enrichment: {e}")

        # Generate document upload suggestions for remaining missing info
        for info in missing_info[:3]:  # Limit to top 3
            suggestions.append(EnrichmentSuggestion(
                type=EnrichmentType.DOCUMENT,
                suggestion=f"Upload documents containing information about: {info}",
                priority="high" if len(suggestions) == 0 else "medium",
                reasoning=f"This information is needed to fully answer the question: '{query}'",
                auto_enrichment_available=enable_auto_enrichment
            ))

        return suggestions
    
    def _create_no_documents_response(self, query: str) -> SearchResponse:
        """Create response when no documents are found."""
        return SearchResponse(
            query=query,
            answer="I couldn't find any relevant documents in the knowledge base to answer your question.",
            confidence=0.0,
            is_complete=False,
            sources=[],
            missing_info=["No documents available in the knowledge base"],
            enrichment_suggestions=[
                EnrichmentSuggestion(
                    type=EnrichmentType.DOCUMENT,
                    suggestion="Upload documents related to this topic",
                    priority="high",
                    reasoning="The knowledge base appears to be empty or doesn't contain relevant information",
                    auto_enrichment_available=False
                )
            ]
        )
    
    def _create_fallback_response(
        self,
        query: str,
        retrieved_docs: List[tuple[Document, float]],
        raw_answer: str
    ) -> SearchResponse:
        """Create fallback response when JSON parsing fails."""
        sources = self._create_source_references(
            retrieved_docs,
            list(range(min(3, len(retrieved_docs))))
        )
        
        return SearchResponse(
            query=query,
            answer=raw_answer,
            confidence=0.5,
            is_complete=False,
            sources=sources,
            missing_info=["Unable to assess completeness"],
            enrichment_suggestions=[]
        )


# Global instance
rag_pipeline = RAGPipeline()

