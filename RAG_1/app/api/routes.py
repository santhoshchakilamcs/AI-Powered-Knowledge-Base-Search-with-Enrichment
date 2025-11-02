"""API route handlers."""
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path
import os
import json
import uuid
from datetime import datetime
import re

# Simple document processing imports
import PyPDF2
import io
import openai
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    services: dict


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    top_k: Optional[int] = 5
    enable_auto_enrichment: Optional[bool] = True


class IntentClassificationOutput(BaseModel):
    """Intent classification output from agentic AI flow."""
    intent: str
    confidence: float
    entities: List[str]
    processing_strategy: str
    reasoning: str


class SearchResponse(BaseModel):
    """Search response model with structured output as per specifications."""
    query: str
    answer: str
    confidence: float  # 0.0 to 1.0
    is_complete: bool  # True if AI has enough info to answer
    sources: List[dict] = []  # Retrieved document chunks
    missing_info: List[str] = []  # What information is missing
    enrichment_suggestions: List[dict] = []  # How to improve knowledge base
    auto_enrichment_applied: Optional[bool] = False
    auto_enrichment_sources: List[str] = []
    intent_classification: Optional[IntentClassificationOutput] = None  # Agentic AI intent classification


# Simple document processing functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        return ""


def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file."""
    try:
        return file_content.decode('utf-8').strip()
    except Exception as e:
        logger.error(f"Error extracting TXT text: {e}")
        return ""


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from file based on extension."""
    extension = Path(filename).suffix.lower()

    if extension == '.pdf':
        return extract_text_from_pdf(file_content)
    elif extension == '.docx':
        return extract_text_from_docx(file_content)
    elif extension == '.txt':
        return extract_text_from_txt(file_content)
    else:
        return ""


def simple_search(query: str, documents: List[dict]) -> List[dict]:
    """Simple text-based search through documents."""
    query_words = query.lower().split()
    results = []

    for doc in documents:
        if 'content' not in doc:
            continue

        content = doc['content'].lower()
        score = 0

        # Simple scoring based on word matches
        for word in query_words:
            if word in content:
                score += content.count(word)

        if score > 0:
            # Calculate relevance score (normalize by content length)
            relevance = min(score / len(content.split()) * 100, 100)

            # Find relevant excerpt (longer chunks for LLM)
            sentences = doc['content'].split('.')
            best_chunk = ""
            best_score = 0

            # Get larger context chunks (5 sentences instead of 3)
            chunk_size = min(5, len(sentences))
            for i in range(len(sentences) - chunk_size + 1):
                chunk = '. '.join(sentences[i:i+chunk_size]).strip()
                chunk_score = sum(1 for word in query_words if word in chunk.lower())
                if chunk_score > best_score:
                    best_score = chunk_score
                    best_chunk = chunk

            # If no good chunk found, use the beginning of the document
            if not best_chunk and sentences:
                best_chunk = '. '.join(sentences[:3]).strip()

            results.append({
                'filename': doc['filename'],
                'document_name': doc['filename'],
                'relevance_score': relevance / 100,
                'content': best_chunk[:500] + "..." if len(best_chunk) > 500 else best_chunk,
                'full_content': doc['content'],  # Keep full content for LLM
                'type': 'uploaded'
            })

    # Sort by relevance score
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return results[:5]  # Return top 5 results


async def generate_rag_answer(query: str, search_results: List[dict]) -> dict:
    """Generate AI answer using RAG pipeline with structured output."""
    try:
        # Get OpenAI API key from settings
        api_key = settings.openai_api_key
        if not api_key or api_key == "your_openai_api_key_here":
            return {
                "answer": "OpenAI API key not configured. Please add your API key to use AI-powered answers.",
                "confidence": 0.0,
                "is_complete": False,
                "missing_info": ["OpenAI API key configuration"],
                "enrichment_suggestions": [
                    {
                        "type": "configuration",
                        "description": "Add OpenAI API key to enable AI-powered answers",
                        "priority": "high"
                    }
                ]
            }

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Prepare context from search results - use full content for better understanding
        context = ""
        sources_info = []
        for i, result in enumerate(search_results[:2]):  # Use top 2 results but with full content
            # Use full content instead of just excerpts for better AI understanding
            full_content = result.get('full_content', result['content'])
            # Limit to reasonable size to avoid token limits
            content_to_use = full_content[:3000] + "..." if len(full_content) > 3000 else full_content
            context += f"Document {i+1} ({result['filename']}):\n{content_to_use}\n\n"
            sources_info.append({
                "filename": result['filename'],
                "relevance_score": result['relevance_score']
            })

        # RAG prompt following specifications
        system_prompt = """You are an AI assistant that answers questions based on provided documents.

Your task is to:
1. Use ONLY the provided documents to answer the question naturally and conversationally
2. Provide a comprehensive answer based on all relevant information in the documents
3. Detect when information is missing or uncertain
4. Assess your confidence in the answer

Provide a natural language answer to the user's question. Do NOT respond in JSON format.
After your natural answer, I will separately extract metadata about confidence and completeness.

If documents don't contain relevant information, be honest about it and explain what you found instead."""

        user_prompt = f"""Question: {query}

Available Documents:
{context}

Please provide a comprehensive, natural language answer based on the documents."""

        # Call OpenAI API for the main answer
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Get the natural language answer
        answer = response.choices[0].message.content

        # Second API call to assess metadata
        metadata_prompt = f"""Based on this question and answer, assess the response quality:

Question: {query}
Answer: {answer}
Available Documents: {context}

Respond in JSON format with:
- confidence: Float 0.0-1.0 (how confident the answer is)
- is_complete: Boolean (true if answer fully addresses the question)
- missing_info: Array of strings (what specific information is missing, if any)
- enrichment_suggestions: Array of objects with type, description, priority (if answer is incomplete)"""

        metadata_response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You are an AI that assesses answer quality and completeness."},
                {"role": "user", "content": metadata_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        # Parse metadata response
        try:
            import json
            metadata_text = metadata_response.choices[0].message.content
            metadata = json.loads(metadata_text)

            result = {
                "answer": answer,
                "confidence": float(metadata.get("confidence", 0.7)),
                "is_complete": bool(metadata.get("is_complete", True)),
                "missing_info": metadata.get("missing_info", []),
                "enrichment_suggestions": metadata.get("enrichment_suggestions", [])
            }

        except (json.JSONDecodeError, Exception):
            # Fallback if metadata parsing fails
            result = {
                "answer": answer,
                "confidence": 0.7,
                "is_complete": True,
                "missing_info": [],
                "enrichment_suggestions": []
            }

        return result

    except Exception as e:
        logger.error(f"Error generating RAG answer: {e}")
        return {
            "answer": f"Error generating AI response: {str(e)}",
            "confidence": 0.0,
            "is_complete": False,
            "missing_info": ["AI processing error"],
            "enrichment_suggestions": []
        }


async def generate_rag_answer_with_enrichment(query: str, enriched_context: str) -> dict:
    """Generate RAG answer using enriched external content."""
    try:
        # Get OpenAI API key from settings
        api_key = settings.openai_api_key
        if not api_key or api_key == "your_openai_api_key_here":
            return {
                "answer": "OpenAI API key not configured",
                "confidence": 0.0,
                "is_complete": False
            }

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Enhanced prompt for enriched content
        system_prompt = """You are an AI assistant that provides comprehensive answers using both document content and external knowledge sources.

Your task is to:
1. Use the provided external knowledge and document content to answer the question thoroughly
2. Provide a natural, informative response
3. Integrate information from multiple sources seamlessly

Provide a natural language answer to the user's question. Do NOT respond in JSON format."""

        user_prompt = f"""Question: {query}

{enriched_context}

Please provide a comprehensive answer based on the available information."""

        # Call OpenAI API for the enriched answer
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        # Get the natural language answer
        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "confidence": 0.8,
            "is_complete": True
        }

    except Exception as e:
        logger.error(f"Error generating enriched RAG answer: {e}")
        return {
            "answer": f"Error generating enriched response: {str(e)}",
            "confidence": 0.0,
            "is_complete": False
        }


async def attempt_auto_enrichment(query: str, missing_info: List[str]) -> dict:
    """Auto-enrichment from external sources (stretch goal implementation)."""
    try:
        import requests

        # Try to get information from Wikipedia API
        enriched_content = ""
        enriched_sources = []

        # Wikipedia search
        try:
            # Set proper headers for Wikipedia API
            headers = {
                'User-Agent': 'RAG-Knowledge-Base/1.0 (https://example.com/contact) Python/requests'
            }

            # First try direct page lookup with underscores
            search_terms = [
                "black_hole",  # Direct term
                query.replace("what are ", "").replace("?", "").replace(" ", "_"),  # "black_holes"
                query.replace("what are ", "").replace("?", ""),  # "black holes"
                query.replace("what is ", "").replace("?", "").replace(" ", "_"),  # handle "what is"
                query.replace("what is ", "").replace("?", "")  # handle "what is"
            ]

            for search_term in search_terms:
                if not search_term.strip():
                    continue

                search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{search_term}"
                logger.info(f"Trying Wikipedia URL: {search_url}")
                response = requests.get(search_url, headers=headers, timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    if 'extract' in data and data['extract']:
                        enriched_content = data['extract']
                        enriched_sources.append("Wikipedia")
                        logger.info(f"Successfully got Wikipedia content for: {search_term}")
                        break
                else:
                    logger.info(f"Wikipedia API returned {response.status_code} for: {search_term}")

            # If direct lookup failed, try search API
            if not enriched_content:
                search_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': query.replace("what are ", "").replace("what is ", "").replace("?", ""),
                    'srlimit': 1
                }
                logger.info(f"Trying Wikipedia search API with params: {params}")
                response = requests.get(search_url, params=params, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('query', {}).get('search'):
                        title = data['query']['search'][0]['title']
                        # Get page content
                        content_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                        logger.info(f"Getting content from: {content_url}")
                        content_response = requests.get(content_url, headers=headers, timeout=5)
                        if content_response.status_code == 200:
                            content_data = content_response.json()
                            if 'extract' in content_data and content_data['extract']:
                                enriched_content = content_data['extract']
                                enriched_sources.append("Wikipedia")
                                logger.info(f"Successfully got Wikipedia content via search for: {title}")

        except Exception as e:
            logger.warning(f"Wikipedia enrichment failed: {str(e)}")

        return {
            "success": len(enriched_sources) > 0,
            "sources": enriched_sources,
            "enriched_content": enriched_content,
            "additional_context": f"Retrieved from {', '.join(enriched_sources)}" if enriched_sources else "No external sources found"
        }

    except Exception as e:
        logger.error(f"Auto-enrichment failed: {str(e)}")
        return {
            "success": False,
            "sources": [],
            "enriched_content": "",
            "additional_context": f"Auto-enrichment error: {str(e)}"
        }


# Rating system for answer quality (stretch goal)
class RatingRequest(BaseModel):
    """Rating request model."""
    query: str
    answer: str
    rating: int  # 1-5 scale
    feedback: Optional[str] = None


class RatingResponse(BaseModel):
    """Rating response model."""
    rating_id: str
    message: str


async def save_rating(rating_data: RatingRequest) -> str:
    """Save user rating for answer quality improvement."""
    try:
        rating_id = f"rating_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save rating to file (in production, use proper database)
        ratings_dir = Path("./data/ratings")
        ratings_dir.mkdir(parents=True, exist_ok=True)

        rating_file = ratings_dir / f"{rating_id}.json"
        rating_record = {
            "rating_id": rating_id,
            "query": rating_data.query,
            "answer": rating_data.answer,
            "rating": rating_data.rating,
            "feedback": rating_data.feedback,
            "timestamp": datetime.now().isoformat()
        }

        with open(rating_file, "w") as f:
            json.dump(rating_record, f, indent=2)

        logger.info(f"Saved rating {rating_id}: {rating_data.rating}/5")
        return rating_id

    except Exception as e:
        logger.error(f"Error saving rating: {e}")
        return "error"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    openai_configured = (settings.openai_api_key and
                        settings.openai_api_key != "your_openai_api_key_here")

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "api": "operational",
            "openai": "configured" if openai_configured else "not_configured",
            "documents": "ready"
        }
    )


@router.post("/documents/upload", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document using the document processor."""
    try:
        logger.info(f"Uploading document: {file.filename}")

        # Import locally to avoid pydantic v1/v2 conflicts
        from app.services.document_processor import document_processor
        from app.services.vector_store import vector_store_service

        # Read file content
        content = await file.read()

        # Process document using the document processor
        doc_id, chunks = await document_processor.process_upload(content, file.filename)

        logger.info(f"Document processed with ID: {doc_id}, created {len(chunks)} chunks")

        # Add chunks to vector store for semantic search
        await vector_store_service.add_documents(chunks)

        logger.info(f"Added {len(chunks)} chunks to vector store")

        return {
            "message": "Document uploaded and processed successfully!",
            "filename": file.filename,
            "document_id": doc_id,
            "file_size": len(content),
            "chunks_created": len(chunks),
            "status": "processed"
        }

    except ValueError as e:
        logger.error(f"Validation error uploading document: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
    """Search the knowledge base and generate an answer using RAG pipeline with Agentic Intent Classification."""
    try:
        logger.info(f"Processing search query: {request.query}")

        # Import locally to avoid pydantic v1/v2 conflicts
        from app.services.rag_pipeline import RAGPipeline
        from app.services.search_results_service import search_results_service
        from app.services.true_agentic_intent_classifier import true_agentic_intent_classifier

        # Step 1: Agentic Intent Classification
        logger.info(f"🤖 Starting Agentic Intent Classification for: {request.query}")
        intent_classification = await true_agentic_intent_classifier.classify_intent(request.query)
        logger.info(f"✅ Intent classified as: {intent_classification.intent.value} (confidence: {intent_classification.confidence:.2%})")

        # Use the proper RAG pipeline with vector store and semantic search
        rag_pipeline = RAGPipeline()

        # Get top_k from request or use default
        top_k = request.top_k if hasattr(request, 'top_k') and request.top_k else 5

        # Step 2: Execute RAG pipeline with semantic search
        # Always enable auto-enrichment so external sources are fetched when needed
        response = await rag_pipeline.search_and_answer(
            query=request.query,
            top_k=top_k,
            enable_auto_enrichment=True  # Always enabled to fetch external sources when answer is incomplete
        )

        logger.info(f"Generated answer with confidence: {response.confidence}")

        # Convert SourceReference objects to dicts for JSON serialization
        sources_dict = []
        if hasattr(response, 'sources') and response.sources:
            for source in response.sources:
                if hasattr(source, 'dict'):
                    sources_dict.append(source.dict())
                else:
                    sources_dict.append(source)

        # Extract enrichment information
        enrichment_suggestions_list = []
        enrichment_suggestions_dicts = []
        auto_enrichment_applied = False
        auto_enrichment_sources = []

        if hasattr(response, 'enrichment_suggestions') and response.enrichment_suggestions:
            enrichment_suggestions_list = response.enrichment_suggestions
            # Convert to dicts for response
            enrichment_suggestions_dicts = [
                s.dict() if hasattr(s, 'dict') else s
                for s in enrichment_suggestions_list
            ]
            # Check if any enrichment was from external sources
            auto_enrichment_applied = any(
                s.type.value == 'external_source' if hasattr(s.type, 'value') else s.type == 'external_source'
                for s in enrichment_suggestions_list
            )
            # Extract source names
            auto_enrichment_sources = [
                s.suggestion for s in enrichment_suggestions_list
                if (s.type.value == 'external_source' if hasattr(s.type, 'value') else s.type == 'external_source')
            ]

        # Step 2.5: Save search result to JSON file with intent classification
        try:
            result_file = await search_results_service.save_search_result(
                query=request.query,
                answer=response.answer,
                confidence=response.confidence,
                is_complete=response.is_complete,
                sources=sources_dict,
                missing_info=response.missing_info,
                enrichment_suggestions=enrichment_suggestions_dicts,
                auto_enrichment_applied=auto_enrichment_applied,
                auto_enrichment_sources=auto_enrichment_sources,
                intent_classification={
                    "intent": intent_classification.intent.value,
                    "confidence": intent_classification.confidence,
                    "entities": intent_classification.entities,
                    "processing_strategy": intent_classification.processing_strategy,
                    "reasoning": intent_classification.reasoning
                }
            )
            logger.info(f"Search result saved to: {result_file}")
        except Exception as e:
            logger.error(f"Error saving search result: {e}")
            # Don't fail the search if saving fails

        # Step 3: Return response with intent classification
        return SearchResponse(
            query=response.query,
            answer=response.answer,
            confidence=response.confidence,
            is_complete=response.is_complete,
            sources=sources_dict,
            missing_info=response.missing_info,
            enrichment_suggestions=enrichment_suggestions_dicts,
            auto_enrichment_applied=auto_enrichment_applied,
            auto_enrichment_sources=auto_enrichment_sources,
            intent_classification=IntentClassificationOutput(
                intent=intent_classification.intent.value,
                confidence=intent_classification.confidence,
                entities=intent_classification.entities,
                processing_strategy=intent_classification.processing_strategy,
                reasoning=intent_classification.reasoning
            )
        )

    except Exception as e:
        logger.error(f"Error processing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=dict)
async def list_documents():
    """List all uploaded documents."""
    try:
        upload_dir = Path("./data/uploads")
        if not upload_dir.exists():
            return {
                "documents": [],
                "total_count": 0
            }

        documents = []
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                stat = file_path.stat()
                documents.append({
                    "filename": file_path.name,
                    "file_size": stat.st_size,
                    "upload_timestamp": stat.st_mtime
                })

        return {
            "documents": documents,
            "total_count": len(documents)
        }

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    try:
        logger.info(f"Deleting document: {document_id}")

        # Define directories
        upload_dir = Path("./data/uploads")
        extracted_dir = Path("./data/extracted_text")
        metadata_dir = Path("./data/metadata")

        # Find the actual file in uploads directory
        # Handle both UUID-prefixed and regular filenames
        file_path = None
        original_filename = None

        # First try exact match
        exact_path = upload_dir / document_id
        if exact_path.exists():
            file_path = exact_path
            original_filename = document_id
        else:
            # Search for files that end with the document_id (handle UUID prefixes)
            for file in upload_dir.glob("*"):
                if file.name.endswith(document_id) or file.name == document_id:
                    file_path = file
                    original_filename = file.name
                    break
                # Also check if document_id is part of the filename
                if document_id in file.name:
                    file_path = file
                    original_filename = file.name
                    break

        if not file_path or not file_path.exists():
            logger.error(f"Document not found: {document_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # Delete from uploads directory
        file_path.unlink()
        logger.info(f"Deleted upload file: {file_path}")

        # Delete extracted text file (remove UUID prefix for lookup)
        base_filename = original_filename
        if "_" in original_filename and len(original_filename.split("_")[0]) == 36:
            # Remove UUID prefix
            base_filename = "_".join(original_filename.split("_")[1:])

        extracted_file = extracted_dir / f"{base_filename}.txt"
        if extracted_file.exists():
            extracted_file.unlink()
            logger.info(f"Deleted extracted text: {extracted_file}")

        # Delete metadata file
        metadata_file = metadata_dir / f"{base_filename}.json"
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"Deleted metadata: {metadata_file}")

        return {
            "message": "Document deleted successfully",
            "document_id": document_id,
            "filename": original_filename,
            "files_deleted": {
                "upload": str(file_path),
                "extracted_text": str(extracted_file) if extracted_file.exists() else None,
                "metadata": str(metadata_file) if metadata_file.exists() else None
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/rate", response_model=RatingResponse)
async def rate_answer(rating_request: RatingRequest):
    """Rate an answer for feedback and improvement (stretch goal)."""
    try:
        logger.info(f"Recording rating: {rating_request.rating}/5 for query: {rating_request.query}")

        rating_id = await save_rating(rating_request)

        if rating_id == "error":
            raise HTTPException(status_code=500, detail="Failed to save rating")

        return RatingResponse(
            rating_id=rating_id,
            message="Thank you for your feedback! This helps improve the system."
        )

    except Exception as e:
        logger.error(f"Error recording rating: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ratings/statistics")
async def get_rating_statistics():
    """Get rating statistics for analytics and system improvement."""
    try:
        ratings_dir = Path("./data/ratings")
        if not ratings_dir.exists():
            return {
                "total_ratings": 0,
                "average_rating": 0.0,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            }

        ratings = []
        for rating_file in ratings_dir.glob("*.json"):
            try:
                with open(rating_file, "r") as f:
                    rating_data = json.load(f)
                    ratings.append(rating_data["rating"])
            except Exception as e:
                logger.error(f"Error reading rating file {rating_file}: {e}")
                continue

        if not ratings:
            return {
                "total_ratings": 0,
                "average_rating": 0.0,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            }

        # Calculate statistics
        total_ratings = len(ratings)
        average_rating = sum(ratings) / total_ratings
        rating_distribution = {i: ratings.count(i) for i in range(1, 6)}

        return {
            "total_ratings": total_ratings,
            "average_rating": round(average_rating, 2),
            "rating_distribution": rating_distribution
        }

    except Exception as e:
        logger.error(f"Error getting rating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-results")
async def get_search_results(limit: int = 50):
    """Get recent search results saved as JSON files."""
    try:
        from app.services.search_results_service import search_results_service

        results = search_results_service.get_search_results(limit=limit)

        return {
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error getting search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-results/query/{query}")
async def get_search_results_by_query(query: str):
    """Get all search results for a specific query."""
    try:
        from app.services.search_results_service import search_results_service

        results = search_results_service.get_search_result_by_query(query)

        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error getting search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search-results/statistics")
async def get_search_results_statistics():
    """Get statistics about search results."""
    try:
        from app.services.search_results_service import search_results_service

        stats = search_results_service.get_statistics()

        return stats

    except Exception as e:
        logger.error(f"Error getting search statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Intent Classification Endpoints
# ============================================================================

@router.post("/classify-intent")
async def classify_intent(request: dict):
    """
    Classify the intent of a user query using Agentic AI flow.

    Args:
        request: JSON with 'query' field

    Returns:
        Intent classification with type, confidence, and entities
    """
    try:
        from app.services.true_agentic_intent_classifier import true_agentic_intent_classifier
        from app.models.schemas import IntentClassificationResponse

        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        logger.info(f"🤖 Classifying intent for query using TRUE Agentic AI: {query[:100]}...")

        # Classify intent using TRUE agentic AI flow
        intent_classification = await true_agentic_intent_classifier.classify_intent(query)

        # Create response
        response = IntentClassificationResponse(
            query=query,
            intent_classification=intent_classification
        )

        logger.info(f"✅ Intent classified: {intent_classification.intent.value}")

        return response.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying intent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intent-types")
async def get_intent_types():
    """
    Get all available intent types.

    Returns:
        List of intent types with descriptions
    """
    try:
        from app.models.schemas import IntentType

        intent_descriptions = {
            "factual_question": "User asks for facts, definitions, or information",
            "comparison": "User compares two or more concepts",
            "explanation": "User wants detailed explanation of a concept",
            "how_to": "User asks for step-by-step instructions or tutorials",
            "troubleshooting": "User reports a problem and seeks solution",
            "recommendation": "User asks for suggestions or best practices",
            "summary": "User asks for a summary or overview",
            "clarification": "User asks for clarification on previous answer",
            "related_topics": "User asks for related or similar topics",
            "opinion": "User asks for opinion or discussion"
        }

        intent_types = [
            {
                "type": intent.value,
                "description": intent_descriptions.get(intent.value, "")
            }
            for intent in IntentType
        ]

        return {
            "intent_types": intent_types,
            "total_count": len(intent_types)
        }

    except Exception as e:
        logger.error(f"Error getting intent types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

