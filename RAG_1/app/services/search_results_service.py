"""Service for saving search results to JSON files."""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from app.config import settings

logger = logging.getLogger(__name__)


class SearchResultsService:
    """Manages saving search results to JSON files."""
    
    def __init__(self):
        self.results_dir = Path(settings.upload_directory).parent / "search_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Search results directory: {self.results_dir}")
    
    async def save_search_result(
        self,
        query: str,
        answer: str,
        confidence: float,
        is_complete: bool,
        sources: List[Dict[str, Any]],
        missing_info: List[str],
        enrichment_suggestions: List[Dict[str, Any]] = None,
        auto_enrichment_applied: bool = False,
        auto_enrichment_sources: List[str] = None,
        intent_classification: Dict[str, Any] = None
    ) -> str:
        """
        Save a search result to a JSON file.

        Args:
            query: The search query
            answer: The generated answer
            confidence: Confidence score (0.0-1.0)
            is_complete: Whether the answer is complete
            sources: List of source documents
            missing_info: List of missing information
            enrichment_suggestions: Enrichment suggestions
            auto_enrichment_applied: Whether auto-enrichment was applied
            auto_enrichment_sources: Sources used for enrichment
            intent_classification: Agentic intent classification output

        Returns:
            Path to the saved JSON file
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Sanitize query for filename (remove special characters)
            safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                                for c in query)[:50]
            
            filename = f"search_{timestamp}_{safe_query}.json"
            filepath = self.results_dir / filename
            
            # Create the result object
            result_data = {
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "is_complete": is_complete,
                "sources": sources,
                "missing_info": missing_info,
                "enrichment_suggestions": enrichment_suggestions or [],
                "auto_enrichment_applied": auto_enrichment_applied,
                "auto_enrichment_sources": auto_enrichment_sources or [],
                "intent_classification": intent_classification or {},
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "num_sources": len(sources),
                    "num_missing_info": len(missing_info),
                    "has_enrichment": auto_enrichment_applied,
                    "intent": intent_classification.get("intent", "unknown") if intent_classification else "unknown"
                }
            }
            
            # Save to JSON file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved search result to: {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Error saving search result: {e}")
            raise
    
    def get_search_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent search results."""
        try:
            results = []
            
            # Get all JSON files sorted by modification time (newest first)
            json_files = sorted(
                self.results_dir.glob("search_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:limit]
            
            for filepath in json_files:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["file_path"] = str(filepath)
                        results.append(data)
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error getting search results: {e}")
            return []
    
    def get_search_result_by_query(self, query: str) -> List[Dict[str, Any]]:
        """Get all search results for a specific query."""
        try:
            results = []
            
            for filepath in self.results_dir.glob("search_*.json"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("query").lower() == query.lower():
                            data["file_path"] = str(filepath)
                            results.append(data)
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error getting search results: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about search results."""
        try:
            results = self.get_search_results(limit=1000)
            
            if not results:
                return {
                    "total_searches": 0,
                    "average_confidence": 0.0,
                    "complete_answers": 0,
                    "incomplete_answers": 0
                }
            
            total = len(results)
            confidence_sum = sum(r.get("confidence", 0) for r in results)
            complete_count = sum(1 for r in results if r.get("is_complete", False))
            
            return {
                "total_searches": total,
                "average_confidence": round(confidence_sum / total, 2) if total > 0 else 0,
                "complete_answers": complete_count,
                "incomplete_answers": total - complete_count,
                "completion_rate": round((complete_count / total * 100), 2) if total > 0 else 0
            }
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# Global instance
search_results_service = SearchResultsService()

