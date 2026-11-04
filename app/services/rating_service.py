"""Rating service for tracking answer quality feedback."""
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)


class RatingService:
    """Manages user ratings and feedback for answers."""
    
    def __init__(self):
        self.ratings_file = Path(settings.upload_directory).parent / "ratings.jsonl"
        self.ratings_file.parent.mkdir(parents=True, exist_ok=True)
    
    async def save_rating(
        self,
        query: str,
        answer: str,
        rating: int,
        feedback: str = None
    ) -> str:
        """
        Save a user rating for an answer.
        
        Args:
            query: The original query
            answer: The answer that was rated
            rating: Rating from 1-5
            feedback: Optional text feedback
            
        Returns:
            Rating ID
        """
        rating_id = str(uuid.uuid4())
        
        rating_data = {
            'rating_id': rating_id,
            'query': query,
            'answer': answer,
            'rating': rating,
            'feedback': feedback,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Append to JSONL file
            with open(self.ratings_file, 'a') as f:
                f.write(json.dumps(rating_data) + '\n')
            
            logger.info(f"Saved rating {rating_id} with score {rating}")
            return rating_id
        
        except Exception as e:
            logger.error(f"Error saving rating: {e}")
            raise
    
    def get_ratings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent ratings."""
        ratings = []
        
        if not self.ratings_file.exists():
            return ratings
        
        try:
            with open(self.ratings_file, 'r') as f:
                for line in f:
                    if line.strip():
                        ratings.append(json.loads(line))
            
            # Return most recent first
            return ratings[-limit:][::-1]
        
        except Exception as e:
            logger.error(f"Error reading ratings: {e}")
            return []
    
    def get_rating_statistics(self) -> Dict[str, Any]:
        """Get statistics about ratings."""
        ratings = self.get_ratings(limit=1000)
        
        if not ratings:
            return {
                'total_ratings': 0,
                'average_rating': 0.0,
                'rating_distribution': {}
            }
        
        total = len(ratings)
        rating_sum = sum(r['rating'] for r in ratings)
        average = rating_sum / total if total > 0 else 0.0
        
        # Distribution
        distribution = {i: 0 for i in range(1, 6)}
        for r in ratings:
            distribution[r['rating']] = distribution.get(r['rating'], 0) + 1
        
        return {
            'total_ratings': total,
            'average_rating': round(average, 2),
            'rating_distribution': distribution,
            'recent_ratings': ratings[:10]
        }
    
    def get_low_rated_queries(self, threshold: int = 3, limit: int = 20) -> List[Dict[str, Any]]:
        """Get queries with low ratings for improvement."""
        ratings = self.get_ratings(limit=1000)
        
        low_rated = [
            r for r in ratings
            if r['rating'] <= threshold
        ]
        
        return low_rated[:limit]


# Global instance
rating_service = RatingService()

