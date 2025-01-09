import logging
import os
import json
from typing import Optional, Dict, List, Iterator
from cachetools import LRUCache

logger = logging.getLogger("websocietysimulator")

class CacheInteractionTool:
    def __init__(self, data_dir: str, cache_size: int = 10000):
        """
        Initialize the tool with the dataset directory.
        Args:
            data_dir: Path to the directory containing Yelp dataset files.
            cache_size: Maximum number of items to keep in each cache.
        """
        logger.info(f"Initializing InteractionTool with data directory: {data_dir}")
        self.data_dir = data_dir
        self.user_cache = LRUCache(maxsize=cache_size)
        self.item_cache = LRUCache(maxsize=cache_size)
        self.review_cache = LRUCache(maxsize=cache_size)
        self.item_reviews_cache = LRUCache(maxsize=cache_size)
        self.user_reviews_cache = LRUCache(maxsize=cache_size)

    def _iter_file(self, filename: str) -> Iterator[Dict]:
        """Iterate through file line by line."""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line)

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Fetch user data based on user_id."""
        user = self.user_cache.get(user_id)
        if user is not None:
            return user
        
        for user in self._iter_file('user.json'):
            if user['user_id'] == user_id:
                self.user_cache[user_id] = user
                return user
        return None

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Fetch item data based on item_id."""
        if not item_id:
            return None
            
        item = self.item_cache.get(item_id)
        if item is not None:
            return item
        
        for item in self._iter_file('item.json'):
            if item['item_id'] == item_id:
                self.item_cache[item_id] = item
                return item
        return None

    def get_reviews(
        self, 
        item_id: Optional[str] = None, 
        user_id: Optional[str] = None, 
        review_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch reviews filtered by various parameters."""
        if review_id:
            review = self.review_cache.get(review_id)
            if review is not None:
                return [review]
            
            for review in self._iter_file('review.json'):
                if review['review_id'] == review_id:
                    self.review_cache[review_id] = review
                    return [review]
            return []

        if item_id:
            cached_reviews = self.item_reviews_cache.get(item_id)
            if cached_reviews is not None:
                return cached_reviews
            
            reviews = []
            for review in self._iter_file('review.json'):
                if review['item_id'] == item_id:
                    reviews.append(review)
            self.item_reviews_cache[item_id] = reviews
            return reviews
            
        elif user_id:
            cached_reviews = self.user_reviews_cache.get(user_id)
            if cached_reviews is not None:
                return cached_reviews
                
            reviews = []
            for review in self._iter_file('review.json'):
                if review['user_id'] == user_id:
                    reviews.append(review)
            self.user_reviews_cache[user_id] = reviews
            return reviews
        
        return []