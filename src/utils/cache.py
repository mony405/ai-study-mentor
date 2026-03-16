"""
Step 5: The "Efficiency Brain" (Semantic Cache)
Before hitting the LLM, check if we've been here before.

File: src/utils/cache.py

Class: SemanticCache

Methods:

query_cache(query_text): Compare current query embedding to previous ones.

hit_cache(response): Return the stored JSON instantly.
"""
import os
from typing import Optional
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from src.schemas.models import MentorResponse
from typing import List, Any, Optional
class SemanticCache:
    def __init__(self, cache_dir: str = "./storage/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index = self._load_or_create_cache()

    def _load_or_create_cache(self) -> VectorStoreIndex:
        if os.path.exists(os.path.join(self.cache_dir, "docstore.json")):
            storage_context = StorageContext.from_defaults(persist_dir=self.cache_dir)
            return load_index_from_storage(storage_context)
        return VectorStoreIndex([])

    async def query_cache(self, query: str, threshold: float = 0.92) -> Optional[str]:
        """Checks if a similar question exists in the Efficiency Brain."""
        retriever = self.index.as_retriever(similarity_top_k=1)
        results = await retriever.aretrieve(query)
        
        if results and results[0].score >= threshold:
            print(f"🚀 Cache Hit! (Similarity: {results[0].score:.2f})")
            return results[0].node.text
        return None

    def update_cache(self, query: str, response: Any):
        """Adds a new successful Q&A pair to the cache, handling objects or strings."""
        # Logic: If 'response' is our MentorResponse object, get the .answer string
        answer_text = response.answer if hasattr(response, 'answer') else str(response)
        
        new_doc = Document(
            text=answer_text, 
            metadata={"original_query": query}
        )
        self.index.insert(new_doc)
        self.index.storage_context.persist(persist_dir=self.cache_dir)