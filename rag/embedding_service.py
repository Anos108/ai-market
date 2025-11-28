"""
Embedding Service for RAG System

This module provides embedding generation functionality for the RAG system,
including text embedding, batch processing, and embedding management.
"""

import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating and managing text embeddings.
    
    Features:
    - Text embedding generation
    - Batch processing
    - Embedding caching
    - Multiple embedding models support
    - Embedding similarity calculation
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 cache_path: str = "data/embedding_cache"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
            cache_path: Path to store embedding cache
        """
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_metadata = {}
        
        # Model configuration
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        self.max_text_length = 512  # Maximum text length for embedding
        
        # Load cache
        self._load_cache()
        
        logger.info(f"Initialized EmbeddingService with model: {model_name}")
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector
        """
        try:
            # Check cache first
            if use_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    logger.debug(f"Using cached embedding for text: {text[:50]}...")
                    return self.embedding_cache[cache_key]
            
            # Process text
            processed_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self._generate_embedding_vector(processed_text)
            
            # Cache the result
            if use_cache:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._get_zero_embedding()
    
    def generate_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            texts_to_process = []
            indices_to_process = []
            
            # Check cache for each text
            for i, text in enumerate(texts):
                if use_cache:
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.embedding_cache:
                        embeddings.append(self.embedding_cache[cache_key])
                        continue
                
                texts_to_process.append(text)
                indices_to_process.append(i)
                embeddings.append(None)  # Placeholder
            
            # Process texts that weren't in cache
            if texts_to_process:
                processed_embeddings = self._generate_embeddings_batch_vectors(texts_to_process)
                
                # Fill in the embeddings
                for i, embedding in enumerate(processed_embeddings):
                    original_index = indices_to_process[i]
                    embeddings[original_index] = embedding
                    
                    # Cache the result
                    if use_cache:
                        self._cache_embedding(texts_to_process[i], embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [self._get_zero_embedding() for _ in texts]
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get embedding cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            total_embeddings = len(self.embedding_cache)
            
            # Calculate cache size
            cache_size = 0
            for embedding in self.embedding_cache.values():
                cache_size += len(embedding) * 4  # 4 bytes per float
            
            # Calculate average embedding norm
            if self.embedding_cache:
                norms = [np.linalg.norm(embedding) for embedding in self.embedding_cache.values()]
                avg_norm = np.mean(norms)
            else:
                avg_norm = 0.0
            
            return {
                'total_cached_embeddings': total_embeddings,
                'cache_size_mb': cache_size / (1024 * 1024),
                'average_embedding_norm': avg_norm,
                'model_name': self.model_name,
                'embedding_dimension': self.embedding_dimension
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def clear_cache(self) -> bool:
        """
        Clear the embedding cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.embedding_cache.clear()
            self.cache_metadata.clear()
            logger.info("Cleared embedding cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def save_cache(self) -> bool:
        """
        Save embedding cache to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save embeddings
            embeddings_file = self.cache_path / 'embeddings.pkl'
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            
            # Save metadata
            metadata_file = self.cache_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
            
            logger.info(f"Saved embedding cache to {self.cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        try:
            # Load embeddings
            embeddings_file = self.cache_path / 'embeddings.pkl'
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
            
            # Load metadata
            metadata_file = self.cache_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.cache_metadata = json.load(f)
            
            logger.info(f"Loaded embedding cache with {len(self.embedding_cache)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            # Initialize empty cache
            self.embedding_cache = {}
            self.cache_metadata = {}
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding generation.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        try:
            # Truncate if too long
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Convert to lowercase
            text = text.lower()
            
            return text
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return text
    
    def _generate_embedding_vector(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Embedding vector
        """
        try:
            # In a real implementation, this would use a proper embedding model
            # For now, we'll create a simple hash-based embedding
            
            # Create a hash-based embedding
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = []
            
            # Convert hash to embedding vector
            for i in range(0, len(text_hash), 2):
                hex_pair = text_hash[i:i+2]
                embedding.append(int(hex_pair, 16) / 255.0)
            
            # Pad or truncate to required dimension
            while len(embedding) < self.embedding_dimension:
                embedding.append(0.0)
            
            # Normalize the embedding
            embedding = embedding[:self.embedding_dimension]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding vector generation failed: {e}")
            return self._get_zero_embedding()
    
    def _generate_embeddings_batch_vectors(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process each text individually
            # In a real implementation, this would use batch processing
            embeddings = []
            for text in texts:
                embedding = self._generate_embedding_vector(text)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [self._get_zero_embedding() for _ in texts]
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        try:
            # Create hash of the text
            return hashlib.md5(text.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            return hashlib.md5(f"{text}{datetime.now()}".encode()).hexdigest()
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        try:
            cache_key = self._get_cache_key(text)
            self.embedding_cache[cache_key] = embedding
            self.cache_metadata[cache_key] = {
                'text_preview': text[:100],
                'cached_at': datetime.now().isoformat(),
                'text_length': len(text)
            }
            
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
    
    def _get_zero_embedding(self) -> List[float]:
        """Get a zero embedding vector."""
        return [0.0] * self.embedding_dimension
