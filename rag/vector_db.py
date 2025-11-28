"""
Vector Database for RAG System

This module provides vector database functionality for storing and retrieving
document embeddings with similarity search capabilities.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import pickle
import hashlib

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Vector database for storing and retrieving document embeddings.
    
    Features:
    - Embedding storage and retrieval
    - Similarity search
    - Persistence to disk
    - Metadata management
    """
    
    def __init__(self, db_path: str, embedding_dimension: int = 384):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to store the database files
            embedding_dimension: Dimension of the embeddings
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dimension = embedding_dimension
        
        # Storage structures
        self.embeddings = {}  # doc_id -> embedding vector
        self.metadata = {}    # doc_id -> metadata dict
        self.index = {}       # doc_id -> document info
        
        # Load existing database
        self._load_database()
        
        logger.info(f"Initialized VectorDatabase at {self.db_path} with {len(self.embeddings)} vectors")
    
    def add_document(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Add a document to the vector database.
        
        Args:
            doc_id: Unique document identifier
            embedding: Document embedding vector
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate embedding dimension
            if len(embedding) != self.embedding_dimension:
                logger.error(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
                return False
            
            # Store embedding and metadata
            self.embeddings[doc_id] = embedding
            self.metadata[doc_id] = metadata
            self.index[doc_id] = {
                'added_at': datetime.now().isoformat(),
                'embedding_dim': len(embedding)
            }
            
            logger.debug(f"Added document {doc_id} to vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
    
    def get_embedding(self, doc_id: str) -> Optional[List[float]]:
        """
        Get embedding for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Embedding vector or None if not found
        """
        return self.embeddings.get(doc_id)
    
    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata.get(doc_id)
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10, 
                      threshold: float = 0.0) -> List[Tuple[str, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        try:
            if not self.embeddings:
                return []
            
            similarities = []
            
            for doc_id, doc_embedding in self.embeddings.items():
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                if similarity >= threshold:
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def search_by_metadata(self, filters: Dict[str, Any]) -> List[str]:
        """
        Search documents by metadata filters.
        
        Args:
            filters: Metadata filters (key-value pairs)
            
        Returns:
            List of document IDs matching the filters
        """
        try:
            matching_docs = []
            
            for doc_id, metadata in self.metadata.items():
                match = True
                for key, value in filters.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    matching_docs.append(doc_id)
            
            return matching_docs
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the database.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]
                del self.metadata[doc_id]
                del self.index[doc_id]
                logger.debug(f"Removed document {doc_id} from vector database")
                return True
            else:
                logger.warning(f"Document {doc_id} not found in database")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            total_docs = len(self.embeddings)
            
            # Count by metadata categories
            categories = {}
            for metadata in self.metadata.values():
                category = metadata.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
            
            # Calculate average embedding norm
            if self.embeddings:
                norms = [np.linalg.norm(embedding) for embedding in self.embeddings.values()]
                avg_norm = np.mean(norms)
            else:
                avg_norm = 0.0
            
            return {
                'total_documents': total_docs,
                'embedding_dimension': self.embedding_dimension,
                'categories': categories,
                'average_embedding_norm': avg_norm,
                'database_size_mb': self._get_database_size()
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def clear_database(self) -> bool:
        """
        Clear all data from the database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.embeddings.clear()
            self.metadata.clear()
            self.index.clear()
            logger.info("Cleared vector database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False
    
    def save_database(self) -> bool:
        """
        Save database to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save embeddings
            embeddings_file = self.db_path / 'embeddings.pkl'
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            metadata_file = self.db_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save index
            index_file = self.db_path / 'index.json'
            with open(index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
            
            logger.info(f"Saved vector database to {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False
    
    def _load_database(self) -> None:
        """Load database from disk."""
        try:
            # Load embeddings
            embeddings_file = self.db_path / 'embeddings.pkl'
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
            
            # Load metadata
            metadata_file = self.db_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load index
            index_file = self.db_path / 'index.json'
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.index = json.load(f)
            
            logger.info(f"Loaded vector database with {len(self.embeddings)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            # Initialize empty database
            self.embeddings = {}
            self.metadata = {}
            self.index = {}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            # Convert to numpy arrays for efficient computation
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    def _get_database_size(self) -> float:
        """Get database size in MB."""
        try:
            total_size = 0
            
            # Calculate size of embeddings
            if self.embeddings:
                total_size += len(pickle.dumps(self.embeddings))
            
            # Calculate size of metadata
            if self.metadata:
                total_size += len(json.dumps(self.metadata).encode())
            
            # Calculate size of index
            if self.index:
                total_size += len(json.dumps(self.index).encode())
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Failed to calculate database size: {e}")
            return 0.0
