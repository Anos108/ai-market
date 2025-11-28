"""
Document Store for RAG System

This module provides document storage and management functionality
for the RAG system, including document ingestion, processing, and retrieval.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the store."""
    doc_id: str
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    category: str
    tags: List[str]
    metadata: Dict[str, Any]
    processed: bool = False
    embedding_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class DocumentStore:
    """
    Document store for managing documents in the RAG system.
    
    Features:
    - Document storage and retrieval
    - Content processing and cleaning
    - Metadata management
    - Search and filtering
    - Persistence to disk
    """
    
    def __init__(self, store_path: str):
        """
        Initialize the document store.
        
        Args:
            store_path: Path to store the document files
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Storage structures
        self.documents = {}  # doc_id -> Document
        self.index = {}      # doc_id -> index info
        self.categories = {} # category -> list of doc_ids
        self.tags = {}       # tag -> list of doc_ids
        
        # Load existing store
        self._load_store()
        
        logger.info(f"Initialized DocumentStore at {self.store_path} with {len(self.documents)} documents")
    
    def add_document(self, title: str, content: str, source: str, url: str = "",
                    category: str = "general", tags: List[str] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the store.
        
        Args:
            title: Document title
            content: Document content
            source: Document source
            url: Document URL
            category: Document category
            tags: Document tags
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = self._generate_doc_id(title, content, source)
            
            # Check if document already exists
            if doc_id in self.documents:
                logger.debug(f"Document {doc_id} already exists")
                return doc_id
            
            # Process content
            processed_content = self._process_content(content)
            
            # Create document
            document = Document(
                doc_id=doc_id,
                title=title,
                content=processed_content,
                source=source,
                url=url,
                timestamp=datetime.now(),
                category=category,
                tags=tags or [],
                metadata=metadata or {},
                processed=True
            )
            
            # Store document
            self.documents[doc_id] = document
            self._update_indexes(document)
            
            logger.debug(f"Added document {doc_id}: {title[:50]}...")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return ""
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document object or None if not found
        """
        return self.documents.get(doc_id)
    
    def search_documents(self, query: str = "", category: str = "", 
                        tags: List[str] = None, limit: int = 100) -> List[Document]:
        """
        Search documents by various criteria.
        
        Args:
            query: Text query to search in title and content
            category: Filter by category
            tags: Filter by tags
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            results = []
            
            for doc in self.documents.values():
                # Filter by category
                if category and doc.category != category:
                    continue
                
                # Filter by tags
                if tags and not any(tag in doc.tags for tag in tags):
                    continue
                
                # Filter by query
                if query:
                    query_lower = query.lower()
                    if (query_lower not in doc.title.lower() and 
                        query_lower not in doc.content.lower()):
                        continue
                
                results.append(doc)
                
                if len(results) >= limit:
                    break
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    def get_documents_by_category(self, category: str) -> List[Document]:
        """
        Get all documents in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of documents in the category
        """
        try:
            doc_ids = self.categories.get(category, [])
            return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]
            
        except Exception as e:
            logger.error(f"Failed to get documents by category {category}: {e}")
            return []
    
    def get_documents_by_tag(self, tag: str) -> List[Document]:
        """
        Get all documents with a specific tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List of documents with the tag
        """
        try:
            doc_ids = self.tags.get(tag, [])
            return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]
            
        except Exception as e:
            logger.error(f"Failed to get documents by tag {tag}: {e}")
            return []
    
    def get_recent_documents(self, hours: int = 24, limit: int = 50) -> List[Document]:
        """
        Get recent documents.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of results
            
        Returns:
            List of recent documents
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_docs = []
            for doc in self.documents.values():
                if doc.timestamp >= cutoff_time:
                    recent_docs.append(doc)
            
            # Sort by timestamp (newest first)
            recent_docs.sort(key=lambda x: x.timestamp, reverse=True)
            
            return recent_docs[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
            return []
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not found")
                return False
            
            document = self.documents[doc_id]
            
            # Remove from indexes
            self._remove_from_indexes(document)
            
            # Remove document
            del self.documents[doc_id]
            
            logger.debug(f"Removed document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    def update_document(self, doc_id: str, **kwargs) -> bool:
        """
        Update a document.
        
        Args:
            doc_id: Document identifier
            **kwargs: Fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not found")
                return False
            
            document = self.documents[doc_id]
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(document, key):
                    setattr(document, key, value)
            
            # Update indexes
            self._update_indexes(document)
            
            logger.debug(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        try:
            total_docs = len(self.documents)
            
            # Count by category
            category_counts = {}
            for doc in self.documents.values():
                category_counts[doc.category] = category_counts.get(doc.category, 0) + 1
            
            # Count by source
            source_counts = {}
            for doc in self.documents.values():
                source_counts[doc.source] = source_counts.get(doc.source, 0) + 1
            
            # Count total tags
            all_tags = set()
            for doc in self.documents.values():
                all_tags.update(doc.tags)
            
            # Calculate average content length
            if self.documents:
                avg_content_length = sum(len(doc.content) for doc in self.documents.values()) / total_docs
            else:
                avg_content_length = 0
            
            return {
                'total_documents': total_docs,
                'categories': category_counts,
                'sources': source_counts,
                'total_tags': len(all_tags),
                'average_content_length': avg_content_length,
                'store_size_mb': self._get_store_size()
            }
            
        except Exception as e:
            logger.error(f"Failed to get store stats: {e}")
            return {}
    
    def clear_store(self) -> bool:
        """
        Clear all documents from the store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.documents.clear()
            self.index.clear()
            self.categories.clear()
            self.tags.clear()
            logger.info("Cleared document store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
            return False
    
    def save_store(self) -> bool:
        """
        Save store to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save documents
            documents_file = self.store_path / 'documents.json'
            documents_data = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}
            with open(documents_file, 'w') as f:
                json.dump(documents_data, f, indent=2)
            
            # Save indexes
            index_file = self.store_path / 'index.json'
            with open(index_file, 'w') as f:
                json.dump({
                    'index': self.index,
                    'categories': self.categories,
                    'tags': self.tags
                }, f, indent=2)
            
            logger.info(f"Saved document store to {self.store_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save store: {e}")
            return False
    
    def _load_store(self) -> None:
        """Load store from disk."""
        try:
            # Load documents
            documents_file = self.store_path / 'documents.json'
            if documents_file.exists():
                with open(documents_file, 'r') as f:
                    documents_data = json.load(f)
                    for doc_id, doc_data in documents_data.items():
                        self.documents[doc_id] = Document.from_dict(doc_data)
            
            # Load indexes
            index_file = self.store_path / 'index.json'
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                    self.index = index_data.get('index', {})
                    self.categories = index_data.get('categories', {})
                    self.tags = index_data.get('tags', {})
            
            logger.info(f"Loaded document store with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load store: {e}")
            # Initialize empty store
            self.documents = {}
            self.index = {}
            self.categories = {}
            self.tags = {}
    
    def _generate_doc_id(self, title: str, content: str, source: str) -> str:
        """Generate a unique document ID."""
        try:
            # Create hash from title, content, and source
            text = f"{title}|{content[:100]}|{source}"
            return hashlib.md5(text.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate document ID: {e}")
            return hashlib.md5(f"{title}{source}{datetime.now()}".encode()).hexdigest()
    
    def _process_content(self, content: str) -> str:
        """Process and clean document content."""
        try:
            # Remove extra whitespace
            content = re.sub(r'\s+', ' ', content)
            
            # Remove HTML tags if present
            content = re.sub(r'<[^>]+>', '', content)
            
            # Remove special characters but keep basic punctuation
            content = re.sub(r'[^\w\s.,!?;:\-()]', '', content)
            
            # Strip leading/trailing whitespace
            content = content.strip()
            
            return content
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            return content
    
    def _update_indexes(self, document: Document) -> None:
        """Update indexes for a document."""
        try:
            doc_id = document.doc_id
            
            # Update category index
            if document.category not in self.categories:
                self.categories[document.category] = []
            if doc_id not in self.categories[document.category]:
                self.categories[document.category].append(doc_id)
            
            # Update tag indexes
            for tag in document.tags:
                if tag not in self.tags:
                    self.tags[tag] = []
                if doc_id not in self.tags[tag]:
                    self.tags[tag].append(doc_id)
            
            # Update main index
            self.index[doc_id] = {
                'title': document.title,
                'category': document.category,
                'tags': document.tags,
                'timestamp': document.timestamp.isoformat(),
                'source': document.source
            }
            
        except Exception as e:
            logger.error(f"Failed to update indexes for document {document.doc_id}: {e}")
    
    def _remove_from_indexes(self, document: Document) -> None:
        """Remove document from indexes."""
        try:
            doc_id = document.doc_id
            
            # Remove from category index
            if document.category in self.categories:
                if doc_id in self.categories[document.category]:
                    self.categories[document.category].remove(doc_id)
                if not self.categories[document.category]:
                    del self.categories[document.category]
            
            # Remove from tag indexes
            for tag in document.tags:
                if tag in self.tags:
                    if doc_id in self.tags[tag]:
                        self.tags[tag].remove(doc_id)
                    if not self.tags[tag]:
                        del self.tags[tag]
            
            # Remove from main index
            if doc_id in self.index:
                del self.index[doc_id]
                
        except Exception as e:
            logger.error(f"Failed to remove document {document.doc_id} from indexes: {e}")
    
    def _get_store_size(self) -> float:
        """Get store size in MB."""
        try:
            total_size = 0
            
            # Calculate size of documents
            if self.documents:
                total_size += len(json.dumps({doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}).encode())
            
            # Calculate size of indexes
            if self.index or self.categories or self.tags:
                total_size += len(json.dumps({
                    'index': self.index,
                    'categories': self.categories,
                    'tags': self.tags
                }).encode())
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.error(f"Failed to calculate store size: {e}")
            return 0.0
