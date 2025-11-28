"""
LLM-RAG Powered Event Agent for AI Market Analysis System

This agent uses Retrieval-Augmented Generation (RAG) with vector databases
and Large Language Models to understand global news impact and provide
context-aware event analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import requests
import json
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import feedparser
import re
import hashlib
import os
from pathlib import Path

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus
from .event_impact_agent import EventType, EventImpact, MarketEvent

logger = logging.getLogger(__name__)


@dataclass
class NewsDocument:
    """Represents a news document for RAG processing."""
    doc_id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    url: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None


@dataclass
class RAGContext:
    """Context for RAG-based event analysis."""
    query: str
    relevant_docs: List[NewsDocument]
    llm_response: str
    confidence: float
    reasoning: str


class RAGEventAgent(BaseAgent):
    """
    LLM-RAG Powered Event Agent for advanced event impact analysis.
    
    This agent uses:
    - Vector database for document storage and retrieval
    - LLM for context-aware analysis
    - Real-time news ingestion
    - Advanced event impact prediction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG Event Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'vector_db_path': 'data/vector_db',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'llm_model': 'gpt-3.5-turbo',  # Can be changed to local model
            'llm_api_key': os.getenv('OPENAI_API_KEY', ''),
            'news_sources': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.marketwatch.com/marketwatch/topstories/',
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.federalreserve.gov/feeds/press_all.xml',
                'https://feeds.reuters.com/news/wealth',
                'https://feeds.reuters.com/reuters/businessNews'
            ],
            'max_documents': 1000,
            'embedding_dimension': 384,
            'similarity_threshold': 0.7,
            'context_window': 5,  # Number of relevant documents to use
            'update_frequency_minutes': 30,
            'enable_llm': True,
            'fallback_to_simple': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="RAGEventAgent",
            version="2.0.0",
            config=default_config
        )
        
        # Initialize RAG components
        self.vector_db = {}
        self.document_store = {}
        self.embeddings_cache = {}
        self.last_update = None
        
        # Initialize paths
        self.vector_db_path = Path(self.config['vector_db_path'])
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RAGEventAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the RAG event agent.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # Initialize document store with historical data
            self._initialize_document_store()
            
            # Load existing vector database if available
            self._load_vector_database()
            
            # Perform initial news ingestion
            self._ingest_initial_news()
            
            self.is_trained = True
            
            logger.info(f"{self.name}: RAG system initialized and ready")
            return {
                "status": "rag_system_ready",
                "documents_loaded": len(self.document_store),
                "vector_db_size": len(self.vector_db)
            }
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate RAG-powered event prediction.
        
        Args:
            context: Current market context
            
        Returns:
            RAG-enhanced trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # Update news if needed
            self._update_news_if_needed()
            
            # If not trained, use simple analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple event analysis (not trained)")
                return self._simple_event_analysis(context)
            
            # Perform RAG-based event analysis
            rag_analysis = self._perform_rag_analysis(context)
            
            # Generate signal based on RAG insights
            signal = self._generate_rag_signal(rag_analysis, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"RAG analysis error: {e}", context)
    
    def _initialize_document_store(self) -> None:
        """Initialize the document store with sample data."""
        try:
            # Create sample documents for demonstration
            sample_docs = [
                {
                    'title': 'Federal Reserve Interest Rate Decision',
                    'content': 'The Federal Reserve announced a 0.25% increase in the federal funds rate, citing persistent inflation concerns and strong labor market conditions.',
                    'source': 'Federal Reserve',
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'url': 'https://federalreserve.gov/news/pressreleases/monetary20241201a.htm'
                },
                {
                    'title': 'Tech Earnings Beat Expectations',
                    'content': 'Major technology companies reported stronger than expected quarterly earnings, driven by cloud computing growth and AI investments.',
                    'source': 'MarketWatch',
                    'timestamp': datetime.now() - timedelta(hours=4),
                    'url': 'https://marketwatch.com/tech-earnings'
                },
                {
                    'title': 'Inflation Data Shows Moderation',
                    'content': 'Consumer Price Index data for November showed inflation moderating to 3.2% year-over-year, below expectations.',
                    'source': 'Reuters',
                    'timestamp': datetime.now() - timedelta(hours=6),
                    'url': 'https://reuters.com/inflation-data'
                }
            ]
            
            for doc_data in sample_docs:
                doc_id = hashlib.md5(doc_data['content'].encode()).hexdigest()
                document = NewsDocument(
                    doc_id=doc_id,
                    title=doc_data['title'],
                    content=doc_data['content'],
                    source=doc_data['source'],
                    timestamp=doc_data['timestamp'],
                    url=doc_data['url'],
                    metadata={'type': 'sample', 'category': 'market_news'}
                )
                
                self.document_store[doc_id] = document
                
                # Generate simple embedding (in real implementation, use proper embedding model)
                embedding = self._generate_embedding(document.content)
                self.vector_db[doc_id] = embedding
            
            logger.info(f"Initialized document store with {len(self.document_store)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
    
    def _load_vector_database(self) -> None:
        """Load existing vector database from disk."""
        try:
            vector_db_file = self.vector_db_path / 'vector_db.json'
            if vector_db_file.exists():
                with open(vector_db_file, 'r') as f:
                    data = json.load(f)
                    self.vector_db = {k: v for k, v in data.items()}
                logger.info(f"Loaded vector database with {len(self.vector_db)} vectors")
            else:
                logger.info("No existing vector database found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
    
    def _save_vector_database(self) -> None:
        """Save vector database to disk."""
        try:
            vector_db_file = self.vector_db_path / 'vector_db.json'
            with open(vector_db_file, 'w') as f:
                json.dump(self.vector_db, f)
            logger.info(f"Saved vector database with {len(self.vector_db)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
    
    def _ingest_initial_news(self) -> None:
        """Perform initial news ingestion."""
        try:
            logger.info("Starting initial news ingestion...")
            
            for source in self.config['news_sources']:
                try:
                    self._ingest_news_from_source(source)
                except Exception as e:
                    logger.warning(f"Failed to ingest from {source}: {e}")
                    continue
            
            # Save updated vector database
            self._save_vector_database()
            
            logger.info(f"Initial news ingestion completed. Total documents: {len(self.document_store)}")
            
        except Exception as e:
            logger.error(f"Initial news ingestion failed: {e}")
    
    def _ingest_news_from_source(self, source_url: str) -> None:
        """Ingest news from a specific RSS source."""
        try:
            feed = feedparser.parse(source_url)
            
            for entry in feed.entries[:20]:  # Limit to recent entries
                try:
                    # Extract content
                    title = entry.get('title', '')
                    description = entry.get('summary', '')
                    content = f"{title} {description}"
                    
                    # Create document
                    doc_id = hashlib.md5(content.encode()).hexdigest()
                    
                    if doc_id not in self.document_store:
                        document = NewsDocument(
                            doc_id=doc_id,
                            title=title,
                            content=content,
                            source=source_url,
                            timestamp=datetime.now(),
                            url=entry.get('link', ''),
                            metadata={'source_type': 'rss', 'entry_id': entry.get('id', '')}
                        )
                        
                        self.document_store[doc_id] = document
                        
                        # Generate embedding
                        embedding = self._generate_embedding(content)
                        self.vector_db[doc_id] = embedding
                        
                except Exception as e:
                    logger.warning(f"Failed to process entry from {source_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to ingest from {source_url}: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (simplified version)."""
        try:
            # In a real implementation, use a proper embedding model like:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            # return model.encode(text).tolist()
            
            # For now, create a simple hash-based embedding
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = []
            
            for i in range(0, len(text_hash), 2):
                hex_pair = text_hash[i:i+2]
                embedding.append(int(hex_pair, 16) / 255.0)
            
            # Pad or truncate to required dimension
            target_dim = self.config['embedding_dimension']
            while len(embedding) < target_dim:
                embedding.append(0.0)
            
            return embedding[:target_dim]
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.config['embedding_dimension']
    
    def _update_news_if_needed(self) -> None:
        """Update news if enough time has passed."""
        try:
            if self.last_update is None:
                self.last_update = datetime.now()
                return
            
            time_since_update = datetime.now() - self.last_update
            if time_since_update.total_seconds() > (self.config['update_frequency_minutes'] * 60):
                logger.info("Updating news feeds...")
                self._ingest_initial_news()
                self.last_update = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to update news: {e}")
    
    def _perform_rag_analysis(self, context: AgentContext) -> RAGContext:
        """
        Perform RAG-based event analysis.
        
        Args:
            context: Current market context
            
        Returns:
            RAG analysis context
        """
        try:
            # Create query for the symbol and current market conditions
            query = f"Market events and news affecting {context.symbol} stock price and trading"
            
            # Retrieve relevant documents
            relevant_docs = self._retrieve_relevant_documents(query)
            
            # Generate LLM response if enabled
            if self.config['enable_llm'] and self.config['llm_api_key']:
                llm_response = self._query_llm(query, relevant_docs, context)
            else:
                llm_response = self._generate_fallback_response(relevant_docs, context)
            
            # Calculate confidence based on document relevance and LLM response quality
            confidence = self._calculate_rag_confidence(relevant_docs, llm_response)
            
            # Generate reasoning
            reasoning = self._generate_rag_reasoning(relevant_docs, llm_response, confidence)
            
            return RAGContext(
                query=query,
                relevant_docs=relevant_docs,
                llm_response=llm_response,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"RAG analysis failed: {e}")
            return RAGContext(
                query="",
                relevant_docs=[],
                llm_response="Analysis failed",
                confidence=0.0,
                reasoning=f"RAG analysis error: {e}"
            )
    
    def _retrieve_relevant_documents(self, query: str) -> List[NewsDocument]:
        """Retrieve relevant documents using vector similarity."""
        try:
            if not self.vector_db:
                return []
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.vector_db.items():
                similarity = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                if similarity > self.config['similarity_threshold']:
                    similarities.append((doc_id, similarity))
            
            # Sort by similarity and take top documents
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_docs = similarities[:self.config['context_window']]
            
            # Return document objects
            relevant_docs = []
            for doc_id, similarity in top_docs:
                if doc_id in self.document_store:
                    doc = self.document_store[doc_id]
                    doc.metadata = doc.metadata or {}
                    doc.metadata['similarity_score'] = similarity
                    relevant_docs.append(doc)
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _query_llm(self, query: str, relevant_docs: List[NewsDocument], context: AgentContext) -> str:
        """Query LLM with relevant documents for context-aware analysis."""
        try:
            if not self.config['llm_api_key']:
                return self._generate_fallback_response(relevant_docs, context)
            
            # Prepare context for LLM
            context_text = f"Query: {query}\n\n"
            context_text += f"Symbol: {context.symbol}\n"
            context_text += f"Current Time: {context.timestamp}\n\n"
            context_text += "Relevant News and Events:\n"
            
            for i, doc in enumerate(relevant_docs[:3]):  # Limit to top 3 docs
                context_text += f"{i+1}. {doc.title}\n{doc.content}\n\n"
            
            # Create LLM prompt
            prompt = f"""
            As a financial market analyst, analyze the following information and provide insights about potential market impact for {context.symbol}:
            
            {context_text}
            
            Please provide:
            1. Key market events and their potential impact
            2. Risk assessment for {context.symbol}
            3. Trading recommendation (buy/sell/hold) with reasoning
            4. Confidence level in your analysis
            
            Keep your response concise and actionable.
            """
            
            # In a real implementation, make API call to LLM
            # For now, generate a structured response
            response = self._generate_llm_response_simulation(prompt, relevant_docs, context)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return self._generate_fallback_response(relevant_docs, context)
    
    def _generate_llm_response_simulation(self, prompt: str, relevant_docs: List[NewsDocument], context: AgentContext) -> str:
        """Generate simulated LLM response (replace with actual LLM call)."""
        try:
            if not relevant_docs:
                return "No relevant news found. Market conditions appear stable with no significant events affecting the stock."
            
            # Analyze the most relevant document
            top_doc = relevant_docs[0]
            
            # Generate response based on document content
            if 'fed' in top_doc.content.lower() or 'federal reserve' in top_doc.content.lower():
                response = f"Federal Reserve policy impact detected. {top_doc.title} suggests potential interest rate implications for {context.symbol}. "
                if 'increase' in top_doc.content.lower() or 'raise' in top_doc.content.lower():
                    response += "Rate increase may create headwinds for growth stocks. Consider reducing position size."
                else:
                    response += "Policy stance appears supportive. Monitor for trading opportunities."
            
            elif 'earnings' in top_doc.content.lower():
                response = f"Earnings-related news detected. {top_doc.title} indicates sector performance trends. "
                if 'beat' in top_doc.content.lower() or 'strong' in top_doc.content.lower():
                    response += "Positive earnings trends suggest sector strength. Consider position increase."
                else:
                    response += "Mixed earnings signals. Maintain current position with close monitoring."
            
            elif 'inflation' in top_doc.content.lower() or 'cpi' in top_doc.content.lower():
                response = f"Inflation data impact. {top_doc.title} affects market sentiment. "
                if 'moderate' in top_doc.content.lower() or 'decrease' in top_doc.content.lower():
                    response += "Inflation moderation is positive for equity markets. Consider bullish positioning."
                else:
                    response += "Inflation concerns may pressure markets. Exercise caution with new positions."
            
            else:
                response = f"Market event detected: {top_doc.title}. Impact assessment suggests monitoring for {context.symbol} price movements. "
                response += "Consider maintaining current position until clearer signals emerge."
            
            return response
            
        except Exception as e:
            logger.error(f"LLM response simulation failed: {e}")
            return "Analysis completed with standard market monitoring recommendations."
    
    def _generate_fallback_response(self, relevant_docs: List[NewsDocument], context: AgentContext) -> str:
        """Generate fallback response when LLM is not available."""
        try:
            if not relevant_docs:
                return f"No significant news events found for {context.symbol}. Market appears stable."
            
            # Simple analysis based on document count and content
            doc_count = len(relevant_docs)
            if doc_count >= 3:
                return f"Multiple news events detected ({doc_count} relevant articles). High market activity expected for {context.symbol}. Monitor closely for trading opportunities."
            elif doc_count >= 1:
                return f"News event detected. {relevant_docs[0].title} may impact {context.symbol}. Moderate market activity expected."
            else:
                return f"Limited news activity for {context.symbol}. Standard market monitoring recommended."
                
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return "Standard market analysis recommended."
    
    def _calculate_rag_confidence(self, relevant_docs: List[NewsDocument], llm_response: str) -> float:
        """Calculate confidence in RAG analysis."""
        try:
            # Base confidence on number of relevant documents
            doc_confidence = min(len(relevant_docs) / 5.0, 1.0)
            
            # Adjust based on document quality (similarity scores)
            if relevant_docs:
                avg_similarity = np.mean([doc.metadata.get('similarity_score', 0.5) for doc in relevant_docs])
                quality_confidence = avg_similarity
            else:
                quality_confidence = 0.0
            
            # Adjust based on response quality (length and content)
            response_confidence = min(len(llm_response) / 200.0, 1.0)
            
            # Combine confidences
            final_confidence = (doc_confidence * 0.4 + quality_confidence * 0.4 + response_confidence * 0.2)
            
            return max(0.0, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_rag_reasoning(self, relevant_docs: List[NewsDocument], llm_response: str, confidence: float) -> str:
        """Generate reasoning for RAG analysis."""
        try:
            reasoning_parts = []
            
            # Document-based reasoning
            if relevant_docs:
                reasoning_parts.append(f"Found {len(relevant_docs)} relevant news articles")
                if relevant_docs[0].metadata.get('similarity_score', 0) > 0.8:
                    reasoning_parts.append("High relevance to current market conditions")
            else:
                reasoning_parts.append("No highly relevant news found")
            
            # Confidence-based reasoning
            if confidence > 0.8:
                reasoning_parts.append("High confidence in analysis")
            elif confidence > 0.6:
                reasoning_parts.append("Moderate confidence in analysis")
            else:
                reasoning_parts.append("Low confidence - limited data available")
            
            # Response-based reasoning
            if len(llm_response) > 100:
                reasoning_parts.append("Comprehensive analysis provided")
            else:
                reasoning_parts.append("Limited analysis available")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return "RAG analysis completed with standard reasoning."
    
    def _generate_rag_signal(self, rag_context: RAGContext, context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on RAG analysis.
        
        Args:
            rag_context: RAG analysis context
            context: Current market context
            
        Returns:
            RAG-enhanced trading signal
        """
        try:
            llm_response = rag_context.llm_response.lower()
            
            # Determine signal based on LLM response
            if 'buy' in llm_response or 'bullish' in llm_response or 'increase' in llm_response:
                signal_type = SignalType.BUY
            elif 'sell' in llm_response or 'bearish' in llm_response or 'reduce' in llm_response:
                signal_type = SignalType.SELL
            elif 'caution' in llm_response or 'monitor' in llm_response or 'stable' in llm_response:
                signal_type = SignalType.HOLD
            else:
                # Default based on confidence and document count
                if rag_context.confidence > 0.7 and len(rag_context.relevant_docs) > 2:
                    signal_type = SignalType.BUY
                elif rag_context.confidence < 0.4:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.HOLD
            
            # Adjust confidence based on RAG analysis quality
            adjusted_confidence = min(rag_context.confidence * 0.9, 0.95)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'rag_context': {
                        'query': rag_context.query,
                        'doc_count': len(rag_context.relevant_docs),
                        'llm_response': rag_context.llm_response[:200] + "..." if len(rag_context.llm_response) > 200 else rag_context.llm_response
                    },
                    'method': 'rag_enhanced_analysis'
                },
                reasoning=rag_context.reasoning
            )
            
        except Exception as e:
            logger.error(f"RAG signal generation failed: {e}")
            return self._create_hold_signal(f"RAG signal generation error: {e}", context)
    
    def _simple_event_analysis(self, context: AgentContext) -> AgentSignal:
        """Simple event analysis when RAG system is not available."""
        try:
            # Use basic market volatility as proxy for event activity
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            # Simple volatility-based analysis
            if len(context.market_data) >= 5:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()
                    
                    if volatility > 0.04:  # 4% volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.HOLD,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility_rag'},
                            reasoning=f"High volatility ({volatility:.2%}) suggests potential event-driven activity. RAG system not available for detailed analysis."
                        )
            
            return self._create_hold_signal("RAG system not available - using basic analysis", context)
            
        except Exception as e:
            logger.error(f"Simple RAG analysis failed: {e}")
            return self._create_hold_signal(f"Simple RAG analysis error: {e}", context)
    
    def _create_hold_signal(self, reason: str, context: AgentContext) -> AgentSignal:
        """Create a hold signal with error information."""
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=context.timestamp,
            asset_symbol=context.symbol,
            metadata={'error': reason, 'agent_version': self.version},
            reasoning=f"Hold signal: {reason}"
        )
    
    def update_model(self, new_data: pd.DataFrame, context: AgentContext) -> None:
        """
        Update the RAG model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update document store with new market data insights
            # In a real implementation, this would update the vector database
            # with new market insights and performance feedback
            
            logger.info(f"Updated RAG model for {self.name}")
            
        except Exception as e:
            logger.error(f"RAG model update failed for {self.name}: {e}")
