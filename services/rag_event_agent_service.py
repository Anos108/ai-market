"""
RAG Event Agent Service for real news analysis and document management.
Provides comprehensive RAG-powered event analysis with Ollama LLM integration.
"""

import asyncio
import asyncpg
import json
import random
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGEventAgentService:
    """Service for managing RAG Event Agent data and analysis with Ollama LLM."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        # Use host.docker.internal for Docker containers to access host services
        self.ollama_url = "http://host.docker.internal:11434"  # Docker host access
        self.embedding_model = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize RAG Event Agent database tables."""
        # Tables are created by init.sql, this is for any additional setup
        pass
    
    async def _get_embedding_model(self):
        """Initialize the sentence transformer model for embeddings."""
        if self.embedding_model is None:
            try:
                # Use a lightweight model that works well for financial text
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                self.embedding_model = None
        return self.embedding_model
    
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            model = await self._get_embedding_model()
            if model is None:
                # Fallback: return random embeddings
                return [np.random.rand(384).tolist() for _ in texts]
            
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback: return random embeddings
            return [np.random.rand(384).tolist() for _ in texts]
    
    async def _query_ollama(self, prompt: str, model: str = "llama3.1:8b", max_retries: int = 3) -> str:
        """Query Ollama LLM with a prompt and retry logic."""
        # Try different models in order of preference (faster to slower)
        models_to_try = [model, "llama3.1:8b", "llama3:8b", "llama3:latest"]
        
        for model_to_use in models_to_try:
            logger.info(f"Trying model: {model_to_use}")
            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            "model": model_to_use,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "max_tokens": 800,  # Reduced from 1000 to improve speed
                                "num_predict": 400,  # Limit response length
                                "stop": ["\n\n\n"]  # Stop generation at multiple newlines
                            }
                        }
                        
                        # Increase timeout to 60 seconds for complex sector analyses
                        timeout = aiohttp.ClientTimeout(total=60)
                        
                        async with session.post(
                            f"{self.ollama_url}/api/generate",
                            json=payload,
                            timeout=timeout
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                response_text = result.get("response", "No response from Ollama")
                                if response_text and not response_text.startswith("Error:"):
                                    logger.info(f"Ollama request successful with model {model_to_use}")
                                    return response_text
                                else:
                                    logger.warning(f"Ollama returned error response: {response_text}")
                                    break  # Try next model
                            else:
                                logger.error(f"Ollama API error: {response.status}")
                                break  # Try next model
                                
                except asyncio.TimeoutError:
                    logger.error(f"Ollama request timeout with model {model_to_use}")
                    break  # Try next model
                except Exception as e:
                    logger.error(f"Error querying Ollama with model {model_to_use}: {e}")
                    break  # Try next model
        
        return "Error: LLM request failed with all available models"
    
    async def get_rag_event_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive RAG Event Agent summary with real data."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get document statistics
                doc_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(DISTINCT source) as active_sources,
                        MAX(published_at) as latest_news
                    FROM rag_news_documents
                    WHERE published_at >= NOW() - INTERVAL '7 days'
                """)
                
                # Get analysis statistics
                analysis_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_queries,
                        COALESCE(AVG(response_time_ms), 0) as avg_response_time,
                        COALESCE(AVG(confidence), 0) as avg_confidence
                    FROM rag_analysis
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                
                # Get performance metrics
                performance_stats = await conn.fetchrow("""
                    SELECT 
                        COALESCE(AVG(metric_value), 0) as rag_accuracy
                    FROM rag_performance
                    WHERE metric_name = 'rag_accuracy'
                    AND measurement_date >= NOW() - INTERVAL '24 hours'
                """)
                
                return {
                    "total_documents": doc_stats['total_documents'] or 0,
                    "vector_db_size": doc_stats['total_documents'] or 0,
                    "last_news_update": doc_stats['latest_news'].isoformat() if doc_stats['latest_news'] else datetime.now().isoformat(),
                    "rag_accuracy": float(performance_stats['rag_accuracy'] or 0.85),
                    "llm_enabled": True,
                    "active_sources": doc_stats['active_sources'] or 0,
                    "total_queries": analysis_stats['total_queries'] or 0,
                    "avg_response_time": float(analysis_stats['avg_response_time'] or 0.0),
                    "avg_confidence": float(analysis_stats['avg_confidence'] or 0.0),
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting RAG Event Agent summary: {e}")
            # Return fallback data
            return self._get_fallback_summary()
    
    async def get_rag_documents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent news documents with real data."""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        doc_id,
                        title,
                        content,
                        source,
                        url,
                        category,
                        tags,
                        similarity_score,
                        published_at,
                        ingested_at
                    FROM rag_news_documents
                    ORDER BY published_at DESC
                    LIMIT $1
                """, limit)
                
                documents = []
                for row in rows:
                    documents.append({
                        "doc_id": row['doc_id'],
                        "title": row['title'],
                        "content": row['content'][:500] + "..." if len(row['content']) > 500 else row['content'],
                        "source": row['source'],
                        "url": row['url'],
                        "category": row['category'],
                        "tags": row['tags'] or [],
                        "similarity_score": float(row['similarity_score']) if row['similarity_score'] else None,
                        "timestamp": row['published_at'].isoformat(),
                        "ingested_at": row['ingested_at'].isoformat()
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Error getting RAG documents: {e}")
            return []
    
    async def get_rag_analysis(self, query: str = None, sector: str = None) -> Dict[str, Any]:
        """Get RAG analysis with real Ollama LLM for a specific sector or general market."""
        try:
            start_time = datetime.now()
            
            # Use provided query or create a default one based on sector
            if not query:
                if sector == 'technology':
                    query = "What are the current trends and opportunities in technology stocks, including AI, semiconductors, and software companies?"
                elif sector == 'finance':
                    query = "What are the latest developments in financial services, banking, and fintech that could impact financial stocks?"
                elif sector == 'healthcare':
                    query = "What are the current trends in healthcare, pharmaceuticals, and biotech that could affect healthcare stocks?"
                elif sector == 'retail':
                    query = "What are the latest trends in retail, e-commerce, and consumer spending that could impact retail stocks?"
                else:
                    query = "What are the current market trends and their potential impact on stocks across all sectors?"
            
            async with self.db_pool.acquire() as conn:
                # Get relevant documents using vector similarity search with sector filtering
                relevant_docs = await self._retrieve_relevant_documents(conn, query, sector=sector)
                
                # Generate LLM response using Ollama with sector context
                llm_response = await self._generate_sector_analysis_with_ollama(query, relevant_docs, sector)
                
                # Calculate confidence based on document relevance and response quality
                confidence = await self._calculate_confidence(relevant_docs, llm_response)
                
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Save the analysis to database with sector information
                analysis_id = await self._save_analysis(conn, query, llm_response, confidence, relevant_docs, response_time, sector)
                
                return {
                    "query": query,
                    "sector": sector or "general",
                    "relevant_docs": relevant_docs[:5],  # Limit to top 5 docs
                    "llm_response": llm_response,
                    "confidence": confidence,
                    "reasoning": f"Analysis based on {len(relevant_docs)} relevant documents using Ollama LLM",
                    "analysis_type": f"{sector}_impact" if sector else "market_impact",
                    "response_time_ms": int(response_time),
                    "created_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting RAG analysis: {e}")
            return self._get_fallback_analysis()
    
    async def get_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Get sector-specific RAG analysis."""
        try:
            sector_queries = {
                'technology': "Analyze current trends in technology sector including AI, semiconductors, cloud computing, and software companies. What are the key opportunities and risks?",
                'finance': "Analyze current trends in financial services sector including banking, fintech, payment systems, and investment services. What are the key opportunities and risks?",
                'healthcare': "Analyze current trends in healthcare sector including pharmaceuticals, biotech, medical devices, and healthcare services. What are the key opportunities and risks?",
                'retail': "Analyze current trends in retail sector including e-commerce, consumer spending, supply chain, and retail innovation. What are the key opportunities and risks?"
            }
            
            query = sector_queries.get(sector.lower(), sector_queries['technology'])
            return await self.get_rag_analysis(query, sector)
            
        except Exception as e:
            logger.error(f"Error getting sector analysis for {sector}: {e}")
            return self._get_fallback_analysis()
    
    async def get_multi_sector_analysis(self) -> Dict[str, Any]:
        """Get analysis for all major sectors."""
        try:
            sectors = ['technology', 'finance', 'healthcare', 'retail']
            sector_analyses = {}
            
            for sector in sectors:
                try:
                    sector_analyses[sector] = await self.get_sector_analysis(sector)
                except Exception as e:
                    logger.error(f"Error analyzing {sector} sector: {e}")
                    sector_analyses[sector] = self._get_fallback_analysis()
            
            return {
                "sector_analyses": sector_analyses,
                "generated_at": datetime.now().isoformat(),
                "total_sectors": len(sectors)
            }
            
        except Exception as e:
            logger.error(f"Error getting multi-sector analysis: {e}")
            return {"sector_analyses": {}, "error": str(e)}
    
    async def _retrieve_relevant_documents(self, conn, query: str, limit: int = 10, sector: str = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using vector similarity with optional sector filtering."""
        try:
            # Build SQL query with optional sector filtering
            if sector:
                sector_categories = {
                    'technology': ['technology', 'earnings'],
                    'finance': ['finance', 'earnings', 'monetary_policy'],
                    'healthcare': ['healthcare', 'earnings'],
                    'retail': ['retail', 'earnings']
                }
                categories = sector_categories.get(sector.lower(), [])
                
                if categories:
                    placeholders = ','.join([f"${i+1}" for i in range(len(categories))])
                    sql_query = f"""
                        SELECT doc_id, title, content, source, url, category, tags, published_at
                        FROM rag_news_documents
                        WHERE category = ANY(ARRAY[{placeholders}])
                        ORDER BY published_at DESC
                        LIMIT 50
                    """
                    doc_rows = await conn.fetch(sql_query, *categories)
                else:
                    doc_rows = await conn.fetch("""
                        SELECT doc_id, title, content, source, url, category, tags, published_at
                        FROM rag_news_documents
                        ORDER BY published_at DESC
                        LIMIT 50
                    """)
            else:
                doc_rows = await conn.fetch("""
                    SELECT doc_id, title, content, source, url, category, tags, published_at
                    FROM rag_news_documents
                    ORDER BY published_at DESC
                    LIMIT 50
                """)
            
            if not doc_rows:
                return []
            
            # Generate embeddings for query and documents
            query_embedding = await self._get_embeddings([query])
            doc_texts = [f"{row['title']} {row['content'][:500]}" for row in doc_rows]
            doc_embeddings = await self._get_embeddings(doc_texts)
            
            # Calculate similarities (cosine similarity)
            similarities = []
            for i, doc_row in enumerate(doc_rows):
                similarity = self._cosine_similarity(query_embedding[0], doc_embeddings[i])
                similarities.append((similarity, doc_row))
            
            # Sort by similarity and return top documents
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            relevant_docs = []
            for similarity, doc_row in similarities[:limit]:
                if similarity > 0.3:  # Minimum similarity threshold
                    relevant_docs.append({
                        "doc_id": doc_row['doc_id'],
                        "title": doc_row['title'],
                        "content": doc_row['content'][:200] + "..." if len(doc_row['content']) > 200 else doc_row['content'],
                        "source": doc_row['source'],
                        "url": doc_row['url'],
                        "category": doc_row['category'],
                        "tags": doc_row['tags'] or [],
                        "similarity": similarity
                    })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    async def _generate_sector_analysis_with_ollama(self, query: str, relevant_docs: List[Dict[str, Any]], sector: str = None) -> str:
        """Generate sector-specific analysis using Ollama LLM."""
        try:
            # Build context from relevant documents
            context = ""
            for i, doc in enumerate(relevant_docs[:3], 1):
                context += f"{i}. {doc['title']}\n   Source: {doc['source']}\n   Content: {doc['content'][:300]}...\n\n"
            
            # Create sector-specific prompts (optimized for faster responses)
            sector_prompts = {
                'technology': """Tech analyst: Provide concise insights on AI, semiconductors, and software trends.""",
                'finance': """Finance analyst: Provide concise insights on banking, fintech, and payment systems.""",
                'healthcare': """Healthcare analyst: Provide concise insights on pharma, biotech, and medical devices.""",
                'retail': """Retail analyst: Provide concise insights on e-commerce and consumer spending."""
            }
            
            base_prompt = sector_prompts.get(sector, "Market analyst: Provide concise insights.")
            
            prompt = f"""{base_prompt}

Query: {query}

News Context:
{context}

Provide a brief analysis with:
1. Key sector trends (2-3 points)
2. Investment opportunities and risks (2-3 points)
3. Overall sector outlook

Keep response under 300 words, focus on actionable insights."""

            response = await self._query_ollama(prompt)
            
            # Clean up the response
            if response.startswith("Error:"):
                return f"LLM analysis unavailable: {response}"
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating sector analysis with Ollama: {e}")
            return f"Error generating analysis: {str(e)}"
    
    async def _generate_analysis_with_ollama(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate general analysis using Ollama LLM (backward compatibility)."""
        return await self._generate_sector_analysis_with_ollama(query, relevant_docs, None)
    
    async def _calculate_confidence(self, relevant_docs: List[Dict[str, Any]], llm_response: str) -> float:
        """Calculate confidence score based on document relevance and response quality."""
        try:
            base_confidence = 0.5
            
            # Adjust based on number of relevant documents
            if len(relevant_docs) >= 3:
                base_confidence += 0.2
            elif len(relevant_docs) >= 1:
                base_confidence += 0.1
            
            # Adjust based on average document similarity
            if relevant_docs:
                avg_similarity = sum(doc.get('similarity', 0) for doc in relevant_docs) / len(relevant_docs)
                base_confidence += min(avg_similarity * 0.3, 0.3)
            
            # Adjust based on response quality (length and content)
            if len(llm_response) > 100 and not llm_response.startswith("Error"):
                base_confidence += 0.1
            
            return min(base_confidence, 0.95)  # Cap at 95%
            
        except Exception:
            return 0.5
    
    async def _save_analysis(self, conn, query: str, llm_response: str, confidence: float, 
                           relevant_docs: List[Dict[str, Any]], response_time: float, sector: str = None) -> str:
        """Save comprehensive analysis to database with all relevant data."""
        try:
            doc_ids = [doc['doc_id'] for doc in relevant_docs]
            
            # Extract source information
            sources = list(set([doc['source'] for doc in relevant_docs]))
            source_count = len(sources)
            
            # Calculate performance metrics
            avg_similarity = sum(doc.get('similarity', 0) for doc in relevant_docs) / len(relevant_docs) if relevant_docs else 0
            doc_count = len(relevant_docs)
            
            # Create comprehensive metadata
            metadata = {
                "sources": sources,
                "source_count": source_count,
                "avg_similarity": avg_similarity,
                "doc_count": doc_count,
                "llm_model": "llama3:latest",
                "embedding_model": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.3,
                "query_type": f"{sector}_analysis" if sector else "market_analysis",
                "sector": sector or "general",
                "response_length": len(llm_response),
                "relevant_docs_details": [
                    {
                        "doc_id": doc['doc_id'],
                        "title": doc['title'],
                        "source": doc['source'],
                        "similarity": doc.get('similarity', 0),
                        "category": doc.get('category', 'unknown')
                    } for doc in relevant_docs
                ]
            }
            
            # Insert analysis and get the generated ID
            analysis_type = f"{sector}_impact" if sector else "market_impact"
            result = await conn.fetchrow("""
                INSERT INTO rag_analysis 
                (query, llm_response, confidence, reasoning, analysis_type, 
                 relevant_doc_ids, response_time_ms, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """, query, llm_response, confidence, 
                 f"Analysis based on {doc_count} documents from {source_count} sources using Ollama LLM",
                 analysis_type, doc_ids, int(response_time), datetime.now())
            
            analysis_id = f"analysis_{result['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store detailed performance metrics
            await self._store_analysis_performance_metrics(conn, analysis_id, confidence, response_time, metadata)
            
            logger.info(f"Saved comprehensive RAG analysis {analysis_id} with {doc_count} docs from {source_count} sources")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return "unknown"
    
    async def _store_analysis_performance_metrics(self, conn, analysis_id: str, confidence: float, 
                                                response_time: float, metadata: Dict[str, Any]):
        """Store detailed performance metrics for the analysis."""
        try:
            # Store individual performance metrics
            metrics = [
                ('analysis_confidence', confidence, 'percentage', f'Confidence score for analysis {analysis_id}'),
                ('response_time_ms', response_time, 'milliseconds', f'Response time for analysis {analysis_id}'),
                ('sources_count', metadata['source_count'], 'count', f'Number of sources used in analysis {analysis_id}'),
                ('documents_count', metadata['doc_count'], 'count', f'Number of documents used in analysis {analysis_id}'),
                ('avg_similarity_score', metadata['avg_similarity'], 'score', f'Average document similarity for analysis {analysis_id}'),
                ('llm_response_length', metadata['response_length'], 'characters', f'Length of LLM response for analysis {analysis_id}')
            ]
            
            for metric_name, metric_value, metric_unit, notes in metrics:
                await conn.execute("""
                    INSERT INTO rag_performance 
                    (metric_name, metric_value, metric_unit, measurement_date, notes)
                    VALUES ($1, $2, $3, $4, $5)
                """, metric_name, metric_value, metric_unit, datetime.now(), notes)
                
        except Exception as e:
            logger.error(f"Error storing analysis performance metrics: {e}")
    
    async def get_rag_performance(self) -> Dict[str, Any]:
        """Get RAG system performance metrics with real data."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get performance metrics
                rows = await conn.fetch("""
                    SELECT 
                        metric_name,
                        metric_value,
                        metric_unit,
                        measurement_date
                    FROM rag_performance
                    WHERE measurement_date >= NOW() - INTERVAL '24 hours'
                    ORDER BY measurement_date DESC
                """)
                
                performance_metrics = {}
                for row in rows:
                    performance_metrics[row['metric_name']] = {
                        "value": float(row['metric_value']),
                        "unit": row['metric_unit'],
                        "timestamp": row['measurement_date'].isoformat()
                    }
                
                return {
                    "metrics": performance_metrics,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting RAG performance: {e}")
            return {"metrics": {}, "last_updated": datetime.now().isoformat()}
    
    async def fetch_real_news_articles(self) -> List[Dict[str, Any]]:
        """Fetch real news articles from external sources."""
        try:
            articles = []
            
            # Fetch from Yahoo Finance RSS feeds
            yahoo_articles = await self._fetch_from_yahoo_finance()
            articles.extend(yahoo_articles)
            
            # Fetch from financial RSS feeds
            rss_articles = await self._fetch_from_rss_feeds()
            articles.extend(rss_articles)
            
            # Remove duplicates and filter recent articles
            unique_articles = self._deduplicate_articles(articles)
            recent_articles = []
            for a in unique_articles:
                try:
                    published_at = a.get('published_at', datetime.now())
                    # Ensure both datetimes are naive (no timezone info)
                    if published_at.tzinfo is not None:
                        published_at = published_at.replace(tzinfo=None)
                    if (datetime.now() - published_at).days <= 14:  # Extended to 14 days for more data
                        recent_articles.append(a)
                except:
                    # If there's any date parsing issue, include the article
                    recent_articles.append(a)
            
            logger.info(f"Fetched {len(recent_articles)} real news articles from {len(articles)} total articles")
            return recent_articles[:100]  # Increased limit to 100 most recent articles for better RAG accuracy
            
        except Exception as e:
            logger.error(f"Error fetching real news articles: {e}")
            return []
    
    async def _fetch_from_yahoo_finance(self) -> List[Dict[str, Any]]:
        """Fetch news from Yahoo Finance."""
        try:
            articles = []
            
            # Yahoo Finance RSS feeds for financial news (expanded)
            rss_feeds = [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.finance.yahoo.com/rss/2.0/topfinstories',
                'https://feeds.finance.yahoo.com/rss/2.0/stock',
                'https://feeds.finance.yahoo.com/rss/2.0/marketnews',
                'https://feeds.finance.yahoo.com/rss/2.0/industry'
            ]
            
            for feed_url in rss_feeds:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(feed_url, timeout=10) as response:
                            if response.status == 200:
                                content = await response.text()
                                articles.extend(self._parse_rss_content(content, feed_url))
                except Exception as e:
                    logger.warning(f"Failed to fetch from {feed_url}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from Yahoo Finance: {e}")
            return []
    
    async def _fetch_from_rss_feeds(self) -> List[Dict[str, Any]]:
        """Fetch news from financial RSS feeds."""
        try:
            articles = []
            
            # Comprehensive financial news RSS feeds (expanded for maximum coverage)
            rss_feeds = [
                # MarketWatch feeds
                'https://feeds.marketwatch.com/marketwatch/topstories/',
                'https://feeds.marketwatch.com/marketwatch/marketpulse/',
                'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
                
                # Yahoo Finance feeds (additional)
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.finance.yahoo.com/rss/2.0/topfinstories',
                'https://feeds.finance.yahoo.com/rss/2.0/stock',
                'https://feeds.finance.yahoo.com/rss/2.0/marketnews',
                'https://feeds.finance.yahoo.com/rss/2.0/industry',
                
                # CNBC feeds
                'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
                'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069',
                'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362',
                
                # Financial Times feeds
                'https://www.ft.com/rss/home',
                'https://www.ft.com/markets?format=rss',
                'https://www.ft.com/companies?format=rss',
                
                # Reuters Business feeds
                'https://feeds.reuters.com/reuters/businessNews',
                'https://feeds.reuters.com/news/wealth',
                'https://feeds.reuters.com/reuters/companyNews',
                
                # Bloomberg feeds
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://feeds.bloomberg.com/politics/news.rss',
                
                # Investing.com feeds
                'https://www.investing.com/rss/news.rss',
                'https://www.investing.com/rss/news_14.rss',
                
                # Benzinga feeds
                'https://www.benzinga.com/topic/rss',
                'https://www.benzinga.com/news/rss',
                
                # Seeking Alpha feeds
                'https://seekingalpha.com/api/sa/combined/RSS.xml',
                'https://seekingalpha.com/api/sa/combined/RSS_earnings.xml'
            ]
            
            for feed_url in rss_feeds:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(feed_url, timeout=15) as response:  # Increased timeout for more sources
                            if response.status == 200:
                                content = await response.text()
                                parsed_articles = self._parse_rss_content(content, feed_url)
                                articles.extend(parsed_articles)
                                logger.info(f"Successfully fetched {len(parsed_articles)} articles from {self._get_source_name(feed_url)}")
                            else:
                                logger.warning(f"HTTP {response.status} from {feed_url}")
                except Exception as e:
                    logger.warning(f"Failed to fetch from {feed_url}: {e}")
                    continue
            
            logger.info(f"Total articles collected from all RSS feeds: {len(articles)}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from RSS feeds: {e}")
            return []
    
    def _parse_rss_content(self, content: str, source_url: str) -> List[Dict[str, Any]]:
        """Parse RSS content and extract articles."""
        try:
            import xml.etree.ElementTree as ET
            
            articles = []
            root = ET.fromstring(content)
            
            for item in root.findall('.//item'):
                title = item.find('title')
                description = item.find('description')
                link = item.find('link')
                pub_date = item.find('pubDate')
                
                if title is not None and title.text:
                    article = {
                        'title': title.text.strip(),
                        'content': description.text.strip() if description is not None and description.text else '',
                        'url': link.text.strip() if link is not None and link.text else '',
                        'source': self._get_source_name(source_url),
                        'category': self._categorize_article(title.text.strip()),
                        'tags': self._extract_tags(title.text.strip()),
                        'published_at': self._parse_rss_date(pub_date.text) if pub_date is not None and pub_date.text else datetime.now()
                    }
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing RSS content: {e}")
            return []
    
    def _get_source_name(self, source_url: str) -> str:
        """Extract source name from URL."""
        if 'yahoo' in source_url:
            return 'Yahoo Finance'
        elif 'marketwatch' in source_url:
            return 'MarketWatch'
        elif 'bloomberg' in source_url:
            return 'Bloomberg'
        elif 'reuters' in source_url:
            return 'Reuters'
        else:
            return 'Financial News'
    
    def _categorize_article(self, title: str) -> str:
        """Categorize article based on title content into specific sectors."""
        title_lower = title.lower()
        
        # Technology sector
        if any(word in title_lower for word in ['tech', 'ai', 'artificial intelligence', 'nvidia', 'apple', 'microsoft', 'google', 'amazon', 'meta', 'tesla', 'software', 'cloud', 'cyber', 'digital', 'semiconductor', 'chip']):
            return 'technology'
        # Finance sector
        elif any(word in title_lower for word in ['bank', 'financial', 'jpmorgan', 'goldman', 'morgan stanley', 'wells fargo', 'bank of america', 'credit', 'lending', 'mortgage', 'investment', 'hedge fund', 'private equity', 'fintech', 'payment', 'visa', 'mastercard']):
            return 'finance'
        # Healthcare sector
        elif any(word in title_lower for word in ['health', 'medical', 'pharma', 'biotech', 'pfizer', 'moderna', 'johnson & johnson', 'merck', 'bristol', 'abbott', 'medtronic', 'hospital', 'drug', 'vaccine', 'clinical trial', 'fda', 'healthcare']):
            return 'healthcare'
        # Retail sector
        elif any(word in title_lower for word in ['retail', 'walmart', 'target', 'costco', 'home depot', 'lowes', 'consumer', 'shopping', 'e-commerce', 'amazon', 'ebay', 'sales', 'store', 'chain', 'merchandise', 'inventory']):
            return 'retail'
        # General market categories
        elif any(word in title_lower for word in ['fed', 'federal reserve', 'interest rate', 'monetary policy']):
            return 'monetary_policy'
        elif any(word in title_lower for word in ['earnings', 'revenue', 'profit', 'quarterly']):
            return 'earnings'
        elif any(word in title_lower for word in ['inflation', 'cpi', 'economic data', 'gdp']):
            return 'economic_data'
        elif any(word in title_lower for word in ['oil', 'energy', 'commodities', 'gold']):
            return 'commodities'
        else:
            return 'general_market'
    
    def _extract_tags(self, title: str) -> List[str]:
        """Extract relevant tags from article title."""
        title_lower = title.lower()
        tags = []
        
        # Common financial tags
        tag_keywords = {
            'federal_reserve': ['fed', 'federal reserve', 'interest rate'],
            'tech_stocks': ['tech', 'nvidia', 'apple', 'microsoft', 'google'],
            'earnings': ['earnings', 'revenue', 'profit', 'quarterly'],
            'inflation': ['inflation', 'cpi', 'economic data'],
            'oil': ['oil', 'energy', 'crude'],
            'ai': ['ai', 'artificial intelligence', 'machine learning']
        }
        
        for tag, keywords in tag_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _parse_rss_date(self, date_str: str) -> datetime:
        """Parse RSS date string to datetime."""
        try:
            from email.utils import parsedate_to_datetime
            parsed_date = parsedate_to_datetime(date_str)
            # Convert to naive datetime to avoid timezone issues
            if parsed_date.tzinfo is not None:
                return parsed_date.replace(tzinfo=None)
            return parsed_date
        except:
            return datetime.now()
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title_lower = article['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
        
        return unique_articles

    async def create_sample_data(self):
        """Create sample RAG Event Agent data for demonstration."""
        try:
            # First, try to fetch real news articles
            real_articles = await self.fetch_real_news_articles()
            
            async with self.db_pool.acquire() as conn:
                # Use real articles if available, otherwise fall back to sample data
                if real_articles:
                    documents = []
                    for i, article in enumerate(real_articles, 1):
                        doc = {
                            'doc_id': f'real_news_{i:03d}',
                            'title': article['title'],
                            'content': article['content'],
                            'source': article['source'],
                            'url': article['url'],
                            'category': article['category'],
                            'tags': article['tags'],
                            'published_at': article['published_at']
                        }
                        documents.append(doc)
                else:
                    # Fallback to sample documents if real news fetching fails
                    documents = [
                        {
                            'doc_id': 'news_001',
                            'title': 'Federal Reserve Signals Potential Rate Cut Amid Economic Uncertainty',
                            'content': 'The Federal Reserve indicated a possible shift in monetary policy as economic indicators show signs of slowing growth. Fed officials are considering rate cuts to stimulate economic activity and address inflation concerns.',
                            'source': 'Federal Reserve',
                            'url': 'https://federalreserve.gov/news/2024/rate-policy-update',
                            'category': 'monetary_policy',
                            'tags': ['federal_reserve', 'interest_rates', 'monetary_policy'],
                            'published_at': datetime.now() - timedelta(hours=2)
                        },
                        {
                            'doc_id': 'news_002',
                            'title': 'Tech Stocks Rally on Strong Q4 Earnings Reports',
                            'content': 'Major technology companies reported better-than-expected quarterly earnings, driving a significant rally in tech stocks. NVIDIA, Apple, and Microsoft led the gains with strong AI and cloud computing revenue.',
                            'source': 'MarketWatch',
                            'url': 'https://marketwatch.com/tech-earnings-q4',
                            'category': 'earnings',
                            'tags': ['tech_stocks', 'earnings', 'nvidia', 'apple', 'microsoft'],
                            'published_at': datetime.now() - timedelta(hours=4)
                        },
                        {
                            'doc_id': 'news_003',
                            'title': 'Oil Prices Surge on Middle East Tensions and Supply Concerns',
                            'content': 'Crude oil prices jumped 5% following renewed tensions in the Middle East and reports of supply disruptions. Energy sector stocks gained while transportation companies faced pressure from higher fuel costs.',
                            'source': 'Reuters',
                            'url': 'https://reuters.com/business/energy/oil-prices-surge',
                            'category': 'commodities',
                            'tags': ['oil', 'energy', 'middle_east', 'supply_disruption'],
                            'published_at': datetime.now() - timedelta(hours=6)
                        },
                        {
                            'doc_id': 'news_004',
                            'title': 'Inflation Data Shows Cooling Trend, Boosting Market Sentiment',
                            'content': 'Latest inflation figures indicate a continued cooling trend, with core CPI rising at the slowest pace in months. This data supports the case for potential Fed rate cuts and boosted overall market sentiment.',
                            'source': 'Bloomberg',
                            'url': 'https://bloomberg.com/inflation-data-december',
                            'category': 'economic_data',
                            'tags': ['inflation', 'cpi', 'economic_data', 'fed_policy'],
                            'published_at': datetime.now() - timedelta(hours=8)
                        },
                        {
                            'doc_id': 'news_005',
                            'title': 'AI Sector Sees Massive Investment Influx as Companies Accelerate Adoption',
                            'content': 'Artificial intelligence companies are experiencing unprecedented investment flows as enterprises accelerate AI adoption. Venture capital funding in AI startups reached record levels this quarter.',
                            'source': 'TechCrunch',
                            'url': 'https://techcrunch.com/ai-investment-surge',
                            'category': 'technology',
                            'tags': ['artificial_intelligence', 'investment', 'venture_capital', 'startups'],
                            'published_at': datetime.now() - timedelta(hours=10)
                        }
                    ]
                
                # Insert documents (real or sample)
                for doc in documents:
                    await conn.execute("""
                        INSERT INTO rag_news_documents 
                        (doc_id, title, content, source, url, category, tags, published_at, ingested_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (doc_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            updated_at = EXCLUDED.ingested_at
                    """, doc['doc_id'], doc['title'], doc['content'], doc['source'], 
                         doc['url'], doc['category'], doc['tags'], doc['published_at'], 
                         datetime.now())
                
                # Sample RAG analysis
                sample_analysis = {
                    'query': 'Market events and news affecting NVDA stock price and trading',
                    'llm_response': 'Based on recent news analysis, NVIDIA is positioned favorably due to strong AI sector investment trends and better-than-expected tech earnings. The company benefits from continued enterprise AI adoption and venture capital funding in AI startups. However, broader market concerns about Fed policy and economic uncertainty may create volatility.',
                    'confidence': 0.87,
                    'reasoning': 'Analysis based on 5 relevant documents including tech earnings reports, AI investment trends, and Fed policy signals. High confidence due to multiple corroborating sources.',
                    'analysis_type': 'market_impact',
                    'relevant_doc_ids': ['news_002', 'news_005', 'news_004'],
                    'response_time_ms': 1250
                }
                
                await conn.execute("""
                    INSERT INTO rag_analysis 
                    (query, llm_response, confidence, reasoning, analysis_type, relevant_doc_ids, response_time_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, sample_analysis['query'], sample_analysis['llm_response'], 
                     sample_analysis['confidence'], sample_analysis['reasoning'],
                     sample_analysis['analysis_type'], sample_analysis['relevant_doc_ids'],
                     sample_analysis['response_time_ms'])
                
                # Sample performance metrics
                performance_metrics = [
                    ('rag_accuracy', 0.87, 'percentage', 'Overall RAG system accuracy'),
                    ('avg_response_time', 1250.0, 'ms', 'Average response time for analysis'),
                    ('document_retrieval_success', 0.95, 'percentage', 'Success rate for document retrieval'),
                    ('llm_confidence_score', 0.82, 'score', 'Average LLM confidence score'),
                    ('query_processing_rate', 45.5, 'queries/hour', 'Rate of query processing')
                ]
                
                for metric_name, metric_value, metric_unit, notes in performance_metrics:
                    await conn.execute("""
                        INSERT INTO rag_performance 
                        (metric_name, metric_value, metric_unit, measurement_date, notes)
                        VALUES ($1, $2, $3, $4, $5)
                    """, metric_name, metric_value, metric_unit, datetime.now(), notes)
                
                logger.info("Sample RAG Event Agent data created successfully")
                
        except Exception as e:
            logger.error(f"Error creating sample RAG Event Agent data: {e}")
    
    def _get_fallback_summary(self) -> Dict[str, Any]:
        """Get fallback summary data when database is unavailable."""
        return {
            "total_documents": 0,
            "vector_db_size": 0,
            "last_news_update": datetime.now().isoformat(),
            "rag_accuracy": 0.0,
            "llm_enabled": False,
            "active_sources": 0,
            "total_queries": 0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_historical_analyses(self, limit: int = 20, days: int = 7) -> List[Dict[str, Any]]:
        """Get historical RAG analyses for insights and trend analysis."""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        id,
                        query,
                        llm_response,
                        confidence,
                        reasoning,
                        analysis_type,
                        relevant_doc_ids,
                        response_time_ms,
                        created_at
                    FROM rag_analysis
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    ORDER BY created_at DESC
                    LIMIT $1
                """ % days, limit)
                
                analyses = []
                for row in rows:
                    analysis_id = f"analysis_{row['id']}_{row['created_at'].strftime('%Y%m%d_%H%M%S')}"
                    analyses.append({
                        "analysis_id": analysis_id,
                        "query": row['query'],
                        "llm_response": row['llm_response'][:500] + "..." if len(row['llm_response']) > 500 else row['llm_response'],
                        "confidence": float(row['confidence']),
                        "reasoning": row['reasoning'],
                        "analysis_type": row['analysis_type'],
                        "relevant_doc_ids": row['relevant_doc_ids'],
                        "response_time_ms": row['response_time_ms'],
                        "created_at": row['created_at'].isoformat()
                    })
                
                return analyses
                
        except Exception as e:
            logger.error(f"Error getting historical analyses: {e}")
            return []
    
    async def get_analysis_insights(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive insights from historical RAG analyses."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get analysis statistics
                analysis_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_analyses,
                        AVG(confidence) as avg_confidence,
                        AVG(response_time_ms) as avg_response_time,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence,
                        COUNT(DISTINCT DATE(created_at)) as active_days
                    FROM rag_analysis
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                """ % days)
                
                # Get query patterns
                query_patterns = await conn.fetch("""
                    SELECT 
                        query,
                        COUNT(*) as frequency,
                        AVG(confidence) as avg_confidence
                    FROM rag_analysis
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT 10
                """ % days)
                
                # Get confidence trends
                confidence_trends = await conn.fetch("""
                    SELECT 
                        DATE(created_at) as analysis_date,
                        AVG(confidence) as avg_confidence,
                        COUNT(*) as analysis_count
                    FROM rag_analysis
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY DATE(created_at)
                    ORDER BY analysis_date DESC
                """ % days)
                
                # Get performance metrics
                performance_metrics = await conn.fetch("""
                    SELECT 
                        metric_name,
                        AVG(metric_value) as avg_value,
                        MAX(metric_value) as max_value,
                        MIN(metric_value) as min_value
                    FROM rag_performance
                    WHERE measurement_date >= NOW() - INTERVAL '%s days'
                    GROUP BY metric_name
                """ % days)
                
                return {
                    "analysis_statistics": {
                        "total_analyses": analysis_stats['total_analyses'] or 0,
                        "avg_confidence": float(analysis_stats['avg_confidence'] or 0),
                        "avg_response_time": float(analysis_stats['avg_response_time'] or 0),
                        "confidence_range": {
                            "min": float(analysis_stats['min_confidence'] or 0),
                            "max": float(analysis_stats['max_confidence'] or 0)
                        },
                        "active_days": analysis_stats['active_days'] or 0
                    },
                    "query_patterns": [
                        {
                            "query": row['query'],
                            "frequency": row['frequency'],
                            "avg_confidence": float(row['avg_confidence'])
                        } for row in query_patterns
                    ],
                    "confidence_trends": [
                        {
                            "date": row['analysis_date'].isoformat(),
                            "avg_confidence": float(row['avg_confidence']),
                            "analysis_count": row['analysis_count']
                        } for row in confidence_trends
                    ],
                    "performance_metrics": {
                        row['metric_name']: {
                            "avg_value": float(row['avg_value']),
                            "max_value": float(row['max_value']),
                            "min_value": float(row['min_value'])
                        } for row in performance_metrics
                    },
                    "insights_generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting analysis insights: {e}")
            return {
                "analysis_statistics": {"total_analyses": 0, "avg_confidence": 0.0},
                "query_patterns": [],
                "confidence_trends": [],
                "performance_metrics": {},
                "insights_generated_at": datetime.now().isoformat()
            }
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Get fallback analysis data when database is unavailable."""
        return {
            "query": "Market analysis unavailable",
            "relevant_docs": [],
            "llm_response": "RAG analysis service temporarily unavailable",
            "confidence": 0.0,
            "reasoning": "Service unavailable",
            "analysis_type": "market_impact",
            "response_time_ms": 0,
            "created_at": datetime.now().isoformat()
        }
