"""
Automated RAG Service - Background News Fetching and Analysis

This service runs automated tasks to:
- Fetch fresh news articles every 30 minutes
- Generate LLM analysis for all sectors (Technology, Finance, Healthcare, Retail)
- Store results in database for forecasting accuracy
- Monitor performance and log activity
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from loguru import logger

from services.rag_event_agent_service import RAGEventAgentService


class AutomatedRAGService:
    """Service for automated RAG news fetching and analysis."""
    
    def __init__(self, db_pool, fetch_interval_minutes: int = 30):
        """
        Initialize the automated RAG service.
        
        Args:
            db_pool: Database connection pool
            fetch_interval_minutes: Interval between news fetches (default: 30 minutes)
        """
        self.db_pool = db_pool
        self.fetch_interval = fetch_interval_minutes * 60  # Convert to seconds
        self.rag_service: Optional[RAGEventAgentService] = None
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        self.last_fetch: Optional[datetime] = None
        self.total_fetches = 0
        self.total_articles_fetched = 0
        self.total_analyses_generated = 0
        
        logger.info(f"AutomatedRAGService initialized with {fetch_interval_minutes}-minute interval")
    
    async def start(self):
        """Start the automated RAG update service."""
        if self.is_running:
            logger.warning("AutomatedRAGService is already running")
            return
        
        # Initialize RAG service
        if not self.rag_service:
            self.rag_service = RAGEventAgentService(self.db_pool)
            logger.info("RAGEventAgentService initialized for automated updates")
        
        self.is_running = True
        self.task = asyncio.create_task(self._run_update_loop())
        logger.info(f"✅ AutomatedRAGService started - will fetch news every {self.fetch_interval // 60} minutes")
    
    async def stop(self):
        """Stop the automated RAG update service."""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("AutomatedRAGService stopped")
    
    async def _run_update_loop(self):
        """Main loop for automated updates."""
        logger.info("🚀 Starting automated RAG update loop...")
        
        # Run initial fetch immediately
        await self._perform_update()
        
        while self.is_running:
            try:
                # Wait for the specified interval
                await asyncio.sleep(self.fetch_interval)
                
                # Perform update
                if self.is_running:
                    await self._perform_update()
                    
            except asyncio.CancelledError:
                logger.info("Automated RAG update loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in automated RAG update loop: {e}")
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _perform_update(self):
        """Perform a complete RAG update cycle."""
        try:
            update_start = datetime.now()
            logger.info("=" * 80)
            logger.info(f"🔄 Starting automated RAG update cycle #{self.total_fetches + 1}")
            logger.info(f"⏰ Time: {update_start.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            
            # Step 1: Fetch fresh news articles
            logger.info("📰 Step 1: Fetching fresh news articles...")
            articles_before = await self._count_recent_articles()
            
            articles = await self.rag_service.fetch_real_news_articles()
            
            # Save fetched articles to database
            new_articles_saved = await self._save_articles_to_database(articles)
            
            articles_after = await self._count_recent_articles()
            new_articles = articles_after - articles_before
            
            logger.info(f"✅ Fetched {len(articles)} articles from sources ({new_articles_saved} new articles saved to database)")
            self.total_articles_fetched += new_articles_saved
            
            # Step 2: Generate analysis for all sectors
            logger.info("🧠 Step 2: Generating LLM analysis for all sectors...")
            sectors = ['technology', 'finance', 'healthcare', 'retail']
            analyses_generated = 0
            
            for sector in sectors:
                try:
                    logger.info(f"  🔍 Analyzing {sector} sector...")
                    analysis = await self.rag_service.get_sector_analysis(sector)
                    
                    if analysis and 'llm_response' in analysis:
                        analyses_generated += 1
                        confidence = analysis.get('confidence', 0)
                        doc_count = len(analysis.get('relevant_docs', []))
                        logger.info(f"  ✅ {sector.capitalize()}: {confidence:.1%} confidence, {doc_count} docs analyzed")
                    else:
                        logger.warning(f"  ⚠️  {sector.capitalize()}: Analysis generation failed")
                        
                except Exception as e:
                    logger.error(f"  ❌ {sector.capitalize()}: Error generating analysis - {e}")
            
            self.total_analyses_generated += analyses_generated
            
            # Step 3: Update statistics
            update_duration = (datetime.now() - update_start).total_seconds()
            self.last_fetch = update_start
            self.total_fetches += 1
            
            logger.info("=" * 80)
            logger.info(f"✅ RAG update cycle completed successfully")
            logger.info(f"📊 Statistics:")
            logger.info(f"  - Duration: {update_duration:.1f}s")
            logger.info(f"  - New articles: {new_articles}")
            logger.info(f"  - Analyses generated: {analyses_generated}/4 sectors")
            logger.info(f"  - Total fetches: {self.total_fetches}")
            logger.info(f"  - Total articles: {self.total_articles_fetched}")
            logger.info(f"  - Total analyses: {self.total_analyses_generated}")
            logger.info(f"⏰ Next update: {(update_start + timedelta(seconds=self.fetch_interval)).strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error performing RAG update: {e}")
            logger.exception(e)
    
    async def _count_recent_articles(self) -> int:
        """Count recent articles in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM rag_news_documents 
                    WHERE published_at >= NOW() - INTERVAL '24 hours'
                """)
                return result or 0
        except Exception as e:
            logger.error(f"Error counting articles: {e}")
            return 0
    
    async def _save_articles_to_database(self, articles: List[Dict[str, Any]]) -> int:
        """Save fetched articles to the database."""
        try:
            if not articles:
                return 0
            
            saved_count = 0
            async with self.db_pool.acquire() as conn:
                for article in articles:
                    try:
                        # Generate a unique doc_id if not present
                        doc_id = article.get('doc_id')
                        if not doc_id:
                            # Create doc_id from title hash and timestamp
                            import hashlib
                            title_hash = hashlib.md5(article.get('title', '').encode()).hexdigest()[:8]
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            doc_id = f"auto_news_{title_hash}_{timestamp}"
                        
                        # Ensure published_at is a datetime object
                        published_at = article.get('published_at', datetime.now())
                        if isinstance(published_at, str):
                            try:
                                published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                            except:
                                published_at = datetime.now()
                        
                        # Remove timezone info if present
                        if published_at.tzinfo is not None:
                            published_at = published_at.replace(tzinfo=None)
                        
                        await conn.execute("""
                            INSERT INTO rag_news_documents 
                            (doc_id, title, content, source, url, category, tags, published_at, ingested_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT (doc_id) DO NOTHING
                        """, 
                        doc_id,
                        article.get('title', ''),
                        article.get('content', ''),
                        article.get('source', 'Unknown'),
                        article.get('url', ''),
                        article.get('category', 'general_market'),
                        article.get('tags', []),
                        published_at,
                        datetime.now()
                        )
                        
                        # Check if the insert was successful (not a conflict)
                        result = await conn.fetchval("""
                            SELECT COUNT(*) FROM rag_news_documents WHERE doc_id = $1
                        """, doc_id)
                        
                        if result and result > 0:
                            saved_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error saving article '{article.get('title', 'Unknown')}': {e}")
                        continue
            
            logger.info(f"💾 Saved {saved_count} new articles to database")
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving articles to database: {e}")
            return 0
    
    async def get_status(self) -> dict:
        """Get current status of the automated RAG service."""
        return {
            "is_running": self.is_running,
            "fetch_interval_minutes": self.fetch_interval // 60,
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "total_fetches": self.total_fetches,
            "total_articles_fetched": self.total_articles_fetched,
            "total_analyses_generated": self.total_analyses_generated,
            "next_fetch": (self.last_fetch + timedelta(seconds=self.fetch_interval)).isoformat() 
                         if self.last_fetch else "Pending initial fetch"
        }
    
    async def force_update(self):
        """Force an immediate update cycle."""
        logger.info("🔄 Forcing immediate RAG update cycle...")
        await self._perform_update()
        return {
            "success": True,
            "message": "Forced update completed",
            "timestamp": datetime.now().isoformat()
        }

