"""
News Ingester for RAG System

This module provides news ingestion functionality for the RAG system,
including RSS feed parsing, web scraping, and content processing.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import hashlib
import re
from urllib.parse import urljoin, urlparse
import time
import json

logger = logging.getLogger(__name__)


class NewsIngester:
    """
    Service for ingesting news from various sources.
    
    Features:
    - RSS feed parsing
    - Web scraping
    - Content extraction and cleaning
    - Duplicate detection
    - Rate limiting
    - Multiple source support
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the news ingester.
        
        Args:
            config: Configuration dictionary
        """
        default_config = {
            'rss_feeds': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.marketwatch.com/marketwatch/topstories/',
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.federalreserve.gov/feeds/press_all.xml',
                'https://feeds.reuters.com/news/wealth',
                'https://feeds.reuters.com/reuters/businessNews',
                'https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US',
                'https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US',
                'https://feeds.finance.yahoo.com/rss/2.0/headline?s=GOOGL&region=US&lang=en-US'
            ],
            'scrape_delay': 1.0,  # Delay between requests in seconds
            'max_articles_per_feed': 20,
            'timeout': 30,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'enable_web_scraping': True,
            'content_extraction': True,
            'duplicate_detection': True
        }
        
        self.config = config or default_config
        self.seen_articles = set()  # For duplicate detection
        self.last_ingestion = {}
        
        logger.info(f"Initialized NewsIngester with {len(self.config['rss_feeds'])} RSS feeds")
    
    def ingest_all_sources(self) -> List[Dict[str, Any]]:
        """
        Ingest news from all configured sources.
        
        Returns:
            List of ingested articles
        """
        try:
            all_articles = []
            
            # Ingest from RSS feeds
            rss_articles = self.ingest_rss_feeds()
            all_articles.extend(rss_articles)
            
            # Ingest from web scraping if enabled
            if self.config['enable_web_scraping']:
                web_articles = self.ingest_web_sources()
                all_articles.extend(web_articles)
            
            # Remove duplicates
            if self.config['duplicate_detection']:
                all_articles = self._remove_duplicates(all_articles)
            
            # Sort by timestamp
            all_articles.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            
            logger.info(f"Ingested {len(all_articles)} articles from all sources")
            return all_articles
            
        except Exception as e:
            logger.error(f"News ingestion failed: {e}")
            return []
    
    def ingest_rss_feeds(self) -> List[Dict[str, Any]]:
        """
        Ingest news from RSS feeds.
        
        Returns:
            List of articles from RSS feeds
        """
        try:
            all_articles = []
            
            for feed_url in self.config['rss_feeds']:
                try:
                    articles = self._ingest_rss_feed(feed_url)
                    all_articles.extend(articles)
                    
                    # Add delay between feeds
                    time.sleep(self.config['scrape_delay'])
                    
                except Exception as e:
                    logger.warning(f"Failed to ingest RSS feed {feed_url}: {e}")
                    continue
            
            logger.info(f"Ingested {len(all_articles)} articles from RSS feeds")
            return all_articles
            
        except Exception as e:
            logger.error(f"RSS feed ingestion failed: {e}")
            return []
    
    def ingest_web_sources(self) -> List[Dict[str, Any]]:
        """
        Ingest news from web sources.
        
        Returns:
            List of articles from web sources
        """
        try:
            # Define web sources to scrape
            web_sources = [
                {
                    'url': 'https://finance.yahoo.com/news/',
                    'title_selector': 'h3 a',
                    'content_selector': '.caas-body',
                    'category': 'finance'
                },
                {
                    'url': 'https://www.marketwatch.com/latest-news',
                    'title_selector': '.article__headline a',
                    'content_selector': '.article__content',
                    'category': 'market'
                }
            ]
            
            all_articles = []
            
            for source in web_sources:
                try:
                    articles = self._scrape_web_source(source)
                    all_articles.extend(articles)
                    
                    # Add delay between sources
                    time.sleep(self.config['scrape_delay'])
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape web source {source['url']}: {e}")
                    continue
            
            logger.info(f"Ingested {len(all_articles)} articles from web sources")
            return all_articles
            
        except Exception as e:
            logger.error(f"Web source ingestion failed: {e}")
            return []
    
    def _ingest_rss_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """
        Ingest articles from a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            List of articles from the feed
        """
        try:
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"RSS feed {feed_url} has parsing issues")
            
            articles = []
            
            # Process feed entries
            for entry in feed.entries[:self.config['max_articles_per_feed']]:
                try:
                    article = self._process_rss_entry(entry, feed_url)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to process RSS entry: {e}")
                    continue
            
            logger.debug(f"Ingested {len(articles)} articles from {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to ingest RSS feed {feed_url}: {e}")
            return []
    
    def _process_rss_entry(self, entry: Dict[str, Any], source_url: str) -> Optional[Dict[str, Any]]:
        """
        Process a single RSS entry.
        
        Args:
            entry: RSS entry data
            source_url: Source URL of the feed
            
        Returns:
            Processed article or None if invalid
        """
        try:
            # Extract basic information
            title = entry.get('title', '').strip()
            description = entry.get('summary', '').strip()
            link = entry.get('link', '')
            
            if not title:
                return None
            
            # Parse timestamp
            timestamp = self._parse_timestamp(entry.get('published', ''))
            
            # Generate article ID
            article_id = self._generate_article_id(title, description, link)
            
            # Check for duplicates
            if self.config['duplicate_detection'] and article_id in self.seen_articles:
                return None
            
            self.seen_articles.add(article_id)
            
            # Extract content if enabled
            content = description
            if self.config['content_extraction'] and link:
                try:
                    scraped_content = self._scrape_article_content(link)
                    if scraped_content:
                        content = scraped_content
                except Exception as e:
                    logger.debug(f"Failed to scrape content from {link}: {e}")
            
            # Determine category
            category = self._determine_category(title, description, source_url)
            
            # Extract tags
            tags = self._extract_tags(title, description)
            
            article = {
                'id': article_id,
                'title': title,
                'content': content,
                'description': description,
                'url': link,
                'source': source_url,
                'timestamp': timestamp,
                'category': category,
                'tags': tags,
                'metadata': {
                    'source_type': 'rss',
                    'entry_id': entry.get('id', ''),
                    'author': entry.get('author', ''),
                    'ingested_at': datetime.now().isoformat()
                }
            }
            
            return article
            
        except Exception as e:
            logger.error(f"Failed to process RSS entry: {e}")
            return None
    
    def _scrape_web_source(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scrape articles from a web source.
        
        Args:
            source: Web source configuration
            
        Returns:
            List of scraped articles
        """
        try:
            headers = {
                'User-Agent': self.config['user_agent']
            }
            
            response = requests.get(
                source['url'],
                headers=headers,
                timeout=self.config['timeout']
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {source['url']}: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find article links
            title_links = soup.select(source['title_selector'])
            
            for link in title_links[:self.config['max_articles_per_feed']]:
                try:
                    article_url = urljoin(source['url'], link.get('href', ''))
                    title = link.get_text().strip()
                    
                    if not title or not article_url:
                        continue
                    
                    # Scrape article content
                    content = self._scrape_article_content(article_url)
                    
                    # Generate article ID
                    article_id = self._generate_article_id(title, content, article_url)
                    
                    # Check for duplicates
                    if self.config['duplicate_detection'] and article_id in self.seen_articles:
                        continue
                    
                    self.seen_articles.add(article_id)
                    
                    # Extract tags
                    tags = self._extract_tags(title, content)
                    
                    article = {
                        'id': article_id,
                        'title': title,
                        'content': content,
                        'description': content[:200] + "..." if len(content) > 200 else content,
                        'url': article_url,
                        'source': source['url'],
                        'timestamp': datetime.now(),
                        'category': source.get('category', 'general'),
                        'tags': tags,
                        'metadata': {
                            'source_type': 'web_scraping',
                            'ingested_at': datetime.now().isoformat()
                        }
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Failed to process web article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to scrape web source {source['url']}: {e}")
            return []
    
    def _scrape_article_content(self, url: str) -> str:
        """
        Scrape content from an article URL.
        
        Args:
            url: Article URL
            
        Returns:
            Scraped content
        """
        try:
            headers = {
                'User-Agent': self.config['user_agent']
            }
            
            response = requests.get(
                url,
                headers=headers,
                timeout=self.config['timeout']
            )
            
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content_selectors = [
                'article',
                '.article-content',
                '.content',
                '.post-content',
                '.entry-content',
                '.article-body',
                '.story-body',
                'main'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text() for elem in elements])
                    break
            
            # If no specific content found, get all text
            if not content:
                content = soup.get_text()
            
            # Clean up content
            content = self._clean_content(content)
            
            return content
            
        except Exception as e:
            logger.debug(f"Failed to scrape content from {url}: {e}")
            return ""
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize content.
        
        Args:
            content: Raw content
            
        Returns:
            Cleaned content
        """
        try:
            # Remove extra whitespace
            content = re.sub(r'\s+', ' ', content)
            
            # Remove special characters but keep basic punctuation
            content = re.sub(r'[^\w\s.,!?;:\-()]', '', content)
            
            # Strip leading/trailing whitespace
            content = content.strip()
            
            return content
            
        except Exception as e:
            logger.error(f"Content cleaning failed: {e}")
            return content
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp from various formats.
        
        Args:
            timestamp_str: Timestamp string
            
        Returns:
            Parsed datetime
        """
        try:
            if not timestamp_str:
                return datetime.now()
            
            # Try to parse with feedparser
            import time
            parsed_time = time.strptime(timestamp_str, "%a, %d %b %Y %H:%M:%S %Z")
            return datetime(*parsed_time[:6])
            
        except Exception:
            try:
                # Try ISO format
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except Exception:
                # Return current time as fallback
                return datetime.now()
    
    def _generate_article_id(self, title: str, content: str, url: str) -> str:
        """
        Generate a unique article ID.
        
        Args:
            title: Article title
            content: Article content
            url: Article URL
            
        Returns:
            Unique article ID
        """
        try:
            # Create hash from title, content, and URL
            text = f"{title}|{content[:100]}|{url}"
            return hashlib.md5(text.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Article ID generation failed: {e}")
            return hashlib.md5(f"{title}{url}{datetime.now()}".encode()).hexdigest()
    
    def _determine_category(self, title: str, content: str, source_url: str) -> str:
        """
        Determine article category based on content.
        
        Args:
            title: Article title
            content: Article content
            source_url: Source URL
            
        Returns:
            Article category
        """
        try:
            text = f"{title} {content}".lower()
            
            # Define category keywords
            categories = {
                'earnings': ['earnings', 'quarterly', 'revenue', 'profit', 'q1', 'q2', 'q3', 'q4'],
                'fed': ['fed', 'federal reserve', 'interest rate', 'fomc', 'monetary policy'],
                'inflation': ['inflation', 'cpi', 'consumer price index', 'price level'],
                'gdp': ['gdp', 'gross domestic product', 'economic growth', 'recession'],
                'unemployment': ['unemployment', 'jobless', 'employment', 'jobs report'],
                'merger': ['merger', 'acquisition', 'takeover', 'buyout'],
                'ipo': ['ipo', 'initial public offering', 'going public'],
                'regulation': ['regulation', 'regulatory', 'sec', 'fda'],
                'geopolitical': ['war', 'conflict', 'sanctions', 'trade war'],
                'technology': ['tech', 'technology', 'ai', 'artificial intelligence', 'software'],
                'finance': ['bank', 'banking', 'financial', 'credit', 'loan'],
                'energy': ['oil', 'gas', 'energy', 'renewable', 'solar', 'wind'],
                'healthcare': ['health', 'medical', 'pharmaceutical', 'drug', 'vaccine']
            }
            
            # Count keyword matches for each category
            category_scores = {}
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    category_scores[category] = score
            
            # Return category with highest score
            if category_scores:
                return max(category_scores, key=category_scores.get)
            
            # Default category based on source
            if 'yahoo' in source_url.lower():
                return 'finance'
            elif 'marketwatch' in source_url.lower():
                return 'market'
            elif 'bloomberg' in source_url.lower():
                return 'business'
            elif 'reuters' in source_url.lower():
                return 'news'
            else:
                return 'general'
                
        except Exception as e:
            logger.error(f"Category determination failed: {e}")
            return 'general'
    
    def _extract_tags(self, title: str, content: str) -> List[str]:
        """
        Extract tags from article content.
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            List of tags
        """
        try:
            text = f"{title} {content}".lower()
            tags = []
            
            # Define tag keywords
            tag_keywords = {
                'breaking': ['breaking', 'urgent', 'alert'],
                'analysis': ['analysis', 'opinion', 'commentary'],
                'forecast': ['forecast', 'prediction', 'outlook'],
                'volatility': ['volatility', 'volatile', 'uncertainty'],
                'bullish': ['bullish', 'positive', 'optimistic', 'rise'],
                'bearish': ['bearish', 'negative', 'pessimistic', 'fall'],
                'high-impact': ['major', 'significant', 'important', 'critical'],
                'low-impact': ['minor', 'small', 'insignificant']
            }
            
            # Extract tags based on keywords
            for tag, keywords in tag_keywords.items():
                if any(keyword in text for keyword in keywords):
                    tags.append(tag)
            
            return tags
            
        except Exception as e:
            logger.error(f"Tag extraction failed: {e}")
            return []
    
    def _remove_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate articles.
        
        Args:
            articles: List of articles
            
        Returns:
            List of unique articles
        """
        try:
            seen_ids = set()
            unique_articles = []
            
            for article in articles:
                article_id = article.get('id', '')
                if article_id and article_id not in seen_ids:
                    seen_ids.add(article_id)
                    unique_articles.append(article)
            
            logger.debug(f"Removed {len(articles) - len(unique_articles)} duplicate articles")
            return unique_articles
            
        except Exception as e:
            logger.error(f"Duplicate removal failed: {e}")
            return articles
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get ingestion statistics.
        
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            return {
                'total_feeds': len(self.config['rss_feeds']),
                'seen_articles': len(self.seen_articles),
                'last_ingestion': self.last_ingestion,
                'config': {
                    'scrape_delay': self.config['scrape_delay'],
                    'max_articles_per_feed': self.config['max_articles_per_feed'],
                    'enable_web_scraping': self.config['enable_web_scraping'],
                    'duplicate_detection': self.config['duplicate_detection']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get ingestion stats: {e}")
            return {}
