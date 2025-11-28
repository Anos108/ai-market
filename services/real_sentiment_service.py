"""
Real Sentiment Analysis Service

This service provides real sentiment analysis using multiple news sources and APIs
to replace simulated sentiment data with actual market sentiment analysis.
"""

import asyncio
import logging
import requests
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import yfinance as yf
from textblob import TextBlob
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Represents a news article with sentiment analysis."""
    title: str
    description: str
    source: str
    published_at: datetime
    url: str
    sentiment_score: float
    sentiment_polarity: str  # 'positive', 'negative', 'neutral'
    relevance_score: float
    keywords: List[str]

@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis results."""
    symbol: str
    overall_sentiment: float  # -1 to 1
    sentiment_confidence: float  # 0 to 1
    news_volume: int
    positive_articles: int
    negative_articles: int
    neutral_articles: int
    sentiment_trend: str  # 'improving', 'deteriorating', 'stable'
    key_themes: List[str]
    recent_articles: List[NewsArticle]
    analysis_timestamp: datetime

class RealSentimentService:
    """Real sentiment analysis service using multiple data sources."""
    
    def __init__(self, news_api_key: str = None, alpha_vantage_key: str = None):
        self.news_api_key = news_api_key
        self.alpha_vantage_key = alpha_vantage_key
        self.session = None
        
        # News sources configuration
        self.news_sources = [
            'bloomberg', 'reuters', 'marketwatch', 'cnbc', 'wsj',
            'yahoo-finance', 'investing', 'benzinga', 'seeking-alpha'
        ]
        
        # Sentiment keywords and weights
        self.positive_keywords = {
            'beat': 0.8, 'exceed': 0.8, 'surge': 0.9, 'rally': 0.8, 'gain': 0.7,
            'strong': 0.6, 'growth': 0.6, 'profit': 0.7, 'revenue': 0.5,
            'upgrade': 0.8, 'bullish': 0.9, 'optimistic': 0.7, 'positive': 0.6,
            'breakthrough': 0.8, 'milestone': 0.6, 'record': 0.7, 'success': 0.8
        }
        
        self.negative_keywords = {
            'miss': 0.8, 'decline': 0.7, 'fall': 0.7, 'drop': 0.7, 'loss': 0.8,
            'weak': 0.6, 'concern': 0.7, 'risk': 0.6, 'downgrade': 0.8,
            'bearish': 0.9, 'pessimistic': 0.7, 'negative': 0.6, 'crisis': 0.9,
            'warning': 0.7, 'disappoint': 0.8, 'struggle': 0.6, 'challenge': 0.5
        }
        
        # Market-specific terms
        self.market_terms = {
            'earnings': 0.8, 'guidance': 0.7, 'outlook': 0.6, 'forecast': 0.6,
            'merger': 0.7, 'acquisition': 0.6, 'partnership': 0.5, 'deal': 0.6,
            'fda': 0.8, 'approval': 0.8, 'trial': 0.7, 'clinical': 0.6,
            'dividend': 0.5, 'buyback': 0.6, 'split': 0.5, 'ipo': 0.7
        }
    
    async def initialize(self):
        """Initialize async session."""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close async session."""
        if self.session:
            await self.session.close()
    
    async def analyze_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Perform comprehensive sentiment analysis for a symbol."""
        try:
            logger.info(f"🔍 Analyzing sentiment for {symbol}")
            
            # Get news articles from multiple sources
            articles = await self._fetch_news_articles(symbol)
            
            # Analyze sentiment for each article
            analyzed_articles = []
            for article in articles:
                analyzed_article = self._analyze_article_sentiment(article, symbol)
                analyzed_articles.append(analyzed_article)
            
            # Calculate overall sentiment metrics
            sentiment_metrics = self._calculate_sentiment_metrics(analyzed_articles, symbol)
            
            # Identify key themes
            key_themes = self._extract_key_themes(analyzed_articles)
            
            # Determine sentiment trend
            sentiment_trend = self._determine_sentiment_trend(analyzed_articles)
            
            return SentimentAnalysis(
                symbol=symbol,
                overall_sentiment=sentiment_metrics['overall_sentiment'],
                sentiment_confidence=sentiment_metrics['confidence'],
                news_volume=len(analyzed_articles),
                positive_articles=sentiment_metrics['positive_count'],
                negative_articles=sentiment_metrics['negative_count'],
                neutral_articles=sentiment_metrics['neutral_count'],
                sentiment_trend=sentiment_trend,
                key_themes=key_themes,
                recent_articles=analyzed_articles[:10],  # Top 10 most relevant
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return self._get_fallback_sentiment(symbol)
    
    async def _fetch_news_articles(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news articles from multiple sources."""
        articles = []
        
        # Try NewsAPI first
        if self.news_api_key:
            try:
                newsapi_articles = await self._fetch_from_newsapi(symbol)
                articles.extend(newsapi_articles)
            except Exception as e:
                logger.warning(f"NewsAPI failed for {symbol}: {e}")
        
        # Try Alpha Vantage News & Sentiment
        if self.alpha_vantage_key:
            try:
                av_articles = await self._fetch_from_alpha_vantage(symbol)
                articles.extend(av_articles)
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
        
        # Fallback to Yahoo Finance news
        try:
            yahoo_articles = await self._fetch_from_yahoo_finance(symbol)
            articles.extend(yahoo_articles)
        except Exception as e:
            logger.warning(f"Yahoo Finance news failed for {symbol}: {e}")
        
        # Remove duplicates and filter recent articles
        unique_articles = self._deduplicate_articles(articles)
        recent_articles = [a for a in unique_articles 
                          if (datetime.now() - a.get('published_at', datetime.now())).days <= 7]
        
        return recent_articles[:50]  # Limit to 50 most recent articles
    
    async def _fetch_from_newsapi(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI."""
        if not self.session:
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'"{symbol}" OR "{self._get_company_name(symbol)}"',
            'sources': ','.join(self.news_sources),
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 20,
            'apiKey': self.news_api_key
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                articles = []
                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'published_at': datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                        'url': article.get('url', ''),
                        'content': article.get('content', '')
                    })
                return articles
            else:
                logger.warning(f"NewsAPI returned status {response.status}")
                return []
    
    async def _fetch_from_alpha_vantage(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage News & Sentiment API."""
        if not self.session:
            return []
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'limit': 20,
            'apikey': self.alpha_vantage_key
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                articles = []
                for item in data.get('feed', []):
                    articles.append({
                        'title': item.get('title', ''),
                        'description': item.get('summary', ''),
                        'source': item.get('source', ''),
                        'published_at': datetime.fromtimestamp(int(item.get('time_published', 0))),
                        'url': item.get('url', ''),
                        'sentiment_score': float(item.get('overall_sentiment_score', 0)),
                        'sentiment_label': item.get('overall_sentiment_label', '')
                    })
                return articles
            else:
                logger.warning(f"Alpha Vantage returned status {response.status}")
                return []
    
    async def _fetch_from_yahoo_finance(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            articles = []
            for item in news_data[:20]:  # Limit to 20 articles
                articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'source': 'Yahoo Finance',
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'url': item.get('link', ''),
                    'publisher': item.get('publisher', '')
                })
            return articles
        except Exception as e:
            logger.warning(f"Yahoo Finance news fetch failed: {e}")
            return []
    
    def _analyze_article_sentiment(self, article: Dict[str, Any], symbol: str) -> NewsArticle:
        """Analyze sentiment of a single article."""
        title = article.get('title', '')
        description = article.get('description', '')
        content = title + ' ' + description
        
        # Use TextBlob for basic sentiment
        blob = TextBlob(content)
        textblob_sentiment = blob.sentiment.polarity
        
        # Enhanced sentiment analysis using keywords
        keyword_sentiment = self._calculate_keyword_sentiment(content, symbol)
        
        # Combine TextBlob and keyword analysis
        combined_sentiment = (textblob_sentiment * 0.4) + (keyword_sentiment * 0.6)
        
        # Determine polarity
        if combined_sentiment > 0.1:
            polarity = 'positive'
        elif combined_sentiment < -0.1:
            polarity = 'negative'
        else:
            polarity = 'neutral'
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(content, symbol)
        
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        return NewsArticle(
            title=title,
            description=description,
            source=article.get('source', 'Unknown'),
            published_at=article.get('published_at', datetime.now()),
            url=article.get('url', ''),
            sentiment_score=combined_sentiment,
            sentiment_polarity=polarity,
            relevance_score=relevance_score,
            keywords=keywords
        )
    
    def _calculate_keyword_sentiment(self, content: str, symbol: str) -> float:
        """Calculate sentiment based on financial keywords."""
        content_lower = content.lower()
        
        positive_score = 0
        negative_score = 0
        
        # Check positive keywords
        for keyword, weight in self.positive_keywords.items():
            if keyword in content_lower:
                positive_score += weight
        
        # Check negative keywords
        for keyword, weight in self.negative_keywords.items():
            if keyword in content_lower:
                negative_score += weight
        
        # Normalize scores
        total_positive = positive_score
        total_negative = negative_score
        
        if total_positive + total_negative == 0:
            return 0.0
        
        return (total_positive - total_negative) / (total_positive + total_negative)
    
    def _calculate_relevance_score(self, content: str, symbol: str) -> float:
        """Calculate how relevant the article is to the symbol."""
        content_lower = content.lower()
        symbol_lower = symbol.lower()
        
        relevance = 0.0
        
        # Direct symbol mention
        if symbol_lower in content_lower:
            relevance += 0.8
        
        # Company name mention
        company_name = self._get_company_name(symbol)
        if company_name and company_name.lower() in content_lower:
            relevance += 0.7
        
        # Market terms
        for term, weight in self.market_terms.items():
            if term in content_lower:
                relevance += weight * 0.1
        
        return min(1.0, relevance)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract relevant keywords from content."""
        keywords = []
        content_lower = content.lower()
        
        # Extract market terms
        for term in self.market_terms.keys():
            if term in content_lower:
                keywords.append(term)
        
        # Extract sentiment keywords
        for keyword in self.positive_keywords.keys():
            if keyword in content_lower:
                keywords.append(keyword)
        
        for keyword in self.negative_keywords.keys():
            if keyword in content_lower:
                keywords.append(keyword)
        
        return list(set(keywords))[:10]  # Limit to 10 unique keywords
    
    def _calculate_sentiment_metrics(self, articles: List[NewsArticle], symbol: str) -> Dict[str, Any]:
        """Calculate overall sentiment metrics."""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Weight articles by relevance
        weighted_sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            weight = article.relevance_score
            weighted_sentiment = article.sentiment_score * weight
            weighted_sentiments.append(weighted_sentiment)
            
            if article.sentiment_polarity == 'positive':
                positive_count += 1
            elif article.sentiment_polarity == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate weighted average sentiment
        total_weight = sum(article.relevance_score for article in articles)
        if total_weight > 0:
            overall_sentiment = sum(weighted_sentiments) / total_weight
        else:
            overall_sentiment = 0.0
        
        # Calculate confidence based on article volume and sentiment distribution
        total_articles = len(articles)
        sentiment_variance = np.var([a.sentiment_score for a in articles])
        confidence = min(1.0, (total_articles / 20) * (1 - sentiment_variance))
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': confidence,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }
    
    def _extract_key_themes(self, articles: List[NewsArticle]) -> List[str]:
        """Extract key themes from articles."""
        theme_counts = defaultdict(int)
        
        for article in articles:
            for keyword in article.keywords:
                theme_counts[keyword] += 1
        
        # Sort by frequency and return top themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5]]
    
    def _determine_sentiment_trend(self, articles: List[NewsArticle]) -> str:
        """Determine if sentiment is improving, deteriorating, or stable."""
        if len(articles) < 3:
            return 'stable'
        
        # Sort articles by publication time
        sorted_articles = sorted(articles, key=lambda x: x.published_at)
        
        # Calculate sentiment for recent vs older articles
        recent_count = max(1, len(sorted_articles) // 3)
        recent_articles = sorted_articles[-recent_count:]
        older_articles = sorted_articles[:-recent_count]
        
        recent_sentiment = np.mean([a.sentiment_score for a in recent_articles])
        older_sentiment = np.mean([a.sentiment_score for a in older_articles])
        
        diff = recent_sentiment - older_sentiment
        
        if diff > 0.2:
            return 'improving'
        elif diff < -0.2:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower()
            # Simple deduplication based on title
            if title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for a symbol."""
        company_names = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc',
            'TSLA': 'Tesla Inc',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms Inc',
            'AMZN': 'Amazon.com Inc',
            'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust',
            'BTC-USD': 'Bitcoin'
        }
        return company_names.get(symbol, symbol)
    
    def _get_fallback_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Get fallback sentiment when real analysis fails."""
        return SentimentAnalysis(
            symbol=symbol,
            overall_sentiment=0.0,
            sentiment_confidence=0.0,
            news_volume=0,
            positive_articles=0,
            negative_articles=0,
            neutral_articles=0,
            sentiment_trend='stable',
            key_themes=[],
            recent_articles=[],
            analysis_timestamp=datetime.now()
        )
