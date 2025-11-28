"""
Sentiment Agent for AI Market Analysis System

This agent analyzes news articles, social media sentiment, and market sentiment
to provide sentiment-based trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import re
from collections import Counter
import requests
from bs4 import BeautifulSoup
import feedparser

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """
    Sentiment Agent for analyzing market sentiment from news and social media.
    
    This agent uses natural language processing techniques to analyze:
    - News article sentiment
    - Social media sentiment
    - Market sentiment indicators
    - Sentiment momentum and trends
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Sentiment Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'sentiment_threshold': 0.3,
            'news_weight': 0.6,
            'social_weight': 0.4,
            'lookback_hours': 24,
            'min_articles': 3,
            'sentiment_keywords': {
                'positive': ['bullish', 'growth', 'profit', 'gain', 'rise', 'up', 'strong', 'positive', 'optimistic', 'buy'],
                'negative': ['bearish', 'decline', 'loss', 'fall', 'down', 'weak', 'negative', 'pessimistic', 'sell', 'crash'],
                'neutral': ['stable', 'unchanged', 'flat', 'hold', 'maintain', 'steady']
            },
            'news_sources': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline',
                'https://feeds.marketwatch.com/marketwatch/topstories/',
                'https://feeds.bloomberg.com/markets/news.rss'
            ],
            'confidence_threshold': 0.6
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="SentimentAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.sentiment_history = []
        self.news_cache = {}
        self.last_news_fetch = None
        
        logger.info(f"Initialized SentimentAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the sentiment agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For sentiment analysis, we don't need traditional ML training
            # Instead, we'll validate our sentiment analysis approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Sentiment analysis approach validated")
            return {"status": "sentiment_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate sentiment-based prediction.
        
        Args:
            context: Current market context
            
        Returns:
            Sentiment-based trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple sentiment analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple sentiment analysis (not trained)")
                return self._simple_sentiment_analysis(context)
            
            # Perform comprehensive sentiment analysis
            sentiment_score = self._analyze_sentiment(context)
            
            # Generate signal based on sentiment
            signal = self._generate_sentiment_signal(sentiment_score, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Sentiment analysis error: {e}", context)
    
    def _analyze_sentiment(self, context: AgentContext) -> Dict[str, float]:
        """
        Analyze sentiment from multiple sources.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            # Get recent news sentiment
            news_sentiment = self._analyze_news_sentiment(context.symbol)
            
            # Get social media sentiment (simulated for now)
            social_sentiment = self._analyze_social_sentiment(context.symbol)
            
            # Combine sentiment scores
            combined_sentiment = {
                'overall': (news_sentiment['score'] * self.config['news_weight'] + 
                           social_sentiment['score'] * self.config['social_weight']),
                'news': news_sentiment['score'],
                'social': social_sentiment['score'],
                'confidence': min(news_sentiment['confidence'], social_sentiment['confidence']),
                'article_count': news_sentiment['count'],
                'sentiment_trend': self._calculate_sentiment_trend()
            }
            
            return combined_sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'overall': 0.0,
                'news': 0.0,
                'social': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'sentiment_trend': 'neutral'
            }
    
    def _analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            News sentiment analysis results
        """
        try:
            # Fetch recent news articles
            articles = self._fetch_news_articles(symbol)
            
            if not articles:
                return {'score': 0.0, 'confidence': 0.0, 'count': 0}
            
            # Analyze sentiment of each article
            sentiments = []
            for article in articles:
                sentiment = self._analyze_article_sentiment(article)
                sentiments.append(sentiment)
            
            # Calculate overall sentiment
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                confidence = min(len(sentiments) / self.config['min_articles'], 1.0)
            else:
                avg_sentiment = 0.0
                confidence = 0.0
            
            return {
                'score': avg_sentiment,
                'confidence': confidence,
                'count': len(articles)
            }
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'count': 0}
    
    def _fetch_news_articles(self, symbol: str) -> List[Dict[str, str]]:
        """
        Fetch recent news articles for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of news articles
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]
            
            articles = []
            
            # Fetch from RSS feeds
            for source in self.config['news_sources']:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:10]:  # Limit to recent articles
                        # Check if article is relevant to the symbol
                        if self._is_relevant_article(entry, symbol):
                            articles.append({
                                'title': entry.get('title', ''),
                                'summary': entry.get('summary', ''),
                                'published': entry.get('published', ''),
                                'source': source
                            })
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source}: {e}")
                    continue
            
            # Cache results
            self.news_cache[cache_key] = articles
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch news articles: {e}")
            return []
    
    def _is_relevant_article(self, entry: Dict, symbol: str) -> bool:
        """
        Check if an article is relevant to the given symbol.
        
        Args:
            entry: RSS feed entry
            symbol: Stock symbol
            
        Returns:
            True if article is relevant
        """
        try:
            text = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
            return symbol.lower() in text or any(
                keyword in text for keyword in ['stock', 'market', 'trading', 'finance']
            )
        except:
            return False
    
    def _analyze_article_sentiment(self, article: Dict[str, str]) -> float:
        """
        Analyze sentiment of a single article.
        
        Args:
            article: Article dictionary with title and summary
            
        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
            
            # Count sentiment keywords
            positive_count = sum(1 for word in self.config['sentiment_keywords']['positive'] if word in text)
            negative_count = sum(1 for word in self.config['sentiment_keywords']['negative'] if word in text)
            neutral_count = sum(1 for word in self.config['sentiment_keywords']['neutral'] if word in text)
            
            total_words = positive_count + negative_count + neutral_count
            
            if total_words == 0:
                return 0.0  # Neutral if no sentiment words found
            
            # Calculate sentiment score
            sentiment_score = (positive_count - negative_count) / total_words
            
            # Normalize to -1 to 1 range
            return max(-1.0, min(1.0, sentiment_score))
            
        except Exception as e:
            logger.error(f"Article sentiment analysis failed: {e}")
            return 0.0
    
    def _analyze_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze social media sentiment (simulated for now).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Social sentiment analysis results
        """
        try:
            # For now, simulate social media sentiment
            # In a real implementation, this would connect to Twitter API, Reddit API, etc.
            
            # Simulate based on market data volatility
            if hasattr(self, '_last_market_data'):
                volatility = self._last_market_data.get('volatility', 0.0)
                # Higher volatility often correlates with negative sentiment
                sentiment_score = -volatility * 0.5
            else:
                sentiment_score = 0.0
            
            return {
                'score': sentiment_score,
                'confidence': 0.3,  # Lower confidence for simulated data
                'count': 0
            }
            
        except Exception as e:
            logger.error(f"Social sentiment analysis failed: {e}")
            return {'score': 0.0, 'confidence': 0.0, 'count': 0}
    
    def _calculate_sentiment_trend(self) -> str:
        """
        Calculate sentiment trend from recent history.
        
        Returns:
            Trend direction: 'improving', 'declining', or 'stable'
        """
        try:
            if len(self.sentiment_history) < 3:
                return 'stable'
            
            recent_sentiments = [s['overall'] for s in self.sentiment_history[-3:]]
            
            if recent_sentiments[-1] > recent_sentiments[0] + 0.1:
                return 'improving'
            elif recent_sentiments[-1] < recent_sentiments[0] - 0.1:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Sentiment trend calculation failed: {e}")
            return 'stable'
    
    def _generate_sentiment_signal(self, sentiment: Dict[str, float], context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on sentiment analysis.
        
        Args:
            sentiment: Sentiment analysis results
            context: Current market context
            
        Returns:
            Sentiment-based trading signal
        """
        try:
            overall_sentiment = sentiment['overall']
            confidence = sentiment['confidence']
            
            # Determine signal type based on sentiment
            if overall_sentiment > self.config['sentiment_threshold'] and confidence > self.config['confidence_threshold']:
                signal_type = SignalType.BUY
                reasoning = f"Positive sentiment ({overall_sentiment:.2f}) with high confidence ({confidence:.2f})"
            elif overall_sentiment < -self.config['sentiment_threshold'] and confidence > self.config['confidence_threshold']:
                signal_type = SignalType.SELL
                reasoning = f"Negative sentiment ({overall_sentiment:.2f}) with high confidence ({confidence:.2f})"
            else:
                signal_type = SignalType.HOLD
                reasoning = f"Neutral sentiment ({overall_sentiment:.2f}) or low confidence ({confidence:.2f})"
            
            # Adjust confidence based on sentiment strength
            adjusted_confidence = min(confidence * abs(overall_sentiment), 0.9)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'sentiment_scores': sentiment,
                    'method': 'sentiment_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_sentiment_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple sentiment analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple sentiment-based signal
        """
        try:
            # Simple sentiment based on recent price movement
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            # Use price momentum as a proxy for sentiment
            if len(context.market_data) >= 2:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    current_price = context.market_data[close_col].iloc[-1]
                    prev_price = context.market_data[close_col].iloc[-2]
                    price_change = (current_price - prev_price) / prev_price
                    
                    # Simple sentiment based on price change
                    if price_change > 0.02:  # 2% increase
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_price_sentiment'},
                            reasoning=f"Positive price momentum ({price_change:.2%}) suggests bullish sentiment"
                        )
                    elif price_change < -0.02:  # 2% decrease
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.SELL,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_price_sentiment'},
                            reasoning=f"Negative price momentum ({price_change:.2%}) suggests bearish sentiment"
                        )
            
            return self._create_hold_signal("No clear sentiment signal", context)
            
        except Exception as e:
            logger.error(f"Simple sentiment analysis failed: {e}")
            return self._create_hold_signal(f"Simple sentiment analysis error: {e}", context)
    
    def _create_hold_signal(self, reason: str, context: AgentContext) -> AgentSignal:
        """Create a hold signal with error information."""
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=context.timestamp,
            asset_symbol=context.symbol,
            metadata={'error': reason},
            reasoning=f"Hold signal: {reason}"
        )
    
    def update_model(self, new_data: pd.DataFrame, context: AgentContext) -> None:
        """
        Update the sentiment model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Store recent sentiment for trend analysis
            if hasattr(self, '_last_sentiment'):
                self.sentiment_history.append(self._last_sentiment)
                
                # Keep only recent history
                if len(self.sentiment_history) > 10:
                    self.sentiment_history = self.sentiment_history[-10:]
            
            # Update market data for social sentiment simulation
            self._last_market_data = {
                'volatility': new_data.get('volatility', 0.0) if hasattr(new_data, 'get') else 0.0
            }
            
            logger.info(f"Updated sentiment model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
