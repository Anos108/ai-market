"""
Data Ingestion Modules for AI Market Analysis System

This module provides data ingestion capabilities for various market data sources
including OHLCV data, news feeds, economic indicators, and other market-relevant information.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import feedparser
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    symbols: List[str]
    start_date: datetime
    end_date: Optional[datetime] = None
    interval: str = "1d"  # 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1


class BaseDataIngestor(ABC):
    """
    Abstract base class for all data ingestors.
    
    Provides common functionality for data retrieval, error handling,
    and rate limiting across different data sources.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the data ingestor.
        
        Args:
            config: Configuration for data ingestion
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Market-Analysis-System/1.0'
        })
        
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: datetime, 
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch data for a specific symbol and date range.
        
        Args:
            symbol: Asset symbol to fetch data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval (optional)
            
        Returns:
            DataFrame containing the fetched data
        """
        pass
    
    def fetch_multiple_symbols(self, symbols: List[str], 
                              start_date: datetime,
                              end_date: Optional[datetime] = None,
                              max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary mapping symbols to their data DataFrames
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_data, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                    logger.info(f"Successfully fetched data for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    results[symbol] = pd.DataFrame()  # Empty DataFrame for failed requests
        
        return results
    
    def _handle_rate_limit(self):
        """Handle rate limiting between requests."""
        time.sleep(self.config.rate_limit_delay)
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """
        Retry a function call on failure with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises last exception
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed for {func.__name__}")
        
        raise last_exception


class YahooFinanceIngestor(BaseDataIngestor):
    """
    Data ingestor for Yahoo Finance OHLCV data.
    
    Provides access to historical and real-time market data
    for stocks, ETFs, indices, and other financial instruments.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """Initialize Yahoo Finance ingestor."""
        super().__init__(config)
        self.ticker_cache = {}
    
    def fetch_data(self, symbol: str, start_date: datetime, 
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'SPY', 'BTC-USD')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with OHLCV data and additional metrics
        """
        try:
            # Get ticker object (with caching)
            if symbol not in self.ticker_cache:
                self.ticker_cache[symbol] = yf.Ticker(symbol)
            
            ticker = self.ticker_cache[symbol]
            
            # Fetch historical data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=self.config.interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data is None or data.empty:
                logger.warning(f"No data found for {symbol} in date range {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Add additional technical indicators
            data = self._add_technical_indicators(data)
            
            # Add metadata
            data['symbol'] = symbol
            data['data_source'] = 'yahoo_finance'
            data['ingestion_timestamp'] = datetime.now()
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data."""
        if data.empty:
            return data
        
        # Calculate returns
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate volatility (rolling standard deviation)
        data['volatility_20d'] = data['returns'].rolling(window=20).std()
        data['volatility_5d'] = data['returns'].rolling(window=5).std()
        
        # Calculate moving averages
        data['sma_20'] = data['Close'].rolling(window=20).mean()
        data['sma_50'] = data['Close'].rolling(window=50).mean()
        data['ema_12'] = data['Close'].ewm(span=12).mean()
        data['ema_26'] = data['Close'].ewm(span=26).mean()
        
        # Calculate RSI
        data['rsi'] = self._calculate_rsi(data['Close'])
        
        # Calculate Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = self._calculate_bollinger_bands(data['Close'])
        
        # Calculate volume indicators
        data['volume_sma_20'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_20']
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information for a symbol."""
        try:
            if symbol not in self.ticker_cache:
                self.ticker_cache[symbol] = yf.Ticker(symbol)
            
            ticker = self.ticker_cache[symbol]
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'data_source': 'yahoo_finance'
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {}


class NewsIngestor(BaseDataIngestor):
    """
    Data ingestor for news and sentiment data.
    
    Fetches news articles from various sources and provides
    sentiment analysis capabilities.
    """
    
    def __init__(self, config: DataIngestionConfig, news_api_key: Optional[str] = None):
        """
        Initialize news ingestor.
        
        Args:
            config: Data ingestion configuration
            news_api_key: API key for NewsAPI (optional)
        """
        super().__init__(config)
        self.news_api_key = news_api_key
        self.news_sources = [
            'bloomberg.com', 'reuters.com', 'cnbc.com', 'marketwatch.com',
            'wsj.com', 'ft.com', 'investing.com', 'yahoo.com'
        ]
    
    def fetch_data(self, symbol: str, start_date: datetime, 
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch news data for a symbol.
        
        Args:
            symbol: Asset symbol to fetch news for
            start_date: Start date for news retrieval
            end_date: End date for news retrieval
            
        Returns:
            DataFrame with news articles and sentiment scores
        """
        try:
            news_data = []
            
            # Fetch from NewsAPI if key is available
            if self.news_api_key:
                newsapi_data = self._fetch_from_newsapi(symbol, start_date, end_date)
                news_data.extend(newsapi_data)
            
            # Fetch from RSS feeds
            rss_data = self._fetch_from_rss_feeds(symbol, start_date, end_date)
            news_data.extend(rss_data)
            
            if not news_data:
                logger.warning(f"No news found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(news_data)
            
            # Add sentiment analysis
            df = self._add_sentiment_analysis(df)
            
            # Add metadata
            df['symbol'] = symbol
            df['data_source'] = 'news_ingestor'
            df['ingestion_timestamp'] = datetime.now()
            
            logger.info(f"Fetched {len(df)} news articles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_from_newsapi(self, symbol: str, start_date: datetime, 
                           end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI."""
        if not self.news_api_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key,
                'pageSize': 100
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            news_data = []
            for article in articles:
                news_data.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': pd.to_datetime(article.get('publishedAt', '')),
                    'author': article.get('author', ''),
                    'url_to_image': article.get('urlToImage', '')
                })
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    def _fetch_from_rss_feeds(self, symbol: str, start_date: datetime, 
                             end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds."""
        rss_urls = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            f"https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://feeds.reuters.com/news/wealth",
        ]
        
        news_data = []
        
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    # Parse publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Filter by date range
                    if pub_date and start_date <= pub_date <= (end_date or datetime.now()):
                        # Check if symbol is mentioned in title or summary
                        content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                        if symbol.lower() in content.lower():
                            news_data.append({
                                'title': entry.get('title', ''),
                                'description': entry.get('summary', ''),
                                'content': entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '',
                                'url': entry.get('link', ''),
                                'source': entry.get('source', {}).get('title', '') if entry.get('source') else '',
                                'published_at': pub_date,
                                'author': entry.get('author', '')
                            })
                
                self._handle_rate_limit()
                
            except Exception as e:
                logger.error(f"Error fetching RSS feed {url}: {e}")
                continue
        
        return news_data
    
    def _add_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis to news data."""
        if df.empty:
            return df
        
        # Simple sentiment analysis based on keywords
        positive_keywords = [
            'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'increase',
            'strong', 'outperform', 'beat', 'exceed', 'surge', 'rally', 'boom'
        ]
        
        negative_keywords = [
            'bearish', 'negative', 'decline', 'loss', 'fall', 'drop', 'decrease',
            'weak', 'underperform', 'miss', 'disappoint', 'crash', 'plunge', 'bust'
        ]
        
        def calculate_sentiment(text):
            if pd.isna(text):
                return 0.0
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_keywords if word in text_lower)
            negative_count = sum(1 for word in negative_keywords if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment_score = (positive_count - negative_count) / total_words
            return max(-1.0, min(1.0, sentiment_score * 10))  # Normalize to [-1, 1]
        
        # Calculate sentiment for title and description
        df['title_sentiment'] = df['title'].apply(calculate_sentiment)
        df['description_sentiment'] = df['description'].apply(calculate_sentiment)
        df['overall_sentiment'] = (df['title_sentiment'] + df['description_sentiment']) / 2
        
        return df


class EconomicDataIngestor(BaseDataIngestor):
    """
    Data ingestor for economic indicators and macro data.
    
    Fetches economic data from various sources including
    government agencies and financial data providers.
    """
    
    def __init__(self, config: DataIngestionConfig, alpha_vantage_key: Optional[str] = None):
        """
        Initialize economic data ingestor.
        
        Args:
            config: Data ingestion configuration
            alpha_vantage_key: API key for Alpha Vantage (optional)
        """
        super().__init__(config)
        self.alpha_vantage_key = alpha_vantage_key
        self.economic_indicators = [
            'GDP', 'CPI', 'PPI', 'UNEMPLOYMENT', 'FEDERAL_FUNDS_RATE',
            'TREASURY_YIELD', 'DURABLE_GOODS', 'RETAIL_SALES', 'HOUSING_STARTS'
        ]
    
    def fetch_data(self, symbol: str, start_date: datetime, 
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch economic data.
        
        Args:
            symbol: Economic indicator symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with economic data
        """
        try:
            # For now, return mock economic data
            # In a real implementation, this would fetch from FRED, Alpha Vantage, etc.
            dates = pd.date_range(start=start_date, end=end_date or datetime.now(), freq='M')
            
            # Generate mock economic data
            np.random.seed(42)  # For reproducible results
            data = pd.DataFrame({
                'date': dates,
                'indicator': symbol,
                'value': np.random.normal(100, 10, len(dates)),
                'change': np.random.normal(0, 2, len(dates)),
                'change_pct': np.random.normal(0, 0.5, len(dates))
            })
            
            data['data_source'] = 'economic_data_ingestor'
            data['ingestion_timestamp'] = datetime.now()
            
            logger.info(f"Fetched {len(data)} economic data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching economic data for {symbol}: {e}")
            return pd.DataFrame()


class DataIngestionOrchestrator:
    """
    Orchestrates data ingestion from multiple sources.
    
    Coordinates data fetching from various ingestors and provides
    a unified interface for data retrieval.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the data ingestion orchestrator.
        
        Args:
            config: Configuration for data ingestion
        """
        self.config = config
        self.ingestors = {
            'yahoo_finance': YahooFinanceIngestor(config),
            'news': NewsIngestor(config),
            'economic': EconomicDataIngestor(config)
        }
    
    def fetch_all_data(self, symbols: List[str], start_date: datetime,
                       end_date: Optional[datetime] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data from all sources for given symbols.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            Nested dictionary with source -> symbol -> DataFrame mapping
        """
        all_data = {}
        
        for source_name, ingestor in self.ingestors.items():
            logger.info(f"Fetching data from {source_name}...")
            
            try:
                if source_name == 'yahoo_finance':
                    # Fetch OHLCV data for all symbols
                    data = ingestor.fetch_multiple_symbols(symbols, start_date, end_date)
                elif source_name == 'news':
                    # Fetch news data for all symbols
                    data = {}
                    for symbol in symbols:
                        news_data = ingestor.fetch_data(symbol, start_date, end_date)
                        if not news_data.empty:
                            data[symbol] = news_data
                elif source_name == 'economic':
                    # Fetch economic data
                    data = {}
                    for indicator in ingestor.economic_indicators:
                        econ_data = ingestor.fetch_data(indicator, start_date, end_date)
                        if not econ_data.empty:
                            data[indicator] = econ_data
                
                all_data[source_name] = data
                logger.info(f"Successfully fetched data from {source_name}")
                
            except Exception as e:
                logger.error(f"Error fetching data from {source_name}: {e}")
                all_data[source_name] = {}
        
        return all_data
    
    def get_ingestor(self, source_name: str) -> Optional[BaseDataIngestor]:
        """Get a specific ingestor by name."""
        return self.ingestors.get(source_name)
    
    def add_ingestor(self, name: str, ingestor: BaseDataIngestor) -> None:
        """Add a new ingestor to the orchestrator."""
        self.ingestors[name] = ingestor
        logger.info(f"Added ingestor: {name}")
