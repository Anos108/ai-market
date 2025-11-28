"""
Alternative Data Sources for AI Market Analysis System

This module provides alternative data sources including social sentiment,
satellite imagery, credit card transactions, and supply chain data.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class AlternativeDataConfig:
    """Configuration for alternative data sources."""
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit_delay: float = 0.1
    max_retries: int = 3
    timeout: int = 30


class BaseAlternativeDataSource(ABC):
    """Base class for alternative data sources."""
    
    def __init__(self, config: AlternativeDataConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Market-Analysis-System/2.0'
        })
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling and retries."""
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        return {}


class SocialSentimentDataSource(BaseAlternativeDataSource):
    """Social media sentiment data source."""
    
    def __init__(self, config: AlternativeDataConfig):
        super().__init__(config)
        self.base_url = "https://api.twitter.com/2"
        if not config.api_key:
            self.api_key = os.getenv('TWITTER_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_sentiment_data(self, symbol: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get social sentiment data for a symbol."""
        if not self.api_key:
            logger.warning("Twitter API key not provided")
            return self._generate_mock_sentiment_data(symbol, start_date, end_date)
        
        try:
            # This is a simplified implementation
            # In practice, you would use Twitter API v2 for sentiment analysis
            return self._generate_mock_sentiment_data(symbol, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data for {symbol}: {e}")
            return self._generate_mock_sentiment_data(symbol, start_date, end_date)
    
    def _generate_mock_sentiment_data(self, symbol: str, start_date: datetime, 
                                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock sentiment data for testing."""
        if end_date is None:
            end_date = datetime.now()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic sentiment scores
        np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol
        sentiment_scores = np.random.normal(0.1, 0.3, len(date_range))
        sentiment_scores = np.clip(sentiment_scores, -1, 1)
        
        # Add some trend based on symbol
        if 'BTC' in symbol.upper():
            trend = np.linspace(0.2, -0.1, len(date_range))
        elif 'TSLA' in symbol.upper():
            trend = np.linspace(-0.1, 0.3, len(date_range))
        else:
            trend = np.linspace(0, 0, len(date_range))
        
        sentiment_scores += trend
        
        df_data = []
        for i, date in enumerate(date_range):
            df_data.append({
                'Date': date,
                'Sentiment_Score': sentiment_scores[i],
                'Positive_Tweets': max(0, int(1000 * (1 + sentiment_scores[i]) / 2)),
                'Negative_Tweets': max(0, int(1000 * (1 - sentiment_scores[i]) / 2)),
                'Total_Tweets': 1000,
                'Engagement_Rate': np.random.uniform(0.02, 0.08),
                'Influence_Score': np.random.uniform(0.3, 0.9)
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        return df


class NewsSentimentDataSource(BaseAlternativeDataSource):
    """News sentiment data source."""
    
    def __init__(self, config: AlternativeDataConfig):
        super().__init__(config)
        self.base_url = "https://newsapi.org/v2"
        if not config.api_key:
            self.api_key = os.getenv('NEWS_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_news_sentiment(self, symbol: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get news sentiment data for a symbol."""
        if not self.api_key:
            logger.warning("News API key not provided")
            return self._generate_mock_news_sentiment(symbol, start_date, end_date)
        
        try:
            # This is a simplified implementation
            # In practice, you would use News API and sentiment analysis
            return self._generate_mock_news_sentiment(symbol, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            return self._generate_mock_news_sentiment(symbol, start_date, end_date)
    
    def _generate_mock_news_sentiment(self, symbol: str, start_date: datetime, 
                                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock news sentiment data."""
        if end_date is None:
            end_date = datetime.now()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic news sentiment
        np.random.seed(hash(symbol + 'news') % 2**32)
        sentiment_scores = np.random.normal(0, 0.4, len(date_range))
        sentiment_scores = np.clip(sentiment_scores, -1, 1)
        
        df_data = []
        for i, date in enumerate(date_range):
            df_data.append({
                'Date': date,
                'News_Sentiment': sentiment_scores[i],
                'Article_Count': np.random.randint(5, 50),
                'Headline_Sentiment': sentiment_scores[i] + np.random.normal(0, 0.2),
                'Source_Diversity': np.random.uniform(0.6, 0.95),
                'Impact_Score': np.random.uniform(0.1, 0.9)
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        return df


class SatelliteDataSource(BaseAlternativeDataSource):
    """Satellite imagery data source (mock implementation)."""
    
    def __init__(self, config: AlternativeDataConfig):
        super().__init__(config)
        self.base_url = "https://api.planet.com/v1"
        if not config.api_key:
            self.api_key = os.getenv('PLANET_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_satellite_data(self, location: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get satellite imagery data for a location."""
        if not self.api_key:
            logger.warning("Planet API key not provided")
            return self._generate_mock_satellite_data(location, start_date, end_date)
        
        try:
            # This is a mock implementation
            # In practice, you would use Planet Labs API for satellite imagery
            return self._generate_mock_satellite_data(location, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching satellite data for {location}: {e}")
            return self._generate_mock_satellite_data(location, start_date, end_date)
    
    def _generate_mock_satellite_data(self, location: str, start_date: datetime, 
                                    end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock satellite data."""
        if end_date is None:
            end_date = datetime.now()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic satellite metrics
        np.random.seed(hash(location) % 2**32)
        
        df_data = []
        for i, date in enumerate(date_range):
            # Simulate different metrics based on location
            if 'oil' in location.lower():
                activity_score = np.random.uniform(0.3, 0.8)
                infrastructure_count = np.random.randint(50, 200)
            elif 'retail' in location.lower():
                activity_score = np.random.uniform(0.4, 0.9)
                infrastructure_count = np.random.randint(100, 500)
            else:
                activity_score = np.random.uniform(0.2, 0.7)
                infrastructure_count = np.random.randint(20, 150)
            
            df_data.append({
                'Date': date,
                'Activity_Score': activity_score,
                'Infrastructure_Count': infrastructure_count,
                'Traffic_Intensity': np.random.uniform(0.1, 0.9),
                'Development_Index': np.random.uniform(0.2, 0.8),
                'Environmental_Score': np.random.uniform(0.3, 0.9)
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        return df


class CreditCardDataSource(BaseAlternativeDataSource):
    """Credit card transaction data source (mock implementation)."""
    
    def __init__(self, config: AlternativeDataConfig):
        super().__init__(config)
        self.base_url = "https://api.spenddata.com/v1"
        if not config.api_key:
            self.api_key = os.getenv('SPEND_DATA_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_spending_data(self, merchant_category: str, start_date: datetime, 
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get credit card spending data for a merchant category."""
        if not self.api_key:
            logger.warning("Spend data API key not provided")
            return self._generate_mock_spending_data(merchant_category, start_date, end_date)
        
        try:
            # This is a mock implementation
            # In practice, you would use real spending data APIs
            return self._generate_mock_spending_data(merchant_category, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching spending data for {merchant_category}: {e}")
            return self._generate_mock_spending_data(merchant_category, start_date, end_date)
    
    def _generate_mock_spending_data(self, merchant_category: str, start_date: datetime, 
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock spending data."""
        if end_date is None:
            end_date = datetime.now()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic spending patterns
        np.random.seed(hash(merchant_category) % 2**32)
        
        # Base spending varies by category
        category_multipliers = {
            'retail': 1.0,
            'restaurants': 0.8,
            'gas': 0.6,
            'travel': 0.4,
            'entertainment': 0.7
        }
        
        base_multiplier = category_multipliers.get(merchant_category.lower(), 0.5)
        
        df_data = []
        for i, date in enumerate(date_range):
            # Add seasonal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add weekend effects
            weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
            
            base_spending = 1000 * base_multiplier * seasonal_factor * weekend_factor
            spending = max(0, np.random.normal(base_spending, base_spending * 0.3))
            
            df_data.append({
                'Date': date,
                'Total_Spending': spending,
                'Transaction_Count': int(spending / 50),  # Average $50 per transaction
                'Average_Transaction': spending / max(1, int(spending / 50)),
                'Weekend_Spending': spending * (0.3 if date.weekday() >= 5 else 0.1),
                'Online_Spending': spending * np.random.uniform(0.2, 0.6)
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        return df


class SupplyChainDataSource(BaseAlternativeDataSource):
    """Supply chain data source (mock implementation)."""
    
    def __init__(self, config: AlternativeDataConfig):
        super().__init__(config)
        self.base_url = "https://api.supplychain.com/v1"
        if not config.api_key:
            self.api_key = os.getenv('SUPPLY_CHAIN_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_supply_chain_data(self, company: str, start_date: datetime, 
                             end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get supply chain data for a company."""
        if not self.api_key:
            logger.warning("Supply chain API key not provided")
            return self._generate_mock_supply_chain_data(company, start_date, end_date)
        
        try:
            # This is a mock implementation
            # In practice, you would use real supply chain data APIs
            return self._generate_mock_supply_chain_data(company, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching supply chain data for {company}: {e}")
            return self._generate_mock_supply_chain_data(company, start_date, end_date)
    
    def _generate_mock_supply_chain_data(self, company: str, start_date: datetime, 
                                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate mock supply chain data."""
        if end_date is None:
            end_date = datetime.now()
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic supply chain metrics
        np.random.seed(hash(company) % 2**32)
        
        df_data = []
        for i, date in enumerate(date_range):
            # Simulate supply chain disruptions
            disruption_factor = 1.0
            if np.random.random() < 0.05:  # 5% chance of disruption
                disruption_factor = np.random.uniform(0.3, 0.8)
            
            df_data.append({
                'Date': date,
                'Supplier_Count': np.random.randint(50, 200),
                'Delivery_Time_Days': np.random.uniform(3, 15) * disruption_factor,
                'Inventory_Level': np.random.uniform(0.6, 1.2) * disruption_factor,
                'Quality_Score': np.random.uniform(0.7, 0.95),
                'Cost_Index': np.random.uniform(0.8, 1.3) / disruption_factor,
                'Risk_Score': np.random.uniform(0.1, 0.6) + (1 - disruption_factor) * 0.4
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        return df


class AlternativeDataManager:
    """Manager for alternative data sources."""
    
    def __init__(self):
        self.sources = {}
        
    def add_source(self, name: str, source: BaseAlternativeDataSource):
        """Add an alternative data source."""
        self.sources[name] = source
        logger.info(f"Added alternative data source: {name}")
    
    def get_sentiment_data(self, symbol: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Get sentiment data from all available sources."""
        sentiment_data = {}
        
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_sentiment_data'):
                try:
                    data = source.get_sentiment_data(symbol, start_date, end_date)
                    if not data.empty:
                        sentiment_data[source_name] = data
                except Exception as e:
                    logger.error(f"Error getting sentiment data from {source_name}: {e}")
        
        return sentiment_data
    
    def get_news_sentiment(self, symbol: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Get news sentiment data from all available sources."""
        news_data = {}
        
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_news_sentiment'):
                try:
                    data = source.get_news_sentiment(symbol, start_date, end_date)
                    if not data.empty:
                        news_data[source_name] = data
                except Exception as e:
                    logger.error(f"Error getting news sentiment from {source_name}: {e}")
        
        return news_data
    
    def get_satellite_data(self, location: str, start_date: datetime, 
                          end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Get satellite data from all available sources."""
        satellite_data = {}
        
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_satellite_data'):
                try:
                    data = source.get_satellite_data(location, start_date, end_date)
                    if not data.empty:
                        satellite_data[source_name] = data
                except Exception as e:
                    logger.error(f"Error getting satellite data from {source_name}: {e}")
        
        return satellite_data
    
    def get_spending_data(self, merchant_category: str, start_date: datetime, 
                         end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Get spending data from all available sources."""
        spending_data = {}
        
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_spending_data'):
                try:
                    data = source.get_spending_data(merchant_category, start_date, end_date)
                    if not data.empty:
                        spending_data[source_name] = data
                except Exception as e:
                    logger.error(f"Error getting spending data from {source_name}: {e}")
        
        return spending_data
    
    def get_supply_chain_data(self, company: str, start_date: datetime, 
                             end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Get supply chain data from all available sources."""
        supply_chain_data = {}
        
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_supply_chain_data'):
                try:
                    data = source.get_supply_chain_data(company, start_date, end_date)
                    if not data.empty:
                        supply_chain_data[source_name] = data
                except Exception as e:
                    logger.error(f"Error getting supply chain data from {source_name}: {e}")
        
        return supply_chain_data


def create_alternative_data_manager() -> AlternativeDataManager:
    """Create and configure alternative data manager with all available sources."""
    manager = AlternativeDataManager()
    
    # Add social sentiment source
    social_config = AlternativeDataConfig(
        api_key=os.getenv('TWITTER_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('social_sentiment', SocialSentimentDataSource(social_config))
    
    # Add news sentiment source
    news_config = AlternativeDataConfig(
        api_key=os.getenv('NEWS_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('news_sentiment', NewsSentimentDataSource(news_config))
    
    # Add satellite data source
    satellite_config = AlternativeDataConfig(
        api_key=os.getenv('PLANET_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('satellite', SatelliteDataSource(satellite_config))
    
    # Add credit card data source
    spending_config = AlternativeDataConfig(
        api_key=os.getenv('SPEND_DATA_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('spending', CreditCardDataSource(spending_config))
    
    # Add supply chain data source
    supply_config = AlternativeDataConfig(
        api_key=os.getenv('SUPPLY_CHAIN_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('supply_chain', SupplyChainDataSource(supply_config))
    
    logger.info("Alternative data manager created with all available sources")
    return manager
