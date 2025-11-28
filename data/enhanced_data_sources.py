"""
Enhanced Data Sources for AI Market Analysis System

This module provides additional data sources including Alpha Vantage, IEX Cloud,
Polygon, FRED, Quandl, and crypto exchanges for comprehensive market data coverage.
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
import time
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for enhanced data sources."""
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit_delay: float = 0.1
    max_retries: int = 3
    timeout: int = 30
    enable_caching: bool = True
    cache_duration: int = 300  # 5 minutes


class BaseEnhancedDataSource(ABC):
    """Base class for enhanced data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Market-Analysis-System/2.0'
        })
        self.cache = {}
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
    
    @abstractmethod
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get price data for a symbol."""
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass


class AlphaVantageDataSource(BaseEnhancedDataSource):
    """Alpha Vantage data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = "https://www.alphavantage.co/query"
        if not config.api_key:
            self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical price data from Alpha Vantage."""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
            return pd.DataFrame()
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            data = self._make_request(self.base_url, params)
            
            if 'Time Series (Daily)' not in data:
                logger.error(f"No time series data for {symbol}: {data}")
                return pd.DataFrame()
            
            time_series = data['Time Series (Daily)']
            df_data = []
            
            for date_str, values in time_series.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')
                if start_date <= date <= (end_date or datetime.now()):
                    df_data.append({
                        'Date': date,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': int(values['5. volume'])
                    })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Alpha Vantage."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            data = self._make_request(self.base_url, params)
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return float(quote['05. price'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0


class IEXCloudDataSource(BaseEnhancedDataSource):
    """IEX Cloud data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = "https://cloud.iexapis.com/stable"
        if not config.api_key:
            self.api_key = os.getenv('IEX_CLOUD_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical price data from IEX Cloud."""
        if not self.api_key:
            logger.warning("IEX Cloud API key not provided")
            return pd.DataFrame()
        
        try:
            # Calculate date range
            days_diff = (end_date or datetime.now()) - start_date
            if days_diff.days > 5 * 365:  # 5 years max
                range_param = "5y"
            elif days_diff.days > 2 * 365:  # 2 years
                range_param = "2y"
            elif days_diff.days > 365:  # 1 year
                range_param = "1y"
            elif days_diff.days > 90:  # 3 months
                range_param = "3m"
            else:
                range_param = "1m"
            
            url = f"{self.base_url}/stock/{symbol}/chart/{range_param}"
            params = {'token': self.api_key}
            
            data = self._make_request(url, params)
            
            if not data:
                return pd.DataFrame()
            
            df_data = []
            for item in data:
                date = datetime.strptime(item['date'], '%Y-%m-%d')
                if start_date <= date <= (end_date or datetime.now()):
                    df_data.append({
                        'Date': date,
                        'Open': item.get('open', 0),
                        'High': item.get('high', 0),
                        'Low': item.get('low', 0),
                        'Close': item.get('close', 0),
                        'Volume': item.get('volume', 0)
                    })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching IEX Cloud data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from IEX Cloud."""
        try:
            url = f"{self.base_url}/stock/{symbol}/quote"
            params = {'token': self.api_key}
            
            data = self._make_request(url, params)
            
            if data and 'latestPrice' in data:
                return float(data['latestPrice'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0


class PolygonDataSource(BaseEnhancedDataSource):
    """Polygon.io data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = "https://api.polygon.io"
        if not config.api_key:
            self.api_key = os.getenv('POLYGON_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical price data from Polygon."""
        if not self.api_key:
            logger.warning("Polygon API key not provided")
            return pd.DataFrame()
        
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date or datetime.now()).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_str}/{end_str}"
            params = {'apikey': self.api_key}
            
            data = self._make_request(url, params)
            
            if 'results' not in data:
                return pd.DataFrame()
            
            df_data = []
            for item in data['results']:
                date = datetime.fromtimestamp(item['t'] / 1000)
                df_data.append({
                    'Date': date,
                    'Open': item['o'],
                    'High': item['h'],
                    'Low': item['l'],
                    'Close': item['c'],
                    'Volume': item['v']
                })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Polygon data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Polygon."""
        try:
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            params = {'apikey': self.api_key}
            
            data = self._make_request(url, params)
            
            if 'results' in data and data['results']:
                return float(data['results']['p'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0


class FREDDataSource(BaseEnhancedDataSource):
    """Federal Reserve Economic Data (FRED) source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = "https://api.stlouisfed.org/fred"
        if not config.api_key:
            self.api_key = os.getenv('FRED_API_KEY')
        else:
            self.api_key = config.api_key
    
    def get_economic_data(self, series_id: str, start_date: datetime, 
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get economic data from FRED."""
        if not self.api_key:
            logger.warning("FRED API key not provided")
            return pd.DataFrame()
        
        try:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date or datetime.now()).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'observation_start': start_str,
                'observation_end': end_str
            }
            
            data = self._make_request(url, params)
            
            if 'observations' not in data:
                return pd.DataFrame()
            
            df_data = []
            for obs in data['observations']:
                if obs['value'] != '.':
                    df_data.append({
                        'Date': datetime.strptime(obs['date'], '%Y-%m-%d'),
                        'Value': float(obs['value'])
                    })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return pd.DataFrame()
    
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """FRED doesn't provide stock price data."""
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """FRED doesn't provide current stock prices."""
        return 0.0


class BinanceDataSource(BaseEnhancedDataSource):
    """Binance crypto data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = "https://api.binance.com/api/v3"
    
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical crypto price data from Binance."""
        try:
            # Convert symbol format (e.g., BTC-USD -> BTCUSDT)
            if symbol.endswith('-USD'):
                symbol = symbol.replace('-USD', 'USDT')
            
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int((end_date or datetime.now()).timestamp() * 1000)
            
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': '1d',
                'startTime': start_ts,
                'endTime': end_ts,
                'limit': 1000
            }
            
            data = self._make_request(url, params)
            
            if not data:
                return pd.DataFrame()
            
            df_data = []
            for kline in data:
                df_data.append({
                    'Date': datetime.fromtimestamp(kline[0] / 1000),
                    'Open': float(kline[1]),
                    'High': float(kline[2]),
                    'Low': float(kline[3]),
                    'Close': float(kline[4]),
                    'Volume': float(kline[5])
                })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Binance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current crypto price from Binance."""
        try:
            # Convert symbol format
            if symbol.endswith('-USD'):
                symbol = symbol.replace('-USD', 'USDT')
            
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': symbol}
            
            data = self._make_request(url, params)
            
            if data and 'price' in data:
                return float(data['price'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0


class CoinbaseDataSource(BaseEnhancedDataSource):
    """Coinbase crypto data source implementation."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = "https://api.exchange.coinbase.com"
    
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get historical crypto price data from Coinbase."""
        try:
            # Convert symbol format (e.g., BTC-USD -> BTC-USD)
            if not symbol.endswith('-USD'):
                symbol = f"{symbol}-USD"
            
            start_iso = start_date.isoformat()
            end_iso = (end_date or datetime.now()).isoformat()
            
            url = f"{self.base_url}/products/{symbol}/candles"
            params = {
                'start': start_iso,
                'end': end_iso,
                'granularity': 86400  # 1 day
            }
            
            data = self._make_request(url, params)
            
            if not data:
                return pd.DataFrame()
            
            df_data = []
            for candle in data:
                df_data.append({
                    'Date': datetime.fromtimestamp(candle[0]),
                    'Low': candle[1],
                    'High': candle[2],
                    'Open': candle[3],
                    'Close': candle[4],
                    'Volume': candle[5]
                })
            
            df = pd.DataFrame(df_data)
            if not df.empty:
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Coinbase data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current crypto price from Coinbase."""
        try:
            # Convert symbol format
            if not symbol.endswith('-USD'):
                symbol = f"{symbol}-USD"
            
            url = f"{self.base_url}/products/{symbol}/ticker"
            
            data = self._make_request(url)
            
            if data and 'price' in data:
                return float(data['price'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0


class EnhancedDataManager:
    """Manager for enhanced data sources with fallback and quality validation."""
    
    def __init__(self):
        self.sources = {}
        self.data_quality_validator = DataQualityValidator()
        
    def add_source(self, name: str, source: BaseEnhancedDataSource):
        """Add a data source."""
        self.sources[name] = source
        logger.info(f"Added data source: {name}")
    
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: Optional[datetime] = None, 
                      preferred_source: str = None) -> pd.DataFrame:
        """Get price data with fallback to multiple sources."""
        sources_to_try = []
        
        if preferred_source and preferred_source in self.sources:
            sources_to_try.append(preferred_source)
        
        # Add other sources as fallbacks
        for source_name in self.sources:
            if source_name != preferred_source:
                sources_to_try.append(source_name)
        
        for source_name in sources_to_try:
            try:
                source = self.sources[source_name]
                df = source.get_price_data(symbol, start_date, end_date)
                
                if not df.empty and self.data_quality_validator.validate_price_data(df):
                    logger.info(f"Successfully fetched data for {symbol} from {source_name}")
                    return df
                else:
                    logger.warning(f"Data quality validation failed for {symbol} from {source_name}")
                    
            except Exception as e:
                logger.error(f"Error fetching data from {source_name}: {e}")
                continue
        
        logger.error(f"Failed to fetch data for {symbol} from all sources")
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str, preferred_source: str = None) -> float:
        """Get current price with fallback to multiple sources."""
        sources_to_try = []
        
        if preferred_source and preferred_source in self.sources:
            sources_to_try.append(preferred_source)
        
        # Add other sources as fallbacks
        for source_name in self.sources:
            if source_name != preferred_source:
                sources_to_try.append(source_name)
        
        for source_name in sources_to_try:
            try:
                source = self.sources[source_name]
                price = source.get_current_price(symbol)
                
                if price > 0 and self.data_quality_validator.validate_price(price):
                    logger.info(f"Successfully fetched current price for {symbol} from {source_name}: ${price}")
                    return price
                else:
                    logger.warning(f"Price validation failed for {symbol} from {source_name}: ${price}")
                    
            except Exception as e:
                logger.error(f"Error fetching current price from {source_name}: {e}")
                continue
        
        logger.error(f"Failed to fetch current price for {symbol} from all sources")
        return 0.0


class DataQualityValidator:
    """Data quality validation for market data."""
    
    def validate_price_data(self, df: pd.DataFrame) -> bool:
        """Validate price data quality."""
        if df.empty:
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for reasonable price values
        if (df['Close'] <= 0).any():
            return False
        
        # Check for reasonable OHLC relationships
        if not ((df['Low'] <= df['Open']) & (df['Open'] <= df['High'])).all():
            return False
        if not ((df['Low'] <= df['Close']) & (df['Close'] <= df['High'])).all():
            return False
        
        # Check for reasonable volume values
        if (df['Volume'] < 0).any():
            return False
        
        # Check for missing values
        if df.isnull().any().any():
            return False
        
        return True
    
    def validate_price(self, price: float) -> bool:
        """Validate single price value."""
        return price > 0 and not np.isnan(price) and not np.isinf(price)


def create_enhanced_data_manager() -> EnhancedDataManager:
    """Create and configure enhanced data manager with all available sources."""
    manager = EnhancedDataManager()
    
    # Add Alpha Vantage
    alpha_config = DataSourceConfig(
        api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('alpha_vantage', AlphaVantageDataSource(alpha_config))
    
    # Add IEX Cloud
    iex_config = DataSourceConfig(
        api_key=os.getenv('IEX_CLOUD_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('iex_cloud', IEXCloudDataSource(iex_config))
    
    # Add Polygon
    polygon_config = DataSourceConfig(
        api_key=os.getenv('POLYGON_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('polygon', PolygonDataSource(polygon_config))
    
    # Add FRED
    fred_config = DataSourceConfig(
        api_key=os.getenv('FRED_API_KEY'),
        rate_limit_delay=0.1
    )
    manager.add_source('fred', FREDDataSource(fred_config))
    
    # Add Binance
    binance_config = DataSourceConfig(rate_limit_delay=0.1)
    manager.add_source('binance', BinanceDataSource(binance_config))
    
    # Add Coinbase
    coinbase_config = DataSourceConfig(rate_limit_delay=0.1)
    manager.add_source('coinbase', CoinbaseDataSource(coinbase_config))
    
    logger.info("Enhanced data manager created with all available sources")
    return manager
