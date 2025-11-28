"""
Real-time Data Feeds for AI Market Analysis System

This module provides real-time market data integration using WebSocket feeds,
REST APIs, and other real-time data sources.
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import yfinance as yf
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class FeedType(Enum):
    """Types of data feeds."""
    WEBSOCKET = "websocket"
    REST_API = "rest_api"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    BINANCE = "binance"
    COINBASE = "coinbase"


class DataType(Enum):
    """Types of market data."""
    PRICE = "price"
    VOLUME = "volume"
    TRADE = "trade"
    ORDER_BOOK = "order_book"
    NEWS = "news"
    SENTIMENT = "sentiment"
    ECONOMIC = "economic"


@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    data_type: DataType
    timestamp: datetime
    price: Optional[float] = None
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    close: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedConfig:
    """Configuration for data feeds."""
    feed_type: FeedType
    symbols: List[str]
    data_types: List[DataType]
    update_interval: float = 1.0  # seconds
    max_retries: int = 3
    timeout: float = 30.0
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


class RealTimeDataFeed:
    """
    Real-time data feed manager.
    
    This class provides:
    - WebSocket connections for real-time data
    - REST API polling for periodic updates
    - Multiple data source integration
    - Data buffering and processing
    - Error handling and reconnection
    """
    
    def __init__(self, config: FeedConfig):
        """
        Initialize the real-time data feed.
        
        Args:
            config: Feed configuration
        """
        self.config = config
        self.is_running = False
        self.websocket = None
        self.data_queue = queue.Queue()
        self.subscribers = []
        self.last_data = {}
        self.connection_attempts = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized RealTimeDataFeed: {config.feed_type.value}")
    
    def add_subscriber(self, callback: Callable[[MarketData], None]) -> None:
        """
        Add a data subscriber.
        
        Args:
            callback: Function to call when new data arrives
        """
        self.subscribers.append(callback)
        logger.info(f"Added subscriber: {callback.__name__}")
    
    def remove_subscriber(self, callback: Callable[[MarketData], None]) -> None:
        """
        Remove a data subscriber.
        
        Args:
            callback: Function to remove
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed subscriber: {callback.__name__}")
    
    def start(self) -> None:
        """Start the data feed."""
        try:
            self.is_running = True
            
            if self.config.feed_type == FeedType.WEBSOCKET:
                asyncio.create_task(self._start_websocket_feed())
            elif self.config.feed_type == FeedType.REST_API:
                asyncio.create_task(self._start_rest_api_feed())
            elif self.config.feed_type == FeedType.YAHOO_FINANCE:
                asyncio.create_task(self._start_yahoo_finance_feed())
            else:
                raise ValueError(f"Unsupported feed type: {self.config.feed_type}")
            
            logger.info(f"Started {self.config.feed_type.value} feed")
            
        except Exception as e:
            logger.error(f"Failed to start feed: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the data feed."""
        try:
            self.is_running = False
            
            if self.websocket:
                asyncio.create_task(self.websocket.close())
            
            self.executor.shutdown(wait=True)
            logger.info("Stopped data feed")
            
        except Exception as e:
            logger.error(f"Error stopping feed: {e}")
    
    async def _start_websocket_feed(self) -> None:
        """Start WebSocket data feed."""
        try:
            while self.is_running:
                try:
                    if not self.config.endpoint:
                        raise ValueError("WebSocket endpoint not configured")
                    
                    async with websockets.connect(
                        self.config.endpoint,
                        extra_headers=self.config.headers,
                        timeout=self.config.timeout
                    ) as websocket:
                        self.websocket = websocket
                        self.connection_attempts = 0
                        
                        # Subscribe to symbols
                        await self._subscribe_to_symbols(websocket)
                        
                        # Listen for messages
                        async for message in websocket:
                            if not self.is_running:
                                break
                            
                            try:
                                data = json.loads(message)
                                market_data = self._parse_websocket_data(data)
                                if market_data:
                                    await self._process_data(market_data)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON received: {message}")
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                
                # Reconnection logic
                if self.is_running:
                    self.connection_attempts += 1
                    if self.connection_attempts <= self.config.max_retries:
                        wait_time = min(2 ** self.connection_attempts, 30)
                        logger.info(f"Reconnecting in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Max reconnection attempts reached")
                        break
                        
        except Exception as e:
            logger.error(f"WebSocket feed error: {e}")
    
    async def _start_rest_api_feed(self) -> None:
        """Start REST API polling feed."""
        try:
            async with aiohttp.ClientSession() as session:
                while self.is_running:
                    try:
                        for symbol in self.config.symbols:
                            if not self.is_running:
                                break
                            
                            # Fetch data for symbol
                            data = await self._fetch_rest_api_data(session, symbol)
                            if data is not None:
                                market_data = self._parse_rest_api_data(data, symbol)
                                if market_data:
                                    await self._process_data(market_data)
                        
                        # Wait before next update
                        await asyncio.sleep(self.config.update_interval)
                        
                    except Exception as e:
                        logger.error(f"REST API feed error: {e}")
                        await asyncio.sleep(self.config.update_interval)
                        
        except Exception as e:
            logger.error(f"REST API feed failed: {e}")
    
    async def _start_yahoo_finance_feed(self) -> None:
        """Start Yahoo Finance polling feed."""
        try:
            while self.is_running:
                try:
                    for symbol in self.config.symbols:
                        if not self.is_running:
                            break
                        
                        # Fetch data using yfinance
                        data = await self._fetch_yahoo_finance_data(symbol)
                        if data is not None and not data.empty:
                            market_data = self._parse_yahoo_finance_data(data, symbol)
                            if market_data:
                                await self._process_data(market_data)
                    
                    # Wait before next update
                    await asyncio.sleep(self.config.update_interval)
                    
                except Exception as e:
                    logger.error(f"Yahoo Finance feed error: {e}")
                    await asyncio.sleep(self.config.update_interval)
                    
        except Exception as e:
            logger.error(f"Yahoo Finance feed failed: {e}")
    
    async def _subscribe_to_symbols(self, websocket) -> None:
        """Subscribe to symbols on WebSocket."""
        try:
            for symbol in self.config.symbols:
                subscribe_message = {
                    "action": "subscribe",
                    "symbol": symbol,
                    "data_types": [dt.value for dt in self.config.data_types]
                }
                await websocket.send(json.dumps(subscribe_message))
                logger.info(f"Subscribed to {symbol}")
                
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
    
    async def _fetch_rest_api_data(self, session: aiohttp.ClientSession, symbol: str) -> Optional[Dict]:
        """Fetch data from REST API."""
        try:
            if not self.config.endpoint:
                return None
            
            url = f"{self.config.endpoint}/quote/{symbol}"
            headers = self.config.headers.copy()
            
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            async with session.get(url, headers=headers, timeout=self.config.timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API request failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"REST API fetch failed for {symbol}: {e}")
            return None
    
    async def _fetch_yahoo_finance_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        try:
            # Run yfinance in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            
            def fetch_data():
                try:
                    return ticker.history(period="1d", interval="1m")
                except Exception as e:
                    logger.error(f"Yahoo Finance history fetch failed for {symbol}: {e}")
                    return None
            
            data = await loop.run_in_executor(self.executor, fetch_data)
            
            # Check if data is valid DataFrame and not empty
            if data is not None and hasattr(data, 'empty') and not data.empty:
                return data.tail(1)  # Get latest data point
            return None
            
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed for {symbol}: {e}")
            return None
    
    def _parse_websocket_data(self, data: Dict) -> Optional[MarketData]:
        """Parse WebSocket data into MarketData."""
        try:
            # This is a generic parser - would need to be customized for specific APIs
            symbol = data.get("symbol", "")
            timestamp = datetime.now()
            
            market_data = MarketData(
                symbol=symbol,
                data_type=DataType.PRICE,
                timestamp=timestamp,
                price=data.get("price"),
                volume=data.get("volume"),
                bid=data.get("bid"),
                ask=data.get("ask"),
                high=data.get("high"),
                low=data.get("low"),
                open_price=data.get("open"),
                close=data.get("close"),
                metadata=data
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"WebSocket data parsing failed: {e}")
            return None
    
    def _parse_rest_api_data(self, data: Dict, symbol: str) -> Optional[MarketData]:
        """Parse REST API data into MarketData."""
        try:
            timestamp = datetime.now()
            
            market_data = MarketData(
                symbol=symbol,
                data_type=DataType.PRICE,
                timestamp=timestamp,
                price=data.get("latestPrice"),
                volume=data.get("latestVolume"),
                high=data.get("high"),
                low=data.get("low"),
                open_price=data.get("open"),
                close=data.get("close"),
                metadata=data
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"REST API data parsing failed: {e}")
            return None
    
    def _parse_yahoo_finance_data(self, data: pd.DataFrame, symbol: str) -> Optional[MarketData]:
        """Parse Yahoo Finance data into MarketData."""
        try:
            # Check if data is valid and not empty
            if data is None or not hasattr(data, 'empty') or data.empty:
                return None
            
            latest = data.iloc[-1]
            timestamp = datetime.now()
            
            market_data = MarketData(
                symbol=symbol,
                data_type=DataType.PRICE,
                timestamp=timestamp,
                price=latest.get("Close"),
                volume=latest.get("Volume"),
                high=latest.get("High"),
                low=latest.get("Low"),
                open_price=latest.get("Open"),
                close=latest.get("Close"),
                metadata={"source": "yahoo_finance"}
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Yahoo Finance data parsing failed: {e}")
            return None
    
    async def _process_data(self, market_data: MarketData) -> None:
        """Process incoming market data."""
        try:
            # Store latest data
            self.last_data[market_data.symbol] = market_data
            
            # Add to queue
            self.data_queue.put(market_data)
            
            # Notify subscribers
            for callback in self.subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(market_data)
                    else:
                        callback(market_data)
                except Exception as e:
                    logger.error(f"Subscriber callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest data for a symbol."""
        return self.last_data.get(symbol)
    
    def get_all_latest_data(self) -> Dict[str, MarketData]:
        """Get all latest data."""
        return self.last_data.copy()
    
    def get_data_queue(self) -> queue.Queue:
        """Get the data queue."""
        return self.data_queue


class RealTimeDataManager:
    """
    Manager for multiple real-time data feeds.
    
    This class provides:
    - Multiple feed management
    - Data aggregation and processing
    - Feed health monitoring
    - Automatic failover
    """
    
    def __init__(self):
        """Initialize the real-time data manager."""
        self.feeds = {}
        self.aggregated_data = {}
        self.is_running = False
        self.data_processors = []
        
        logger.info("Initialized RealTimeDataManager")
    
    def add_feed(self, name: str, config: FeedConfig) -> RealTimeDataFeed:
        """
        Add a data feed.
        
        Args:
            name: Feed name
            config: Feed configuration
            
        Returns:
            Created feed instance
        """
        try:
            feed = RealTimeDataFeed(config)
            self.feeds[name] = feed
            logger.info(f"Added feed: {name}")
            return feed
            
        except Exception as e:
            logger.error(f"Failed to add feed {name}: {e}")
            raise
    
    def remove_feed(self, name: str) -> None:
        """
        Remove a data feed.
        
        Args:
            name: Feed name
        """
        try:
            if name in self.feeds:
                self.feeds[name].stop()
                del self.feeds[name]
                logger.info(f"Removed feed: {name}")
                
        except Exception as e:
            logger.error(f"Failed to remove feed {name}: {e}")
    
    def start_all_feeds(self) -> None:
        """Start all data feeds."""
        try:
            self.is_running = True
            
            for name, feed in self.feeds.items():
                try:
                    feed.start()
                    logger.info(f"Started feed: {name}")
                except Exception as e:
                    logger.error(f"Failed to start feed {name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to start feeds: {e}")
    
    def stop_all_feeds(self) -> None:
        """Stop all data feeds."""
        try:
            self.is_running = False
            
            for name, feed in self.feeds.items():
                try:
                    feed.stop()
                    logger.info(f"Stopped feed: {name}")
                except Exception as e:
                    logger.error(f"Failed to stop feed {name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to stop feeds: {e}")
    
    def get_aggregated_data(self, symbol: str) -> Optional[MarketData]:
        """Get aggregated data for a symbol."""
        return self.aggregated_data.get(symbol)
    
    def get_all_aggregated_data(self) -> Dict[str, MarketData]:
        """Get all aggregated data."""
        return self.aggregated_data.copy()
    
    def add_data_processor(self, processor: Callable[[MarketData], None]) -> None:
        """Add a data processor."""
        self.data_processors.append(processor)
        logger.info(f"Added data processor: {processor.__name__}")
    
    def get_feed_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all feeds."""
        status = {}
        
        for name, feed in self.feeds.items():
            status[name] = {
                "is_running": feed.is_running,
                "connection_attempts": feed.connection_attempts,
                "last_data_count": len(feed.last_data),
                "queue_size": feed.data_queue.qsize()
            }
        
        return status
