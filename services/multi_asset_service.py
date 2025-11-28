"""
Multi-Asset Data Service for AI Market Analysis System

This service provides data integration for multiple asset classes:
- Commodities: Gold (GC=F), Oil (CL=F), Silver (SI=F)
- Forex: EUR/USD (EURUSD=X), GBP/USD (GBPUSD=X)
- Cryptocurrencies: Bitcoin (BTC-USD), Ethereum (ETH-USD)
- Equities: Existing stock symbols

Part of Sprint 7: Multi-Asset Support
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from queue import Queue, Empty
from enum import Enum

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Asset class types."""
    EQUITY = "equity"
    COMMODITY = "commodity"
    FOREX = "forex"
    CRYPTOCURRENCY = "cryptocurrency"
    BOND = "bond"
    ETF = "etf"

class AssetType(Enum):
    """Asset type categories."""
    STOCK = "stock"
    GOLD = "gold"
    OIL = "oil"
    SILVER = "silver"
    EUR_USD = "eur_usd"
    GBP_USD = "gbp_usd"
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"

@dataclass
class AssetInfo:
    """Information about an asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    asset_type: AssetType
    currency: str
    exchange: str
    lot_size: float = 1.0
    tick_size: float = 0.01
    margin_requirement: float = 1.0
    volatility_multiplier: float = 1.0
    risk_weight: float = 1.0

@dataclass
class MultiAssetConfig:
    """Configuration for multi-asset service."""
    symbols: List[str] = field(default_factory=lambda: [
        # Commodities
        'GC=F', 'CL=F', 'SI=F',
        # Forex
        'EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'CHFUSD=X',
        # Cryptocurrencies
        'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD',
        # Additional commodities
        'NG=F', 'HG=F', 'PL=F', 'PA=F'
    ])
    update_interval_seconds: int = 30
    lookback_days: int = 30
    enable_real_time: bool = True
    cache_duration_minutes: int = 5
    max_concurrent_requests: int = 10

class MultiAssetService:
    """
    Multi-Asset Data Service that provides data for different asset classes.
    
    This service:
    - Fetches real market data from Yahoo Finance for multiple asset classes
    - Provides normalized data across different asset types
    - Handles different trading hours and market sessions
    - Provides risk-adjusted position sizing
    - Normalizes volatility across asset classes
    """
    
    def __init__(self, config: Optional[MultiAssetConfig] = None):
        """Initialize the multi-asset service."""
        self.config = config or MultiAssetConfig()
        self.asset_info: Dict[str, AssetInfo] = {}
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests)
        
        # Initialize asset information
        self._initialize_asset_info()
        
        logger.info("MultiAssetService initialized with %d symbols across %d asset classes", 
                   len(self.config.symbols), len(set(info.asset_class for info in self.asset_info.values())))
    
    def _initialize_asset_info(self):
        """Initialize asset information for all symbols."""
        asset_mappings = {
            # Commodities
            'GC=F': AssetInfo('GC=F', 'Gold Futures', AssetClass.COMMODITY, AssetType.GOLD, 'USD', 'COMEX', 100, 0.1, 0.05, 1.2, 1.1),
            'CL=F': AssetInfo('CL=F', 'Crude Oil Futures', AssetClass.COMMODITY, AssetType.OIL, 'USD', 'NYMEX', 1000, 0.01, 0.08, 1.5, 1.3),
            'SI=F': AssetInfo('SI=F', 'Silver Futures', AssetClass.COMMODITY, AssetType.SILVER, 'USD', 'COMEX', 5000, 0.001, 0.06, 1.3, 1.2),
            'NG=F': AssetInfo('NG=F', 'Natural Gas Futures', AssetClass.COMMODITY, AssetType.OIL, 'USD', 'NYMEX', 10000, 0.001, 0.10, 1.8, 1.4),
            'HG=F': AssetInfo('HG=F', 'Copper Futures', AssetClass.COMMODITY, AssetType.OIL, 'USD', 'COMEX', 25000, 0.0005, 0.07, 1.4, 1.2),
            'PL=F': AssetInfo('PL=F', 'Platinum Futures', AssetClass.COMMODITY, AssetType.GOLD, 'USD', 'NYMEX', 50, 0.1, 0.05, 1.1, 1.0),
            'PA=F': AssetInfo('PA=F', 'Palladium Futures', AssetClass.COMMODITY, AssetType.GOLD, 'USD', 'NYMEX', 100, 0.05, 0.06, 1.2, 1.1),
            
            # Forex
            'EURUSD=X': AssetInfo('EURUSD=X', 'EUR/USD', AssetClass.FOREX, AssetType.EUR_USD, 'USD', 'FOREX', 100000, 0.00001, 0.02, 0.8, 0.9),
            'GBPUSD=X': AssetInfo('GBPUSD=X', 'GBP/USD', AssetClass.FOREX, AssetType.GBP_USD, 'USD', 'FOREX', 100000, 0.00001, 0.03, 1.0, 1.0),
            'JPYUSD=X': AssetInfo('JPYUSD=X', 'JPY/USD', AssetClass.FOREX, AssetType.EUR_USD, 'USD', 'FOREX', 100000, 0.00001, 0.02, 0.9, 0.9),
            'CHFUSD=X': AssetInfo('CHFUSD=X', 'CHF/USD', AssetClass.FOREX, AssetType.EUR_USD, 'USD', 'FOREX', 100000, 0.00001, 0.02, 0.8, 0.9),
            
            # Cryptocurrencies
            'BTC-USD': AssetInfo('BTC-USD', 'Bitcoin', AssetClass.CRYPTOCURRENCY, AssetType.BITCOIN, 'USD', 'CRYPTO', 1, 0.01, 0.20, 2.0, 1.8),
            'ETH-USD': AssetInfo('ETH-USD', 'Ethereum', AssetClass.CRYPTOCURRENCY, AssetType.ETHEREUM, 'USD', 'CRYPTO', 1, 0.01, 0.25, 2.2, 2.0),
            'ADA-USD': AssetInfo('ADA-USD', 'Cardano', AssetClass.CRYPTOCURRENCY, AssetType.BITCOIN, 'USD', 'CRYPTO', 1, 0.0001, 0.30, 2.5, 2.2),
            'DOT-USD': AssetInfo('DOT-USD', 'Polkadot', AssetClass.CRYPTOCURRENCY, AssetType.BITCOIN, 'USD', 'CRYPTO', 1, 0.01, 0.28, 2.3, 2.1),
        }
        
        # Add default asset info for any symbols not in the mapping
        for symbol in self.config.symbols:
            if symbol not in asset_mappings:
                # Default to equity if not specified
                asset_mappings[symbol] = AssetInfo(
                    symbol=symbol,
                    name=symbol,
                    asset_class=AssetClass.EQUITY,
                    asset_type=AssetType.STOCK,
                    currency='USD',
                    exchange='NYSE',
                    lot_size=1.0,
                    tick_size=0.01,
                    margin_requirement=0.5,
                    volatility_multiplier=1.0,
                    risk_weight=1.0
                )
        
        self.asset_info = asset_mappings
    
    async def get_asset_info(self, symbol: str) -> Optional[AssetInfo]:
        """Get asset information for a symbol."""
        return self.asset_info.get(symbol)
    
    async def get_all_asset_info(self) -> Dict[str, AssetInfo]:
        """Get all asset information."""
        return self.asset_info.copy()
    
    async def get_price_history(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get price history for a symbol."""
        try:
            with self.cache_lock:
                # Check if we have recent data
                if symbol in self.price_data and symbol in self.last_update:
                    if datetime.now() - self.last_update[symbol] < timedelta(minutes=self.config.cache_duration_minutes):
                        return self.price_data[symbol].tail(days)
                
                # Fetch new data
                data = await self._fetch_price_data(symbol, days)
                if data is not None and not data.empty:
                    self.price_data[symbol] = data
                    self.last_update[symbol] = datetime.now()
                    return data.tail(days)
                
                return None
                
        except Exception as e:
            logger.error("Error getting price history for %s: %s", symbol, e)
            return None
    
    async def _fetch_price_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch price data from Yahoo Finance."""
        try:
            # Use yfinance to get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d")
            
            if data.empty:
                logger.warning("No data available for symbol: %s", symbol)
                return None
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.warning("Missing column %s for symbol %s", col, symbol)
                    return None
            
            # Rename columns to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            logger.error("Error fetching price data for %s: %s", symbol, e)
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            data = await self.get_price_history(symbol, 1)
            if data is not None and not data.empty:
                return float(data['close'].iloc[-1])
            return None
        except Exception as e:
            logger.error("Error getting current price for %s: %s", symbol, e)
            return None
    
    async def get_normalized_volatility(self, symbol: str, days: int = 20) -> Optional[float]:
        """Get normalized volatility for a symbol."""
        try:
            data = await self.get_price_history(symbol, days)
            if data is None or len(data) < 10:
                return None
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Get asset info for normalization
            asset_info = await self.get_asset_info(symbol)
            if asset_info:
                # Apply volatility multiplier
                normalized_volatility = volatility * asset_info.volatility_multiplier
            else:
                normalized_volatility = volatility
            
            return float(normalized_volatility)
            
        except Exception as e:
            logger.error("Error calculating normalized volatility for %s: %s", symbol, e)
            return None
    
    async def get_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Get Average True Range (ATR) for a symbol."""
        try:
            data = await self.get_price_history(symbol, period + 10)
            if data is None or len(data) < period:
                return None
            
            # Calculate True Range
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr)
            
        except Exception as e:
            logger.error("Error calculating ATR for %s: %s", symbol, e)
            return None
    
    async def calculate_position_size(self, symbol: str, account_balance: float, risk_percent: float = 0.02) -> Optional[float]:
        """Calculate position size based on risk management."""
        try:
            asset_info = await self.get_asset_info(symbol)
            if not asset_info:
                return None
            
            # Get current price
            current_price = await self.get_current_price(symbol)
            if not current_price:
                return None
            
            # Get ATR for stop loss calculation
            atr = await self.get_atr(symbol)
            if not atr:
                return None
            
            # Calculate risk amount
            risk_amount = account_balance * risk_percent
            
            # Calculate stop loss distance (2 * ATR)
            stop_loss_distance = 2 * atr
            
            # Calculate position size
            position_size = risk_amount / stop_loss_distance
            
            # Apply lot size and risk weight
            position_size = (position_size / asset_info.lot_size) * asset_info.risk_weight
            
            return float(position_size)
            
        except Exception as e:
            logger.error("Error calculating position size for %s: %s", symbol, e)
            return None
    
    async def get_asset_class_data(self, asset_class: AssetClass) -> Dict[str, Any]:
        """Get data for all symbols in an asset class."""
        try:
            symbols = [symbol for symbol, info in self.asset_info.items() if info.asset_class == asset_class]
            
            data = {}
            for symbol in symbols:
                price_data = await self.get_price_history(symbol, 30)
                if price_data is not None:
                    data[symbol] = {
                        'current_price': float(price_data['close'].iloc[-1]) if not price_data.empty else None,
                        'volatility': await self.get_normalized_volatility(symbol),
                        'atr': await self.get_atr(symbol),
                        'asset_info': self.asset_info[symbol]
                    }
            
            return data
            
        except Exception as e:
            logger.error("Error getting asset class data for %s: %s", asset_class.value, e)
            return {}
    
    async def get_multi_asset_summary(self) -> Dict[str, Any]:
        """Get summary of all multi-asset data."""
        try:
            summary = {
                'total_symbols': len(self.config.symbols),
                'asset_classes': {},
                'last_update': datetime.now().isoformat(),
                'symbols_by_class': {}
            }
            
            # Group symbols by asset class
            for symbol, info in self.asset_info.items():
                asset_class = info.asset_class.value
                if asset_class not in summary['symbols_by_class']:
                    summary['symbols_by_class'][asset_class] = []
                summary['symbols_by_class'][asset_class].append(symbol)
            
            # Get data for each asset class
            for asset_class in AssetClass:
                class_data = await self.get_asset_class_data(asset_class)
                if class_data:
                    summary['asset_classes'][asset_class.value] = {
                        'symbol_count': len(class_data),
                        'symbols': list(class_data.keys()),
                        'avg_volatility': np.mean([data['volatility'] for data in class_data.values() if data['volatility'] is not None]) if class_data else 0.0
                    }
            
            return summary
            
        except Exception as e:
            logger.error("Error getting multi-asset summary: %s", e)
            return {}
    
    async def start_real_time_updates(self):
        """Start real-time data updates."""
        if not self.config.enable_real_time:
            return
        
        logger.info("Starting real-time multi-asset data updates")
        
        while True:
            try:
                # Update all symbols
                tasks = []
                for symbol in self.config.symbols:
                    task = self.get_price_history(symbol, 1)
                    tasks.append(task)
                
                # Wait for all updates to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                logger.error("Error in real-time updates: %s", e)
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop(self):
        """Stop the service."""
        logger.info("Stopping MultiAssetService")
        self.executor.shutdown(wait=True)
