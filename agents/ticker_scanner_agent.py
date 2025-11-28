"""
TickerScannerAgent - Market Ticker Discovery Engine

This agent scans the market universe for promising tickers based on:
- Volatility triggers (unusual volume, price movements)
- News sentiment triggers (positive/negative news impact)
- Technical indicators (breakouts, momentum)
- Sector rotation patterns
- Market regime changes

Part of Sprint 5-6: Market Ticker Discovery Engine
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from services.real_data_service import RealDataService

class ScanTrigger(Enum):
    """Types of scan triggers."""
    VOLATILITY = "volatility"
    NEWS_SENTIMENT = "news_sentiment"
    TECHNICAL_BREAKOUT = "technical_breakout"
    MOMENTUM = "momentum"
    SECTOR_ROTATION = "sector_rotation"
    VOLUME_SURGE = "volume_surge"
    EARNINGS = "earnings"
    DIVIDEND = "dividend"

class ScanPriority(Enum):
    """Scan priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ScanResult:
    """Result of a ticker scan."""
    symbol: str
    trigger: ScanTrigger
    priority: ScanPriority
    confidence: float
    score: float
    description: str
    metadata: Dict
    timestamp: datetime

@dataclass
class TickerScannerSummary:
    """Summary of ticker scanner performance."""
    total_scanned: int
    triggers_found: int
    high_priority_finds: int
    medium_priority_finds: int
    low_priority_finds: int
    top_triggers: List[ScanResult]
    scan_duration: float
    last_scan: datetime
    sectors_scanned: List[str]
    avg_confidence: float

class TickerScannerAgent:
    """Agent for scanning market universe for promising tickers."""
    
    def __init__(self, real_data_service: Optional[RealDataService] = None):
        """Initialize the ticker scanner agent."""
        self.real_data_service = real_data_service
        self.scan_results: List[ScanResult] = []
        self.scan_history: List[ScanResult] = []
        
        # Market universe - can be expanded
        self.market_universe = [
            # Tech stocks
            'BTC-USD', 'SOXL', 'NVDA', 'RIVN', 'TSLA', 'META', 'TQQQ', 'SPXU', 'AMD', 'INTC',
            # Financial stocks
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'V',
            # Healthcare stocks
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
            # Energy stocks
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'WMB', 'MPC', 'VLO',
            # Consumer stocks
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
            # Industrial stocks
            'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
            # ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'GLD', 'SLV', 'TLT', 'HYG'
        ]
        
        # Sector mapping
        self.sector_mapping = {
            'BTC-USD': 'Cryptocurrency', 'SOXL': 'Technology ETF', 'NVDA': 'Technology', 'RIVN': 'Consumer Discretionary',
            'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology', 'NFLX': 'Technology',
            'AMD': 'Technology', 'INTC': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
            'MS': 'Financials', 'C': 'Financials', 'AXP': 'Financials', 'BLK': 'Financials',
            'SPGI': 'Financials', 'V': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'MRK': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'DHR': 'Healthcare',
            'BMY': 'Healthcare', 'AMGN': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy', 'SLB': 'Energy',
            'PXD': 'Energy', 'KMI': 'Energy', 'WMB': 'Energy', 'MPC': 'Energy', 'VLO': 'Energy',
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
            'HD': 'Consumer', 'MCD': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer',
            'TGT': 'Consumer', 'LOW': 'Consumer',
            'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial', 'HON': 'Industrial',
            'MMM': 'Industrial', 'UPS': 'Industrial', 'FDX': 'Industrial', 'RTX': 'Industrial',
            'LMT': 'Industrial', 'NOC': 'Industrial',
            'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'VTI': 'ETF', 'VEA': 'ETF',
            'VWO': 'ETF', 'GLD': 'ETF', 'SLV': 'ETF', 'TLT': 'ETF', 'HYG': 'ETF'
        }
        
        # Scan thresholds
        self.volatility_threshold = 0.05  # 5% daily volatility
        self.volume_threshold = 2.0  # 2x average volume
        self.momentum_threshold = 0.03  # 3% price momentum
        self.breakout_threshold = 0.02  # 2% breakout from resistance
        
        logger.info("TickerScannerAgent initialized with market universe of %d tickers", len(self.market_universe))
    
    async def scan_market_universe(self, sector: Optional[str] = None) -> List[ScanResult]:
        """Scan the entire market universe for opportunities, optionally filtered by sector."""
        start_time = datetime.now()
        
        # Filter universe by sector if specified
        if sector:
            filtered_universe = [symbol for symbol in self.market_universe 
                               if self.sector_mapping.get(symbol, '').lower() == sector.lower()]
            logger.info("Starting sector-specific scan for %s sector: %d tickers", sector, len(filtered_universe))
        else:
            filtered_universe = self.market_universe
            logger.info("Starting market universe scan for %d tickers", len(filtered_universe))
        
        scan_results = []
        
        # Scan in batches to avoid overwhelming the data service
        batch_size = 10
        for i in range(0, len(filtered_universe), batch_size):
            batch = filtered_universe[i:i + batch_size]
            batch_results = await self._scan_batch(batch)
            scan_results.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Sort by score and priority
        scan_results.sort(key=lambda x: (x.priority.value, x.score), reverse=True)
        
        # Store results
        self.scan_results = scan_results
        self.scan_history.extend(scan_results)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(days=7)
        self.scan_history = [r for r in self.scan_history if r.timestamp > cutoff_time]
        
        scan_duration = (datetime.now() - start_time).total_seconds()
        logger.info("Market scan completed in %.2fs, found %d triggers", scan_duration, len(scan_results))
        
        return scan_results
    
    async def scan_sector_opportunities(self, sector: str) -> List[ScanResult]:
        """Scan a specific sector for market opportunities."""
        sector_mapping = {
            'technology': ['Technology'],
            'finance': ['Financials'],
            'healthcare': ['Healthcare'],
            'retail': ['Consumer']
        }
        
        # Map sector names to the internal sector categories
        target_sectors = sector_mapping.get(sector.lower(), [sector])
        
        # Get all symbols in the target sectors
        sector_symbols = [symbol for symbol, sec in self.sector_mapping.items() 
                         if sec in target_sectors]
        
        logger.info("Scanning %s sector for opportunities: %d tickers", sector, len(sector_symbols))
        
        if not sector_symbols:
            logger.warning("No tickers found for sector: %s", sector)
            return []
        
        # Scan only the symbols in this sector
        scan_results = []
        batch_size = 8  # Smaller batches for sector-specific scans
        
        for i in range(0, len(sector_symbols), batch_size):
            batch = sector_symbols[i:i + batch_size]
            batch_results = await self._scan_batch(batch)
            scan_results.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Sort by score and priority
        scan_results.sort(key=lambda x: (x.priority.value, x.score), reverse=True)
        
        # Store results with sector metadata
        for result in scan_results:
            result.metadata = result.metadata or {}
            result.metadata['sector'] = sector
            result.metadata['sector_category'] = self.sector_mapping.get(result.symbol, 'Unknown')
        
        logger.info("Sector scan completed for %s: found %d opportunities", sector, len(scan_results))
        return scan_results
    
    async def scan_all_sectors(self) -> Dict[str, List[ScanResult]]:
        """Scan all major sectors for opportunities."""
        sectors = ['technology', 'finance', 'healthcare', 'retail']
        sector_results = {}
        
        for sector in sectors:
            try:
                sector_results[sector] = await self.scan_sector_opportunities(sector)
            except Exception as e:
                logger.error(f"Error scanning {sector} sector: {e}")
                sector_results[sector] = []
        
        return sector_results
    
    async def _scan_batch(self, symbols: List[str]) -> List[ScanResult]:
        """Scan a batch of symbols."""
        results = []
        
        for symbol in symbols:
            try:
                symbol_results = await self._scan_symbol(symbol)
                results.extend(symbol_results)
            except Exception as e:
                logger.warning("Error scanning symbol %s: %s", symbol, e)
                continue
        
        return results
    
    async def _scan_symbol(self, symbol: str) -> List[ScanResult]:
        """Scan a single symbol for triggers."""
        results = []
        
        try:
            # Get market data
            data = await self._get_market_data(symbol, periods=50)
            if data is None or len(data) < 20:
                return results
            
            # Check for volatility triggers
            volatility_result = self._check_volatility_trigger(symbol, data)
            if volatility_result:
                results.append(volatility_result)
            
            # Check for volume surge
            volume_result = self._check_volume_trigger(symbol, data)
            if volume_result:
                results.append(volume_result)
            
            # Check for technical breakouts
            breakout_result = self._check_breakout_trigger(symbol, data)
            if breakout_result:
                results.append(breakout_result)
            
            # Check for momentum
            momentum_result = self._check_momentum_trigger(symbol, data)
            if momentum_result:
                results.append(momentum_result)
            
            # Check for news sentiment (mock for now)
            news_result = self._check_news_trigger(symbol)
            if news_result:
                results.append(news_result)
            
        except Exception as e:
                logger.warning("Error scanning symbol %s: %s", symbol, e)
        
        return results
    
    def _check_volatility_trigger(self, symbol: str, data: pd.DataFrame) -> Optional[ScanResult]:
        """Check for volatility triggers."""
        try:
            if len(data) < 20:
                return None
            
            # Calculate recent volatility
            returns = data['close'].pct_change().dropna()
            recent_volatility = returns.tail(5).std()
            historical_volatility = returns.std()
            
            # Check if volatility is significantly higher than historical
            volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
            
            if volatility_ratio > 1.5:  # 50% higher than historical
                confidence = min(0.9, volatility_ratio / 2)
                score = volatility_ratio * 0.7
                
                priority = ScanPriority.HIGH if volatility_ratio > 2 else ScanPriority.MEDIUM
                
                return ScanResult(
                    symbol=symbol,
                    trigger=ScanTrigger.VOLATILITY,
                    priority=priority,
                    confidence=confidence,
                    score=score,
                    description=f"High volatility detected: {volatility_ratio:.2f}x historical average",
                    metadata={
                        'recent_volatility': recent_volatility,
                        'historical_volatility': historical_volatility,
                        'volatility_ratio': volatility_ratio,
                        'sector': self.sector_mapping.get(symbol, 'Unknown')
                    },
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.warning("Error checking volatility for %s: %s", symbol, e)
        
        return None
    
    def _check_volume_trigger(self, symbol: str, data: pd.DataFrame) -> Optional[ScanResult]:
        """Check for volume surge triggers."""
        try:
            if len(data) < 20:
                return None
            
            # Calculate recent vs historical volume
            recent_volume = data['volume'].tail(5).mean()
            historical_volume = data['volume'].mean()
            
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
            
            if volume_ratio > self.volume_threshold:
                confidence = min(0.9, volume_ratio / 3)
                score = volume_ratio * 0.6
                
                priority = ScanPriority.HIGH if volume_ratio > 3 else ScanPriority.MEDIUM
                
                return ScanResult(
                    symbol=symbol,
                    trigger=ScanTrigger.VOLUME_SURGE,
                    priority=priority,
                    confidence=confidence,
                    score=score,
                    description=f"Volume surge detected: {volume_ratio:.2f}x average volume",
                    metadata={
                        'recent_volume': recent_volume,
                        'historical_volume': historical_volume,
                        'volume_ratio': volume_ratio,
                        'sector': self.sector_mapping.get(symbol, 'Unknown')
                    },
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.warning("Error checking volume for %s: %s", symbol, e)
        
        return None
    
    def _check_breakout_trigger(self, symbol: str, data: pd.DataFrame) -> Optional[ScanResult]:
        """Check for technical breakout triggers."""
        try:
            if len(data) < 20:
                return None
            
            # Calculate resistance and support levels
            recent_high = data['high'].tail(20).max()
            recent_low = data['low'].tail(20).min()
            current_price = data['close'].iloc[-1]
            
            # Check for breakout above resistance
            resistance_breakout = (current_price - recent_high) / recent_high
            support_breakdown = (recent_low - current_price) / recent_low
            
            if resistance_breakout > self.breakout_threshold:
                confidence = min(0.9, resistance_breakout * 10)
                score = resistance_breakout * 0.8
                
                priority = ScanPriority.HIGH if resistance_breakout > 0.03 else ScanPriority.MEDIUM
                
                return ScanResult(
                    symbol=symbol,
                    trigger=ScanTrigger.TECHNICAL_BREAKOUT,
                    priority=priority,
                    confidence=confidence,
                    score=score,
                    description=f"Resistance breakout: {resistance_breakout:.2%} above recent high",
                    metadata={
                        'resistance_breakout': resistance_breakout,
                        'recent_high': recent_high,
                        'current_price': current_price,
                        'sector': self.sector_mapping.get(symbol, 'Unknown')
                    },
                    timestamp=datetime.now()
                )
            
            elif support_breakdown > self.breakout_threshold:
                confidence = min(0.9, support_breakdown * 10)
                score = support_breakdown * 0.8
                
                priority = ScanPriority.MEDIUM
                
                return ScanResult(
                    symbol=symbol,
                    trigger=ScanTrigger.TECHNICAL_BREAKOUT,
                    priority=priority,
                    confidence=confidence,
                    score=score,
                    description=f"Support breakdown: {support_breakdown:.2%} below recent low",
                    metadata={
                        'support_breakdown': support_breakdown,
                        'recent_low': recent_low,
                        'current_price': current_price,
                        'sector': self.sector_mapping.get(symbol, 'Unknown')
                    },
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.warning("Error checking breakout for %s: %s", symbol, e)
        
        return None
    
    def _check_momentum_trigger(self, symbol: str, data: pd.DataFrame) -> Optional[ScanResult]:
        """Check for momentum triggers."""
        try:
            if len(data) < 20:
                return None
            
            # Calculate momentum indicators
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            current_price = data['close'].iloc[-1]
            
            # Price above moving averages
            price_vs_sma20 = (current_price - sma_20) / sma_20
            price_vs_sma50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            
            # RSI calculation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Check for strong momentum
            if price_vs_sma20 > self.momentum_threshold and price_vs_sma50 > 0 and current_rsi > 50:
                confidence = min(0.9, (price_vs_sma20 + price_vs_sma50) * 5)
                score = (price_vs_sma20 + price_vs_sma50) * 0.6
                
                priority = ScanPriority.HIGH if price_vs_sma20 > 0.05 else ScanPriority.MEDIUM
                
                return ScanResult(
                    symbol=symbol,
                    trigger=ScanTrigger.MOMENTUM,
                    priority=priority,
                    confidence=confidence,
                    score=score,
                    description=f"Strong momentum: {price_vs_sma20:.2%} above SMA20, RSI {current_rsi:.1f}",
                    metadata={
                        'price_vs_sma20': price_vs_sma20,
                        'price_vs_sma50': price_vs_sma50,
                        'rsi': current_rsi,
                        'sma_20': sma_20,
                        'sma_50': sma_50,
                        'sector': self.sector_mapping.get(symbol, 'Unknown')
                    },
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.warning("Error checking momentum for %s: %s", symbol, e)
        
        return None
    
    def _check_news_trigger(self, symbol: str) -> Optional[ScanResult]:
        """Check for news sentiment triggers (mock implementation)."""
        try:
            # Mock news sentiment analysis
            # In a real implementation, this would integrate with news APIs
            np.random.seed(hash(symbol) % 2**32)
            
            # Simulate news sentiment
            sentiment_score = np.random.normal(0, 0.3)
            
            if abs(sentiment_score) > 0.5:  # Strong sentiment
                confidence = min(0.9, abs(sentiment_score))
                score = abs(sentiment_score) * 0.7
                
                priority = ScanPriority.HIGH if abs(sentiment_score) > 0.7 else ScanPriority.MEDIUM
                
                sentiment_type = "positive" if sentiment_score > 0 else "negative"
                
                return ScanResult(
                    symbol=symbol,
                    trigger=ScanTrigger.NEWS_SENTIMENT,
                    priority=priority,
                    confidence=confidence,
                    score=score,
                    description=f"Strong {sentiment_type} news sentiment: {sentiment_score:.2f}",
                    metadata={
                        'sentiment_score': sentiment_score,
                        'sentiment_type': sentiment_type,
                        'sector': self.sector_mapping.get(symbol, 'Unknown')
                    },
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.warning("Error checking news for %s: %s", symbol, e)
        
        return None
    
    async def _get_market_data(self, symbol: str, periods: int = 50) -> Optional[pd.DataFrame]:
        """Get market data for analysis."""
        try:
            if not self.real_data_service:
                logger.warning("Real data service not available, using mock data")
                return self._generate_mock_data(symbol, periods)
            
            # Get historical data
            data = self.real_data_service.get_price_history(symbol, days=periods)
            if data is None or data.empty:
                logger.warning(f"No real data available for {symbol}, using mock data")
                return self._generate_mock_data(symbol, periods)
            
            return data
            
        except Exception as e:
            logger.error("Error getting market data for %s: %s", symbol, e)
            logger.warning("Using mock data for %s", symbol)
            return self._generate_mock_data(symbol, periods)
    
    def _generate_mock_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate mock market data for testing."""
        try:
            # Base prices for different symbols
            base_prices = {
                'BTC-USD': 69000.0, 'SOXL': 45.0, 'NVDA': 480.0, 'RIVN': 15.0, 'TSLA': 250.0, 'SPY': 450.0,
                'META': 300.0, 'TQQQ': 120.0, 'SPXL': 80.0, 'AMD': 110.0,
                'JPM': 150.0, 'BAC': 30.0, 'WFC': 45.0, 'GS': 400.0, 'MS': 90.0,
                'JNJ': 160.0, 'PFE': 30.0, 'UNH': 500.0, 'ABBV': 150.0, 'MRK': 120.0,
                'XOM': 110.0, 'CVX': 150.0, 'COP': 120.0, 'EOG': 130.0, 'SLB': 50.0,
                'PG': 150.0, 'KO': 60.0, 'PEP': 170.0, 'WMT': 160.0, 'HD': 350.0,
                'BA': 200.0, 'CAT': 250.0, 'GE': 100.0, 'HON': 200.0, 'MMM': 100.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Generate time series data
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            # Generate price data with some trend and volatility
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.001, 0.02, periods)  # 0.1% daily return, 2% volatility
            prices = [base_price]
            
            for i in range(1, periods):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(new_price)
            
            # Generate OHLCV data
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                # Generate OHLC from close price
                volatility = 0.01  # 1% intraday volatility
                high = close * (1 + np.random.uniform(0, volatility))
                low = close * (1 - np.random.uniform(0, volatility))
                open_price = close * (1 + np.random.uniform(-volatility/2, volatility/2))
                
                # Ensure OHLC consistency
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # Generate volume
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'date': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error("Error generating mock data: %s", e)
            return None
    
    async def get_scan_summary(self) -> TickerScannerSummary:
        """Get summary of ticker scanner performance."""
        try:
            total_scanned = len(self.market_universe)
            triggers_found = len(self.scan_results)
            
            high_priority = len([r for r in self.scan_results if r.priority == ScanPriority.HIGH])
            medium_priority = len([r for r in self.scan_results if r.priority == ScanPriority.MEDIUM])
            low_priority = len([r for r in self.scan_results if r.priority == ScanPriority.LOW])
            
            top_triggers = sorted(self.scan_results, key=lambda x: x.score, reverse=True)[:10]
            
            sectors_scanned = list(set([r.metadata.get('sector', 'Unknown') for r in self.scan_results]))
            
            avg_confidence = np.mean([r.confidence for r in self.scan_results]) if self.scan_results else 0.0
            
            return TickerScannerSummary(
                total_scanned=total_scanned,
                triggers_found=triggers_found,
                high_priority_finds=high_priority,
                medium_priority_finds=medium_priority,
                low_priority_finds=low_priority,
                top_triggers=top_triggers,
                scan_duration=0.0,  # Will be updated by caller
                last_scan=datetime.now(),
                sectors_scanned=sectors_scanned,
                avg_confidence=avg_confidence
            )
            
        except Exception as e:
            logger.error("Error getting scan summary: %s", e)
            return TickerScannerSummary(
                total_scanned=0,
                triggers_found=0,
                high_priority_finds=0,
                medium_priority_finds=0,
                low_priority_finds=0,
                top_triggers=[],
                scan_duration=0.0,
                last_scan=datetime.now(),
                sectors_scanned=[],
                avg_confidence=0.0
            )
    
    def get_top_opportunities(self, limit: int = 5) -> List[ScanResult]:
        """Get top opportunities from recent scan."""
        return sorted(self.scan_results, key=lambda x: (x.priority.value, x.score), reverse=True)[:limit]
    
    def get_opportunities_by_sector(self, sector: str) -> List[ScanResult]:
        """Get opportunities filtered by sector."""
        return [r for r in self.scan_results if r.metadata.get('sector') == sector]
    
    def get_opportunities_by_trigger(self, trigger: ScanTrigger) -> List[ScanResult]:
        """Get opportunities filtered by trigger type."""
        return [r for r in self.scan_results if r.trigger == trigger]
