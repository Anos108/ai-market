import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
from pathlib import Path

from services.real_data_service import RealDataService, RealDataConfig
from services.postgres_database import PostgreSQLDatabase, SymbolInfo, ManagedSymbol, SymbolStatus, SymbolSource

logger = logging.getLogger(__name__)

@dataclass
class SymbolPerformance:
    symbol: str
    current_price: float
    change_percent: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    last_updated: str
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingDecision:
    symbol: str
    action: str  # "buy", "sell", "hold", "watch"
    confidence: float
    reasoning: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    timeframe: str = "short_term"
    generated_at: str = ""

@dataclass
class SymbolManagerSummary:
    total_symbols: int
    active_symbols: int
    monitoring_symbols: int
    watchlist_symbols: int
    symbols_by_source: Dict[str, int]
    symbols_by_sector: Dict[str, int]
    last_updated: str
    performance_summary: Dict[str, Any] = field(default_factory=dict)

class SymbolManagerPostgreSQL:
    def __init__(self, 
                 real_data_service: RealDataService,
                 db_host: str = "localhost",
                 db_port: int = 5432,
                 db_name: str = "ai_market_system",
                 db_user: str = "postgres",
                 db_password: str = "password"):
        self.real_data_service = real_data_service
        self.db = PostgreSQLDatabase(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        self.performance_cache: Dict[str, SymbolPerformance] = {}
        self.trading_decisions: Dict[str, TradingDecision] = {}
        self._initialized = False
        
        logger.info(f"SymbolManagerPostgreSQL initialized with PostgreSQL at {db_host}:{db_port}/{db_name}")

    async def initialize(self):
        """Initialize the database connection and migrate data if needed."""
        if self._initialized:
            return
        
        try:
            # Initialize PostgreSQL database
            await self.db.initialize()
            
            # Migrate from existing JSON/SQLite if they exist
            await self._migrate_existing_data()
            
            # Initialize with default symbols if empty
            if not await self.db.get_managed_symbols():
                await self._initialize_default_symbols()
            
            self._initialized = True
            logger.info("SymbolManagerPostgreSQL initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SymbolManagerPostgreSQL: {e}")
            raise

    async def close(self):
        """Close the database connection."""
        if self.db:
            await self.db.close()
            logger.info("SymbolManagerPostgreSQL closed")

    async def _migrate_existing_data(self):
        """Migrate from existing JSON or SQLite files if they exist."""
        # Try to migrate from JSON first
        json_file = Path("symbols.json")
        if json_file.exists():
            logger.info("Found existing symbols.json, migrating to PostgreSQL...")
            if await self.db.migrate_from_json("symbols.json"):
                # Backup the JSON file
                backup_path = f"symbols_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                json_file.rename(backup_path)
                logger.info(f"JSON file backed up to {backup_path}")
            else:
                logger.error("Failed to migrate from JSON file")
        
        # Try to migrate from SQLite
        sqlite_file = Path("symbols.db")
        if sqlite_file.exists():
            logger.info("Found existing symbols.db, migrating to PostgreSQL...")
            if await self.db.migrate_from_sqlite("symbols.db"):
                # Backup the SQLite file
                backup_path = f"symbols_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                sqlite_file.rename(backup_path)
                logger.info(f"SQLite file backed up to {backup_path}")
            else:
                logger.error("Failed to migrate from SQLite file")

    async def _initialize_default_symbols(self):
        """Initialize with default symbols."""
        default_symbols = [
            ("AAPL", "Apple Inc.", "Technology", "Consumer Electronics"),
            ("MSFT", "Microsoft Corporation", "Technology", "Software"),
            ("GOOGL", "Alphabet Inc.", "Technology", "Internet Content & Information"),
            ("TSLA", "Tesla Inc.", "Consumer Discretionary", "Auto Manufacturers"),
            ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "Internet Retail"),
            ("NVDA", "NVIDIA Corporation", "Technology", "Semiconductors"),
            ("META", "Meta Platforms Inc.", "Technology", "Internet Content & Information"),
            ("NFLX", "Netflix Inc.", "Communication Services", "Entertainment"),
            ("AMD", "Advanced Micro Devices Inc.", "Technology", "Semiconductors"),
            ("SPY", "SPDR S&P 500 ETF", "Financial Services", "Asset Management")
        ]
        
        for symbol, name, sector, industry in default_symbols:
            await self.add_symbol(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry,
                source=SymbolSource.MANUAL,
                status=SymbolStatus.ACTIVE
            )

    async def add_symbol(
        self,
        symbol: str,
        name: str,
        sector: str,
        industry: str,
        source: SymbolSource = SymbolSource.MANUAL,
        status: SymbolStatus = SymbolStatus.ACTIVE,
        notes: Optional[str] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        priority: int = 1,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Add a new symbol to the managed list."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Add symbol info
            symbol_info = SymbolInfo(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry
            )
            
            if not await self.db.add_symbol(symbol_info):
                logger.warning(f"Failed to add symbol info for {symbol}")
                return False
            
            # Add managed symbol
            managed_symbol = ManagedSymbol(
                symbol=symbol,
                status=status,
                source=source,
                added_date=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                notes=notes,
                target_price=target_price,
                stop_loss=stop_loss,
                priority=priority,
                tags=tags or [],
                alerts_enabled=True,
                auto_trade_enabled=False
            )
            
            success = await self.db.add_managed_symbol(managed_symbol)
            if success:
                logger.info(f"Added symbol {symbol} ({name}) to managed list")
            return success
            
        except Exception as e:
            logger.error(f"Error adding symbol {symbol}: {e}")
            return False

    async def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the managed list."""
        try:
            if not self._initialized:
                await self.initialize()
            
            success = await self.db.remove_symbol(symbol)
            if success:
                # Remove from caches
                self.performance_cache.pop(symbol, None)
                self.trading_decisions.pop(symbol, None)
                logger.info(f"Removed symbol {symbol} from managed list")
            return success
            
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False

    async def update_symbol_status(self, symbol: str, status: SymbolStatus) -> bool:
        """Update the status of a managed symbol."""
        try:
            if not self._initialized:
                await self.initialize()
            
            success = await self.db.update_symbol_status(symbol, status)
            if success:
                logger.info(f"Updated symbol {symbol} status to {status}")
            return success
            
        except Exception as e:
            logger.error(f"Error updating symbol {symbol} status: {e}")
            return False

    async def update_symbol_info(
        self,
        symbol: str,
        notes: Optional[str] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        priority: Optional[int] = None,
        tags: Optional[List[str]] = None,
        alerts_enabled: Optional[bool] = None,
        auto_trade_enabled: Optional[bool] = None
    ) -> bool:
        """Update symbol information."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get current managed symbol
            managed_symbols = await self.db.get_managed_symbols()
            current_symbol = next((s for s in managed_symbols if s.symbol == symbol), None)
            
            if not current_symbol:
                logger.warning(f"Symbol {symbol} not found")
                return False
            
            # Update fields
            if notes is not None:
                current_symbol.notes = notes
            if target_price is not None:
                current_symbol.target_price = target_price
            if stop_loss is not None:
                current_symbol.stop_loss = stop_loss
            if priority is not None:
                current_symbol.priority = priority
            if tags is not None:
                current_symbol.tags = tags
            if alerts_enabled is not None:
                current_symbol.alerts_enabled = alerts_enabled
            if auto_trade_enabled is not None:
                current_symbol.auto_trade_enabled = auto_trade_enabled
            
            current_symbol.last_updated = datetime.now().isoformat()
            
            # Save to database
            success = await self.db.add_managed_symbol(current_symbol)
            if success:
                logger.info(f"Updated symbol {symbol} information")
            return success
            
        except Exception as e:
            logger.error(f"Error updating symbol {symbol} info: {e}")
            return False

    async def get_managed_symbols(self, status: Optional[SymbolStatus] = None) -> List[ManagedSymbol]:
        """Get all managed symbols, optionally filtered by status."""
        try:
            if not self._initialized:
                await self.initialize()
            
            return await self.db.get_managed_symbols(status)
        except Exception as e:
            logger.error(f"Error getting managed symbols: {e}")
            return []

    async def get_symbol_performance(self, symbol: str) -> Optional[SymbolPerformance]:
        """Get performance data for a symbol."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get current price data
            current_data = await self.real_data_service.get_current_data(symbol)
            if not current_data:
                return None
            
            # Calculate performance metrics
            price = current_data.get('close', 0)
            volume = current_data.get('volume', 0)
            
            # Get historical data for change calculation
            historical_data = await self.real_data_service.get_historical_data(symbol, days=2)
            if historical_data and len(historical_data) >= 2:
                prev_close = historical_data.iloc[-2]['close']
                change_percent = ((price - prev_close) / prev_close) * 100
            else:
                change_percent = 0.0
            
            performance = SymbolPerformance(
                symbol=symbol,
                current_price=price,
                change_percent=change_percent,
                volume=volume,
                market_cap=None,  # Would need additional data source
                pe_ratio=None,    # Would need additional data source
                last_updated=datetime.now().isoformat(),
                performance_metrics={
                    "rsi": self._calculate_rsi(historical_data) if historical_data is not None else None,
                    "sma_20": self._calculate_sma(historical_data, 20) if historical_data is not None else None,
                    "volatility": self._calculate_volatility(historical_data) if historical_data is not None else None
                }
            )
            
            # Store in database
            await self.db.add_performance_data(symbol, {
                'price': price,
                'volume': volume,
                'change_percent': change_percent,
                'rsi': performance.performance_metrics.get('rsi'),
                'sma_20': performance.performance_metrics.get('sma_20'),
                'volatility': performance.performance_metrics.get('volatility')
            })
            
            self.performance_cache[symbol] = performance
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance for {symbol}: {e}")
            return None

    async def get_all_symbols_performance(self) -> Dict[str, SymbolPerformance]:
        """Get performance data for all managed symbols."""
        performance_data = {}
        
        managed_symbols = await self.get_managed_symbols()
        for symbol in [s.symbol for s in managed_symbols]:
            performance = await self.get_symbol_performance(symbol)
            if performance:
                performance_data[symbol] = performance
        
        return performance_data

    async def generate_trading_decision(self, symbol: str) -> Optional[TradingDecision]:
        """Generate a trading decision for a symbol."""
        try:
            if not self._initialized:
                await self.initialize()
            
            managed_symbols = await self.db.get_managed_symbols()
            managed_symbol = next((s for s in managed_symbols if s.symbol == symbol), None)
            
            if not managed_symbol:
                return None
            
            performance = await self.get_symbol_performance(symbol)
            if not performance:
                return None
            
            # Simple trading logic based on performance and targets
            action = "hold"
            confidence = 0.5
            reasoning = "No clear signal"
            
            # Check against target price
            if managed_symbol.target_price:
                if performance.current_price >= managed_symbol.target_price * 1.02:  # 2% above target
                    action = "sell"
                    confidence = 0.8
                    reasoning = f"Price reached target price of ${managed_symbol.target_price:.2f}"
                elif performance.current_price <= managed_symbol.target_price * 0.95:  # 5% below target
                    action = "buy"
                    confidence = 0.7
                    reasoning = f"Price below target price of ${managed_symbol.target_price:.2f}"
            
            # Check against stop loss
            if managed_symbol.stop_loss and performance.current_price <= managed_symbol.stop_loss:
                action = "sell"
                confidence = 0.9
                reasoning = f"Stop loss triggered at ${managed_symbol.stop_loss:.2f}"
            
            # Check performance metrics
            if performance.performance_metrics.get("rsi"):
                rsi = performance.performance_metrics["rsi"]
                if rsi < 30:  # Oversold
                    if action == "hold":
                        action = "buy"
                        confidence = 0.6
                        reasoning = "RSI indicates oversold conditions"
                elif rsi > 70:  # Overbought
                    if action == "hold":
                        action = "sell"
                        confidence = 0.6
                        reasoning = "RSI indicates overbought conditions"
            
            # Check volume
            if performance.volume > 1000000:  # High volume
                confidence = min(confidence + 0.1, 1.0)
                reasoning += " (High volume confirms signal)"
            
            decision = TradingDecision(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                target_price=managed_symbol.target_price,
                stop_loss=managed_symbol.stop_loss,
                position_size=managed_symbol.position_size,
                timeframe="short_term",
                generated_at=datetime.now().isoformat()
            )
            
            # Store in database
            await self.db.add_trading_decision(symbol, {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'target_price': managed_symbol.target_price,
                'stop_loss': managed_symbol.stop_loss,
                'position_size': managed_symbol.position_size,
                'timeframe': 'short_term'
            })
            
            self.trading_decisions[symbol] = decision
            return decision
            
        except Exception as e:
            logger.error(f"Error generating trading decision for {symbol}: {e}")
            return None

    async def get_trading_decisions(self) -> Dict[str, TradingDecision]:
        """Get trading decisions for all managed symbols."""
        decisions = {}
        
        managed_symbols = await self.get_managed_symbols()
        for symbol in [s.symbol for s in managed_symbols]:
            decision = await self.generate_trading_decision(symbol)
            if decision:
                decisions[symbol] = decision
        
        return decisions

    async def add_from_ticker_discovery(
        self,
        symbol: str,
        name: str,
        sector: str,
        industry: str,
        score: float,
        confidence: float,
        notes: Optional[str] = None
    ) -> bool:
        """Add a symbol from ticker discovery results."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Determine status based on score and confidence
            if score >= 0.8 and confidence >= 0.7:
                status = SymbolStatus.ACTIVE
                priority = 5
            elif score >= 0.6 and confidence >= 0.5:
                status = SymbolStatus.MONITORING
                priority = 3
            else:
                status = SymbolStatus.WATCHLIST
                priority = 2
            
            # Add notes about discovery
            discovery_notes = f"Added from ticker discovery - Score: {score:.2f}, Confidence: {confidence:.2f}"
            if notes:
                discovery_notes += f" | {notes}"
            
            success = await self.add_symbol(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry,
                source=SymbolSource.TICKER_DISCOVERY,
                status=status,
                notes=discovery_notes,
                priority=priority,
                tags=["ticker_discovery"]
            )
            
            if success:
                logger.info(f"Added {symbol} from ticker discovery with status {status}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding {symbol} from ticker discovery: {e}")
            return False

    async def get_manager_summary(self) -> SymbolManagerSummary:
        """Get summary of symbol manager."""
        try:
            if not self._initialized:
                await self.initialize()
            
            db_summary = await self.db.get_symbol_summary()
            
            # Get performance summary
            performance_data = await self.get_all_symbols_performance()
            performance_summary = {
                "total_symbols_with_data": len(performance_data),
                "positive_performers": len([p for p in performance_data.values() if p.change_percent > 0]),
                "negative_performers": len([p for p in performance_data.values() if p.change_percent < 0]),
                "avg_change_percent": sum(p.change_percent for p in performance_data.values()) / len(performance_data) if performance_data else 0
            }
            
            return SymbolManagerSummary(
                total_symbols=db_summary.get('total_symbols', 0),
                active_symbols=db_summary.get('active_symbols', 0),
                monitoring_symbols=db_summary.get('monitoring_symbols', 0),
                watchlist_symbols=db_summary.get('watchlist_symbols', 0),
                symbols_by_source=db_summary.get('symbols_by_source', {}),
                symbols_by_sector=db_summary.get('symbols_by_sector', {}),
                last_updated=datetime.now().isoformat(),
                performance_summary=performance_summary
            )
            
        except Exception as e:
            logger.error(f"Error getting manager summary: {e}")
            return SymbolManagerSummary(
                total_symbols=0,
                active_symbols=0,
                monitoring_symbols=0,
                watchlist_symbols=0,
                symbols_by_source={},
                symbols_by_sector={},
                last_updated=datetime.now().isoformat()
            )

    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Search for symbols by name or symbol."""
        try:
            if not self._initialized:
                await self.initialize()
            
            return await self.db.search_symbols(query)
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

    async def get_symbol_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a symbol."""
        try:
            if not self._initialized:
                await self.initialize()
            
            managed_symbols = await self.db.get_managed_symbols()
            managed_symbol = next((s for s in managed_symbols if s.symbol == symbol), None)
            
            if not managed_symbol:
                return None
            
            performance = await self.get_symbol_performance(symbol)
            decision = await self.generate_trading_decision(symbol)
            
            return {
                "managed_symbol": asdict(managed_symbol),
                "performance": asdict(performance) if performance else None,
                "trading_decision": asdict(decision) if decision else None
            }
            
        except Exception as e:
            logger.error(f"Error getting details for {symbol}: {e}")
            return None

    def _calculate_rsi(self, data, period: int = 14) -> Optional[float]:
        """Calculate RSI for the given data."""
        if data is None or len(data) < period + 1:
            return None
        
        try:
            closes = data['close'].values
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception:
            return None

    def _calculate_sma(self, data, period: int) -> Optional[float]:
        """Calculate Simple Moving Average for the given data."""
        if data is None or len(data) < period:
            return None
        
        try:
            return data['close'].tail(period).mean()
        except Exception:
            return None

    def _calculate_volatility(self, data, period: int = 20) -> Optional[float]:
        """Calculate volatility for the given data."""
        if data is None or len(data) < period:
            return None
        
        try:
            returns = data['close'].pct_change().dropna()
            return returns.tail(period).std() * (252 ** 0.5)  # Annualized volatility
        except Exception:
            return None

    async def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        return await self.db.backup_database(backup_path)
