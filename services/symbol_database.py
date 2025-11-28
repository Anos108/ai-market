import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SymbolStatus(str, Enum):
    ACTIVE = "active"
    MONITORING = "monitoring"
    INACTIVE = "inactive"
    WATCHLIST = "watchlist"

class SymbolSource(str, Enum):
    MANUAL = "manual"
    TICKER_DISCOVERY = "ticker_discovery"
    PORTFOLIO = "portfolio"
    RECOMMENDATION = "recommendation"

@dataclass
class SymbolInfo:
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    price: Optional[float] = None
    volume: Optional[int] = None
    description: Optional[str] = None
    exchange: str = "NASDAQ"
    currency: str = "USD"

@dataclass
class ManagedSymbol:
    symbol: str
    status: SymbolStatus
    source: SymbolSource
    added_date: str
    last_updated: str
    notes: Optional[str] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    priority: int = 1
    tags: List[str] = None
    alerts_enabled: bool = True
    auto_trade_enabled: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class SymbolDatabase:
    def __init__(self, db_path: str = "symbols.db"):
        self.db_path = Path(db_path)
        self.init_database()
        logger.info(f"SymbolDatabase initialized with SQLite at {self.db_path}")

    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create symbols table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    industry TEXT NOT NULL,
                    market_cap REAL,
                    price REAL,
                    volume INTEGER,
                    description TEXT,
                    exchange TEXT DEFAULT 'NASDAQ',
                    currency TEXT DEFAULT 'USD',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create managed_symbols table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS managed_symbols (
                    symbol TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('active', 'monitoring', 'inactive', 'watchlist')),
                    source TEXT NOT NULL CHECK (source IN ('manual', 'ticker_discovery', 'portfolio', 'recommendation')),
                    added_date TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    notes TEXT,
                    target_price REAL,
                    stop_loss REAL,
                    position_size REAL,
                    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
                    tags TEXT,  -- JSON array of tags
                    alerts_enabled BOOLEAN DEFAULT 1,
                    auto_trade_enabled BOOLEAN DEFAULT 0,
                    FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
                )
            ''')
            
            # Create symbol_performance table for historical data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS symbol_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    price REAL NOT NULL,
                    volume INTEGER,
                    change_percent REAL,
                    rsi REAL,
                    sma_20 REAL,
                    volatility REAL,
                    market_cap REAL,
                    pe_ratio REAL,
                    FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
                )
            ''')
            
            # Create trading_decisions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL CHECK (action IN ('buy', 'sell', 'hold', 'watch')),
                    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    reasoning TEXT NOT NULL,
                    target_price REAL,
                    stop_loss REAL,
                    position_size REAL,
                    timeframe TEXT DEFAULT 'short_term',
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_managed_symbols_status ON managed_symbols(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_managed_symbols_source ON managed_symbols(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_managed_symbols_priority ON managed_symbols(priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_performance_symbol ON symbol_performance(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_performance_timestamp ON symbol_performance(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_decisions_symbol ON trading_decisions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trading_decisions_generated_at ON trading_decisions(generated_at)')
            
            conn.commit()
            logger.info("Database tables and indexes created successfully")

    def add_symbol(self, symbol_info: SymbolInfo) -> bool:
        """Add a new symbol to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO symbols 
                    (symbol, name, sector, industry, market_cap, price, volume, description, exchange, currency, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    symbol_info.symbol, symbol_info.name, symbol_info.sector, symbol_info.industry,
                    symbol_info.market_cap, symbol_info.price, symbol_info.volume, symbol_info.description,
                    symbol_info.exchange, symbol_info.currency
                ))
                conn.commit()
                logger.info(f"Added symbol {symbol_info.symbol} to database")
                return True
        except Exception as e:
            logger.error(f"Error adding symbol {symbol_info.symbol}: {e}")
            return False

    def add_managed_symbol(self, managed_symbol: ManagedSymbol) -> bool:
        """Add a symbol to managed symbols."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO managed_symbols 
                    (symbol, status, source, added_date, last_updated, notes, target_price, stop_loss, 
                     position_size, priority, tags, alerts_enabled, auto_trade_enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    managed_symbol.symbol, managed_symbol.status.value, managed_symbol.source.value,
                    managed_symbol.added_date, managed_symbol.last_updated, managed_symbol.notes,
                    managed_symbol.target_price, managed_symbol.stop_loss, managed_symbol.position_size,
                    managed_symbol.priority, json.dumps(managed_symbol.tags), 
                    managed_symbol.alerts_enabled, managed_symbol.auto_trade_enabled
                ))
                conn.commit()
                logger.info(f"Added managed symbol {managed_symbol.symbol}")
                return True
        except Exception as e:
            logger.error(f"Error adding managed symbol {managed_symbol.symbol}: {e}")
            return False

    def get_managed_symbols(self, status: Optional[SymbolStatus] = None) -> List[ManagedSymbol]:
        """Get all managed symbols, optionally filtered by status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if status:
                    cursor.execute('''
                        SELECT ms.*, s.name, s.sector, s.industry 
                        FROM managed_symbols ms
                        JOIN symbols s ON ms.symbol = s.symbol
                        WHERE ms.status = ?
                        ORDER BY ms.priority DESC, ms.symbol
                    ''', (status.value,))
                else:
                    cursor.execute('''
                        SELECT ms.*, s.name, s.sector, s.industry 
                        FROM managed_symbols ms
                        JOIN symbols s ON ms.symbol = s.symbol
                        ORDER BY ms.priority DESC, ms.symbol
                    ''')
                
                symbols = []
                for row in cursor.fetchall():
                    tags = json.loads(row['tags']) if row['tags'] else []
                    symbol = ManagedSymbol(
                        symbol=row['symbol'],
                        status=SymbolStatus(row['status']),
                        source=SymbolSource(row['source']),
                        added_date=row['added_date'],
                        last_updated=row['last_updated'],
                        notes=row['notes'],
                        target_price=row['target_price'],
                        stop_loss=row['stop_loss'],
                        position_size=row['position_size'],
                        priority=row['priority'],
                        tags=tags,
                        alerts_enabled=bool(row['alerts_enabled']),
                        auto_trade_enabled=bool(row['auto_trade_enabled'])
                    )
                    symbols.append(symbol)
                
                return symbols
        except Exception as e:
            logger.error(f"Error getting managed symbols: {e}")
            return []

    def update_symbol_status(self, symbol: str, status: SymbolStatus) -> bool:
        """Update symbol status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE managed_symbols 
                    SET status = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = ?
                ''', (status.value, symbol))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Updated symbol {symbol} status to {status.value}")
                    return True
                else:
                    logger.warning(f"Symbol {symbol} not found")
                    return False
        except Exception as e:
            logger.error(f"Error updating symbol {symbol} status: {e}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from managed symbols."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM managed_symbols WHERE symbol = ?', (symbol,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Removed symbol {symbol} from managed symbols")
                    return True
                else:
                    logger.warning(f"Symbol {symbol} not found in managed symbols")
                    return False
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False

    def add_performance_data(self, symbol: str, performance_data: Dict[str, Any]) -> bool:
        """Add performance data for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO symbol_performance 
                    (symbol, price, volume, change_percent, rsi, sma_20, volatility, market_cap, pe_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, performance_data.get('price'), performance_data.get('volume'),
                    performance_data.get('change_percent'), performance_data.get('rsi'),
                    performance_data.get('sma_20'), performance_data.get('volatility'),
                    performance_data.get('market_cap'), performance_data.get('pe_ratio')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding performance data for {symbol}: {e}")
            return False

    def get_latest_performance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest performance data for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM symbol_performance 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (symbol,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting performance data for {symbol}: {e}")
            return None

    def add_trading_decision(self, symbol: str, decision_data: Dict[str, Any]) -> bool:
        """Add a trading decision for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_decisions 
                    (symbol, action, confidence, reasoning, target_price, stop_loss, position_size, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, decision_data.get('action'), decision_data.get('confidence'),
                    decision_data.get('reasoning'), decision_data.get('target_price'),
                    decision_data.get('stop_loss'), decision_data.get('position_size'),
                    decision_data.get('timeframe', 'short_term')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding trading decision for {symbol}: {e}")
            return False

    def get_latest_trading_decision(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest trading decision for a symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM trading_decisions 
                    WHERE symbol = ? 
                    ORDER BY generated_at DESC 
                    LIMIT 1
                ''', (symbol,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting trading decision for {symbol}: {e}")
            return None

    def get_symbol_summary(self) -> Dict[str, Any]:
        """Get summary statistics for managed symbols."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total symbols
                cursor.execute('SELECT COUNT(*) FROM managed_symbols')
                total_symbols = cursor.fetchone()[0]
                
                # Symbols by status
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM managed_symbols 
                    GROUP BY status
                ''')
                status_counts = dict(cursor.fetchall())
                
                # Symbols by source
                cursor.execute('''
                    SELECT source, COUNT(*) 
                    FROM managed_symbols 
                    GROUP BY source
                ''')
                source_counts = dict(cursor.fetchall())
                
                # Symbols by sector
                cursor.execute('''
                    SELECT s.sector, COUNT(*) 
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    GROUP BY s.sector
                ''')
                sector_counts = dict(cursor.fetchall())
                
                return {
                    'total_symbols': total_symbols,
                    'active_symbols': status_counts.get('active', 0),
                    'monitoring_symbols': status_counts.get('monitoring', 0),
                    'watchlist_symbols': status_counts.get('watchlist', 0),
                    'inactive_symbols': status_counts.get('inactive', 0),
                    'symbols_by_source': source_counts,
                    'symbols_by_sector': sector_counts,
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting symbol summary: {e}")
            return {}

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Search symbols by name, symbol, sector, or industry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                search_term = f"%{query.lower()}%"
                cursor.execute('''
                    SELECT s.symbol, s.name, s.sector, s.industry,
                           CASE WHEN ms.symbol IS NOT NULL THEN 1 ELSE 0 END as is_managed,
                           ms.status
                    FROM symbols s
                    LEFT JOIN managed_symbols ms ON s.symbol = ms.symbol
                    WHERE LOWER(s.symbol) LIKE ? 
                       OR LOWER(s.name) LIKE ? 
                       OR LOWER(s.sector) LIKE ? 
                       OR LOWER(s.industry) LIKE ?
                    ORDER BY is_managed DESC, s.symbol
                    LIMIT 20
                ''', (search_term, search_term, search_term, search_term))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'sector': row['sector'],
                        'industry': row['industry'],
                        'is_managed': bool(row['is_managed']),
                        'status': row['status'] if row['is_managed'] else None
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

    def migrate_from_json(self, json_file_path: str) -> bool:
        """Migrate data from JSON file to database."""
        try:
            json_path = Path(json_file_path)
            if not json_path.exists():
                logger.warning(f"JSON file {json_file_path} not found")
                return False
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            migrated_count = 0
            for symbol, symbol_data in data.items():
                # Create symbol info
                symbol_info = SymbolInfo(
                    symbol=symbol,
                    name=symbol_data.get('name', ''),
                    sector=symbol_data.get('sector', ''),
                    industry=symbol_data.get('industry', '')
                )
                
                if self.add_symbol(symbol_info):
                    # Create managed symbol
                    managed_symbol = ManagedSymbol(
                        symbol=symbol,
                        status=SymbolStatus(symbol_data.get('status', 'active')),
                        source=SymbolSource(symbol_data.get('source', 'manual')),
                        added_date=symbol_data.get('added_date', datetime.now().isoformat()),
                        last_updated=symbol_data.get('last_updated', datetime.now().isoformat()),
                        notes=symbol_data.get('notes'),
                        target_price=symbol_data.get('target_price'),
                        stop_loss=symbol_data.get('stop_loss'),
                        position_size=symbol_data.get('position_size'),
                        priority=symbol_data.get('priority', 1),
                        tags=symbol_data.get('tags', []),
                        alerts_enabled=symbol_data.get('alerts_enabled', True),
                        auto_trade_enabled=symbol_data.get('auto_trade_enabled', False)
                    )
                    
                    if self.add_managed_symbol(managed_symbol):
                        migrated_count += 1
            
            logger.info(f"Successfully migrated {migrated_count} symbols from JSON to database")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")
            return False

    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
