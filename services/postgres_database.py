import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
from contextlib import asynccontextmanager
import os
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

class PostgreSQLDatabase:
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "ai_market_system",
                 user: str = "postgres",
                 password: str = "password"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection_pool = None
        logger.info(f"PostgreSQLDatabase initialized for {user}@{host}:{port}/{database}")

    async def initialize(self):
        """Initialize the database connection pool and create tables."""
        try:
            # Create connection pool
            self.connection_pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables
            await self._create_tables()
            logger.info("PostgreSQL database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise

    async def close(self):
        """Close the database connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("PostgreSQL connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self.connection_pool:
            raise RuntimeError("Database not initialized")
        
        async with self.connection_pool.acquire() as connection:
            yield connection

    async def _create_tables(self):
        """Create all required tables."""
        async with self.get_connection() as conn:
            # Create symbols table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS symbols (
                    symbol VARCHAR(20) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    sector VARCHAR(100) NOT NULL,
                    industry VARCHAR(100) NOT NULL,
                    market_cap BIGINT,
                    price DECIMAL(10,2),
                    volume BIGINT,
                    description TEXT,
                    exchange VARCHAR(20) DEFAULT 'NASDAQ',
                    currency VARCHAR(3) DEFAULT 'USD',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create managed_symbols table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS managed_symbols (
                    symbol VARCHAR(20) PRIMARY KEY,
                    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'monitoring', 'inactive', 'watchlist')),
                    source VARCHAR(20) NOT NULL CHECK (source IN ('manual', 'ticker_discovery', 'portfolio', 'recommendation')),
                    added_date TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP NOT NULL,
                    notes TEXT,
                    target_price DECIMAL(10,2),
                    stop_loss DECIMAL(10,2),
                    position_size DECIMAL(15,2),
                    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
                    tags JSONB DEFAULT '[]'::jsonb,
                    alerts_enabled BOOLEAN DEFAULT TRUE,
                    auto_trade_enabled BOOLEAN DEFAULT FALSE,
                    CONSTRAINT fk_managed_symbols_symbol 
                        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
                )
            ''')
            
            # Create symbol_performance table for historical data
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS symbol_performance (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    price DECIMAL(10,2) NOT NULL,
                    volume BIGINT,
                    change_percent DECIMAL(8,4),
                    rsi DECIMAL(5,2),
                    sma_20 DECIMAL(10,2),
                    volatility DECIMAL(8,4),
                    market_cap BIGINT,
                    pe_ratio DECIMAL(8,2),
                    CONSTRAINT fk_symbol_performance_symbol 
                        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
                )
            ''')
            
            # Create trading_decisions table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_decisions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    action VARCHAR(10) NOT NULL CHECK (action IN ('buy', 'sell', 'hold', 'watch')),
                    confidence DECIMAL(3,2) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    reasoning TEXT NOT NULL,
                    target_price DECIMAL(10,2),
                    stop_loss DECIMAL(10,2),
                    position_size DECIMAL(15,2),
                    timeframe VARCHAR(20) DEFAULT 'short_term',
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_trading_decisions_symbol 
                        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better performance
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_managed_symbols_status ON managed_symbols(status)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_managed_symbols_source ON managed_symbols(source)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_managed_symbols_priority ON managed_symbols(priority)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_performance_symbol ON symbol_performance(symbol)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_performance_timestamp ON symbol_performance(timestamp)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_trading_decisions_symbol ON trading_decisions(symbol)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_trading_decisions_generated_at ON trading_decisions(generated_at)')
            
            # Create GIN index for JSONB tags
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_managed_symbols_tags ON managed_symbols USING GIN (tags)')
            
            logger.info("PostgreSQL tables and indexes created successfully")

    async def add_symbol(self, symbol_info: SymbolInfo) -> bool:
        """Add a new symbol to the database."""
        try:
            async with self.get_connection() as conn:
                await conn.execute('''
                    INSERT INTO symbols 
                    (symbol, name, sector, industry, market_cap, price, volume, description, exchange, currency)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (symbol) DO UPDATE SET
                        name = EXCLUDED.name,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        market_cap = EXCLUDED.market_cap,
                        price = EXCLUDED.price,
                        volume = EXCLUDED.volume,
                        description = EXCLUDED.description,
                        exchange = EXCLUDED.exchange,
                        currency = EXCLUDED.currency,
                        updated_at = CURRENT_TIMESTAMP
                ''', (
                    symbol_info.symbol, symbol_info.name, symbol_info.sector, symbol_info.industry,
                    symbol_info.market_cap, symbol_info.price, symbol_info.volume, symbol_info.description,
                    symbol_info.exchange, symbol_info.currency
                ))
                logger.info(f"Added symbol {symbol_info.symbol} to PostgreSQL database")
                return True
        except Exception as e:
            logger.error(f"Error adding symbol {symbol_info.symbol}: {e}")
            return False

    async def add_managed_symbol(self, managed_symbol: ManagedSymbol) -> bool:
        """Add a symbol to managed symbols."""
        try:
            async with self.get_connection() as conn:
                await conn.execute('''
                    INSERT INTO managed_symbols 
                    (symbol, status, source, added_date, last_updated, notes, target_price, stop_loss, 
                     position_size, priority, tags, alerts_enabled, auto_trade_enabled)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (symbol) DO UPDATE SET
                        status = EXCLUDED.status,
                        source = EXCLUDED.source,
                        last_updated = EXCLUDED.last_updated,
                        notes = EXCLUDED.notes,
                        target_price = EXCLUDED.target_price,
                        stop_loss = EXCLUDED.stop_loss,
                        position_size = EXCLUDED.position_size,
                        priority = EXCLUDED.priority,
                        tags = EXCLUDED.tags,
                        alerts_enabled = EXCLUDED.alerts_enabled,
                        auto_trade_enabled = EXCLUDED.auto_trade_enabled
                ''', (
                    managed_symbol.symbol, managed_symbol.status.value, managed_symbol.source.value,
                    managed_symbol.added_date, managed_symbol.last_updated, managed_symbol.notes,
                    managed_symbol.target_price, managed_symbol.stop_loss, managed_symbol.position_size,
                    managed_symbol.priority, json.dumps(managed_symbol.tags), 
                    managed_symbol.alerts_enabled, managed_symbol.auto_trade_enabled
                ))
                logger.info(f"Added managed symbol {managed_symbol.symbol} to PostgreSQL")
                return True
        except Exception as e:
            logger.error(f"Error adding managed symbol {managed_symbol.symbol}: {e}")
            return False

    async def get_managed_symbols(self, status: Optional[SymbolStatus] = None) -> List[ManagedSymbol]:
        """Get all managed symbols, optionally filtered by status."""
        try:
            async with self.get_connection() as conn:
                if status:
                    rows = await conn.fetch('''
                        SELECT ms.*, s.name, s.sector, s.industry 
                        FROM managed_symbols ms
                        JOIN symbols s ON ms.symbol = s.symbol
                        WHERE ms.status = $1
                        ORDER BY ms.priority DESC, ms.symbol
                    ''', status.value)
                else:
                    rows = await conn.fetch('''
                        SELECT ms.*, s.name, s.sector, s.industry 
                        FROM managed_symbols ms
                        JOIN symbols s ON ms.symbol = s.symbol
                        ORDER BY ms.priority DESC, ms.symbol
                    ''')
                
                symbols = []
                for row in rows:
                    tags = row['tags'] if row['tags'] else []
                    symbol = ManagedSymbol(
                        symbol=row['symbol'],
                        status=SymbolStatus(row['status']),
                        source=SymbolSource(row['source']),
                        added_date=row['added_date'].isoformat(),
                        last_updated=row['last_updated'].isoformat(),
                        notes=row['notes'],
                        target_price=float(row['target_price']) if row['target_price'] else None,
                        stop_loss=float(row['stop_loss']) if row['stop_loss'] else None,
                        position_size=float(row['position_size']) if row['position_size'] else None,
                        priority=row['priority'],
                        tags=tags,
                        alerts_enabled=row['alerts_enabled'],
                        auto_trade_enabled=row['auto_trade_enabled']
                    )
                    symbols.append(symbol)
                
                return symbols
        except Exception as e:
            logger.error(f"Error getting managed symbols: {e}")
            return []

    async def update_symbol_status(self, symbol: str, status: SymbolStatus) -> bool:
        """Update symbol status."""
        try:
            async with self.get_connection() as conn:
                result = await conn.execute('''
                    UPDATE managed_symbols 
                    SET status = $1, last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = $2
                ''', status.value, symbol)
                
                if result == "UPDATE 1":
                    logger.info(f"Updated symbol {symbol} status to {status.value}")
                    return True
                else:
                    logger.warning(f"Symbol {symbol} not found")
                    return False
        except Exception as e:
            logger.error(f"Error updating symbol {symbol} status: {e}")
            return False

    async def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from managed symbols."""
        try:
            async with self.get_connection() as conn:
                result = await conn.execute('DELETE FROM managed_symbols WHERE symbol = $1', symbol)
                
                if result == "DELETE 1":
                    logger.info(f"Removed symbol {symbol} from managed symbols")
                    return True
                else:
                    logger.warning(f"Symbol {symbol} not found in managed symbols")
                    return False
        except Exception as e:
            logger.error(f"Error removing symbol {symbol}: {e}")
            return False

    async def add_performance_data(self, symbol: str, performance_data: Dict[str, Any]) -> bool:
        """Add performance data for a symbol."""
        try:
            async with self.get_connection() as conn:
                await conn.execute('''
                    INSERT INTO symbol_performance 
                    (symbol, price, volume, change_percent, rsi, sma_20, volatility, market_cap, pe_ratio)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''', (
                    symbol, performance_data.get('price'), performance_data.get('volume'),
                    performance_data.get('change_percent'), performance_data.get('rsi'),
                    performance_data.get('sma_20'), performance_data.get('volatility'),
                    performance_data.get('market_cap'), performance_data.get('pe_ratio')
                ))
                return True
        except Exception as e:
            logger.error(f"Error adding performance data for {symbol}: {e}")
            return False

    async def get_latest_performance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest performance data for a symbol."""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT * FROM symbol_performance 
                    WHERE symbol = $1 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', symbol)
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting performance data for {symbol}: {e}")
            return None

    async def add_trading_decision(self, symbol: str, decision_data: Dict[str, Any]) -> bool:
        """Add a trading decision for a symbol."""
        try:
            async with self.get_connection() as conn:
                await conn.execute('''
                    INSERT INTO trading_decisions 
                    (symbol, action, confidence, reasoning, target_price, stop_loss, position_size, timeframe)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''', (
                    symbol, decision_data.get('action'), decision_data.get('confidence'),
                    decision_data.get('reasoning'), decision_data.get('target_price'),
                    decision_data.get('stop_loss'), decision_data.get('position_size'),
                    decision_data.get('timeframe', 'short_term')
                ))
                return True
        except Exception as e:
            logger.error(f"Error adding trading decision for {symbol}: {e}")
            return False

    async def get_latest_trading_decision(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest trading decision for a symbol."""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow('''
                    SELECT * FROM trading_decisions 
                    WHERE symbol = $1 
                    ORDER BY generated_at DESC 
                    LIMIT 1
                ''', symbol)
                
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Error getting trading decision for {symbol}: {e}")
            return None

    async def get_symbol_summary(self) -> Dict[str, Any]:
        """Get summary statistics for managed symbols."""
        try:
            async with self.get_connection() as conn:
                # Total symbols
                total_symbols = await conn.fetchval('SELECT COUNT(*) FROM managed_symbols')
                
                # Symbols by status
                status_rows = await conn.fetch('''
                    SELECT status, COUNT(*) 
                    FROM managed_symbols 
                    GROUP BY status
                ''')
                status_counts = {row['status']: row['count'] for row in status_rows}
                
                # Symbols by source
                source_rows = await conn.fetch('''
                    SELECT source, COUNT(*) 
                    FROM managed_symbols 
                    GROUP BY source
                ''')
                source_counts = {row['source']: row['count'] for row in source_rows}
                
                # Symbols by sector
                sector_rows = await conn.fetch('''
                    SELECT s.sector, COUNT(*) 
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    GROUP BY s.sector
                ''')
                sector_counts = {row['sector']: row['count'] for row in sector_rows}
                
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

    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Search symbols by name, symbol, sector, or industry."""
        try:
            async with self.get_connection() as conn:
                search_term = f"%{query.lower()}%"
                rows = await conn.fetch('''
                    SELECT s.symbol, s.name, s.sector, s.industry,
                           CASE WHEN ms.symbol IS NOT NULL THEN TRUE ELSE FALSE END as is_managed,
                           ms.status
                    FROM symbols s
                    LEFT JOIN managed_symbols ms ON s.symbol = ms.symbol
                    WHERE LOWER(s.symbol) LIKE $1 
                       OR LOWER(s.name) LIKE $1 
                       OR LOWER(s.sector) LIKE $1 
                       OR LOWER(s.industry) LIKE $1
                    ORDER BY is_managed DESC, s.symbol
                    LIMIT 20
                ''', search_term)
                
                results = []
                for row in rows:
                    results.append({
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'sector': row['sector'],
                        'industry': row['industry'],
                        'is_managed': row['is_managed'],
                        'status': row['status'] if row['is_managed'] else None
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

    async def migrate_from_json(self, json_file_path: str) -> bool:
        """Migrate data from JSON file to PostgreSQL."""
        try:
            json_path = Path(json_file_path)
            if not json_path.exists():
                logger.warning(f"JSON file {json_file_path} not found")
                return False
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            migrated_count = 0
            for symbol, symbol_data in data.items():
                # Create symbol info with default values if not present
                symbol_info = SymbolInfo(
                    symbol=symbol,
                    name=symbol_data.get('name', f'{symbol} Corporation'),
                    sector=symbol_data.get('sector', 'Technology'),
                    industry=symbol_data.get('industry', 'Software')
                )
                
                if await self.add_symbol(symbol_info):
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
                    
                    if await self.add_managed_symbol(managed_symbol):
                        migrated_count += 1
            
            logger.info(f"Successfully migrated {migrated_count} symbols from JSON to PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")
            return False

    async def migrate_from_sqlite(self, sqlite_file_path: str) -> bool:
        """Migrate data from SQLite database to PostgreSQL."""
        try:
            import sqlite3
            import aiosqlite
            
            if not Path(sqlite_file_path).exists():
                logger.warning(f"SQLite file {sqlite_file_path} not found")
                return False
            
            migrated_count = 0
            
            # Migrate symbols
            async with aiosqlite.connect(sqlite_file_path) as sqlite_conn:
                # Get symbols
                async with sqlite_conn.execute('SELECT * FROM symbols') as cursor:
                    symbols = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    for symbol_row in symbols:
                        symbol_dict = dict(zip(columns, symbol_row))
                        symbol_info = SymbolInfo(
                            symbol=symbol_dict['symbol'],
                            name=symbol_dict['name'],
                            sector=symbol_dict['sector'],
                            industry=symbol_dict['industry'],
                            market_cap=symbol_dict.get('market_cap'),
                            price=symbol_dict.get('price'),
                            volume=symbol_dict.get('volume'),
                            description=symbol_dict.get('description'),
                            exchange=symbol_dict.get('exchange', 'NASDAQ'),
                            currency=symbol_dict.get('currency', 'USD')
                        )
                        
                        if await self.add_symbol(symbol_info):
                            migrated_count += 1
                
                # Get managed symbols
                async with sqlite_conn.execute('SELECT * FROM managed_symbols') as cursor:
                    managed_symbols = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    
                    for managed_row in managed_symbols:
                        managed_dict = dict(zip(columns, managed_row))
                        tags = json.loads(managed_dict.get('tags', '[]'))
                        
                        managed_symbol = ManagedSymbol(
                            symbol=managed_dict['symbol'],
                            status=SymbolStatus(managed_dict['status']),
                            source=SymbolSource(managed_dict['source']),
                            added_date=managed_dict['added_date'],
                            last_updated=managed_dict['last_updated'],
                            notes=managed_dict.get('notes'),
                            target_price=managed_dict.get('target_price'),
                            stop_loss=managed_dict.get('stop_loss'),
                            position_size=managed_dict.get('position_size'),
                            priority=managed_dict.get('priority', 1),
                            tags=tags,
                            alerts_enabled=bool(managed_dict.get('alerts_enabled', True)),
                            auto_trade_enabled=bool(managed_dict.get('auto_trade_enabled', False))
                        )
                        
                        if await self.add_managed_symbol(managed_symbol):
                            migrated_count += 1
            
            logger.info(f"Successfully migrated {migrated_count} records from SQLite to PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating from SQLite: {e}")
            return False

    async def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database using pg_dump."""
        try:
            import subprocess
            import os
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.password
            
            # Run pg_dump
            cmd = [
                'pg_dump',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '-f', backup_path,
                '--verbose'
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"PostgreSQL database backed up to {backup_path}")
                return True
            else:
                logger.error(f"Backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
