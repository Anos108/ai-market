"""
Execution Agent Service for real order management and execution tracking.
Provides comprehensive order management, position tracking, and execution strategy monitoring.
"""

import asyncio
import asyncpg
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)

class ExecutionAgentService:
    """Service for managing execution agent data and order processing."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self._initialize_database()

        self.mt5_service = None
        if os.getenv('USE_MT5', 'false').lower() == 'true':
            try:
                from services.mt5_service import MT5Service
                self.mt5_service = MT5Service()
                logger.info("MT5 Execution enabled in ExecutionAgentService")
            except ImportError:
                logger.warning("MT5Service could not be imported")
    
    def _initialize_database(self):
        """Initialize execution agent database tables."""
        # Tables are created by init.sql, this is for any additional setup
        pass
    
    async def get_execution_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution agent summary with real data."""
        try:
            async with self.db_pool.acquire() as conn:
                # Get order statistics
                order_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_orders,
                        COUNT(CASE WHEN status IN ('pending', 'submitted') THEN 1 END) as active_orders,
                        COUNT(CASE WHEN status = 'filled' THEN 1 END) as filled_orders,
                        COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_orders,
                        COALESCE(SUM(commission), 0) as total_commission,
                        COALESCE(AVG(execution_time_seconds), 0) as avg_execution_time
                    FROM execution_orders
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                
                # Get position statistics
                position_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_positions,
                        COALESCE(SUM(market_value), 0) as total_market_value,
                        COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl
                    FROM execution_positions
                """)
                
                # Get strategy statistics
                strategy_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_strategies,
                        COUNT(*) as total_strategies
                    FROM execution_strategies
                """)
                
                # Calculate success rate
                success_rate = 0.0
                if order_stats['total_orders'] > 0:
                    success_rate = (order_stats['filled_orders'] / order_stats['total_orders'])
                
                return {
                    "total_orders": order_stats['total_orders'] or 0,
                    "active_orders": order_stats['active_orders'] or 0,
                    "filled_orders": order_stats['filled_orders'] or 0,
                    "cancelled_orders": order_stats['cancelled_orders'] or 0,
                    "total_volume": position_stats['total_market_value'] or 0.0,
                    "total_commission": float(order_stats['total_commission'] or 0.0),
                    "avg_execution_time": float(order_stats['avg_execution_time'] or 0.0),
                    "execution_success_rate": round(success_rate, 3),
                    "active_strategies": strategy_stats['active_strategies'] or 0,
                    "total_positions": position_stats['total_positions'] or 0,
                    "total_market_value": float(position_stats['total_market_value'] or 0.0),
                    "total_unrealized_pnl": float(position_stats['total_unrealized_pnl'] or 0.0),
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting execution agent summary: {e}")
            # Return fallback data
            return self._get_fallback_summary()
    
    async def get_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent orders with real data."""
        # If MT5 is enabled, fetch from MT5 first (or merge)
        mt5_orders = []
        if self.mt5_service:
            try:
                mt5_orders_raw = await self.mt5_service.get_orders()
                # Format MT5 orders to match our schema
                for order in mt5_orders_raw:
                     # MT5 order structure needs mapping
                     mt5_orders.append({
                        "order_id": str(order.get('ticket')),
                        "symbol": order.get('symbol'),
                        "order_type": "limit" if "LIMIT" in str(order.get('type')) else "market", # simplified
                        "side": "buy" if order.get('type') in [0, 2, 4] else "sell", # 0=buy, 1=sell
                        "quantity": float(order.get('volume_initial', 0)),
                        "price": float(order.get('price_open', 0)),
                        "status": "pending", # Active orders in MT5 are pending
                        "strategy": "Manual/MT5",
                        "commission": 0.0,
                        "execution_time_seconds": 0,
                        "created_at": datetime.fromtimestamp(order.get('time_setup', 0)).isoformat(),
                        "filled_at": None
                     })
            except Exception as e:
                logger.error(f"Error fetching MT5 orders: {e}")

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        order_id,
                        symbol,
                        order_type,
                        side,
                        quantity,
                        price,
                        status,
                        strategy,
                        commission,
                        execution_time_seconds,
                        created_at,
                        filled_at
                    FROM execution_orders
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit)
                
                orders = []
                for row in rows:
                    orders.append({
                        "order_id": row['order_id'],
                        "symbol": row['symbol'],
                        "order_type": row['order_type'],
                        "side": row['side'],
                        "quantity": float(row['quantity']),
                        "price": float(row['price']) if row['price'] else None,
                        "status": row['status'],
                        "strategy": row['strategy'],
                        "commission": float(row['commission']),
                        "execution_time_seconds": row['execution_time_seconds'],
                        "created_at": row['created_at'].isoformat(),
                        "filled_at": row['filled_at'].isoformat() if row['filled_at'] else None
                    })
                
                # Combine MT5 orders with DB orders, avoiding duplicates
                # Use a dictionary keyed by order_id to merge
                all_orders_map = {o['order_id']: o for o in orders}

                # Merge MT5 orders (they take precedence if we want real-time status,
                # but DB might have more metadata like 'strategy' name if we saved it)
                for mt5_order in mt5_orders:
                    if mt5_order['order_id'] in all_orders_map:
                        # Update status from MT5
                        all_orders_map[mt5_order['order_id']]['status'] = mt5_order['status']
                    else:
                        all_orders_map[mt5_order['order_id']] = mt5_order

                return list(all_orders_map.values())
                
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return mt5_orders # Return at least MT5 orders if DB fails
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions with real data."""
        mt5_positions = []
        if self.mt5_service:
            try:
                raw_positions = await self.mt5_service.get_positions()
                for pos in raw_positions:
                    mt5_positions.append({
                        "symbol": pos.get('symbol'),
                        "quantity": float(pos.get('volume', 0)),
                        "average_price": float(pos.get('price_open', 0)),
                        "market_value": float(pos.get('volume', 0) * pos.get('price_current', 0)),
                        "unrealized_pnl": float(pos.get('profit', 0)),
                        "realized_pnl": 0.0, # Not readily available in positions_get
                        "total_commission": 0.0,
                        "position_type": "long" if pos.get('type') == 0 else "short",
                        "strategy": "MT5",
                        "created_at": datetime.fromtimestamp(pos.get('time', 0)).isoformat(),
                        "updated_at": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error fetching MT5 positions: {e}")

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        symbol,
                        quantity,
                        average_price,
                        market_value,
                        unrealized_pnl,
                        realized_pnl,
                        total_commission,
                        position_type,
                        strategy,
                        created_at,
                        updated_at
                    FROM execution_positions
                    ORDER BY market_value DESC
                """)
                
                positions = []
                for row in rows:
                    positions.append({
                        "symbol": row['symbol'],
                        "quantity": float(row['quantity']),
                        "average_price": float(row['average_price']),
                        "market_value": float(row['market_value']),
                        "unrealized_pnl": float(row['unrealized_pnl']),
                        "realized_pnl": float(row['realized_pnl']),
                        "total_commission": float(row['total_commission']),
                        "position_type": row['position_type'],
                        "strategy": row['strategy'],
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat()
                    })
                
                # If using MT5, we might want to prefer MT5 positions over DB ones or show both
                # For now, let's append them. Duplication might occur if DB syncs with MT5 separately.
                return mt5_positions + positions
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return mt5_positions

    async def execute_order(self, symbol: str, action: str, quantity: float,
                           order_type: str = "market", price: Optional[float] = None,
                           sl: Optional[float] = None, tp: Optional[float] = None,
                           strategy: str = "Manual") -> Dict[str, Any]:
        """Execute a trade order."""

        # 1. Execute on MT5 if enabled
        mt5_result = None
        status = "pending"
        filled_at = None
        execution_time = None

        if self.mt5_service:
            try:
                # Map internal action to MT5 action types
                mt5_action = action.lower()
                if order_type.lower() == 'limit':
                    mt5_action = f"{action}_limit"
                elif order_type.lower() == 'stop':
                     mt5_action = f"{action}_stop"

                res = await self.mt5_service.execute_order(
                    symbol=symbol,
                    action_type=mt5_action,
                    volume=quantity,
                    price=price,
                    sl=sl,
                    tp=tp,
                    comment=strategy
                )

                mt5_result = res
                if res.get('retcode') == 10009: # TRADE_RETCODE_DONE
                    status = "filled"
                    filled_at = datetime.now()
                    execution_time = 0.5 # Estimated
                else:
                    logger.warning(f"MT5 Order failed: {res.get('comment')}")
                    # We might still want to record the attempt?
                    # For now proceed to DB recording
            except Exception as e:
                logger.error(f"MT5 execution error: {e}")

        # 2. Record in Database
        order_id = f"ORD-{int(datetime.now().timestamp())}"

        try:
             async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO execution_orders
                    (order_id, symbol, order_type, side, quantity, price, status, strategy,
                     commission, execution_time_seconds, created_at, filled_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, order_id, symbol, order_type, action, quantity, price, status,
                     strategy, 0.0, execution_time, datetime.now(), filled_at)

                logger.info(f"Order {order_id} recorded in DB (Status: {status})")

        except Exception as e:
            logger.error(f"Error recording order to DB: {e}")

        return {
            "order_id": order_id,
            "status": status,
            "mt5_result": mt5_result
        }
    
    async def get_execution_strategies(self) -> List[Dict[str, Any]]:
        """Get execution strategies with real data."""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        strategy_name,
                        strategy_type,
                        parameters,
                        performance_metrics,
                        is_active,
                        total_orders,
                        successful_orders,
                        avg_execution_time,
                        success_rate,
                        created_at,
                        updated_at
                    FROM execution_strategies
                    ORDER BY success_rate DESC, total_orders DESC
                """)
                
                strategies = []
                for row in rows:
                    strategies.append({
                        "strategy_name": row['strategy_name'],
                        "strategy_type": row['strategy_type'],
                        "parameters": row['parameters'],
                        "performance_metrics": row['performance_metrics'],
                        "is_active": row['is_active'],
                        "total_orders": row['total_orders'],
                        "successful_orders": row['successful_orders'],
                        "avg_execution_time": float(row['avg_execution_time']),
                        "success_rate": float(row['success_rate']),
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat()
                    })
                
                return strategies
                
        except Exception as e:
            logger.error(f"Error getting execution strategies: {e}")
            return []
    
    async def create_sample_data(self):
        """Create sample execution agent data for demonstration."""
        try:
            async with self.db_pool.acquire() as conn:
                # Create sample orders
                symbols = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
                order_types = ['market', 'limit', 'stop']
                sides = ['buy', 'sell']
                statuses = ['filled', 'pending', 'cancelled', 'submitted']
                strategies = ['TWAP', 'VWAP', 'Iceberg', 'Adaptive', 'Aggressive']
                
                for i in range(50):
                    symbol = random.choice(symbols)
                    order_type = random.choice(order_types)
                    side = random.choice(sides)
                    status = random.choice(statuses)
                    strategy = random.choice(strategies)
                    
                    quantity = round(random.uniform(1, 1000), 4)
                    price = round(random.uniform(100, 1000), 2) if order_type in ['limit', 'stop'] else None
                    commission = round(random.uniform(0.5, 5.0), 2)
                    execution_time = random.randint(1, 300) if status == 'filled' else None
                    
                    created_at = datetime.now() - timedelta(days=random.randint(0, 30))
                    filled_at = created_at + timedelta(seconds=execution_time) if execution_time else None
                    
                    await conn.execute("""
                        INSERT INTO execution_orders 
                        (order_id, symbol, order_type, side, quantity, price, status, strategy, 
                         commission, execution_time_seconds, created_at, filled_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (order_id) DO NOTHING
                    """, f"ORD-{i+1:04d}", symbol, order_type, side, quantity, price, status, 
                         strategy, commission, execution_time, created_at, filled_at)
                
                # Create sample positions
                for i, symbol in enumerate(['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL']):
                    quantity = round(random.uniform(10, 500), 4)
                    average_price = round(random.uniform(150, 800), 2)
                    market_value = round(quantity * average_price * random.uniform(0.95, 1.05), 2)
                    unrealized_pnl = round(market_value - (quantity * average_price), 2)
                    realized_pnl = round(random.uniform(-1000, 2000), 2)
                    total_commission = round(random.uniform(10, 100), 2)
                    strategy = random.choice(strategies)
                    
                    await conn.execute("""
                        INSERT INTO execution_positions 
                        (symbol, quantity, average_price, market_value, unrealized_pnl, 
                         realized_pnl, total_commission, strategy, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (symbol) DO UPDATE SET
                            quantity = EXCLUDED.quantity,
                            market_value = EXCLUDED.market_value,
                            unrealized_pnl = EXCLUDED.unrealized_pnl,
                            updated_at = EXCLUDED.updated_at
                    """, symbol, quantity, average_price, market_value, unrealized_pnl,
                         realized_pnl, total_commission, strategy, 
                         datetime.now() - timedelta(days=random.randint(1, 30)),
                         datetime.now())
                
                # Create sample strategies
                strategy_configs = [
                    {
                        'name': 'TWAP Strategy',
                        'type': 'twap',
                        'parameters': {'time_horizon': '1h', 'slices': 10},
                        'metrics': {'fill_ratio': 0.95, 'slippage': 0.02}
                    },
                    {
                        'name': 'VWAP Strategy',
                        'type': 'vwap',
                        'parameters': {'benchmark': 'vwap', 'participation_rate': 0.15},
                        'metrics': {'fill_ratio': 0.98, 'slippage': 0.01}
                    },
                    {
                        'name': 'Iceberg Strategy',
                        'type': 'iceberg',
                        'parameters': {'display_size': 100, 'reserve_ratio': 0.8},
                        'metrics': {'fill_ratio': 0.92, 'slippage': 0.03}
                    },
                    {
                        'name': 'Adaptive Strategy',
                        'type': 'adaptive',
                        'parameters': {'volatility_threshold': 0.02, 'momentum_factor': 0.5},
                        'metrics': {'fill_ratio': 0.89, 'slippage': 0.025}
                    },
                    {
                        'name': 'Aggressive Strategy',
                        'type': 'aggressive',
                        'parameters': {'urgency': 'high', 'max_slippage': 0.05},
                        'metrics': {'fill_ratio': 0.99, 'slippage': 0.04}
                    }
                ]
                
                for config in strategy_configs:
                    total_orders = random.randint(20, 200)
                    successful_orders = int(total_orders * random.uniform(0.85, 0.98))
                    avg_execution_time = round(random.uniform(30, 180), 2)
                    success_rate = successful_orders / total_orders if total_orders > 0 else 0.0
                    
                    await conn.execute("""
                        INSERT INTO execution_strategies 
                        (strategy_name, strategy_type, parameters, performance_metrics, 
                         total_orders, successful_orders, avg_execution_time, success_rate, 
                         created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (strategy_name) DO UPDATE SET
                            performance_metrics = EXCLUDED.performance_metrics,
                            total_orders = EXCLUDED.total_orders,
                            successful_orders = EXCLUDED.successful_orders,
                            avg_execution_time = EXCLUDED.avg_execution_time,
                            success_rate = EXCLUDED.success_rate,
                            updated_at = EXCLUDED.updated_at
                    """, config['name'], config['type'], json.dumps(config['parameters']),
                         json.dumps(config['metrics']), total_orders, successful_orders,
                         avg_execution_time, success_rate, datetime.now() - timedelta(days=random.randint(1, 30)),
                         datetime.now())
                
                logger.info("Sample execution agent data created successfully")
                
        except Exception as e:
            logger.error(f"Error creating sample execution agent data: {e}")
    
    def _get_fallback_summary(self) -> Dict[str, Any]:
        """Get fallback summary data when database is unavailable."""
        return {
            "total_orders": 0,
            "active_orders": 0,
            "filled_orders": 0,
            "cancelled_orders": 0,
            "total_volume": 0.0,
            "total_commission": 0.0,
            "avg_execution_time": 0.0,
            "execution_success_rate": 0.0,
            "active_strategies": 0,
            "total_positions": 0,
            "total_market_value": 0.0,
            "total_unrealized_pnl": 0.0,
            "last_updated": datetime.now().isoformat()
        }
