"""
Data Models for Portfolio Management
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from .order_types import OrderType, OrderSide, OrderStatus

@dataclass
class Holding:
    """Portfolio holding representation."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float
    last_updated: datetime

@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    filled_price: Optional[float]
    created_at: datetime
    updated_at: datetime
    agent_signal: Optional[str]
    confidence: Optional[float]

@dataclass
class Transaction:
    """Transaction record."""
    transaction_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    fees: float
    agent_signal: Optional[str]

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_percent: float
    daily_pnl: float
    daily_pnl_percent: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    alpha: float
    last_updated: datetime
