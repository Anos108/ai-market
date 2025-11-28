"""
Order Types and Enums for Portfolio Management
"""

from enum import Enum

class OrderType(Enum):
    """Order types for portfolio management."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides for trading."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status tracking."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
