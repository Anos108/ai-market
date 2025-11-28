"""
Execution Router Agent

This agent handles order routing to different brokers and execution venues.
Supports paper trading, live trading via Alpaca, Binance, and Interactive Brokers.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Execution modes for the router."""
    PAPER = "paper"
    LIVE = "live"

class OrderType(Enum):
    """Order types supported by the execution router."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status tracking."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ExecutionOrder:
    """Order representation for execution."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    trailing_distance: Optional[float]
    time_in_force: str = "GTC"  # Good Till Cancelled
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    agent_signal: Optional[str] = None
    confidence: Optional[float] = None
    broker: Optional[str] = None
    fees: float = 0.0
    commission: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class ExecutionResult:
    """Result of order execution."""
    order_id: str
    success: bool
    filled_quantity: float
    filled_price: Optional[float]
    fees: float
    commission: float
    execution_time: datetime
    broker: str
    error_message: Optional[str] = None

class BrokerInterface(ABC):
    """Abstract base class for broker interfaces."""
    
    @abstractmethod
    async def submit_order(self, order: ExecutionOrder) -> ExecutionResult:
        """Submit an order to the broker."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        pass

class PaperTradingBroker(BrokerInterface):
    """Paper trading broker implementation."""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.orders: Dict[str, ExecutionOrder] = {}
        self.order_counter = 0
        
    async def submit_order(self, order: ExecutionOrder) -> ExecutionResult:
        """Simulate order execution for paper trading."""
        try:
            # Simulate market data delay
            await asyncio.sleep(0.1)
            
            # Generate execution price (simulate slippage)
            if order.order_type == OrderType.MARKET:
                # Simulate market execution with small slippage
                slippage = 0.001  # 0.1% slippage
                if order.side == OrderSide.BUY:
                    execution_price = order.price * (1 + slippage) if order.price else 100.0
                else:
                    execution_price = order.price * (1 - slippage) if order.price else 100.0
            else:
                execution_price = order.price or 100.0
            
            # Check if we have enough cash/shares
            if order.side == OrderSide.BUY:
                required_cash = order.quantity * execution_price
                if required_cash > self.cash:
                    return ExecutionResult(
                        order_id=order.order_id,
                        success=False,
                        filled_quantity=0.0,
                        filled_price=None,
                        fees=0.0,
                        commission=0.0,
                        execution_time=datetime.now(),
                        broker="paper",
                        error_message="Insufficient cash"
                    )
            else:
                current_position = self.positions.get(order.symbol, 0.0)
                if order.quantity > current_position:
                    return ExecutionResult(
                        order_id=order.order_id,
                        success=False,
                        filled_quantity=0.0,
                        filled_price=None,
                        fees=0.0,
                        commission=0.0,
                        execution_time=datetime.now(),
                        broker="paper",
                        error_message="Insufficient shares"
                    )
            
            # Execute the trade
            if order.side == OrderSide.BUY:
                self.cash -= (order.quantity * execution_price)
                self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + order.quantity
            else:
                self.cash += (order.quantity * execution_price)
                self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) - order.quantity
                if self.positions[order.symbol] <= 0:
                    del self.positions[order.symbol]
            
            # Calculate fees (simplified)
            fees = order.quantity * execution_price * 0.001  # 0.1% fee
            commission = 0.0  # No commission for paper trading
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.fees = fees
            order.commission = commission
            order.updated_at = datetime.now()
            
            self.orders[order.order_id] = order
            
            logger.info(f"Paper trade executed: {order.side.value} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
            
            return ExecutionResult(
                order_id=order.order_id,
                success=True,
                filled_quantity=order.quantity,
                filled_price=execution_price,
                fees=fees,
                commission=commission,
                execution_time=datetime.now(),
                broker="paper"
            )
            
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0.0,
                filled_price=None,
                fees=0.0,
                commission=0.0,
                execution_time=datetime.now(),
                broker="paper",
                error_message=str(e)
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in paper trading."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            self.orders[order_id].updated_at = datetime.now()
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status in paper trading."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get paper trading account info."""
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            # Use a simple price model for paper trading
            price = 100.0  # Simplified
            total_value += quantity * price
        
        return {
            "cash": self.cash,
            "total_value": total_value,
            "buying_power": self.cash,
            "positions": self.positions,
            "account_type": "paper"
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get paper trading positions."""
        positions = []
        for symbol, quantity in self.positions.items():
            positions.append({
                "symbol": symbol,
                "quantity": quantity,
                "market_value": quantity * 100.0,  # Simplified
                "avg_cost": 100.0,  # Simplified
                "unrealized_pnl": 0.0,  # Simplified
                "unrealized_pnl_percent": 0.0
            })
        return positions

class AlpacaBroker(BrokerInterface):
    """Alpaca broker interface (placeholder for real implementation)."""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.orders: Dict[str, ExecutionOrder] = {}
    
    async def submit_order(self, order: ExecutionOrder) -> ExecutionResult:
        """Submit order to Alpaca (placeholder implementation)."""
        # This would integrate with Alpaca's API
        # For now, return a placeholder response
        logger.info(f"Alpaca order submission (placeholder): {order.symbol} {order.side.value} {order.quantity}")
        
        # Simulate successful execution
        execution_price = order.price or 100.0
        fees = order.quantity * execution_price * 0.001
        
        return ExecutionResult(
            order_id=order.order_id,
            success=True,
            filled_quantity=order.quantity,
            filled_price=execution_price,
            fees=fees,
            commission=0.0,
            execution_time=datetime.now(),
            broker="alpaca"
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in Alpaca."""
        logger.info(f"Alpaca order cancellation (placeholder): {order_id}")
        return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Alpaca."""
        return OrderStatus.FILLED
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from Alpaca."""
        return {
            "cash": 100000.0,
            "total_value": 100000.0,
            "buying_power": 100000.0,
            "account_type": "alpaca"
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Alpaca."""
        return []

class BinanceBroker(BrokerInterface):
    """Binance broker interface (placeholder for real implementation)."""
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.orders: Dict[str, ExecutionOrder] = {}
    
    async def submit_order(self, order: ExecutionOrder) -> ExecutionResult:
        """Submit order to Binance (placeholder implementation)."""
        logger.info(f"Binance order submission (placeholder): {order.symbol} {order.side.value} {order.quantity}")
        
        # Simulate successful execution
        execution_price = order.price or 100.0
        fees = order.quantity * execution_price * 0.001
        
        return ExecutionResult(
            order_id=order.order_id,
            success=True,
            filled_quantity=order.quantity,
            filled_price=execution_price,
            fees=fees,
            commission=0.0,
            execution_time=datetime.now(),
            broker="binance"
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in Binance."""
        logger.info(f"Binance order cancellation (placeholder): {order_id}")
        return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Binance."""
        return OrderStatus.FILLED
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from Binance."""
        return {
            "cash": 100000.0,
            "total_value": 100000.0,
            "buying_power": 100000.0,
            "account_type": "binance"
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Binance."""
        return []

class InteractiveBrokersBroker(BrokerInterface):
    """Interactive Brokers interface (placeholder for real implementation)."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.orders: Dict[str, ExecutionOrder] = {}
    
    async def submit_order(self, order: ExecutionOrder) -> ExecutionResult:
        """Submit order to Interactive Brokers (placeholder implementation)."""
        logger.info(f"IB order submission (placeholder): {order.symbol} {order.side.value} {order.quantity}")
        
        # Simulate successful execution
        execution_price = order.price or 100.0
        fees = order.quantity * execution_price * 0.005  # IB has higher fees
        
        return ExecutionResult(
            order_id=order.order_id,
            success=True,
            filled_quantity=order.quantity,
            filled_price=execution_price,
            fees=fees,
            commission=0.0,
            execution_time=datetime.now(),
            broker="interactive_brokers"
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in Interactive Brokers."""
        logger.info(f"IB order cancellation (placeholder): {order_id}")
        return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Interactive Brokers."""
        return OrderStatus.FILLED
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info from Interactive Brokers."""
        return {
            "cash": 100000.0,
            "total_value": 100000.0,
            "buying_power": 100000.0,
            "account_type": "interactive_brokers"
        }
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions from Interactive Brokers."""
        return []

class ExecutionRouterAgent:
    """
    Execution Router Agent for managing order routing to different brokers.
    
    This agent handles:
    - Order routing to appropriate brokers
    - Execution mode management (paper vs live)
    - Order status tracking
    - Broker selection based on asset type and requirements
    """
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.PAPER):
        self.execution_mode = execution_mode
        self.brokers: Dict[str, BrokerInterface] = {}
        self.orders: Dict[str, ExecutionOrder] = {}
        self.order_counter = 0
        
        # Initialize brokers
        self._initialize_brokers()
        
        logger.info(f"ExecutionRouterAgent initialized in {execution_mode.value} mode")
    
    def _initialize_brokers(self):
        """Initialize available brokers."""
        # Paper trading broker (always available)
        self.brokers["paper"] = PaperTradingBroker()
        
        # Live trading brokers (placeholder initialization)
        # In production, these would be initialized with real API keys
        if self.execution_mode == ExecutionMode.LIVE:
            # Alpaca (for US stocks)
            self.brokers["alpaca"] = AlpacaBroker(
                api_key=os.getenv("ALPACA_API_KEY", ""),
                secret_key=os.getenv("ALPACA_SECRET_KEY", "")
            )
            
            # Binance (for crypto)
            self.brokers["binance"] = BinanceBroker(
                api_key=os.getenv("BINANCE_API_KEY", ""),
                secret_key=os.getenv("BINANCE_SECRET_KEY", "")
            )
            
            # Interactive Brokers (for international markets)
            self.brokers["interactive_brokers"] = InteractiveBrokersBroker()
    
    def _select_broker(self, symbol: str, order_type: OrderType) -> str:
        """Select appropriate broker based on symbol and order type."""
        if self.execution_mode == ExecutionMode.PAPER:
            return "paper"
        
        # Broker selection logic for live trading
        symbol_upper = symbol.upper()
        
        # Crypto symbols
        if any(crypto in symbol_upper for crypto in ["BTC", "ETH", "ADA", "DOT", "LINK"]):
            return "binance"
        
        # US stocks (default to Alpaca)
        if symbol_upper.endswith(".US") or len(symbol_upper) <= 5:
            return "alpaca"
        
        # International markets
        if symbol_upper.endswith((".L", ".HK", ".T", ".PA")):
            return "interactive_brokers"
        
        # Default to Alpaca for US stocks
        return "alpaca"
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"EXE_{timestamp}_{self.order_counter:04d}"
    
    async def submit_order(self, symbol: str, side: OrderSide, quantity: float,
                          order_type: OrderType = OrderType.MARKET, price: Optional[float] = None,
                          stop_price: Optional[float] = None, trailing_distance: Optional[float] = None,
                          agent_signal: Optional[str] = None, confidence: Optional[float] = None) -> str:
        """Submit an order for execution."""
        try:
            # Generate order ID
            order_id = self._generate_order_id()
            
            # Select appropriate broker
            broker_name = self._select_broker(symbol, order_type)
            
            # Create execution order
            order = ExecutionOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                trailing_distance=trailing_distance,
                agent_signal=agent_signal,
                confidence=confidence,
                broker=broker_name
            )
            
            # Store order
            self.orders[order_id] = order
            
            # Submit to broker
            broker = self.brokers[broker_name]
            result = await broker.submit_order(order)
            
            # Update order based on result
            if result.success:
                order.status = OrderStatus.FILLED
                order.filled_quantity = result.filled_quantity
                order.filled_price = result.filled_price
                order.fees = result.fees
                order.commission = result.commission
            else:
                order.status = OrderStatus.REJECTED
            
            order.updated_at = datetime.now()
            
            logger.info(f"Order {order_id} submitted to {broker_name}: {result.success}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            broker = self.brokers[order.broker]
            
            success = await broker.cancel_order(order_id)
            
            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
            
            logger.info(f"Order {order_id} cancellation: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        broker = self.brokers[order.broker]
        
        try:
            status = await broker.get_order_status(order_id)
            order.status = status
            order.updated_at = datetime.now()
            return status
        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return order.status
    
    async def get_account_info(self, broker_name: Optional[str] = None) -> Dict[str, Any]:
        """Get account information from broker(s)."""
        if broker_name:
            if broker_name in self.brokers:
                return await self.brokers[broker_name].get_account_info()
            else:
                raise ValueError(f"Broker {broker_name} not found")
        
        # Get info from all brokers
        account_info = {}
        for name, broker in self.brokers.items():
            try:
                info = await broker.get_account_info()
                account_info[name] = info
            except Exception as e:
                logger.error(f"Error getting account info from {name}: {e}")
                account_info[name] = {"error": str(e)}
        
        return account_info
    
    async def get_positions(self, broker_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get positions from broker(s)."""
        if broker_name:
            if broker_name in self.brokers:
                positions = await self.brokers[broker_name].get_positions()
                return {broker_name: positions}
            else:
                raise ValueError(f"Broker {broker_name} not found")
        
        # Get positions from all brokers
        all_positions = {}
        for name, broker in self.brokers.items():
            try:
                positions = await broker.get_positions()
                all_positions[name] = positions
            except Exception as e:
                logger.error(f"Error getting positions from {name}: {e}")
                all_positions[name] = []
        
        return all_positions
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution router summary."""
        total_orders = len(self.orders)
        filled_orders = sum(1 for order in self.orders.values() if order.status == OrderStatus.FILLED)
        pending_orders = sum(1 for order in self.orders.values() if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED])
        cancelled_orders = sum(1 for order in self.orders.values() if order.status == OrderStatus.CANCELLED)
        rejected_orders = sum(1 for order in self.orders.values() if order.status == OrderStatus.REJECTED)
        
        total_fees = sum(order.fees for order in self.orders.values())
        total_commission = sum(order.commission for order in self.orders.values())
        
        return {
            "execution_mode": self.execution_mode.value,
            "available_brokers": list(self.brokers.keys()),
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "pending_orders": pending_orders,
            "cancelled_orders": cancelled_orders,
            "rejected_orders": rejected_orders,
            "total_fees": total_fees,
            "total_commission": total_commission,
            "success_rate": (filled_orders / total_orders * 100) if total_orders > 0 else 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    def set_execution_mode(self, mode: ExecutionMode):
        """Change execution mode."""
        self.execution_mode = mode
        self._initialize_brokers()
        logger.info(f"Execution mode changed to {mode.value}")
    
    def get_orders(self, status_filter: Optional[OrderStatus] = None) -> List[ExecutionOrder]:
        """Get orders, optionally filtered by status."""
        orders = list(self.orders.values())
        if status_filter:
            orders = [order for order in orders if order.status == status_filter]
        return sorted(orders, key=lambda x: x.created_at, reverse=True)
