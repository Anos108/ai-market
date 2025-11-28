"""
Rebalance Agent

This agent handles automated portfolio rebalancing based on target allocations,
risk parameters, and market conditions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
import numpy as np

logger = logging.getLogger(__name__)

class RebalanceStrategy(Enum):
    """Rebalancing strategies."""
    TARGET_WEIGHTS = "target_weights"
    RISK_PARITY = "risk_parity"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_TARGETING = "volatility_targeting"

class RebalanceTrigger(Enum):
    """Rebalancing triggers."""
    TIME_BASED = "time_based"
    THRESHOLD_BASED = "threshold_based"
    VOLATILITY_BASED = "volatility_based"
    CORRELATION_BASED = "correlation_based"
    MANUAL = "manual"

@dataclass
class TargetAllocation:
    """Target allocation definition."""
    symbol: str
    target_weight: float
    min_weight: float
    max_weight: float
    asset_class: str
    sector: str
    is_active: bool = True
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class RebalanceOrder:
    """Rebalancing order."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    target_weight: float
    current_weight: float
    weight_diff: float
    order_value: float
    priority: int  # 1 = highest priority
    reason: str
    created_at: datetime

@dataclass
class RebalancePlan:
    """Rebalancing plan."""
    plan_id: str
    strategy: RebalanceStrategy
    trigger: RebalanceTrigger
    target_allocations: List[TargetAllocation]
    rebalance_orders: List[RebalanceOrder]
    total_trades: int
    estimated_cost: float
    estimated_impact: float
    created_at: datetime
    status: str  # 'pending', 'executing', 'completed', 'cancelled'

class RebalanceAgent:
    """
    Rebalance Agent for automated portfolio rebalancing.
    
    This agent handles:
    - Target allocation management
    - Rebalancing strategy implementation
    - Order generation and execution
    - Performance tracking
    - Risk-aware rebalancing
    """
    
    def __init__(self, execution_router=None, position_tracker=None, risk_control=None):
        self.execution_router = execution_router
        self.position_tracker = position_tracker
        self.risk_control = risk_control
        
        # Rebalancing parameters
        self.rebalance_threshold = 0.05  # 5% threshold for rebalancing
        self.min_trade_size = 1000.0  # $1000 minimum trade size
        self.max_trades_per_rebalance = 10  # Limit number of trades
        self.rebalance_frequency = 7  # Days between rebalances
        
        # Target allocations
        self.target_allocations: Dict[str, TargetAllocation] = {}
        self.rebalance_history: List[RebalancePlan] = []
        
        # Initialize default allocations
        self._initialize_default_allocations()
        
        logger.info("RebalanceAgent initialized")
    
    def _initialize_default_allocations(self):
        """Initialize default target allocations."""
        default_allocations = [
            TargetAllocation("AAPL", 0.20, 0.10, 0.30, "equity", "technology"),
            TargetAllocation("MSFT", 0.15, 0.05, 0.25, "equity", "technology"),
            TargetAllocation("GOOGL", 0.10, 0.05, 0.20, "equity", "technology"),
            TargetAllocation("TSLA", 0.08, 0.03, 0.15, "equity", "automotive"),
            TargetAllocation("SPY", 0.25, 0.15, 0.35, "equity", "broad_market"),
            TargetAllocation("AMZN", 0.12, 0.05, 0.20, "equity", "consumer_discretionary"),
            TargetAllocation("NVDA", 0.10, 0.05, 0.18, "equity", "technology")
        ]
        
        for allocation in default_allocations:
            self.target_allocations[allocation.symbol] = allocation
    
    async def check_rebalance_needed(self) -> Tuple[bool, str]:
        """Check if rebalancing is needed."""
        try:
            if not self.position_tracker:
                return False, "Position tracker not available"
            
            # Update positions
            await self.position_tracker.update_positions_from_brokers()
            
            if not self.position_tracker.portfolio_metrics:
                return False, "Portfolio metrics not available"
            
            # Check time-based trigger
            if self._is_time_based_rebalance_needed():
                return True, "Time-based rebalancing needed"
            
            # Check threshold-based trigger
            if self._is_threshold_based_rebalance_needed():
                return True, "Threshold-based rebalancing needed"
            
            # Check volatility-based trigger
            if self._is_volatility_based_rebalance_needed():
                return True, "Volatility-based rebalancing needed"
            
            return False, "No rebalancing needed"
            
        except Exception as e:
            logger.error(f"Error checking rebalance need: {e}")
            return False, f"Error: {str(e)}"
    
    def _is_time_based_rebalance_needed(self) -> bool:
        """Check if time-based rebalancing is needed."""
        if not self.rebalance_history:
            return True  # First rebalance
        
        last_rebalance = self.rebalance_history[-1]
        days_since_rebalance = (datetime.now() - last_rebalance.created_at).days
        
        return days_since_rebalance >= self.rebalance_frequency
    
    def _is_threshold_based_rebalance_needed(self) -> bool:
        """Check if threshold-based rebalancing is needed."""
        if not self.position_tracker or not self.position_tracker.positions:
            return False
        
        for symbol, position in self.position_tracker.positions.items():
            if symbol in self.target_allocations:
                target_allocation = self.target_allocations[symbol]
                current_weight = position.weight
                target_weight = target_allocation.target_weight * 100
                
                weight_diff = abs(current_weight - target_weight)
                if weight_diff > self.rebalance_threshold * 100:
                    return True
        
        return False
    
    def _is_volatility_based_rebalance_needed(self) -> bool:
        """Check if volatility-based rebalancing is needed."""
        if not self.position_tracker or not self.position_tracker.portfolio_metrics:
            return False
        
        portfolio_volatility = self.position_tracker.portfolio_metrics.portfolio_volatility
        volatility_threshold = 0.25  # 25% volatility threshold
        
        return portfolio_volatility > volatility_threshold
    
    async def create_rebalance_plan(self, strategy: RebalanceStrategy = RebalanceStrategy.TARGET_WEIGHTS,
                                   trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD_BASED) -> RebalancePlan:
        """Create a rebalancing plan."""
        try:
            plan_id = f"REBAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get current positions
            if not self.position_tracker:
                raise ValueError("Position tracker not available")
            
            await self.position_tracker.update_positions_from_brokers()
            
            if not self.position_tracker.portfolio_metrics:
                raise ValueError("Portfolio metrics not available")
            
            portfolio_value = self.position_tracker.portfolio_metrics.total_value
            positions = self.position_tracker.positions
            
            # Generate rebalancing orders based on strategy
            rebalance_orders = []
            
            if strategy == RebalanceStrategy.TARGET_WEIGHTS:
                rebalance_orders = await self._create_target_weight_orders(positions, portfolio_value)
            elif strategy == RebalanceStrategy.RISK_PARITY:
                rebalance_orders = await self._create_risk_parity_orders(positions, portfolio_value)
            elif strategy == RebalanceStrategy.MOMENTUM:
                rebalance_orders = await self._create_momentum_orders(positions, portfolio_value)
            elif strategy == RebalanceStrategy.VOLATILITY_TARGETING:
                rebalance_orders = await self._create_volatility_targeting_orders(positions, portfolio_value)
            
            # Filter orders by minimum trade size
            rebalance_orders = [order for order in rebalance_orders if order.order_value >= self.min_trade_size]
            
            # Limit number of trades
            if len(rebalance_orders) > self.max_trades_per_rebalance:
                # Sort by priority and take top trades
                rebalance_orders.sort(key=lambda x: x.priority)
                rebalance_orders = rebalance_orders[:self.max_trades_per_rebalance]
            
            # Calculate estimated cost and impact
            estimated_cost = sum(order.order_value for order in rebalance_orders)
            estimated_impact = self._calculate_rebalance_impact(rebalance_orders, positions)
            
            # Create rebalancing plan
            plan = RebalancePlan(
                plan_id=plan_id,
                strategy=strategy,
                trigger=trigger,
                target_allocations=list(self.target_allocations.values()),
                rebalance_orders=rebalance_orders,
                total_trades=len(rebalance_orders),
                estimated_cost=estimated_cost,
                estimated_impact=estimated_impact,
                created_at=datetime.now(),
                status="pending"
            )
            
            logger.info(f"Created rebalancing plan {plan_id} with {len(rebalance_orders)} orders")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating rebalancing plan: {e}")
            raise
    
    async def _create_target_weight_orders(self, positions: Dict[str, Any], portfolio_value: float) -> List[RebalanceOrder]:
        """Create orders to achieve target weights."""
        orders = []
        
        for symbol, target_allocation in self.target_allocations.items():
            if not target_allocation.is_active:
                continue
            
            current_position = positions.get(symbol)
            current_weight = current_position.weight if current_position else 0.0
            target_weight = target_allocation.target_weight * 100
            
            weight_diff = target_weight - current_weight
            
            # Check if rebalancing is needed
            if abs(weight_diff) < self.rebalance_threshold * 100:
                continue
            
            # Calculate order details
            target_value = portfolio_value * target_allocation.target_weight
            current_value = current_position.market_value if current_position else 0.0
            order_value = target_value - current_value
            
            # Determine order side and quantity
            if order_value > 0:
                side = "buy"
                quantity = order_value / (current_position.current_price if current_position else 100.0)
            else:
                side = "sell"
                quantity = abs(order_value) / (current_position.current_price if current_position else 100.0)
            
            # Set priority based on weight difference
            priority = int(abs(weight_diff) * 10)  # Higher weight diff = higher priority
            
            # Create order
            order = RebalanceOrder(
                order_id=f"REBAL_ORDER_{symbol}_{datetime.now().strftime('%H%M%S')}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                target_weight=target_weight,
                current_weight=current_weight,
                weight_diff=weight_diff,
                order_value=abs(order_value),
                priority=priority,
                reason=f"Target weight rebalancing: {current_weight:.1f}% -> {target_weight:.1f}%",
                created_at=datetime.now()
            )
            
            orders.append(order)
        
        return orders
    
    async def _create_risk_parity_orders(self, positions: Dict[str, Any], portfolio_value: float) -> List[RebalanceOrder]:
        """Create orders for risk parity rebalancing."""
        orders = []
        
        # Calculate risk contributions (simplified)
        risk_contributions = {}
        total_risk = 0.0
        
        for symbol, position in positions.items():
            # Simplified risk calculation (volatility * weight)
            risk = position.volatility * (position.weight / 100)
            risk_contributions[symbol] = risk
            total_risk += risk
        
        if total_risk == 0:
            return orders
        
        # Calculate target weights based on risk parity
        for symbol, risk in risk_contributions.items():
            target_weight = (risk / total_risk) * 100
            
            current_position = positions.get(symbol)
            current_weight = current_position.weight if current_position else 0.0
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < self.rebalance_threshold * 100:
                continue
            
            # Create order (similar to target weight orders)
            target_value = portfolio_value * (target_weight / 100)
            current_value = current_position.market_value if current_position else 0.0
            order_value = target_value - current_value
            
            if abs(order_value) < self.min_trade_size:
                continue
            
            side = "buy" if order_value > 0 else "sell"
            quantity = abs(order_value) / (current_position.current_price if current_position else 100.0)
            
            order = RebalanceOrder(
                order_id=f"RISK_PARITY_{symbol}_{datetime.now().strftime('%H%M%S')}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                target_weight=target_weight,
                current_weight=current_weight,
                weight_diff=weight_diff,
                order_value=abs(order_value),
                priority=int(abs(weight_diff) * 10),
                reason=f"Risk parity rebalancing: {current_weight:.1f}% -> {target_weight:.1f}%",
                created_at=datetime.now()
            )
            
            orders.append(order)
        
        return orders
    
    async def _create_momentum_orders(self, positions: Dict[str, Any], portfolio_value: float) -> List[RebalanceOrder]:
        """Create orders for momentum-based rebalancing."""
        orders = []
        
        # Simplified momentum calculation (using recent performance)
        momentum_scores = {}
        
        for symbol, position in positions.items():
            # Use unrealized P&L as momentum proxy
            momentum = position.unrealized_pnl_percent
            momentum_scores[symbol] = momentum
        
        # Sort by momentum
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Increase weight for high momentum, decrease for low momentum
        for i, (symbol, momentum) in enumerate(sorted_symbols):
            current_position = positions.get(symbol)
            if not current_position:
                continue
            
            # Calculate momentum-based target weight
            base_weight = current_position.weight
            momentum_adjustment = momentum * 0.1  # 10% of momentum as adjustment
            target_weight = base_weight + momentum_adjustment
            
            # Apply min/max constraints
            if symbol in self.target_allocations:
                target_allocation = self.target_allocations[symbol]
                target_weight = max(target_allocation.min_weight * 100, 
                                  min(target_weight, target_allocation.max_weight * 100))
            
            weight_diff = target_weight - current_position.weight
            
            if abs(weight_diff) < self.rebalance_threshold * 100:
                continue
            
            # Create order
            target_value = portfolio_value * (target_weight / 100)
            order_value = target_value - current_position.market_value
            
            if abs(order_value) < self.min_trade_size:
                continue
            
            side = "buy" if order_value > 0 else "sell"
            quantity = abs(order_value) / current_position.current_price
            
            order = RebalanceOrder(
                order_id=f"MOMENTUM_{symbol}_{datetime.now().strftime('%H%M%S')}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                target_weight=target_weight,
                current_weight=current_position.weight,
                weight_diff=weight_diff,
                order_value=abs(order_value),
                priority=int(abs(weight_diff) * 10),
                reason=f"Momentum rebalancing: {current_position.weight:.1f}% -> {target_weight:.1f}%",
                created_at=datetime.now()
            )
            
            orders.append(order)
        
        return orders
    
    async def _create_volatility_targeting_orders(self, positions: Dict[str, Any], portfolio_value: float) -> List[RebalanceOrder]:
        """Create orders for volatility targeting rebalancing."""
        orders = []
        
        target_volatility = 0.15  # 15% target portfolio volatility
        
        # Calculate current portfolio volatility
        current_volatility = 0.0
        if self.position_tracker and self.position_tracker.portfolio_metrics:
            current_volatility = self.position_tracker.portfolio_metrics.portfolio_volatility
        
        # If current volatility is close to target, no rebalancing needed
        if abs(current_volatility - target_volatility) < 0.02:  # 2% threshold
            return orders
        
        # Adjust weights to target volatility
        volatility_adjustment = target_volatility / current_volatility if current_volatility > 0 else 1.0
        
        for symbol, position in positions.items():
            current_weight = position.weight
            target_weight = current_weight * volatility_adjustment
            
            # Apply constraints
            if symbol in self.target_allocations:
                target_allocation = self.target_allocations[symbol]
                target_weight = max(target_allocation.min_weight * 100,
                                  min(target_weight, target_allocation.max_weight * 100))
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) < self.rebalance_threshold * 100:
                continue
            
            # Create order
            target_value = portfolio_value * (target_weight / 100)
            order_value = target_value - position.market_value
            
            if abs(order_value) < self.min_trade_size:
                continue
            
            side = "buy" if order_value > 0 else "sell"
            quantity = abs(order_value) / position.current_price
            
            order = RebalanceOrder(
                order_id=f"VOL_TARGET_{symbol}_{datetime.now().strftime('%H%M%S')}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                target_weight=target_weight,
                current_weight=current_weight,
                weight_diff=weight_diff,
                order_value=abs(order_value),
                priority=int(abs(weight_diff) * 10),
                reason=f"Volatility targeting: {current_weight:.1f}% -> {target_weight:.1f}%",
                created_at=datetime.now()
            )
            
            orders.append(order)
        
        return orders
    
    def _calculate_rebalance_impact(self, orders: List[RebalanceOrder], positions: Dict[str, Any]) -> float:
        """Calculate estimated impact of rebalancing."""
        total_impact = 0.0
        
        for order in orders:
            # Estimate impact based on order size and current position
            current_position = positions.get(order.symbol)
            if current_position:
                impact = (order.order_value / current_position.market_value) * 100
                total_impact += abs(impact)
        
        return total_impact
    
    async def execute_rebalance_plan(self, plan: RebalancePlan) -> Dict[str, Any]:
        """Execute a rebalancing plan."""
        try:
            if not self.execution_router:
                raise ValueError("Execution router not available")
            
            plan.status = "executing"
            executed_orders = []
            failed_orders = []
            
            # Execute orders in priority order
            sorted_orders = sorted(plan.rebalance_orders, key=lambda x: x.priority, reverse=True)
            
            for order in sorted_orders:
                try:
                    # Validate order with risk control
                    if self.risk_control:
                        current_positions = {symbol: asdict(pos) for symbol, pos in self.position_tracker.positions.items()}
                        is_valid, message = await self.risk_control.validate_order(
                            order.symbol, order.side, order.quantity, 100.0, current_positions
                        )
                        
                        if not is_valid:
                            failed_orders.append({
                                "order": asdict(order),
                                "reason": message
                            })
                            continue
                    
                    # Submit order to execution router
                    order_id = await self.execution_router.submit_order(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        order_type="market",
                        agent_signal="RebalanceAgent",
                        confidence=0.8
                    )
                    
                    executed_orders.append({
                        "order": asdict(order),
                        "execution_order_id": order_id
                    })
                    
                    logger.info(f"Executed rebalancing order: {order.symbol} {order.side} {order.quantity}")
                    
                except Exception as e:
                    failed_orders.append({
                        "order": asdict(order),
                        "reason": str(e)
                    })
                    logger.error(f"Failed to execute rebalancing order: {e}")
            
            plan.status = "completed"
            
            # Store plan in history
            self.rebalance_history.append(plan)
            
            return {
                "plan_id": plan.plan_id,
                "status": "completed",
                "executed_orders": executed_orders,
                "failed_orders": failed_orders,
                "total_executed": len(executed_orders),
                "total_failed": len(failed_orders),
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            plan.status = "failed"
            logger.error(f"Error executing rebalancing plan: {e}")
            return {
                "plan_id": plan.plan_id,
                "status": "failed",
                "error": str(e),
                "execution_time": datetime.now().isoformat()
            }
    
    async def get_rebalance_summary(self) -> Dict[str, Any]:
        """Get rebalancing summary."""
        # Check if rebalancing is needed
        is_needed, reason = await self.check_rebalance_needed()
        
        # Get recent rebalancing history
        recent_plans = self.rebalance_history[-5:] if self.rebalance_history else []
        
        return {
            "rebalance_needed": is_needed,
            "reason": reason,
            "target_allocations": [asdict(allocation) for allocation in self.target_allocations.values()],
            "recent_plans": [asdict(plan) for plan in recent_plans],
            "rebalance_threshold": self.rebalance_threshold,
            "min_trade_size": self.min_trade_size,
            "max_trades_per_rebalance": self.max_trades_per_rebalance,
            "rebalance_frequency": self.rebalance_frequency,
            "last_updated": datetime.now().isoformat()
        }
    
    def update_target_allocation(self, symbol: str, target_weight: float, 
                                min_weight: float = None, max_weight: float = None):
        """Update target allocation for a symbol."""
        if symbol in self.target_allocations:
            allocation = self.target_allocations[symbol]
            allocation.target_weight = target_weight
            if min_weight is not None:
                allocation.min_weight = min_weight
            if max_weight is not None:
                allocation.max_weight = max_weight
            allocation.last_updated = datetime.now()
            
            logger.info(f"Updated target allocation for {symbol}: {target_weight:.1%}")
        else:
            # Create new allocation
            allocation = TargetAllocation(
                symbol=symbol,
                target_weight=target_weight,
                min_weight=min_weight or target_weight * 0.5,
                max_weight=max_weight or target_weight * 1.5,
                asset_class="equity",
                sector="unknown"
            )
            self.target_allocations[symbol] = allocation
            
            logger.info(f"Created new target allocation for {symbol}: {target_weight:.1%}")
    
    def set_rebalance_parameters(self, threshold: float = None, min_trade_size: float = None,
                                max_trades: int = None, frequency: int = None):
        """Update rebalancing parameters."""
        if threshold is not None:
            self.rebalance_threshold = threshold
        if min_trade_size is not None:
            self.min_trade_size = min_trade_size
        if max_trades is not None:
            self.max_trades_per_rebalance = max_trades
        if frequency is not None:
            self.rebalance_frequency = frequency
        
        logger.info("Updated rebalancing parameters")
