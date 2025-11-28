"""
Position Tracker Agent

This agent handles real-time position tracking with leverage, margin, and enhanced P&L calculations.
Integrates with the ExecutionRouterAgent to maintain accurate position data.
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

class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"
    OPTION = "option"
    FUTURE = "future"

class MarginType(Enum):
    """Margin types."""
    CASH = "cash"
    MARGIN = "margin"
    PORTFOLIO_MARGIN = "portfolio_margin"

@dataclass
class Position:
    """Enhanced position representation."""
    symbol: str
    position_type: PositionType
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    realized_pnl: float
    total_pnl: float
    weight: float
    leverage: float
    margin_used: float
    margin_available: float
    margin_ratio: float
    beta: float
    volatility: float
    last_updated: datetime
    broker: str
    account_id: str

@dataclass
class PortfolioMetrics:
    """Enhanced portfolio metrics."""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_percent: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    daily_pnl_percent: float
    cash: float
    margin_used: float
    margin_available: float
    buying_power: float
    leverage: float
    portfolio_beta: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    last_updated: datetime

@dataclass
class RiskMetrics:
    """Risk metrics for position tracking."""
    position_concentration: float
    sector_concentration: Dict[str, float]
    correlation_risk: float
    liquidity_risk: float
    margin_risk: float
    leverage_risk: float
    overall_risk_score: float
    risk_level: str  # Low, Medium, High, Critical

class PositionTrackerAgent:
    """
    Position Tracker Agent for real-time position management.
    
    This agent handles:
    - Real-time position tracking across multiple brokers
    - Leverage and margin calculations
    - Risk metrics and portfolio analytics
    - Position rebalancing recommendations
    - Performance attribution
    """
    
    def __init__(self, execution_router=None):
        self.execution_router = execution_router
        self.positions: Dict[str, Position] = {}
        self.portfolio_history: List[Dict[str, Any]] = []
        self.risk_metrics: Optional[RiskMetrics] = None
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        
        # Risk parameters
        self.max_leverage = 2.0
        self.max_position_weight = 0.2  # 20% max per position
        self.max_sector_weight = 0.4    # 40% max per sector
        self.margin_maintenance_ratio = 0.25  # 25% maintenance margin
        
        # Market data cache
        self.price_cache: Dict[str, float] = {}
        self.last_price_update = datetime.now()
        
        logger.info("PositionTrackerAgent initialized")
    
    async def update_positions_from_brokers(self):
        """Update positions from all connected brokers."""
        if not self.execution_router:
            logger.warning("No execution router connected")
            return
        
        try:
            # Get positions from all brokers
            all_positions = await self.execution_router.get_positions()
            
            # Clear existing positions
            self.positions.clear()
            
            # Process positions from each broker
            for broker_name, positions in all_positions.items():
                for pos_data in positions:
                    await self._process_broker_position(pos_data, broker_name)
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
            # Update risk metrics
            await self._update_risk_metrics()
            
            # Save portfolio snapshot
            self._save_portfolio_snapshot()
            
            logger.info(f"Updated {len(self.positions)} positions from {len(all_positions)} brokers")
            
        except Exception as e:
            logger.error(f"Error updating positions from brokers: {e}")
    
    async def _process_broker_position(self, pos_data: Dict[str, Any], broker_name: str):
        """Process a position from a broker."""
        try:
            symbol = pos_data.get("symbol", "")
            if not symbol:
                return
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            
            # Calculate position metrics
            quantity = pos_data.get("quantity", 0.0)
            avg_cost = pos_data.get("avg_cost", current_price)
            market_value = quantity * current_price
            cost_basis = quantity * avg_cost
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            
            # Determine position type
            position_type = PositionType.LONG if quantity > 0 else PositionType.SHORT
            
            # Calculate leverage and margin (simplified)
            leverage = 1.0  # Default for cash positions
            margin_used = 0.0
            margin_available = 0.0
            margin_ratio = 0.0
            
            # Calculate beta and volatility (simplified)
            beta = 1.0  # Default
            volatility = 0.2  # 20% default
            
            # Create position
            position = Position(
                symbol=symbol,
                position_type=position_type,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                market_value=market_value,
                cost_basis=cost_basis,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_percent=unrealized_pnl_percent,
                realized_pnl=0.0,  # Would be tracked separately
                total_pnl=unrealized_pnl,
                weight=0.0,  # Will be calculated in portfolio metrics
                leverage=leverage,
                margin_used=margin_used,
                margin_available=margin_available,
                margin_ratio=margin_ratio,
                beta=beta,
                volatility=volatility,
                last_updated=datetime.now(),
                broker=broker_name,
                account_id=f"{broker_name}_account"
            )
            
            self.positions[symbol] = position
            
        except Exception as e:
            logger.error(f"Error processing position for {pos_data.get('symbol', 'unknown')}: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # Check cache first
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        
        # In a real implementation, this would fetch from market data provider
        # For now, use a simple price model
        base_price = 100.0
        
        # Add some variation based on symbol
        if "AAPL" in symbol.upper():
            base_price = 150.0
        elif "MSFT" in symbol.upper():
            base_price = 300.0
        elif "GOOGL" in symbol.upper():
            base_price = 2500.0
        elif "TSLA" in symbol.upper():
            base_price = 200.0
        elif "SPY" in symbol.upper():
            base_price = 400.0
        
        # Add some random variation
        variation = np.random.normal(0, 0.02)  # 2% standard deviation
        price = base_price * (1 + variation)
        
        # Cache the price
        self.price_cache[symbol] = price
        
        return price
    
    async def _update_portfolio_metrics(self):
        """Update portfolio-level metrics."""
        try:
            # Calculate total values
            total_market_value = sum(pos.market_value for pos in self.positions.values())
            total_cost = sum(pos.cost_basis for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Get cash from brokers
            total_cash = 0.0
            total_margin_used = 0.0
            total_margin_available = 0.0
            
            if self.execution_router:
                account_info = await self.execution_router.get_account_info()
                for broker_info in account_info.values():
                    if isinstance(broker_info, dict) and "error" not in broker_info:
                        total_cash += broker_info.get("cash", 0.0)
                        total_margin_used += broker_info.get("margin_used", 0.0)
                        total_margin_available += broker_info.get("margin_available", 0.0)
            
            total_value = total_market_value + total_cash
            total_pnl = total_unrealized_pnl
            total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
            
            # Calculate daily P&L (simplified)
            daily_pnl = total_pnl * 0.01  # Assume 1% daily variation
            daily_pnl_percent = (daily_pnl / total_value * 100) if total_value > 0 else 0.0
            
            # Calculate leverage
            leverage = (total_market_value / total_value) if total_value > 0 else 0.0
            
            # Calculate portfolio beta (weighted average)
            portfolio_beta = 0.0
            if total_market_value > 0:
                portfolio_beta = sum(pos.beta * (pos.market_value / total_market_value) for pos in self.positions.values())
            
            # Calculate portfolio volatility (simplified)
            portfolio_volatility = 0.15  # 15% default
            
            # Calculate risk metrics
            sharpe_ratio = 1.2  # Placeholder
            max_drawdown = 0.05  # Placeholder
            var_95 = total_value * 0.02  # 2% VaR
            var_99 = total_value * 0.03  # 3% VaR
            
            # Update position weights
            for position in self.positions.values():
                position.weight = (position.market_value / total_value * 100) if total_value > 0 else 0.0
            
            # Create portfolio metrics
            self.portfolio_metrics = PortfolioMetrics(
                total_value=total_value,
                total_cost=total_cost,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=0.0,  # Would be tracked separately
                daily_pnl=daily_pnl,
                daily_pnl_percent=daily_pnl_percent,
                cash=total_cash,
                margin_used=total_margin_used,
                margin_available=total_margin_available,
                buying_power=total_cash + total_margin_available,
                leverage=leverage,
                portfolio_beta=portfolio_beta,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _update_risk_metrics(self):
        """Update risk metrics."""
        try:
            if not self.portfolio_metrics:
                return
            
            # Calculate position concentration
            position_weights = [pos.weight for pos in self.positions.values()]
            position_concentration = max(position_weights) if position_weights else 0.0
            
            # Calculate sector concentration (simplified)
            sector_concentration = {
                "Technology": 0.4,
                "Healthcare": 0.2,
                "Financial": 0.15,
                "Energy": 0.1,
                "Other": 0.15
            }
            
            # Calculate correlation risk (simplified)
            correlation_risk = 0.3  # 30% correlation risk
            
            # Calculate liquidity risk (simplified)
            liquidity_risk = 0.1  # 10% liquidity risk
            
            # Calculate margin risk
            margin_risk = 0.0
            if self.portfolio_metrics.margin_used > 0:
                margin_ratio = self.portfolio_metrics.margin_used / self.portfolio_metrics.total_value
                margin_risk = min(margin_ratio / self.margin_maintenance_ratio, 1.0)
            
            # Calculate leverage risk
            leverage_risk = min(self.portfolio_metrics.leverage / self.max_leverage, 1.0)
            
            # Calculate overall risk score
            risk_factors = [
                position_concentration / 100,  # Convert to 0-1 scale
                max(sector_concentration.values()),
                correlation_risk,
                liquidity_risk,
                margin_risk,
                leverage_risk
            ]
            overall_risk_score = np.mean(risk_factors) * 100
            
            # Determine risk level
            if overall_risk_score < 25:
                risk_level = "Low"
            elif overall_risk_score < 50:
                risk_level = "Medium"
            elif overall_risk_score < 75:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            self.risk_metrics = RiskMetrics(
                position_concentration=position_concentration,
                sector_concentration=sector_concentration,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                margin_risk=margin_risk,
                leverage_risk=leverage_risk,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def _save_portfolio_snapshot(self):
        """Save portfolio snapshot to history."""
        if not self.portfolio_metrics:
            return
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "total_value": self.portfolio_metrics.total_value,
            "total_pnl": self.portfolio_metrics.total_pnl,
            "total_pnl_percent": self.portfolio_metrics.total_pnl_percent,
            "cash": self.portfolio_metrics.cash,
            "leverage": self.portfolio_metrics.leverage,
            "positions_count": len(self.positions),
            "risk_score": self.risk_metrics.overall_risk_score if self.risk_metrics else 0.0,
            "risk_level": self.risk_metrics.risk_level if self.risk_metrics else "Unknown"
        }
        
        self.portfolio_history.append(snapshot)
        
        # Keep only last 1000 snapshots
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary."""
        await self.update_positions_from_brokers()
        
        return {
            "positions": [asdict(pos) for pos in self.positions.values()],
            "portfolio_metrics": asdict(self.portfolio_metrics) if self.portfolio_metrics else None,
            "risk_metrics": asdict(self.risk_metrics) if self.risk_metrics else None,
            "portfolio_history": self.portfolio_history[-30:],  # Last 30 snapshots
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_positions_by_broker(self) -> Dict[str, List[Position]]:
        """Get positions grouped by broker."""
        positions_by_broker = {}
        
        for position in self.positions.values():
            broker = position.broker
            if broker not in positions_by_broker:
                positions_by_broker[broker] = []
            positions_by_broker[broker].append(position)
        
        return positions_by_broker
    
    async def get_risk_alerts(self) -> List[str]:
        """Get risk alerts based on current positions."""
        alerts = []
        
        if not self.risk_metrics or not self.portfolio_metrics:
            return alerts
        
        # Position concentration alert
        if self.risk_metrics.position_concentration > self.max_position_weight * 100:
            alerts.append(f"High position concentration: {self.risk_metrics.position_concentration:.1f}%")
        
        # Leverage alert
        if self.portfolio_metrics.leverage > self.max_leverage:
            alerts.append(f"High leverage: {self.portfolio_metrics.leverage:.2f}x")
        
        # Margin risk alert
        if self.risk_metrics.margin_risk > 0.8:
            alerts.append(f"High margin risk: {self.risk_metrics.margin_risk:.1%}")
        
        # Overall risk alert
        if self.risk_metrics.overall_risk_score > 75:
            alerts.append(f"Critical risk level: {self.risk_metrics.overall_risk_score:.1f}")
        
        return alerts
    
    async def get_rebalancing_recommendations(self, target_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get rebalancing recommendations based on target weights."""
        recommendations = []
        
        if not self.portfolio_metrics:
            return recommendations
        
        total_value = self.portfolio_metrics.total_value
        
        for symbol, target_weight in target_weights.items():
            current_position = self.positions.get(symbol)
            current_weight = current_position.weight if current_position else 0.0
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 1.0:  # 1% threshold
                target_value = total_value * target_weight / 100
                current_value = current_position.market_value if current_position else 0.0
                
                if target_value > current_value:
                    # Need to buy
                    buy_value = target_value - current_value
                    current_price = await self._get_current_price(symbol)
                    buy_quantity = buy_value / current_price
                    
                    recommendations.append({
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": buy_quantity,
                        "value": buy_value,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "weight_diff": weight_diff
                    })
                else:
                    # Need to sell
                    sell_value = current_value - target_value
                    sell_quantity = sell_value / current_position.current_price
                    
                    recommendations.append({
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": sell_quantity,
                        "value": sell_value,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                        "weight_diff": weight_diff
                    })
        
        return recommendations
    
    def get_performance_attribution(self, days: int = 30) -> Dict[str, Any]:
        """Get performance attribution analysis."""
        if len(self.portfolio_history) < 2:
            return {"error": "Insufficient history data"}
        
        # Get recent history
        recent_history = self.portfolio_history[-days:] if len(self.portfolio_history) >= days else self.portfolio_history
        
        if len(recent_history) < 2:
            return {"error": "Insufficient recent data"}
        
        # Calculate performance metrics
        start_value = recent_history[0]["total_value"]
        end_value = recent_history[-1]["total_value"]
        total_return = (end_value - start_value) / start_value * 100
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(recent_history)):
            prev_value = recent_history[i-1]["total_value"]
            curr_value = recent_history[i]["total_value"]
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)
        
        # Calculate volatility
        volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
        
        # Calculate Sharpe ratio (simplified)
        avg_daily_return = np.mean(daily_returns)
        sharpe_ratio = (avg_daily_return * 252) / (volatility / 100) if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = start_value
        max_drawdown = 0
        for snapshot in recent_history:
            if snapshot["total_value"] > peak:
                peak = snapshot["total_value"]
            drawdown = (peak - snapshot["total_value"]) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "period_days": days,
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown * 100,
            "start_value": start_value,
            "end_value": end_value,
            "daily_returns": daily_returns[-10:],  # Last 10 daily returns
            "last_updated": datetime.now().isoformat()
        }
