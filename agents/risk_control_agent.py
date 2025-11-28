"""
Risk Control Agent

This agent enforces risk limits and provides real-time risk monitoring.
Handles exposure limits, drawdown controls, and position sizing validation.
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

class RiskLevel(Enum):
    """Risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(Enum):
    """Types of risk controls."""
    POSITION_SIZE = "position_size"
    SECTOR_EXPOSURE = "sector_exposure"
    LEVERAGE = "leverage"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    MARGIN = "margin"
    VOLATILITY = "volatility"

@dataclass
class RiskLimit:
    """Risk limit definition."""
    risk_type: RiskType
    limit_value: float
    current_value: float
    threshold: float  # Percentage of limit that triggers warning
    is_active: bool = True
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class RiskAlert:
    """Risk alert definition."""
    alert_id: str
    risk_type: RiskType
    severity: RiskLevel
    message: str
    current_value: float
    limit_value: float
    recommendation: str
    timestamp: datetime
    is_acknowledged: bool = False

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    portfolio_var_95: float
    portfolio_var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    portfolio_beta: float
    portfolio_volatility: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    leverage_risk: float
    margin_risk: float
    overall_risk_score: float
    risk_level: RiskLevel
    last_updated: datetime

class RiskControlAgent:
    """
    Risk Control Agent for comprehensive risk management.
    
    This agent handles:
    - Real-time risk monitoring
    - Risk limit enforcement
    - Position sizing validation
    - Drawdown controls
    - Exposure limits
    - Risk alerts and notifications
    """
    
    def __init__(self, position_tracker=None):
        self.position_tracker = position_tracker
        self.risk_limits: Dict[RiskType, RiskLimit] = {}
        self.risk_alerts: List[RiskAlert] = []
        self.risk_metrics: Optional[RiskMetrics] = None
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Risk parameters
        self.max_position_size = 0.15  # 15% max per position
        self.max_sector_exposure = 0.30  # 30% max per sector
        self.max_leverage = 2.0  # 2x max leverage
        self.max_drawdown = 0.15  # 15% max drawdown
        self.max_correlation = 0.7  # 70% max correlation
        self.min_liquidity = 1000000  # $1M min daily volume
        self.margin_maintenance = 0.25  # 25% maintenance margin
        
        # Initialize default risk limits
        self._initialize_default_limits()
        
        logger.info("RiskControlAgent initialized")
    
    def _initialize_default_limits(self):
        """Initialize default risk limits."""
        self.risk_limits = {
            RiskType.POSITION_SIZE: RiskLimit(
                risk_type=RiskType.POSITION_SIZE,
                limit_value=self.max_position_size * 100,  # 15%
                current_value=0.0,
                threshold=0.8  # Warning at 80% of limit
            ),
            RiskType.SECTOR_EXPOSURE: RiskLimit(
                risk_type=RiskType.SECTOR_EXPOSURE,
                limit_value=self.max_sector_exposure * 100,  # 30%
                current_value=0.0,
                threshold=0.8
            ),
            RiskType.LEVERAGE: RiskLimit(
                risk_type=RiskType.LEVERAGE,
                limit_value=self.max_leverage,
                current_value=0.0,
                threshold=0.8
            ),
            RiskType.DRAWDOWN: RiskLimit(
                risk_type=RiskType.DRAWDOWN,
                limit_value=self.max_drawdown * 100,  # 15%
                current_value=0.0,
                threshold=0.8
            ),
            RiskType.CORRELATION: RiskLimit(
                risk_type=RiskType.CORRELATION,
                limit_value=self.max_correlation * 100,  # 70%
                current_value=0.0,
                threshold=0.8
            ),
            RiskType.LIQUIDITY: RiskLimit(
                risk_type=RiskType.LIQUIDITY,
                limit_value=self.min_liquidity,
                current_value=0.0,
                threshold=1.2  # Warning at 120% of minimum
            ),
            RiskType.MARGIN: RiskLimit(
                risk_type=RiskType.MARGIN,
                limit_value=self.margin_maintenance * 100,  # 25%
                current_value=0.0,
                threshold=0.8
            ),
            RiskType.VOLATILITY: RiskLimit(
                risk_type=RiskType.VOLATILITY,
                limit_value=0.30 * 100,  # 30% max volatility
                current_value=0.0,
                threshold=0.8
            )
        }
    
    async def validate_order(self, symbol: str, side: str, quantity: float, 
                           price: float, current_positions: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate an order against risk limits."""
        try:
            # Calculate order value
            order_value = quantity * price
            
            # Get current portfolio metrics
            if self.position_tracker and self.position_tracker.portfolio_metrics:
                portfolio_value = self.position_tracker.portfolio_metrics.total_value
            else:
                portfolio_value = 100000.0  # Default portfolio value
            
            # Check position size limit
            current_position = current_positions.get(symbol, {"quantity": 0, "market_value": 0})
            new_position_value = current_position["market_value"] + order_value
            position_weight = (new_position_value / portfolio_value) * 100
            
            if position_weight > self.risk_limits[RiskType.POSITION_SIZE].limit_value:
                return False, f"Position size limit exceeded: {position_weight:.1f}% > {self.risk_limits[RiskType.POSITION_SIZE].limit_value}%"
            
            # Check leverage limit
            total_market_value = sum(pos.get("market_value", 0) for pos in current_positions.values())
            new_total_market_value = total_market_value + order_value
            leverage = new_total_market_value / portfolio_value
            
            if leverage > self.risk_limits[RiskType.LEVERAGE].limit_value:
                return False, f"Leverage limit exceeded: {leverage:.2f}x > {self.risk_limits[RiskType.LEVERAGE].limit_value}x"
            
            # Check drawdown limit
            if self.position_tracker and self.position_tracker.portfolio_metrics:
                current_drawdown = abs(self.position_tracker.portfolio_metrics.max_drawdown) * 100
                if current_drawdown > self.risk_limits[RiskType.DRAWDOWN].limit_value:
                    return False, f"Drawdown limit exceeded: {current_drawdown:.1f}% > {self.risk_limits[RiskType.DRAWDOWN].limit_value}%"
            
            # Check liquidity (simplified)
            # In a real implementation, this would check actual market liquidity
            if order_value > self.min_liquidity * 0.1:  # 10% of daily volume
                return False, f"Order size too large for market liquidity: ${order_value:,.0f}"
            
            return True, "Order validated successfully"
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False, f"Validation error: {str(e)}"
    
    async def update_risk_metrics(self):
        """Update comprehensive risk metrics."""
        try:
            if not self.position_tracker:
                return
            
            # Get current positions and portfolio metrics
            await self.position_tracker.update_positions_from_brokers()
            
            if not self.position_tracker.portfolio_metrics:
                return
            
            portfolio_metrics = self.position_tracker.portfolio_metrics
            positions = self.position_tracker.positions
            
            # Calculate VaR (simplified)
            portfolio_var_95 = portfolio_metrics.var_95
            portfolio_var_99 = portfolio_metrics.var_99
            
            # Calculate Expected Shortfall (simplified)
            expected_shortfall = portfolio_var_99 * 1.2
            
            # Calculate current drawdown
            current_drawdown = 0.0
            if len(self.portfolio_history) > 0:
                peak_value = max(snapshot["total_value"] for snapshot in self.portfolio_history)
                current_value = portfolio_metrics.total_value
                current_drawdown = (peak_value - current_value) / peak_value * 100
            
            # Calculate Sharpe ratio
            sharpe_ratio = portfolio_metrics.sharpe_ratio
            
            # Calculate Sortino ratio (simplified)
            sortino_ratio = sharpe_ratio * 1.2  # Typically higher than Sharpe
            
            # Calculate Calmar ratio (simplified)
            calmar_ratio = (portfolio_metrics.total_pnl_percent / 252) / (portfolio_metrics.max_drawdown * 100) if portfolio_metrics.max_drawdown > 0 else 0
            
            # Calculate portfolio beta
            portfolio_beta = portfolio_metrics.portfolio_beta
            
            # Calculate portfolio volatility
            portfolio_volatility = portfolio_metrics.portfolio_volatility * 100
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk(positions)
            
            # Calculate concentration risk
            concentration_risk = await self._calculate_concentration_risk(positions)
            
            # Calculate liquidity risk
            liquidity_risk = await self._calculate_liquidity_risk(positions)
            
            # Calculate leverage risk
            leverage_risk = (portfolio_metrics.leverage / self.max_leverage) * 100
            
            # Calculate margin risk
            margin_risk = 0.0
            if portfolio_metrics.margin_used > 0:
                margin_ratio = portfolio_metrics.margin_used / portfolio_metrics.total_value
                margin_risk = (margin_ratio / self.margin_maintenance) * 100
            
            # Calculate overall risk score
            risk_factors = [
                min(concentration_risk / 100, 1.0),
                min(leverage_risk / 100, 1.0),
                min(correlation_risk / 100, 1.0),
                min(liquidity_risk / 100, 1.0),
                min(margin_risk / 100, 1.0),
                min(current_drawdown / (self.max_drawdown * 100), 1.0)
            ]
            overall_risk_score = np.mean(risk_factors) * 100
            
            # Determine risk level
            if overall_risk_score < 25:
                risk_level = RiskLevel.LOW
            elif overall_risk_score < 50:
                risk_level = RiskLevel.MEDIUM
            elif overall_risk_score < 75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            # Create risk metrics
            self.risk_metrics = RiskMetrics(
                portfolio_var_95=portfolio_var_95,
                portfolio_var_99=portfolio_var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=portfolio_metrics.max_drawdown * 100,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                portfolio_beta=portfolio_beta,
                portfolio_volatility=portfolio_volatility,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                leverage_risk=leverage_risk,
                margin_risk=margin_risk,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                last_updated=datetime.now()
            )
            
            # Update risk limits with current values
            self._update_risk_limits()
            
            # Check for risk alerts
            await self._check_risk_alerts()
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    async def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate correlation risk between positions."""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation calculation
        # In a real implementation, this would use actual correlation data
        position_weights = [pos.weight for pos in positions.values()]
        
        # Calculate weighted average correlation (simplified)
        avg_correlation = 0.3  # 30% average correlation
        max_correlation = 0.7  # 70% max correlation
        
        # Weight by position sizes
        total_weight = sum(position_weights)
        if total_weight > 0:
            weighted_correlation = avg_correlation * (total_weight / 100)
            return min(weighted_correlation * 100, max_correlation * 100)
        
        return avg_correlation * 100
    
    async def _calculate_concentration_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate concentration risk."""
        if not positions:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        position_weights = [pos.weight for pos in positions.values()]
        hhi = sum(weight ** 2 for weight in position_weights)
        
        # Convert to percentage (0-100 scale)
        concentration_risk = min(hhi, 100.0)
        
        return concentration_risk
    
    async def _calculate_liquidity_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate liquidity risk."""
        if not positions:
            return 0.0
        
        # Simplified liquidity risk calculation
        # In a real implementation, this would use actual liquidity data
        total_market_value = sum(pos.market_value for pos in positions.values())
        
        # Assume liquidity risk increases with position size
        liquidity_risk = min((total_market_value / 10000000) * 100, 100.0)  # 10M threshold
        
        return liquidity_risk
    
    def _update_risk_limits(self):
        """Update risk limits with current values."""
        if not self.risk_metrics:
            return
        
        # Update current values
        self.risk_limits[RiskType.LEVERAGE].current_value = self.risk_metrics.leverage_risk
        self.risk_limits[RiskType.DRAWDOWN].current_value = self.risk_metrics.current_drawdown
        self.risk_limits[RiskType.CORRELATION].current_value = self.risk_metrics.correlation_risk
        self.risk_limits[RiskType.LIQUIDITY].current_value = self.risk_metrics.liquidity_risk
        self.risk_limits[RiskType.MARGIN].current_value = self.risk_metrics.margin_risk
        self.risk_limits[RiskType.VOLATILITY].current_value = self.risk_metrics.portfolio_volatility
        self.risk_limits[RiskType.CONCENTRATION].current_value = self.risk_metrics.concentration_risk
        
        # Update timestamps
        for limit in self.risk_limits.values():
            limit.last_updated = datetime.now()
    
    async def _check_risk_alerts(self):
        """Check for risk limit violations and create alerts."""
        if not self.risk_metrics:
            return
        
        # Check each risk limit
        for risk_type, limit in self.risk_limits.items():
            if not limit.is_active:
                continue
            
            # Check if limit is exceeded
            if limit.current_value > limit.limit_value:
                severity = RiskLevel.CRITICAL
                message = f"{risk_type.value.replace('_', ' ').title()} limit exceeded: {limit.current_value:.1f}% > {limit.limit_value:.1f}%"
                recommendation = self._get_risk_recommendation(risk_type)
                
                # Create alert
                alert = RiskAlert(
                    alert_id=f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{risk_type.value}",
                    risk_type=risk_type,
                    severity=severity,
                    message=message,
                    current_value=limit.current_value,
                    limit_value=limit.limit_value,
                    recommendation=recommendation,
                    timestamp=datetime.now()
                )
                
                self.risk_alerts.append(alert)
                
            # Check if threshold is exceeded (warning level)
            elif limit.current_value > limit.limit_value * limit.threshold:
                severity = RiskLevel.HIGH
                message = f"{risk_type.value.replace('_', ' ').title()} approaching limit: {limit.current_value:.1f}% > {limit.limit_value * limit.threshold:.1f}%"
                recommendation = self._get_risk_recommendation(risk_type)
                
                # Create alert
                alert = RiskAlert(
                    alert_id=f"WARNING_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{risk_type.value}",
                    risk_type=risk_type,
                    severity=severity,
                    message=message,
                    current_value=limit.current_value,
                    limit_value=limit.limit_value,
                    recommendation=recommendation,
                    timestamp=datetime.now()
                )
                
                self.risk_alerts.append(alert)
        
        # Keep only recent alerts (last 100)
        self.risk_alerts = self.risk_alerts[-100:]
    
    def _get_risk_recommendation(self, risk_type: RiskType) -> str:
        """Get risk mitigation recommendation."""
        recommendations = {
            RiskType.POSITION_SIZE: "Reduce position size or diversify across more positions",
            RiskType.SECTOR_EXPOSURE: "Reduce sector concentration by rebalancing across sectors",
            RiskType.LEVERAGE: "Reduce leverage by closing positions or adding cash",
            RiskType.DRAWDOWN: "Consider reducing position sizes or implementing stop-losses",
            RiskType.CORRELATION: "Diversify into less correlated assets",
            RiskType.LIQUIDITY: "Reduce position sizes in illiquid assets",
            RiskType.MARGIN: "Add cash to account or reduce margin positions",
            RiskType.VOLATILITY: "Reduce exposure to high-volatility assets"
        }
        
        return recommendations.get(risk_type, "Review and adjust portfolio allocation")
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        await self.update_risk_metrics()
        
        # Get active alerts
        active_alerts = [alert for alert in self.risk_alerts if not alert.is_acknowledged]
        critical_alerts = [alert for alert in active_alerts if alert.severity == RiskLevel.CRITICAL]
        high_alerts = [alert for alert in active_alerts if alert.severity == RiskLevel.HIGH]
        
        return {
            "risk_metrics": asdict(self.risk_metrics) if self.risk_metrics else None,
            "risk_limits": {limit.risk_type.value: asdict(limit) for limit in self.risk_limits.values()},
            "active_alerts": [asdict(alert) for alert in active_alerts],
            "critical_alerts": [asdict(alert) for alert in critical_alerts],
            "high_alerts": [asdict(alert) for alert in high_alerts],
            "total_alerts": len(active_alerts),
            "risk_level": self.risk_metrics.risk_level.value if self.risk_metrics else "unknown",
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_risk_alerts(self, severity: Optional[RiskLevel] = None) -> List[RiskAlert]:
        """Get risk alerts, optionally filtered by severity."""
        alerts = [alert for alert in self.risk_alerts if not alert.is_acknowledged]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert."""
        for alert in self.risk_alerts:
            if alert.alert_id == alert_id:
                alert.is_acknowledged = True
                return True
        return False
    
    def update_risk_limit(self, risk_type: RiskType, new_limit: float, threshold: float = 0.8):
        """Update a risk limit."""
        if risk_type in self.risk_limits:
            self.risk_limits[risk_type].limit_value = new_limit
            self.risk_limits[risk_type].threshold = threshold
            self.risk_limits[risk_type].last_updated = datetime.now()
            logger.info(f"Updated {risk_type.value} limit to {new_limit}")
    
    def enable_risk_limit(self, risk_type: RiskType):
        """Enable a risk limit."""
        if risk_type in self.risk_limits:
            self.risk_limits[risk_type].is_active = True
            logger.info(f"Enabled {risk_type.value} limit")
    
    def disable_risk_limit(self, risk_type: RiskType):
        """Disable a risk limit."""
        if risk_type in self.risk_limits:
            self.risk_limits[risk_type].is_active = False
            logger.info(f"Disabled {risk_type.value} limit")
    
    async def get_position_sizing_recommendation(self, symbol: str, signal_confidence: float, 
                                                current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Get position sizing recommendation based on risk limits."""
        try:
            if not self.position_tracker or not self.position_tracker.portfolio_metrics:
                return {"error": "Portfolio metrics not available"}
            
            portfolio_value = self.position_tracker.portfolio_metrics.total_value
            
            # Calculate base position size (2% of portfolio per 1% confidence)
            base_position_size = portfolio_value * 0.02 * signal_confidence
            
            # Apply risk limits
            max_position_value = portfolio_value * self.max_position_size
            
            # Check current position
            current_position = current_positions.get(symbol, {"market_value": 0})
            current_value = current_position["market_value"]
            
            # Calculate recommended position size
            recommended_value = min(base_position_size, max_position_value - current_value)
            
            # Ensure positive value
            recommended_value = max(0, recommended_value)
            
            # Get current price (simplified)
            current_price = 100.0  # Would get from market data
            recommended_quantity = recommended_value / current_price
            
            return {
                "symbol": symbol,
                "recommended_quantity": recommended_quantity,
                "recommended_value": recommended_value,
                "current_position_value": current_value,
                "max_position_value": max_position_value,
                "signal_confidence": signal_confidence,
                "risk_adjusted": True,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating position sizing: {e}")
            return {"error": str(e)}
