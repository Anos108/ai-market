"""
Portfolio Management System for AI Market Analysis System

This module provides comprehensive portfolio management capabilities including
optimization, risk management, and performance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optimization imports
try:
    from scipy.optimize import minimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Portfolio optimization will be limited.")

logger = logging.getLogger(__name__)


class PortfolioStrategy(Enum):
    """Portfolio strategies."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"


class RebalancingFrequency(Enum):
    """Rebalancing frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    DYNAMIC = "dynamic"


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    weight: float
    last_updated: datetime


@dataclass
class Transaction:
    """Portfolio transaction."""
    id: str
    symbol: str
    transaction_type: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    notes: str = ""


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_percent: float
    daily_return: float
    daily_return_percent: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    var_95: float
    var_99: float
    expected_shortfall: float


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""
    initial_capital: float = 100000.0
    strategy: PortfolioStrategy = PortfolioStrategy.EQUAL_WEIGHT
    rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY
    max_position_size: float = 0.1  # 10% max per position
    min_position_size: float = 0.01  # 1% min per position
    transaction_cost: float = 0.001  # 0.1% transaction cost
    risk_free_rate: float = 0.02  # 2% risk-free rate
    target_volatility: float = 0.15  # 15% target volatility
    max_turnover: float = 0.5  # 50% max turnover per rebalancing


class PortfolioManager:
    """
    Portfolio management system.
    
    This class provides:
    - Portfolio optimization
    - Risk management
    - Performance tracking
    - Rebalancing strategies
    - Transaction management
    """
    
    def __init__(self, config: PortfolioConfig):
        """
        Initialize the portfolio manager.
        
        Args:
            config: Portfolio configuration
        """
        self.config = config
        self.positions = {}
        self.transactions = []
        self.cash = config.initial_capital
        self.portfolio_history = []
        self.benchmark_data = None
        self.current_metrics = None
        
        logger.info(f"Initialized PortfolioManager with {config.initial_capital} capital")
    
    def add_position(self, symbol: str, quantity: float, price: float, 
                    timestamp: Optional[datetime] = None) -> None:
        """
        Add a position to the portfolio.
        
        Args:
            symbol: Asset symbol
            quantity: Number of shares
            price: Purchase price
            timestamp: Transaction timestamp
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Create transaction
            transaction = Transaction(
                id=f"{symbol}_{timestamp.isoformat()}",
                symbol=symbol,
                transaction_type="buy",
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                fees=quantity * price * self.config.transaction_cost
            )
            
            self.transactions.append(transaction)
            
            # Update cash
            total_cost = quantity * price + transaction.fees
            self.cash -= total_cost
            
            # Update position
            if symbol in self.positions:
                # Average down/up
                existing_pos = self.positions[symbol]
                total_quantity = existing_pos.quantity + quantity
                total_cost_basis = (existing_pos.quantity * existing_pos.average_price + 
                                  quantity * price)
                new_average_price = total_cost_basis / total_quantity
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    average_price=new_average_price,
                    current_price=price,
                    market_value=total_quantity * price,
                    unrealized_pnl=total_quantity * (price - new_average_price),
                    unrealized_pnl_percent=(price - new_average_price) / new_average_price,
                    weight=0.0,  # Will be calculated
                    last_updated=timestamp
                )
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_percent=0.0,
                    weight=0.0,  # Will be calculated
                    last_updated=timestamp
                )
            
            self._update_portfolio_weights()
            logger.info(f"Added position: {quantity} shares of {symbol} at ${price}")
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            raise
    
    def remove_position(self, symbol: str, quantity: float, price: float,
                       timestamp: Optional[datetime] = None) -> None:
        """
        Remove a position from the portfolio.
        
        Args:
            symbol: Asset symbol
            quantity: Number of shares to sell
            price: Sale price
            timestamp: Transaction timestamp
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if symbol not in self.positions:
                raise ValueError(f"Position {symbol} not found")
            
            position = self.positions[symbol]
            
            if quantity > position.quantity:
                raise ValueError(f"Insufficient shares. Have {position.quantity}, trying to sell {quantity}")
            
            # Create transaction
            transaction = Transaction(
                id=f"{symbol}_{timestamp.isoformat()}_sell",
                symbol=symbol,
                transaction_type="sell",
                quantity=quantity,
                price=price,
                timestamp=timestamp,
                fees=quantity * price * self.config.transaction_cost
            )
            
            self.transactions.append(transaction)
            
            # Update cash
            proceeds = quantity * price - transaction.fees
            self.cash += proceeds
            
            # Update position
            if quantity == position.quantity:
                # Close position
                del self.positions[symbol]
            else:
                # Partial sale
                remaining_quantity = position.quantity - quantity
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=remaining_quantity,
                    average_price=position.average_price,
                    current_price=price,
                    market_value=remaining_quantity * price,
                    unrealized_pnl=remaining_quantity * (price - position.average_price),
                    unrealized_pnl_percent=(price - position.average_price) / position.average_price,
                    weight=0.0,  # Will be calculated
                    last_updated=timestamp
                )
            
            self._update_portfolio_weights()
            logger.info(f"Removed position: {quantity} shares of {symbol} at ${price}")
            
        except Exception as e:
            logger.error(f"Failed to remove position: {e}")
            raise
    
    def update_prices(self, price_data: Dict[str, float]) -> None:
        """
        Update current prices for all positions.
        
        Args:
            price_data: Dictionary of symbol -> current price
        """
        try:
            for symbol, position in self.positions.items():
                if symbol in price_data:
                    new_price = price_data[symbol]
                    position.current_price = new_price
                    position.market_value = position.quantity * new_price
                    position.unrealized_pnl = position.quantity * (new_price - position.average_price)
                    position.unrealized_pnl_percent = (new_price - position.average_price) / position.average_price
                    position.last_updated = datetime.now()
            
            self._update_portfolio_weights()
            
        except Exception as e:
            logger.error(f"Failed to update prices: {e}")
    
    def _update_portfolio_weights(self) -> None:
        """Update portfolio weights."""
        try:
            total_value = self.get_total_value()
            
            if total_value > 0:
                for position in self.positions.values():
                    position.weight = position.market_value / total_value
                    
        except Exception as e:
            logger.error(f"Failed to update weights: {e}")
    
    def get_total_value(self) -> float:
        """Get total portfolio value."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    def get_total_cost(self) -> float:
        """Get total cost basis."""
        return sum(pos.quantity * pos.average_price for pos in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """Get total profit/loss."""
        return self.get_total_value() - self.config.initial_capital
    
    def get_total_pnl_percent(self) -> float:
        """Get total profit/loss percentage."""
        return self.get_total_pnl() / self.config.initial_capital
    
    def optimize_portfolio(self, expected_returns: Dict[str, float], 
                          covariance_matrix: pd.DataFrame,
                          target_return: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            target_return: Target portfolio return (optional)
            
        Returns:
            Optimized weights
        """
        try:
            if not SCIPY_AVAILABLE:
                logger.warning("SciPy not available. Using equal weights.")
                return self._get_equal_weights()
            
            symbols = list(expected_returns.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Convert to numpy arrays
            returns = np.array([expected_returns[symbol] for symbol in symbols])
            cov_matrix = covariance_matrix.loc[symbols, symbols].values
            
            if self.config.strategy == PortfolioStrategy.MINIMUM_VARIANCE:
                weights = self._optimize_minimum_variance(cov_matrix)
            elif self.config.strategy == PortfolioStrategy.MAXIMUM_SHARPE:
                weights = self._optimize_maximum_sharpe(returns, cov_matrix)
            elif self.config.strategy == PortfolioStrategy.RISK_PARITY:
                weights = self._optimize_risk_parity(cov_matrix)
            else:
                weights = self._get_equal_weights()
            
            # Apply constraints
            weights = self._apply_weight_constraints(weights, symbols)
            
            return dict(zip(symbols, weights))
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._get_equal_weights()
    
    def _optimize_minimum_variance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize for minimum variance."""
        try:
            n_assets = cov_matrix.shape[0]
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints: weights sum to 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Bounds: weights between 0 and 1
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            return result.x if result.success else np.ones(n_assets) / n_assets
            
        except Exception as e:
            logger.error(f"Minimum variance optimization failed: {e}")
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
    
    def _optimize_maximum_sharpe(self, returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize for maximum Sharpe ratio."""
        try:
            n_assets = len(returns)
            
            # Objective function: minimize negative Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_std = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_std
                return -sharpe_ratio
            
            # Constraints: weights sum to 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Bounds: weights between 0 and 1
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            return result.x if result.success else np.ones(n_assets) / n_assets
            
        except Exception as e:
            logger.error(f"Maximum Sharpe optimization failed: {e}")
            return np.ones(len(returns)) / len(returns)
    
    def _optimize_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize for risk parity."""
        try:
            n_assets = cov_matrix.shape[0]
            
            # Objective function: minimize sum of squared risk contributions
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                risk_contributions = (weights * np.dot(cov_matrix, weights)) / portfolio_variance
                target_contributions = np.ones(n_assets) / n_assets
                return np.sum((risk_contributions - target_contributions) ** 2)
            
            # Constraints: weights sum to 1
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            
            # Bounds: weights between 0 and 1
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess: equal weights
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            return result.x if result.success else np.ones(n_assets) / n_assets
            
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return np.ones(cov_matrix.shape[0]) / cov_matrix.shape[0]
    
    def _get_equal_weights(self) -> Dict[str, float]:
        """Get equal weights for all positions."""
        if not self.positions:
            return {}
        
        weight = 1.0 / len(self.positions)
        return {symbol: weight for symbol in self.positions.keys()}
    
    def _apply_weight_constraints(self, weights: np.ndarray, symbols: List[str]) -> np.ndarray:
        """Apply weight constraints."""
        try:
            # Apply min/max position size constraints
            weights = np.clip(weights, self.config.min_position_size, self.config.max_position_size)
            
            # Renormalize to sum to 1
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            logger.error(f"Weight constraint application failed: {e}")
            return weights
    
    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                           current_prices: Dict[str, float]) -> List[Transaction]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Target weights for each asset
            current_prices: Current prices for each asset
            
        Returns:
            List of transactions executed
        """
        try:
            transactions = []
            total_value = self.get_total_value()
            
            for symbol, target_weight in target_weights.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                target_value = total_value * target_weight
                
                if symbol in self.positions:
                    current_value = self.positions[symbol].market_value
                    current_weight = current_value / total_value
                else:
                    current_value = 0.0
                    current_weight = 0.0
                
                # Calculate required change
                value_change = target_value - current_value
                
                if abs(value_change) > total_value * 0.001:  # 0.1% threshold
                    if value_change > 0:
                        # Buy more
                        quantity = value_change / current_price
                        self.add_position(symbol, quantity, current_price)
                        transactions.append(Transaction(
                            id=f"rebalance_{symbol}_{datetime.now().isoformat()}",
                            symbol=symbol,
                            transaction_type="buy",
                            quantity=quantity,
                            price=current_price,
                            timestamp=datetime.now(),
                            notes="Rebalancing"
                        ))
                    else:
                        # Sell some
                        quantity = abs(value_change) / current_price
                        if symbol in self.positions and quantity <= self.positions[symbol].quantity:
                            self.remove_position(symbol, quantity, current_price)
                            transactions.append(Transaction(
                                id=f"rebalance_{symbol}_{datetime.now().isoformat()}_sell",
                                symbol=symbol,
                                transaction_type="sell",
                                quantity=quantity,
                                price=current_price,
                                timestamp=datetime.now(),
                                notes="Rebalancing"
                            ))
            
            logger.info(f"Portfolio rebalanced with {len(transactions)} transactions")
            return transactions
            
        except Exception as e:
            logger.error(f"Portfolio rebalancing failed: {e}")
            return []
    
    def calculate_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> PortfolioMetrics:
        """
        Calculate portfolio performance metrics.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Portfolio metrics
        """
        try:
            total_value = self.get_total_value()
            total_cost = self.get_total_cost()
            total_pnl = self.get_total_pnl()
            total_pnl_percent = self.get_total_pnl_percent()
            
            # Calculate daily return (simplified)
            daily_return = 0.0
            daily_return_percent = 0.0
            
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2]['total_value']
                daily_return = total_value - prev_value
                daily_return_percent = daily_return / prev_value
            
            # Calculate volatility (simplified)
            volatility = 0.0
            if len(self.portfolio_history) > 20:
                returns = [h['daily_return_percent'] for h in self.portfolio_history[-20:]]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate Sharpe ratio
            sharpe_ratio = 0.0
            if volatility > 0:
                avg_return = np.mean([h['daily_return_percent'] for h in self.portfolio_history[-252:]]) if len(self.portfolio_history) > 252 else 0.0
                sharpe_ratio = (avg_return * 252 - self.config.risk_free_rate) / volatility
            
            # Calculate max drawdown
            max_drawdown = 0.0
            if len(self.portfolio_history) > 1:
                peak = self.portfolio_history[0]['total_value']
                for h in self.portfolio_history:
                    if h['total_value'] > peak:
                        peak = h['total_value']
                    drawdown = (peak - h['total_value']) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Calmar ratio
            calmar_ratio = 0.0
            if max_drawdown > 0:
                annual_return = np.mean([h['daily_return_percent'] for h in self.portfolio_history[-252:]]) * 252 if len(self.portfolio_history) > 252 else 0.0
                calmar_ratio = annual_return / max_drawdown
            
            # Calculate VaR (simplified)
            var_95 = 0.0
            var_99 = 0.0
            if len(self.portfolio_history) > 20:
                returns = [h['daily_return_percent'] for h in self.portfolio_history[-252:]]
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
            
            # Calculate expected shortfall
            expected_shortfall = 0.0
            if len(self.portfolio_history) > 20:
                returns = [h['daily_return_percent'] for h in self.portfolio_history[-252:]]
                tail_returns = [r for r in returns if r <= var_95]
                expected_shortfall = np.mean(tail_returns) if tail_returns else 0.0
            
            metrics = PortfolioMetrics(
                total_value=total_value,
                total_cost=total_cost,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                daily_return=daily_return,
                daily_return_percent=daily_return_percent,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=0.0,  # Would need more complex calculation
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                beta=0.0,  # Would need benchmark data
                alpha=0.0,  # Would need benchmark data
                information_ratio=0.0,  # Would need benchmark data
                tracking_error=0.0,  # Would need benchmark data
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall
            )
            
            self.current_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return PortfolioMetrics(
                total_value=self.get_total_value(),
                total_cost=self.get_total_cost(),
                total_pnl=self.get_total_pnl(),
                total_pnl_percent=self.get_total_pnl_percent(),
                daily_return=0.0,
                daily_return_percent=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                beta=0.0,
                alpha=0.0,
                information_ratio=0.0,
                tracking_error=0.0,
                var_95=0.0,
                var_99=0.0,
                expected_shortfall=0.0
            )
    
    def save_portfolio_state(self, filepath: str) -> None:
        """Save portfolio state to file."""
        try:
            state = {
                'config': self.config.__dict__,
                'positions': {symbol: {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_percent': pos.unrealized_pnl_percent,
                    'weight': pos.weight,
                    'last_updated': pos.last_updated.isoformat()
                } for symbol, pos in self.positions.items()},
                'transactions': [{
                    'id': txn.id,
                    'symbol': txn.symbol,
                    'transaction_type': txn.transaction_type,
                    'quantity': txn.quantity,
                    'price': txn.price,
                    'timestamp': txn.timestamp.isoformat(),
                    'fees': txn.fees,
                    'notes': txn.notes
                } for txn in self.transactions],
                'cash': self.cash,
                'portfolio_history': self.portfolio_history
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Portfolio state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")
    
    def load_portfolio_state(self, filepath: str) -> None:
        """Load portfolio state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load positions
            self.positions = {}
            for symbol, pos_data in state['positions'].items():
                self.positions[symbol] = Position(
                    symbol=pos_data['symbol'],
                    quantity=pos_data['quantity'],
                    average_price=pos_data['average_price'],
                    current_price=pos_data['current_price'],
                    market_value=pos_data['market_value'],
                    unrealized_pnl=pos_data['unrealized_pnl'],
                    unrealized_pnl_percent=pos_data['unrealized_pnl_percent'],
                    weight=pos_data['weight'],
                    last_updated=datetime.fromisoformat(pos_data['last_updated'])
                )
            
            # Load transactions
            self.transactions = []
            for txn_data in state['transactions']:
                self.transactions.append(Transaction(
                    id=txn_data['id'],
                    symbol=txn_data['symbol'],
                    transaction_type=txn_data['transaction_type'],
                    quantity=txn_data['quantity'],
                    price=txn_data['price'],
                    timestamp=datetime.fromisoformat(txn_data['timestamp']),
                    fees=txn_data['fees'],
                    notes=txn_data['notes']
                ))
            
            # Load other data
            self.cash = state['cash']
            self.portfolio_history = state['portfolio_history']
            
            logger.info(f"Portfolio state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        try:
            metrics = self.calculate_metrics()
            
            summary = {
                'total_value': metrics.total_value,
                'total_pnl': metrics.total_pnl,
                'total_pnl_percent': metrics.total_pnl_percent,
                'cash': self.cash,
                'num_positions': len(self.positions),
                'num_transactions': len(self.transactions),
                'strategy': self.config.strategy.value,
                'rebalancing_frequency': self.config.rebalancing_frequency.value,
                'positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'market_value': pos.market_value,
                        'weight': pos.weight,
                        'unrealized_pnl_percent': pos.unrealized_pnl_percent
                    } for symbol, pos in self.positions.items()
                },
                'metrics': metrics.__dict__
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Portfolio summary failed: {e}")
            return {}
