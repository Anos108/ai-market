"""
Market Environment for Reinforcement Learning

This module provides a market simulation environment for training RL agents
with realistic market dynamics, transaction costs, and risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import random

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market states for environment."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    pnl: float


@dataclass
class EnvironmentState:
    """Represents the current state of the market environment."""
    current_step: int
    current_price: float
    market_state: MarketState
    portfolio_value: float
    cash: float
    positions: Dict[str, Position]
    trades: List[Trade]
    market_data: pd.DataFrame
    technical_indicators: Dict[str, float]
    timestamp: datetime


class MarketEnvironment:
    """
    Market simulation environment for RL training.
    
    This environment simulates realistic market conditions including:
    - Price movements with trend and volatility
    - Transaction costs and slippage
    - Portfolio management and risk constraints
    - Market regime changes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the market environment.
        
        Args:
            config: Configuration dictionary for the environment
        """
        default_config = {
            'initial_cash': 100000.0,
            'initial_price': 100.0,
            'transaction_cost': 0.001,  # 0.1% transaction cost
            'slippage': 0.0005,  # 0.05% slippage
            'max_position_size': 0.2,  # 20% max position size
            'max_leverage': 1.0,
            'volatility': 0.02,  # 2% daily volatility
            'trend_strength': 0.0,
            'regime_change_probability': 0.05,
            'episode_length': 252,  # 1 year of trading days
            'market_hours': True,
            'weekend_trading': False,
            'holiday_trading': False
        }
        
        self.config = config or default_config
        
        # Environment state
        self.current_step = 0
        self.current_price = self.config['initial_price']
        self.cash = self.config['initial_cash']
        self.positions = {}
        self.trades = []
        self.market_data = pd.DataFrame()
        self.technical_indicators = {}
        
        # Market dynamics
        self.market_state = MarketState.SIDEWAYS
        self.trend_direction = 0.0
        self.volatility = self.config['volatility']
        self.regime_duration = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_returns = []
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        logger.info(f"Initialized MarketEnvironment with config: {self.config}")
    
    def reset(self) -> EnvironmentState:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial environment state
        """
        try:
            # Reset environment variables
            self.current_step = 0
            self.current_price = self.config['initial_price']
            self.cash = self.config['initial_cash']
            self.positions = {}
            self.trades = []
            self.market_data = pd.DataFrame()
            self.technical_indicators = {}
            
            # Reset market dynamics
            self.market_state = MarketState.SIDEWAYS
            self.trend_direction = 0.0
            self.volatility = self.config['volatility']
            self.regime_duration = 0
            
            # Generate initial market data
            self._generate_initial_market_data()
            
            # Calculate initial technical indicators
            self._calculate_technical_indicators()
            
            # Create initial state
            initial_state = self._create_environment_state()
            
            logger.info("Environment reset to initial state")
            return initial_state
            
        except Exception as e:
            logger.error(f"Environment reset failed: {e}")
            return self._create_default_state()
    
    def step(self, action: int) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=buy, 1=hold, 2=sell, 3=strong_buy, 4=strong_sell)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        try:
            # Execute action
            reward = self._execute_action(action)
            
            # Update market
            self._update_market()
            
            # Update positions
            self._update_positions()
            
            # Calculate technical indicators
            self._calculate_technical_indicators()
            
            # Check if episode is done
            done = self._is_done()
            
            # Create next state
            next_state = self._create_environment_state()
            
            # Create info dictionary
            info = self._create_info_dict()
            
            # Increment step
            self.current_step += 1
            
            return next_state, reward, done, info
            
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            return self._create_default_state(), 0.0, True, {"error": str(e)}
    
    def _execute_action(self, action: int) -> float:
        """Execute trading action and return reward."""
        try:
            # Map action to trading decision
            action_map = {
                0: "buy",
                1: "hold", 
                2: "sell",
                3: "strong_buy",
                4: "strong_sell"
            }
            
            action_type = action_map.get(action, "hold")
            
            # Execute trade if not hold
            if action_type != "hold":
                trade_result = self._execute_trade(action_type)
                return self._calculate_reward(trade_result)
            
            return 0.0  # No reward for hold action
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return 0.0
    
    def _execute_trade(self, action_type: str) -> Dict[str, Any]:
        """Execute a trade based on action type."""
        try:
            # Determine trade parameters
            if action_type in ["buy", "strong_buy"]:
                side = "buy"
                quantity = self._calculate_trade_quantity(action_type)
            elif action_type in ["sell", "strong_sell"]:
                side = "sell"
                quantity = self._calculate_trade_quantity(action_type)
            else:
                return {"success": False, "reason": "invalid_action"}
            
            # Check if trade is valid
            if not self._is_valid_trade(side, quantity):
                return {"success": False, "reason": "invalid_trade"}
            
            # Calculate trade price with slippage
            trade_price = self._calculate_trade_price(side)
            
            # Calculate commission
            commission = abs(quantity * trade_price * self.config['transaction_cost'])
            
            # Execute trade
            if side == "buy":
                # Buy trade
                total_cost = quantity * trade_price + commission
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    
                    # Update position
                    if "STOCK" in self.positions:
                        # Add to existing position
                        pos = self.positions["STOCK"]
                        total_quantity = pos.quantity + quantity
                        avg_price = ((pos.quantity * pos.entry_price) + (quantity * trade_price)) / total_quantity
                        pos.quantity = total_quantity
                        pos.entry_price = avg_price
                    else:
                        # Create new position
                        self.positions["STOCK"] = Position(
                            symbol="STOCK",
                            quantity=quantity,
                            entry_price=trade_price,
                            current_price=trade_price,
                            entry_time=datetime.now(),
                            unrealized_pnl=0.0
                        )
                    
                    # Record trade
                    trade = Trade(
                        symbol="STOCK",
                        side=side,
                        quantity=quantity,
                        price=trade_price,
                        timestamp=datetime.now(),
                        commission=commission,
                        pnl=0.0
                    )
                    self.trades.append(trade)
                    
                    return {"success": True, "trade": trade}
                else:
                    return {"success": False, "reason": "insufficient_cash"}
            
            else:  # sell
                # Sell trade
                if "STOCK" in self.positions and self.positions["STOCK"].quantity >= quantity:
                    pos = self.positions["STOCK"]
                    
                    # Calculate PnL
                    pnl = (trade_price - pos.entry_price) * quantity - commission
                    
                    # Update cash
                    self.cash += quantity * trade_price - commission
                    
                    # Update position
                    pos.quantity -= quantity
                    pos.realized_pnl += pnl
                    
                    # Remove position if quantity is zero
                    if pos.quantity <= 0:
                        del self.positions["STOCK"]
                    
                    # Record trade
                    trade = Trade(
                        symbol="STOCK",
                        side=side,
                        quantity=quantity,
                        price=trade_price,
                        timestamp=datetime.now(),
                        commission=commission,
                        pnl=pnl
                    )
                    self.trades.append(trade)
                    
                    return {"success": True, "trade": trade}
                else:
                    return {"success": False, "reason": "insufficient_position"}
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"success": False, "reason": "execution_error"}
    
    def _calculate_trade_quantity(self, action_type: str) -> float:
        """Calculate trade quantity based on action type."""
        try:
            # Base quantity calculation
            portfolio_value = self._get_portfolio_value()
            max_position_value = portfolio_value * self.config['max_position_size']
            
            if action_type in ["buy", "strong_buy"]:
                # Calculate how much to buy
                if action_type == "strong_buy":
                    quantity = max_position_value * 0.8  # 80% of max position
                else:
                    quantity = max_position_value * 0.4  # 40% of max position
                
                return quantity / self.current_price
            
            elif action_type in ["sell", "strong_sell"]:
                # Calculate how much to sell
                if "STOCK" in self.positions:
                    pos = self.positions["STOCK"]
                    if action_type == "strong_sell":
                        return pos.quantity  # Sell all
                    else:
                        return pos.quantity * 0.5  # Sell half
                
                return 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Trade quantity calculation failed: {e}")
            return 0.0
    
    def _is_valid_trade(self, side: str, quantity: float) -> bool:
        """Check if trade is valid."""
        try:
            if quantity <= 0:
                return False
            
            if side == "buy":
                # Check if we have enough cash
                total_cost = quantity * self.current_price * (1 + self.config['transaction_cost'])
                return self.cash >= total_cost
            
            elif side == "sell":
                # Check if we have enough position
                if "STOCK" not in self.positions:
                    return False
                return self.positions["STOCK"].quantity >= quantity
            
            return False
            
        except Exception as e:
            logger.error(f"Trade validation failed: {e}")
            return False
    
    def _calculate_trade_price(self, side: str) -> float:
        """Calculate trade price with slippage."""
        try:
            slippage = self.config['slippage']
            
            if side == "buy":
                # Buy price is higher due to slippage
                return self.current_price * (1 + slippage)
            else:
                # Sell price is lower due to slippage
                return self.current_price * (1 - slippage)
                
        except Exception as e:
            logger.error(f"Trade price calculation failed: {e}")
            return self.current_price
    
    def _calculate_reward(self, trade_result: Dict[str, Any]) -> float:
        """Calculate reward for trade execution."""
        try:
            if not trade_result.get("success", False):
                return -0.01  # Small penalty for failed trades
            
            trade = trade_result.get("trade")
            if not trade:
                return 0.0
            
            # Base reward from PnL
            pnl_reward = trade.pnl / 1000.0  # Normalize PnL
            
            # Transaction cost penalty
            cost_penalty = -trade.commission / 1000.0
            
            # Risk penalty for large positions
            risk_penalty = 0.0
            if "STOCK" in self.positions:
                pos = self.positions["STOCK"]
                position_ratio = (pos.quantity * pos.current_price) / self._get_portfolio_value()
                if position_ratio > self.config['max_position_size']:
                    risk_penalty = -0.1
            
            total_reward = pnl_reward + cost_penalty + risk_penalty
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            return 0.0
    
    def _update_market(self) -> None:
        """Update market state and price."""
        try:
            # Check for regime change
            if random.random() < self.config['regime_change_probability']:
                self._change_market_regime()
            
            # Update trend direction
            self._update_trend()
            
            # Generate price movement
            price_change = self._generate_price_change()
            self.current_price *= (1 + price_change)
            
            # Ensure price doesn't go negative
            self.current_price = max(self.current_price, 0.01)
            
            # Update market data
            self._update_market_data()
            
            # Update regime duration
            self.regime_duration += 1
            
        except Exception as e:
            logger.error(f"Market update failed: {e}")
    
    def _change_market_regime(self) -> None:
        """Change market regime."""
        try:
            # Select new regime
            regimes = list(MarketState)
            current_regime = self.market_state
            
            # Avoid immediate regime repetition
            available_regimes = [r for r in regimes if r != current_regime]
            new_regime = random.choice(available_regimes)
            
            self.market_state = new_regime
            self.regime_duration = 0
            
            # Update market parameters based on regime
            if new_regime == MarketState.BULL:
                self.trend_direction = random.uniform(0.001, 0.005)  # 0.1-0.5% daily trend
                self.volatility = random.uniform(0.015, 0.025)  # 1.5-2.5% volatility
            elif new_regime == MarketState.BEAR:
                self.trend_direction = random.uniform(-0.005, -0.001)  # -0.5 to -0.1% daily trend
                self.volatility = random.uniform(0.020, 0.030)  # 2-3% volatility
            elif new_regime == MarketState.VOLATILE:
                self.trend_direction = 0.0
                self.volatility = random.uniform(0.030, 0.050)  # 3-5% volatility
            elif new_regime == MarketState.TRENDING:
                self.trend_direction = random.uniform(-0.003, 0.003)  # -0.3 to 0.3% daily trend
                self.volatility = random.uniform(0.010, 0.020)  # 1-2% volatility
            else:  # SIDEWAYS
                self.trend_direction = 0.0
                self.volatility = random.uniform(0.015, 0.025)  # 1.5-2.5% volatility
            
            logger.debug(f"Market regime changed to {new_regime.value}")
            
        except Exception as e:
            logger.error(f"Market regime change failed: {e}")
    
    def _update_trend(self) -> None:
        """Update trend direction."""
        try:
            # Add some trend persistence with random walk
            trend_change = random.uniform(-0.0001, 0.0001)
            self.trend_direction += trend_change
            
            # Keep trend within reasonable bounds
            self.trend_direction = max(-0.01, min(0.01, self.trend_direction))
            
        except Exception as e:
            logger.error(f"Trend update failed: {e}")
    
    def _generate_price_change(self) -> float:
        """Generate realistic price change."""
        try:
            # Base price change from trend
            trend_component = self.trend_direction
            
            # Random component from volatility
            random_component = np.random.normal(0, self.volatility)
            
            # Market regime adjustments
            regime_multiplier = 1.0
            if self.market_state == MarketState.VOLATILE:
                regime_multiplier = 1.5
            elif self.market_state == MarketState.TRENDING:
                regime_multiplier = 0.8
            
            # Combine components
            price_change = (trend_component + random_component) * regime_multiplier
            
            # Add some mean reversion
            if abs(price_change) > self.volatility * 2:
                price_change *= 0.5
            
            return price_change
            
        except Exception as e:
            logger.error(f"Price change generation failed: {e}")
            return 0.0
    
    def _update_market_data(self) -> None:
        """Update market data DataFrame."""
        try:
            # Create new row
            new_row = {
                'timestamp': datetime.now(),
                'price': self.current_price,
                'volume': random.uniform(1000000, 5000000),  # Random volume
                'market_state': self.market_state.value,
                'volatility': self.volatility,
                'trend_direction': self.trend_direction
            }
            
            # Add to DataFrame
            new_df = pd.DataFrame([new_row])
            self.market_data = pd.concat([self.market_data, new_df], ignore_index=True)
            
            # Keep only recent data (last 1000 rows)
            if len(self.market_data) > 1000:
                self.market_data = self.market_data.tail(1000).reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Market data update failed: {e}")
    
    def _update_positions(self) -> None:
        """Update position values."""
        try:
            for symbol, position in self.positions.items():
                position.current_price = self.current_price
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
            
        except Exception as e:
            logger.error(f"Position update failed: {e}")
    
    def _calculate_technical_indicators(self) -> None:
        """Calculate technical indicators."""
        try:
            if len(self.market_data) < 20:
                self.technical_indicators = {}
                return
            
            prices = self.market_data['price']
            
            # Simple Moving Averages
            self.technical_indicators['sma_5'] = prices.tail(5).mean()
            self.technical_indicators['sma_20'] = prices.tail(20).mean()
            
            # RSI
            self.technical_indicators['rsi'] = self._calculate_rsi(prices)
            
            # MACD
            self.technical_indicators['macd'] = self._calculate_macd(prices)
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(prices)
            self.technical_indicators['bb_upper'] = bb_upper
            self.technical_indicators['bb_lower'] = bb_lower
            
            # Volatility
            self.technical_indicators['volatility'] = prices.pct_change().std()
            
            # Price momentum
            self.technical_indicators['momentum'] = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10] if len(prices) >= 10 else 0
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            self.technical_indicators = {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """Calculate MACD indicator."""
        try:
            if len(prices) < 26:
                return 0.0
            
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            
            return float(macd.iloc[-1]) if not np.isnan(macd.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        try:
            if len(prices) < period:
                return 0.0, 0.0
            
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            upper = float(upper_band.iloc[-1]) if not np.isnan(upper_band.iloc[-1]) else 0.0
            lower = float(lower_band.iloc[-1]) if not np.isnan(lower_band.iloc[-1]) else 0.0
            
            return upper, lower
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return 0.0, 0.0
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        try:
            # Episode ends after specified number of steps
            return self.current_step >= self.config['episode_length']
            
        except Exception as e:
            logger.error(f"Done check failed: {e}")
            return True
    
    def _create_environment_state(self) -> EnvironmentState:
        """Create current environment state."""
        try:
            return EnvironmentState(
                current_step=self.current_step,
                current_price=self.current_price,
                market_state=self.market_state,
                portfolio_value=self._get_portfolio_value(),
                cash=self.cash,
                positions=self.positions.copy(),
                trades=self.trades.copy(),
                market_data=self.market_data.copy(),
                technical_indicators=self.technical_indicators.copy(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Environment state creation failed: {e}")
            return self._create_default_state()
    
    def _create_default_state(self) -> EnvironmentState:
        """Create default environment state."""
        return EnvironmentState(
            current_step=0,
            current_price=self.config['initial_price'],
            market_state=MarketState.SIDEWAYS,
            portfolio_value=self.config['initial_cash'],
            cash=self.config['initial_cash'],
            positions={},
            trades=[],
            market_data=pd.DataFrame(),
            technical_indicators={},
            timestamp=datetime.now()
        )
    
    def _create_info_dict(self) -> Dict[str, Any]:
        """Create info dictionary for step."""
        try:
            return {
                'portfolio_value': self._get_portfolio_value(),
                'cash': self.cash,
                'positions': len(self.positions),
                'trades': len(self.trades),
                'market_state': self.market_state.value,
                'volatility': self.volatility,
                'trend_direction': self.trend_direction,
                'regime_duration': self.regime_duration
            }
            
        except Exception as e:
            logger.error(f"Info dict creation failed: {e}")
            return {}
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        try:
            total_value = self.cash
            
            for position in self.positions.values():
                total_value += position.quantity * position.current_price
            
            return total_value
            
        except Exception as e:
            logger.error(f"Portfolio value calculation failed: {e}")
            return self.cash
    
    def _generate_initial_market_data(self) -> None:
        """Generate initial market data."""
        try:
            # Generate some historical data for technical indicators
            initial_data = []
            base_price = self.config['initial_price']
            
            for i in range(50):  # 50 days of historical data
                # Generate price with some trend and volatility
                price_change = np.random.normal(0, self.volatility)
                base_price *= (1 + price_change)
                
                initial_data.append({
                    'timestamp': datetime.now() - timedelta(days=50-i),
                    'price': base_price,
                    'volume': random.uniform(1000000, 5000000),
                    'market_state': self.market_state.value,
                    'volatility': self.volatility,
                    'trend_direction': self.trend_direction
                })
            
            self.market_data = pd.DataFrame(initial_data)
            self.current_price = base_price
            
        except Exception as e:
            logger.error(f"Initial market data generation failed: {e}")
            self.market_data = pd.DataFrame()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the episode."""
        try:
            if not self.trades:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'avg_trade_pnl': 0.0
                }
            
            # Calculate returns
            initial_value = self.config['initial_cash']
            final_value = self._get_portfolio_value()
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate trade statistics
            trade_pnls = [trade.pnl for trade in self.trades]
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0
            avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0.0
            
            # Calculate Sharpe ratio (simplified)
            if len(trade_pnls) > 1:
                sharpe_ratio = np.mean(trade_pnls) / np.std(trade_pnls) if np.std(trade_pnls) > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown (simplified)
            portfolio_values = [initial_value]
            for trade in self.trades:
                portfolio_values.append(portfolio_values[-1] + trade.pnl)
            
            peak = initial_value
            max_drawdown = 0.0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_trade_pnl': avg_trade_pnl
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0
            }
