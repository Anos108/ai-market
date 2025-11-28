"""
Reward Functions for Reinforcement Learning

This module provides various reward functions for training RL agents,
including return-based, risk-adjusted, and multi-objective rewards.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of reward functions."""
    SIMPLE_RETURN = "simple_return"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MULTI_OBJECTIVE = "multi_objective"
    CUSTOM = "custom"


@dataclass
class RewardComponents:
    """Components of a reward calculation."""
    return_reward: float
    risk_penalty: float
    drawdown_penalty: float
    transaction_cost: float
    volatility_penalty: float
    diversification_bonus: float
    total_reward: float


class RewardFunction:
    """
    Reward function for RL agent training.
    
    This class provides various reward calculation methods to optimize
    different aspects of trading performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reward function.
        
        Args:
            config: Configuration dictionary for reward calculation
        """
        default_config = {
            'reward_type': RewardType.MULTI_OBJECTIVE,
            'weights': {
                'return': 1.0,
                'risk_adjusted': 0.5,
                'drawdown': -0.3,
                'transaction_cost': -0.1,
                'volatility': -0.2,
                'diversification': 0.1
            },
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'lookback_period': 20,  # Days to look back for risk calculation
            'max_drawdown_threshold': 0.2,  # 20% max drawdown threshold
            'volatility_threshold': 0.3,  # 30% volatility threshold
            'transaction_cost_rate': 0.001,  # 0.1% transaction cost
            'diversification_threshold': 0.7,  # 70% diversification threshold
            'normalize_rewards': True,
            'reward_scaling': 100.0  # Scale rewards by this factor
        }
        
        self.config = config or default_config
        self.reward_type = RewardType(self.config['reward_type'])
        
        # Performance tracking
        self.returns_history = []
        self.portfolio_values = []
        self.trade_history = []
        self.drawdown_history = []
        
        logger.info(f"Initialized RewardFunction with type: {self.reward_type.value}")
    
    def calculate_reward(self, 
                        current_portfolio_value: float,
                        previous_portfolio_value: float,
                        action: str,
                        market_data: Dict[str, Any],
                        trade_result: Optional[Dict[str, Any]] = None) -> RewardComponents:
        """
        Calculate reward based on current state and action.
        
        Args:
            current_portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            action: Action taken
            market_data: Current market data
            trade_result: Result of trade execution
            
        Returns:
            Reward components and total reward
        """
        try:
            # Calculate return
            return_reward = self._calculate_return_reward(current_portfolio_value, previous_portfolio_value)
            
            # Calculate risk penalty
            risk_penalty = self._calculate_risk_penalty()
            
            # Calculate drawdown penalty
            drawdown_penalty = self._calculate_drawdown_penalty(current_portfolio_value)
            
            # Calculate transaction cost
            transaction_cost = self._calculate_transaction_cost(trade_result)
            
            # Calculate volatility penalty
            volatility_penalty = self._calculate_volatility_penalty(market_data)
            
            # Calculate diversification bonus
            diversification_bonus = self._calculate_diversification_bonus(market_data)
            
            # Calculate total reward based on type
            if self.reward_type == RewardType.SIMPLE_RETURN:
                total_reward = return_reward
            elif self.reward_type == RewardType.RISK_ADJUSTED:
                total_reward = return_reward + risk_penalty
            elif self.reward_type == RewardType.SHARPE_RATIO:
                total_reward = self._calculate_sharpe_reward()
            elif self.reward_type == RewardType.SORTINO_RATIO:
                total_reward = self._calculate_sortino_reward()
            elif self.reward_type == RewardType.CALMAR_RATIO:
                total_reward = self._calculate_calmar_reward()
            elif self.reward_type == RewardType.MULTI_OBJECTIVE:
                total_reward = self._calculate_multi_objective_reward(
                    return_reward, risk_penalty, drawdown_penalty, 
                    transaction_cost, volatility_penalty, diversification_bonus
                )
            else:
                total_reward = return_reward
            
            # Normalize reward if enabled
            if self.config['normalize_rewards']:
                total_reward *= self.config['reward_scaling']
            
            # Update history
            self._update_history(current_portfolio_value, return_reward)
            
            return RewardComponents(
                return_reward=return_reward,
                risk_penalty=risk_penalty,
                drawdown_penalty=drawdown_penalty,
                transaction_cost=transaction_cost,
                volatility_penalty=volatility_penalty,
                diversification_bonus=diversification_bonus,
                total_reward=total_reward
            )
            
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            return RewardComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_return_reward(self, current_value: float, previous_value: float) -> float:
        """Calculate return-based reward."""
        try:
            if previous_value <= 0:
                return 0.0
            
            return (current_value - previous_value) / previous_value
            
        except Exception as e:
            logger.error(f"Return reward calculation failed: {e}")
            return 0.0
    
    def _calculate_risk_penalty(self) -> float:
        """Calculate risk penalty based on volatility."""
        try:
            if len(self.returns_history) < 2:
                return 0.0
            
            # Calculate volatility
            returns = np.array(self.returns_history[-self.config['lookback_period']:])
            volatility = np.std(returns)
            
            # Penalty increases with volatility
            if volatility > self.config['volatility_threshold']:
                penalty = -((volatility - self.config['volatility_threshold']) / self.config['volatility_threshold'])
            else:
                penalty = 0.0
            
            return penalty
            
        except Exception as e:
            logger.error(f"Risk penalty calculation failed: {e}")
            return 0.0
    
    def _calculate_drawdown_penalty(self, current_value: float) -> float:
        """Calculate drawdown penalty."""
        try:
            if not self.portfolio_values:
                return 0.0
            
            # Calculate current drawdown
            peak_value = max(self.portfolio_values)
            current_drawdown = (peak_value - current_value) / peak_value
            
            # Penalty increases with drawdown
            if current_drawdown > self.config['max_drawdown_threshold']:
                penalty = -((current_drawdown - self.config['max_drawdown_threshold']) / self.config['max_drawdown_threshold'])
            else:
                penalty = 0.0
            
            return penalty
            
        except Exception as e:
            logger.error(f"Drawdown penalty calculation failed: {e}")
            return 0.0
    
    def _calculate_transaction_cost(self, trade_result: Optional[Dict[str, Any]]) -> float:
        """Calculate transaction cost penalty."""
        try:
            if not trade_result or not trade_result.get('success', False):
                return 0.0
            
            trade = trade_result.get('trade')
            if not trade:
                return 0.0
            
            # Penalty is the negative of transaction cost
            return -trade.commission / 1000.0  # Normalize
            
        except Exception as e:
            logger.error(f"Transaction cost calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_penalty(self, market_data: Dict[str, Any]) -> float:
        """Calculate volatility penalty based on market conditions."""
        try:
            market_volatility = market_data.get('volatility', 0.02)
            
            # Penalty for high market volatility
            if market_volatility > 0.05:  # 5% volatility threshold
                penalty = -(market_volatility - 0.05) * 10  # Scale penalty
            else:
                penalty = 0.0
            
            return penalty
            
        except Exception as e:
            logger.error(f"Volatility penalty calculation failed: {e}")
            return 0.0
    
    def _calculate_diversification_bonus(self, market_data: Dict[str, Any]) -> float:
        """Calculate diversification bonus."""
        try:
            # This is a simplified diversification calculation
            # In a real implementation, this would consider portfolio composition
            
            positions = market_data.get('positions', {})
            num_positions = len(positions)
            
            # Bonus for having multiple positions
            if num_positions > 1:
                bonus = min(0.1, (num_positions - 1) * 0.05)  # Max 10% bonus
            else:
                bonus = 0.0
            
            return bonus
            
        except Exception as e:
            logger.error(f"Diversification bonus calculation failed: {e}")
            return 0.0
    
    def _calculate_sharpe_reward(self) -> float:
        """Calculate Sharpe ratio-based reward."""
        try:
            if len(self.returns_history) < 2:
                return 0.0
            
            returns = np.array(self.returns_history[-self.config['lookback_period']:])
            
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            
            # Calculate Sharpe ratio
            excess_returns = returns - (self.config['risk_free_rate'] / 252)  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / np.std(returns)
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Sharpe reward calculation failed: {e}")
            return 0.0
    
    def _calculate_sortino_reward(self) -> float:
        """Calculate Sortino ratio-based reward."""
        try:
            if len(self.returns_history) < 2:
                return 0.0
            
            returns = np.array(self.returns_history[-self.config['lookback_period']:])
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate Sortino ratio (downside deviation)
            excess_returns = returns - (self.config['risk_free_rate'] / 252)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                return np.mean(excess_returns) if len(excess_returns) > 0 else 0.0
            
            sortino_ratio = np.mean(excess_returns) / np.std(downside_returns)
            
            return sortino_ratio
            
        except Exception as e:
            logger.error(f"Sortino reward calculation failed: {e}")
            return 0.0
    
    def _calculate_calmar_reward(self) -> float:
        """Calculate Calmar ratio-based reward."""
        try:
            if len(self.portfolio_values) < 2:
                return 0.0
            
            # Calculate annual return
            initial_value = self.portfolio_values[0]
            final_value = self.portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value
            
            # Calculate max drawdown
            peak = initial_value
            max_drawdown = 0.0
            for value in self.portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            if max_drawdown == 0:
                return total_return
            
            # Calculate Calmar ratio
            calmar_ratio = total_return / max_drawdown
            
            return calmar_ratio
            
        except Exception as e:
            logger.error(f"Calmar reward calculation failed: {e}")
            return 0.0
    
    def _calculate_multi_objective_reward(self, 
                                        return_reward: float,
                                        risk_penalty: float,
                                        drawdown_penalty: float,
                                        transaction_cost: float,
                                        volatility_penalty: float,
                                        diversification_bonus: float) -> float:
        """Calculate multi-objective reward."""
        try:
            weights = self.config['weights']
            
            total_reward = (
                weights['return'] * return_reward +
                weights['risk_adjusted'] * risk_penalty +
                weights['drawdown'] * drawdown_penalty +
                weights['transaction_cost'] * transaction_cost +
                weights['volatility'] * volatility_penalty +
                weights['diversification'] * diversification_bonus
            )
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Multi-objective reward calculation failed: {e}")
            return return_reward
    
    def _update_history(self, portfolio_value: float, return_reward: float) -> None:
        """Update performance history."""
        try:
            self.portfolio_values.append(portfolio_value)
            self.returns_history.append(return_reward)
            
            # Keep only recent history
            max_history = self.config['lookback_period'] * 2
            if len(self.portfolio_values) > max_history:
                self.portfolio_values = self.portfolio_values[-max_history:]
            if len(self.returns_history) > max_history:
                self.returns_history = self.returns_history[-max_history:]
                
        except Exception as e:
            logger.error(f"History update failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        try:
            if not self.returns_history:
                return {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'volatility': 0.0,
                    'win_rate': 0.0
                }
            
            returns = np.array(self.returns_history)
            
            # Basic metrics
            total_return = np.sum(returns)
            volatility = np.std(returns)
            win_rate = np.mean(returns > 0)
            
            # Risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_reward()
            sortino_ratio = self._calculate_sortino_reward()
            calmar_ratio = self._calculate_calmar_reward()
            
            # Drawdown
            if self.portfolio_values:
                peak = self.portfolio_values[0]
                max_drawdown = 0.0
                for value in self.portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'win_rate': win_rate
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'win_rate': 0.0
            }
    
    def reset(self) -> None:
        """Reset reward function state."""
        try:
            self.returns_history = []
            self.portfolio_values = []
            self.trade_history = []
            self.drawdown_history = []
            
            logger.info("Reward function reset")
            
        except Exception as e:
            logger.error(f"Reward function reset failed: {e}")
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update reward weights."""
        try:
            self.config['weights'].update(weights)
            logger.info(f"Reward weights updated: {weights}")
            
        except Exception as e:
            logger.error(f"Weight update failed: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        try:
            self.config.update(new_config)
            self.reward_type = RewardType(self.config['reward_type'])
            logger.info(f"Configuration updated: {new_config}")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
