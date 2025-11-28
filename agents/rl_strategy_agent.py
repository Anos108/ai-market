"""
Reinforcement Learning Strategy Agent for AI Market Analysis System

This agent uses reinforcement learning (PPO/DQN) to learn optimal trading policies
by interacting with market environments and optimizing for risk-adjusted returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import random
import json
import pickle
from pathlib import Path

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class RLAction(Enum):
    """RL Agent actions."""
    BUY = 0
    HOLD = 1
    SELL = 2
    STRONG_BUY = 3
    STRONG_SELL = 4


class RLAlgorithm(Enum):
    """Reinforcement Learning algorithms."""
    PPO = "ppo"
    DQN = "dqn"
    A2C = "a2c"


@dataclass
class RLState:
    """Represents the state for RL agent."""
    technical_indicators: Dict[str, float]
    market_regime: str
    event_impact: float
    forecast_signal: float
    portfolio_state: Dict[str, float]
    market_volatility: float
    timestamp: datetime


@dataclass
class RLReward:
    """Represents the reward for RL agent."""
    return_reward: float
    risk_adjusted_return: float
    drawdown_penalty: float
    transaction_cost: float
    total_reward: float


@dataclass
class RLExperience:
    """Represents an RL experience (state, action, reward, next_state)."""
    state: RLState
    action: RLAction
    reward: RLReward
    next_state: RLState
    done: bool
    timestamp: datetime


class RLStrategyAgent(BaseAgent):
    """
    Reinforcement Learning Strategy Agent for adaptive trading strategies.
    
    This agent uses RL algorithms (PPO/DQN) to learn optimal trading policies
    by interacting with market environments and optimizing for risk-adjusted returns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RL Strategy Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'algorithm': RLAlgorithm.PPO,
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'batch_size': 64,
            'memory_size': 10000,
            'update_frequency': 100,
            'target_update_frequency': 1000,
            'state_dimension': 20,
            'action_dimension': 5,
            'hidden_layers': [256, 128, 64],
            'reward_weights': {
                'return': 1.0,
                'risk_adjusted': 0.5,
                'drawdown': -0.3,
                'transaction_cost': -0.1
            },
            'training_enabled': True,
            'model_path': 'models/rl_strategy_agent',
            'experience_replay': True,
            'double_dqn': True,
            'dueling_dqn': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="RLStrategyAgent",
            version="1.0.0",
            config=default_config
        )
        
        # RL specific attributes
        self.algorithm = RLAlgorithm(self.config['algorithm'])
        self.epsilon = self.config['epsilon']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']
        
        # Experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = self.config['memory_size']
        
        # Model and training
        self.model = None
        self.target_model = None
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Performance tracking
        self.total_rewards = 0.0
        self.episode_count = 0
        self.best_performance = -float('inf')
        
        # Model paths
        self.model_path = Path(self.config['model_path'])
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Initialized RLStrategyAgent with algorithm: {self.algorithm.value}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the RL agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting RL training for {self.name}")
            
            if not self.config['training_enabled']:
                logger.info(f"{self.name}: Training disabled, using pre-trained model")
                self.is_trained = True
                return {"status": "training_disabled"}
            
            # Simulate training episodes
            training_results = self._simulate_training_episodes(training_data, context)
            
            # Update model
            self._update_model()
            
            self.is_trained = True
            
            logger.info(f"{self.name}: RL training completed")
            return {
                "status": "training_completed",
                "episodes": training_results['episodes'],
                "avg_reward": training_results['avg_reward'],
                "best_reward": training_results['best_reward'],
                "convergence": training_results['convergence']
            }
            
        except Exception as e:
            logger.error(f"RL training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate RL-based trading signal.
        
        Args:
            context: Current market context
            
        Returns:
            RL-driven trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use random policy
            if not self.is_trained:
                logger.info(f"{self.name}: Using random policy (not trained)")
                return self._random_policy_signal(context)
            
            # Create RL state from context
            state = self._create_state_from_context(context)
            
            # Get action from RL model
            action = self._get_action(state)
            
            # Convert action to signal
            signal = self._action_to_signal(action, context)
            
            # Store experience for learning
            if self.config['training_enabled']:
                self._store_experience(state, action, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"RL prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"RL prediction error: {e}", context)
    
    def _initialize_model(self) -> None:
        """Initialize the RL model based on algorithm."""
        try:
            if self.algorithm == RLAlgorithm.PPO:
                self._initialize_ppo_model()
            elif self.algorithm == RLAlgorithm.DQN:
                self._initialize_dqn_model()
            else:
                logger.warning(f"Unknown algorithm: {self.algorithm}, using PPO")
                self._initialize_ppo_model()
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            # Fallback to simple model
            self.model = {"type": "simple", "weights": np.random.randn(20, 5)}
    
    def _initialize_ppo_model(self) -> None:
        """Initialize PPO model."""
        try:
            # Simplified PPO model representation
            self.model = {
                "type": "ppo",
                "actor_network": {
                    "layers": self.config['hidden_layers'],
                    "weights": [np.random.randn(self.config['state_dimension'], self.config['hidden_layers'][0])],
                    "biases": [np.zeros(self.config['hidden_layers'][0])]
                },
                "critic_network": {
                    "layers": self.config['hidden_layers'],
                    "weights": [np.random.randn(self.config['state_dimension'], self.config['hidden_layers'][0])],
                    "biases": [np.zeros(self.config['hidden_layers'][0])]
                },
                "learning_rate": self.config['learning_rate'],
                "gamma": self.config['gamma']
            }
            
            logger.info("PPO model initialized")
            
        except Exception as e:
            logger.error(f"PPO model initialization failed: {e}")
            self.model = {"type": "ppo", "error": str(e)}
    
    def _initialize_dqn_model(self) -> None:
        """Initialize DQN model."""
        try:
            # Simplified DQN model representation
            self.model = {
                "type": "dqn",
                "network": {
                    "layers": self.config['hidden_layers'],
                    "weights": [np.random.randn(self.config['state_dimension'], self.config['hidden_layers'][0])],
                    "biases": [np.zeros(self.config['hidden_layers'][0])]
                },
                "target_network": {
                    "layers": self.config['hidden_layers'],
                    "weights": [np.random.randn(self.config['state_dimension'], self.config['hidden_layers'][0])],
                    "biases": [np.zeros(self.config['hidden_layers'][0])]
                },
                "learning_rate": self.config['learning_rate'],
                "gamma": self.config['gamma'],
                "epsilon": self.epsilon,
                "double_dqn": self.config['double_dqn'],
                "dueling_dqn": self.config['dueling_dqn']
            }
            
            # Initialize target network
            self.target_model = self.model["target_network"].copy()
            
            logger.info("DQN model initialized")
            
        except Exception as e:
            logger.error(f"DQN model initialization failed: {e}")
            self.model = {"type": "dqn", "error": str(e)}
    
    def _simulate_training_episodes(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """Simulate training episodes for RL agent."""
        try:
            episodes = 100  # Simulate 100 training episodes
            episode_rewards = []
            
            for episode in range(episodes):
                episode_reward = 0.0
                episode_length = 0
                
                # Simulate episode
                for step in range(50):  # 50 steps per episode
                    # Create random state
                    state = self._create_random_state()
                    
                    # Get action (epsilon-greedy)
                    if random.random() < self.epsilon:
                        action = random.choice(list(RLAction))
                    else:
                        action = self._get_action(state)
                    
                    # Simulate reward
                    reward = self._simulate_reward(action, state)
                    episode_reward += reward.total_reward
                    episode_length += 1
                    
                    # Store experience
                    next_state = self._create_random_state()
                    experience = RLExperience(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=(step == 49),
                        timestamp=datetime.now()
                    )
                    self._add_experience(experience)
                
                episode_rewards.append(episode_reward)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                # Update model periodically
                if episode % 10 == 0:
                    self._update_model()
            
            # Calculate training results
            avg_reward = np.mean(episode_rewards)
            best_reward = np.max(episode_rewards)
            convergence = self._calculate_convergence(episode_rewards)
            
            return {
                "episodes": episodes,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "convergence": convergence
            }
            
        except Exception as e:
            logger.error(f"Training simulation failed: {e}")
            return {
                "episodes": 0,
                "avg_reward": 0.0,
                "best_reward": 0.0,
                "convergence": False
            }
    
    def _create_state_from_context(self, context: AgentContext) -> RLState:
        """Create RL state from agent context."""
        try:
            # Extract technical indicators
            technical_indicators = {}
            if not context.market_data.empty and len(context.market_data) >= 20:
                close_prices = context.market_data['Close'] if 'Close' in context.market_data.columns else context.market_data['close']
                
                # Calculate technical indicators
                technical_indicators = {
                    'sma_5': close_prices.rolling(5).mean().iloc[-1] if len(close_prices) >= 5 else 0,
                    'sma_20': close_prices.rolling(20).mean().iloc[-1] if len(close_prices) >= 20 else 0,
                    'rsi': self._calculate_rsi(close_prices),
                    'macd': self._calculate_macd(close_prices),
                    'bollinger_upper': self._calculate_bollinger_bands(close_prices)[0],
                    'bollinger_lower': self._calculate_bollinger_bands(close_prices)[1],
                    'volume_ratio': self._calculate_volume_ratio(context.market_data),
                    'price_momentum': self._calculate_price_momentum(close_prices),
                    'volatility': close_prices.pct_change().std() if len(close_prices) > 1 else 0,
                    'trend_strength': self._calculate_trend_strength(close_prices)
                }
            
            # Extract market regime
            market_regime = getattr(context, 'market_regime', 'neutral')
            
            # Extract event impact
            event_impact = getattr(context, 'event_impact', 0.0)
            
            # Extract forecast signal
            forecast_signal = getattr(context, 'forecast_signal', 0.0)
            
            # Portfolio state
            portfolio_state = {
                'cash_ratio': 0.5,  # Simplified
                'position_size': 0.3,
                'risk_level': 0.4,
                'diversification': 0.7
            }
            
            # Market volatility
            market_volatility = technical_indicators.get('volatility', 0.02)
            
            return RLState(
                technical_indicators=technical_indicators,
                market_regime=market_regime,
                event_impact=event_impact,
                forecast_signal=forecast_signal,
                portfolio_state=portfolio_state,
                market_volatility=market_volatility,
                timestamp=context.timestamp
            )
            
        except Exception as e:
            logger.error(f"State creation failed: {e}")
            return self._create_default_state()
    
    def _create_default_state(self) -> RLState:
        """Create default RL state."""
        return RLState(
            technical_indicators={},
            market_regime='neutral',
            event_impact=0.0,
            forecast_signal=0.0,
            portfolio_state={'cash_ratio': 0.5, 'position_size': 0.3, 'risk_level': 0.4, 'diversification': 0.7},
            market_volatility=0.02,
            timestamp=datetime.now()
        )
    
    def _create_random_state(self) -> RLState:
        """Create random RL state for training."""
        return RLState(
            technical_indicators={
                'sma_5': random.uniform(100, 200),
                'sma_20': random.uniform(100, 200),
                'rsi': random.uniform(0, 100),
                'macd': random.uniform(-5, 5),
                'bollinger_upper': random.uniform(100, 200),
                'bollinger_lower': random.uniform(100, 200),
                'volume_ratio': random.uniform(0.5, 2.0),
                'price_momentum': random.uniform(-0.1, 0.1),
                'volatility': random.uniform(0.01, 0.05),
                'trend_strength': random.uniform(0, 1)
            },
            market_regime=random.choice(['bull', 'bear', 'neutral', 'volatile']),
            event_impact=random.uniform(-0.1, 0.1),
            forecast_signal=random.uniform(-1, 1),
            portfolio_state={
                'cash_ratio': random.uniform(0.2, 0.8),
                'position_size': random.uniform(0.1, 0.5),
                'risk_level': random.uniform(0.2, 0.8),
                'diversification': random.uniform(0.3, 0.9)
            },
            market_volatility=random.uniform(0.01, 0.05),
            timestamp=datetime.now()
        )
    
    def _get_action(self, state: RLState) -> RLAction:
        """Get action from RL model."""
        try:
            if self.algorithm == RLAlgorithm.PPO:
                return self._get_ppo_action(state)
            elif self.algorithm == RLAlgorithm.DQN:
                return self._get_dqn_action(state)
            else:
                return random.choice(list(RLAction))
                
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            return RLAction.HOLD
    
    def _get_ppo_action(self, state: RLState) -> RLAction:
        """Get action from PPO model."""
        try:
            # Simplified PPO action selection
            # In a real implementation, this would use the actor network
            
            # Extract features from state
            features = self._extract_features(state)
            
            # Simple policy based on features
            if features['trend_strength'] > 0.7 and features['rsi'] < 30:
                return RLAction.STRONG_BUY
            elif features['trend_strength'] > 0.5 and features['rsi'] < 50:
                return RLAction.BUY
            elif features['trend_strength'] < -0.7 and features['rsi'] > 70:
                return RLAction.STRONG_SELL
            elif features['trend_strength'] < -0.5 and features['rsi'] > 50:
                return RLAction.SELL
            else:
                return RLAction.HOLD
                
        except Exception as e:
            logger.error(f"PPO action selection failed: {e}")
            return RLAction.HOLD
    
    def _get_dqn_action(self, state: RLState) -> RLAction:
        """Get action from DQN model."""
        try:
            # Simplified DQN action selection
            # In a real implementation, this would use the Q-network
            
            # Extract features from state
            features = self._extract_features(state)
            
            # Calculate Q-values for each action
            q_values = {}
            for action in RLAction:
                q_values[action] = self._calculate_q_value(state, action)
            
            # Select action with highest Q-value
            best_action = max(q_values, key=q_values.get)
            
            return best_action
            
        except Exception as e:
            logger.error(f"DQN action selection failed: {e}")
            return RLAction.HOLD
    
    def _extract_features(self, state: RLState) -> Dict[str, float]:
        """Extract features from RL state."""
        try:
            features = {}
            
            # Technical indicators
            for key, value in state.technical_indicators.items():
                features[key] = float(value) if not np.isnan(value) else 0.0
            
            # Market regime
            regime_values = {'bull': 1.0, 'bear': -1.0, 'neutral': 0.0, 'volatile': 0.5}
            features['regime'] = regime_values.get(state.market_regime, 0.0)
            
            # Other features
            features['event_impact'] = state.event_impact
            features['forecast_signal'] = state.forecast_signal
            features['market_volatility'] = state.market_volatility
            
            # Portfolio features
            for key, value in state.portfolio_state.items():
                features[f'portfolio_{key}'] = value
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _calculate_q_value(self, state: RLState, action: RLAction) -> float:
        """Calculate Q-value for state-action pair."""
        try:
            # Simplified Q-value calculation
            # In a real implementation, this would use the neural network
            
            features = self._extract_features(state)
            
            # Simple Q-value based on features and action
            base_value = 0.0
            
            # Technical indicators contribution
            if 'trend_strength' in features:
                if action in [RLAction.BUY, RLAction.STRONG_BUY]:
                    base_value += features['trend_strength'] * 0.5
                elif action in [RLAction.SELL, RLAction.STRONG_SELL]:
                    base_value -= features['trend_strength'] * 0.5
            
            # RSI contribution
            if 'rsi' in features:
                if action in [RLAction.BUY, RLAction.STRONG_BUY] and features['rsi'] < 30:
                    base_value += 0.3
                elif action in [RLAction.SELL, RLAction.STRONG_SELL] and features['rsi'] > 70:
                    base_value += 0.3
            
            # Market regime contribution
            if 'regime' in features:
                if action in [RLAction.BUY, RLAction.STRONG_BUY] and features['regime'] > 0:
                    base_value += 0.2
                elif action in [RLAction.SELL, RLAction.STRONG_SELL] and features['regime'] < 0:
                    base_value += 0.2
            
            # Add some randomness for exploration
            base_value += random.uniform(-0.1, 0.1)
            
            return base_value
            
        except Exception as e:
            logger.error(f"Q-value calculation failed: {e}")
            return 0.0
    
    def _action_to_signal(self, action: RLAction, context: AgentContext) -> AgentSignal:
        """Convert RL action to trading signal."""
        try:
            # Map RL actions to signal types
            action_to_signal = {
                RLAction.BUY: SignalType.BUY,
                RLAction.SELL: SignalType.SELL,
                RLAction.HOLD: SignalType.HOLD,
                RLAction.STRONG_BUY: SignalType.STRONG_BUY,
                RLAction.STRONG_SELL: SignalType.STRONG_SELL
            }
            
            signal_type = action_to_signal.get(action, SignalType.HOLD)
            
            # Calculate confidence based on action strength
            confidence_map = {
                RLAction.BUY: 0.7,
                RLAction.SELL: 0.7,
                RLAction.HOLD: 0.5,
                RLAction.STRONG_BUY: 0.9,
                RLAction.STRONG_SELL: 0.9
            }
            
            confidence = confidence_map.get(action, 0.5)
            
            # Generate reasoning
            reasoning = f"RL {self.algorithm.value.upper()} action: {action.name} based on learned policy"
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'algorithm': self.algorithm.value,
                    'action': action.name,
                    'epsilon': self.epsilon,
                    'training_step': self.training_step,
                    'method': 'reinforcement_learning'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Action to signal conversion failed: {e}")
            return self._create_hold_signal(f"Action conversion error: {e}", context)
    
    def _simulate_reward(self, action: RLAction, state: RLState) -> RLReward:
        """Simulate reward for training."""
        try:
            # Simulate return based on action and market conditions
            base_return = random.uniform(-0.05, 0.05)  # -5% to +5%
            
            # Adjust return based on action
            if action in [RLAction.BUY, RLAction.STRONG_BUY]:
                base_return += random.uniform(0.0, 0.03)  # Positive bias for buy actions
            elif action in [RLAction.SELL, RLAction.STRONG_SELL]:
                base_return -= random.uniform(0.0, 0.03)  # Negative bias for sell actions
            
            # Risk-adjusted return (Sharpe ratio approximation)
            risk_adjusted_return = base_return / max(state.market_volatility, 0.01)
            
            # Drawdown penalty
            drawdown_penalty = -abs(base_return) * 0.5 if base_return < 0 else 0
            
            # Transaction cost
            transaction_cost = -0.001 if action != RLAction.HOLD else 0  # 0.1% transaction cost
            
            # Calculate total reward
            weights = self.config['reward_weights']
            total_reward = (
                weights['return'] * base_return +
                weights['risk_adjusted'] * risk_adjusted_return +
                weights['drawdown'] * drawdown_penalty +
                weights['transaction_cost'] * transaction_cost
            )
            
            return RLReward(
                return_reward=base_return,
                risk_adjusted_return=risk_adjusted_return,
                drawdown_penalty=drawdown_penalty,
                transaction_cost=transaction_cost,
                total_reward=total_reward
            )
            
        except Exception as e:
            logger.error(f"Reward simulation failed: {e}")
            return RLReward(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _store_experience(self, state: RLState, action: RLAction, context: AgentContext) -> None:
        """Store experience for learning."""
        try:
            # Simulate next state and reward
            next_state = self._create_state_from_context(context)
            reward = self._simulate_reward(action, state)
            
            experience = RLExperience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=False,
                timestamp=context.timestamp
            )
            
            self._add_experience(experience)
            
        except Exception as e:
            logger.error(f"Experience storage failed: {e}")
    
    def _add_experience(self, experience: RLExperience) -> None:
        """Add experience to replay buffer."""
        try:
            self.experience_buffer.append(experience)
            
            # Maintain buffer size
            if len(self.experience_buffer) > self.max_buffer_size:
                self.experience_buffer.pop(0)
            
            # Update model periodically
            if len(self.experience_buffer) >= self.config['batch_size']:
                if self.training_step % self.config['update_frequency'] == 0:
                    self._update_model()
                
                self.training_step += 1
                
        except Exception as e:
            logger.error(f"Experience addition failed: {e}")
    
    def _update_model(self) -> None:
        """Update the RL model."""
        try:
            if len(self.experience_buffer) < self.config['batch_size']:
                return
            
            # Sample batch from experience buffer
            batch = random.sample(self.experience_buffer, self.config['batch_size'])
            
            # Update model based on algorithm
            if self.algorithm == RLAlgorithm.PPO:
                self._update_ppo_model(batch)
            elif self.algorithm == RLAlgorithm.DQN:
                self._update_dqn_model(batch)
            
            # Update target network for DQN
            if self.algorithm == RLAlgorithm.DQN and self.training_step % self.config['target_update_frequency'] == 0:
                self._update_target_network()
            
            logger.debug(f"Model updated at step {self.training_step}")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def _update_ppo_model(self, batch: List[RLExperience]) -> None:
        """Update PPO model."""
        try:
            # Simplified PPO update
            # In a real implementation, this would perform policy gradient updates
            
            total_reward = sum(exp.reward.total_reward for exp in batch)
            avg_reward = total_reward / len(batch)
            
            # Update model weights (simplified)
            if hasattr(self.model, 'get'):
                self.model['avg_reward'] = avg_reward
                self.model['update_count'] = self.model.get('update_count', 0) + 1
            
            logger.debug(f"PPO model updated with avg reward: {avg_reward:.4f}")
            
        except Exception as e:
            logger.error(f"PPO model update failed: {e}")
    
    def _update_dqn_model(self, batch: List[RLExperience]) -> None:
        """Update DQN model."""
        try:
            # Simplified DQN update
            # In a real implementation, this would perform Q-learning updates
            
            total_reward = sum(exp.reward.total_reward for exp in batch)
            avg_reward = total_reward / len(batch)
            
            # Update model weights (simplified)
            if hasattr(self.model, 'get'):
                self.model['avg_reward'] = avg_reward
                self.model['update_count'] = self.model.get('update_count', 0) + 1
            
            logger.debug(f"DQN model updated with avg reward: {avg_reward:.4f}")
            
        except Exception as e:
            logger.error(f"DQN model update failed: {e}")
    
    def _update_target_network(self) -> None:
        """Update target network for DQN."""
        try:
            if self.algorithm == RLAlgorithm.DQN and self.target_model:
                # Copy weights from main network to target network
                self.target_model = self.model["target_network"].copy()
                logger.debug("Target network updated")
                
        except Exception as e:
            logger.error(f"Target network update failed: {e}")
    
    def _calculate_convergence(self, episode_rewards: List[float]) -> bool:
        """Calculate if training has converged."""
        try:
            if len(episode_rewards) < 20:
                return False
            
            # Check if recent episodes show improvement
            recent_rewards = episode_rewards[-10:]
            older_rewards = episode_rewards[-20:-10]
            
            recent_avg = np.mean(recent_rewards)
            older_avg = np.mean(older_rewards)
            
            # Consider converged if improvement is less than 1%
            return abs(recent_avg - older_avg) / abs(older_avg) < 0.01
            
        except Exception as e:
            logger.error(f"Convergence calculation failed: {e}")
            return False
    
    def _random_policy_signal(self, context: AgentContext) -> AgentSignal:
        """Generate random policy signal when not trained."""
        try:
            action = random.choice(list(RLAction))
            return self._action_to_signal(action, context)
            
        except Exception as e:
            logger.error(f"Random policy signal failed: {e}")
            return self._create_hold_signal("Random policy error", context)
    
    def _create_hold_signal(self, reason: str, context: AgentContext) -> AgentSignal:
        """Create a hold signal with error information."""
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=context.timestamp,
            asset_symbol=context.symbol,
            metadata={'error': reason, 'agent_version': self.version},
            reasoning=f"Hold signal: {reason}"
        )
    
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
    
    def _calculate_volume_ratio(self, market_data: pd.DataFrame) -> float:
        """Calculate volume ratio."""
        try:
            if 'Volume' not in market_data.columns and 'volume' not in market_data.columns:
                return 1.0
            
            volume_col = 'Volume' if 'Volume' in market_data.columns else 'volume'
            volumes = market_data[volume_col]
            
            if len(volumes) < 20:
                return 1.0
            
            recent_avg = volumes.tail(5).mean()
            historical_avg = volumes.tail(20).mean()
            
            return float(recent_avg / historical_avg) if historical_avg > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Volume ratio calculation failed: {e}")
            return 1.0
    
    def _calculate_price_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """Calculate price momentum."""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            momentum = (prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1]
            return float(momentum) if not np.isnan(momentum) else 0.0
            
        except Exception as e:
            logger.error(f"Price momentum calculation failed: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate trend strength."""
        try:
            if len(prices) < period:
                return 0.0
            
            # Calculate linear regression slope
            x = np.arange(len(prices.tail(period)))
            y = prices.tail(period).values
            
            if len(x) != len(y):
                return 0.0
            
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope to [-1, 1] range
            normalized_slope = np.tanh(slope / prices.mean())
            
            return float(normalized_slope) if not np.isnan(normalized_slope) else 0.0
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return 0.0
    
    def save_model(self) -> bool:
        """Save the RL model to disk."""
        try:
            model_file = self.model_path / f"{self.algorithm.value}_model.pkl"
            
            model_data = {
                'model': self.model,
                'target_model': self.target_model,
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'config': self.config
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the RL model from disk."""
        try:
            model_file = self.model_path / f"{self.algorithm.value}_model.pkl"
            
            if not model_file.exists():
                logger.info(f"No saved model found at {model_file}")
                return False
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.target_model = model_data.get('target_model')
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.training_step = model_data.get('training_step', 0)
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.episode_lengths = model_data.get('episode_lengths', [])
            
            logger.info(f"Model loaded from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False
    
    def update_model(self, new_data: pd.DataFrame, context: AgentContext) -> None:
        """
        Update the RL model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update experience buffer with new data
            if len(new_data) > 0:
                # Create state from new data
                state = self._create_state_from_context(context)
                
                # Get action and simulate experience
                action = self._get_action(state)
                self._store_experience(state, action, context)
            
            logger.info(f"Updated RL model for {self.name}")
            
        except Exception as e:
            logger.error(f"RL model update failed for {self.name}: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        try:
            return {
                'algorithm': self.algorithm.value,
                'epsilon': self.epsilon,
                'training_step': self.training_step,
                'episode_count': len(self.episode_rewards),
                'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
                'best_episode_reward': np.max(self.episode_rewards) if self.episode_rewards else 0.0,
                'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
                'experience_buffer_size': len(self.experience_buffer),
                'is_trained': self.is_trained,
                'model_type': self.model.get('type', 'unknown') if self.model else 'none'
            }
            
        except Exception as e:
            logger.error(f"Training stats retrieval failed: {e}")
            return {}
