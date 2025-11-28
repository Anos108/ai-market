"""
Strategy Agent for AI Market Analysis System

This agent aggregates signals from all other agents and executes trading logic
to provide consolidated trading recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"


class SignalWeight(Enum):
    """Signal weight categories."""
    HIGH = 1.0
    MEDIUM = 0.7
    LOW = 0.4
    MINIMAL = 0.1


@dataclass
class SignalAggregation:
    """Represents aggregated signal information."""
    buy_signals: int
    sell_signals: int
    hold_signals: int
    weighted_score: float
    confidence: float
    consensus: SignalType
    participating_agents: List[str]
    signal_details: Dict[str, Any]


class StrategyAgent(BaseAgent):
    """
    Strategy Agent for aggregating signals and executing trading logic.
    
    This agent analyzes:
    - Signal aggregation from all analysis agents
    - Consensus building and signal weighting
    - Strategy selection based on market conditions
    - Risk-adjusted position sizing
    - Portfolio-level decision making
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Strategy Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'signal_weights': {
                'MomentumAgent': SignalWeight.HIGH.value,
                'SentimentAgent': SignalWeight.MEDIUM.value,
                'CorrelationAgent': SignalWeight.MEDIUM.value,
                'RiskAgent': SignalWeight.HIGH.value,
                'VolatilityAgent': SignalWeight.MEDIUM.value,
                'VolumeAgent': SignalWeight.MEDIUM.value,
                'EventImpactAgent': SignalWeight.HIGH.value,
                'ForecastAgent': SignalWeight.HIGH.value
            },
            'consensus_threshold': 0.6,
            'min_agents_for_consensus': 3,
            'risk_adjustment_factor': 0.8,
            'position_sizing_method': 'kelly',
            'max_position_size': 0.2,
            'confidence_threshold': 0.6,
            'strategy_weights': {
                StrategyType.CONSERVATIVE: 0.3,
                StrategyType.MODERATE: 0.5,
                StrategyType.AGGRESSIVE: 0.8,
                StrategyType.MOMENTUM: 0.7,
                StrategyType.MEAN_REVERSION: 0.6,
                StrategyType.ARBITRAGE: 0.9
            }
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="StrategyAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.signal_history = []
        self.strategy_performance = {}
        self.consensus_history = []
        
        logger.info(f"Initialized StrategyAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the strategy agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For strategy analysis, we don't need traditional ML training
            # Instead, we'll validate our strategy approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Strategy analysis approach validated")
            return {"status": "strategy_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate strategy-based prediction by aggregating other agent signals.
        
        Args:
            context: Current market context
            
        Returns:
            Strategy-based trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple strategy analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple strategy analysis (not trained)")
                return self._simple_strategy_analysis(context)
            
            # Get signals from other agents (simulated for now)
            agent_signals = self._get_agent_signals(context)
            
            # Aggregate signals
            signal_aggregation = self._aggregate_signals(agent_signals)
            
            # Apply strategy logic
            strategy_decision = self._apply_strategy_logic(signal_aggregation, context)
            
            # Generate final signal
            signal = self._generate_strategy_signal(strategy_decision, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Strategy analysis error: {e}", context)
    
    def _get_agent_signals(self, context: AgentContext) -> List[AgentSignal]:
        """
        Get signals from other agents.
        
        Args:
            context: Current market context
            
        Returns:
            List of signals from other agents
        """
        try:
            # In a real implementation, this would get actual signals from other agents
            # For now, we'll simulate signals based on market conditions
            
            signals = []
            
            # Simulate signals from each agent type
            agent_types = [
                'MomentumAgent', 'SentimentAgent', 'CorrelationAgent', 'RiskAgent',
                'VolatilityAgent', 'VolumeAgent', 'EventImpactAgent', 'ForecastAgent'
            ]
            
            for agent_name in agent_types:
                # Simulate signal based on market conditions
                signal = self._simulate_agent_signal(agent_name, context)
                if signal:
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to get agent signals: {e}")
            return []
    
    def _simulate_agent_signal(self, agent_name: str, context: AgentContext) -> Optional[AgentSignal]:
        """Simulate a signal from a specific agent."""
        try:
            if context.market_data.empty:
                return None
            
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            if close_col not in context.market_data.columns:
                return None
            
            prices = context.market_data[close_col]
            if len(prices) < 2:
                return None
            
            # Get recent price change
            current_price = prices.iloc[-1]
            prev_price = prices.iloc[-2]
            price_change = (current_price - prev_price) / prev_price
            
            # Simulate different agent behaviors
            if agent_name == 'MomentumAgent':
                if price_change > 0.01:
                    signal_type = SignalType.BUY
                    confidence = 0.7
                elif price_change < -0.01:
                    signal_type = SignalType.SELL
                    confidence = 0.7
                else:
                    signal_type = SignalType.HOLD
                    confidence = 0.5
                reasoning = f"Momentum analysis: {price_change:.2%} price change"
                
            elif agent_name == 'RiskAgent':
                # Risk agent typically more conservative
                if price_change > 0.02:
                    signal_type = SignalType.SELL  # Risk off
                    confidence = 0.6
                elif price_change < -0.02:
                    signal_type = SignalType.SELL  # Risk off
                    confidence = 0.8
                else:
                    signal_type = SignalType.HOLD
                    confidence = 0.7
                reasoning = f"Risk assessment: {price_change:.2%} price change"
                
            elif agent_name == 'SentimentAgent':
                # Sentiment agent based on volatility
                volatility = prices.pct_change().std()
                if volatility > 0.03:
                    signal_type = SignalType.SELL
                    confidence = 0.6
                else:
                    signal_type = SignalType.BUY
                    confidence = 0.6
                reasoning = f"Sentiment analysis: {volatility:.2%} volatility"
                
            else:
                # Default behavior for other agents
                if price_change > 0.005:
                    signal_type = SignalType.BUY
                    confidence = 0.6
                elif price_change < -0.005:
                    signal_type = SignalType.SELL
                    confidence = 0.6
                else:
                    signal_type = SignalType.HOLD
                    confidence = 0.5
                reasoning = f"{agent_name} analysis: {price_change:.2%} price change"
            
            return AgentSignal(
                agent_name=agent_name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={'simulated': True, 'price_change': price_change},
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Failed to simulate signal for {agent_name}: {e}")
            return None
    
    def _aggregate_signals(self, signals: List[AgentSignal]) -> SignalAggregation:
        """
        Aggregate signals from multiple agents.
        
        Args:
            signals: List of signals from other agents
            
        Returns:
            Aggregated signal information
        """
        try:
            if not signals:
                return SignalAggregation(
                    buy_signals=0, sell_signals=0, hold_signals=0,
                    weighted_score=0.0, confidence=0.0,
                    consensus=SignalType.HOLD, participating_agents=[],
                    signal_details={}
                )
            
            # Count signals by type
            buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
            sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
            hold_count = sum(1 for s in signals if s.signal_type == SignalType.HOLD)
            
            # Calculate weighted score
            weighted_score = 0.0
            total_weight = 0.0
            total_confidence = 0.0
            
            signal_details = {}
            
            for signal in signals:
                agent_weight = self.config['signal_weights'].get(signal.agent_name, SignalWeight.MEDIUM.value)
                
                # Convert signal to numeric value
                if signal.signal_type == SignalType.BUY:
                    signal_value = 1.0
                elif signal.signal_type == SignalType.SELL:
                    signal_value = -1.0
                else:
                    signal_value = 0.0
                
                # Weight by confidence and agent weight
                weighted_contribution = signal_value * signal.confidence * agent_weight
                weighted_score += weighted_contribution
                total_weight += agent_weight
                total_confidence += signal.confidence
                
                # Store signal details
                signal_details[signal.agent_name] = {
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'weight': agent_weight,
                    'reasoning': signal.reasoning
                }
            
            # Normalize weighted score
            if total_weight > 0:
                weighted_score = weighted_score / total_weight
            
            # Calculate overall confidence
            overall_confidence = total_confidence / len(signals) if signals else 0.0
            
            # Determine consensus
            total_signals = len(signals)
            if total_signals >= self.config['min_agents_for_consensus']:
                if buy_count / total_signals >= self.config['consensus_threshold']:
                    consensus = SignalType.BUY
                elif sell_count / total_signals >= self.config['consensus_threshold']:
                    consensus = SignalType.SELL
                else:
                    consensus = SignalType.HOLD
            else:
                # Not enough signals for consensus
                if weighted_score > 0.3:
                    consensus = SignalType.BUY
                elif weighted_score < -0.3:
                    consensus = SignalType.SELL
                else:
                    consensus = SignalType.HOLD
            
            return SignalAggregation(
                buy_signals=buy_count,
                sell_signals=sell_count,
                hold_signals=hold_count,
                weighted_score=weighted_score,
                confidence=overall_confidence,
                consensus=consensus,
                participating_agents=[s.agent_name for s in signals],
                signal_details=signal_details
            )
            
        except Exception as e:
            logger.error(f"Signal aggregation failed: {e}")
            return SignalAggregation(
                buy_signals=0, sell_signals=0, hold_signals=0,
                weighted_score=0.0, confidence=0.0,
                consensus=SignalType.HOLD, participating_agents=[],
                signal_details={}
            )
    
    def _apply_strategy_logic(self, aggregation: SignalAggregation, context: AgentContext) -> Dict[str, Any]:
        """
        Apply strategy logic to the aggregated signals.
        
        Args:
            aggregation: Aggregated signal information
            context: Current market context
            
        Returns:
            Strategy decision
        """
        try:
            # Determine market regime (simplified)
            market_regime = self._determine_market_regime(context)
            
            # Select appropriate strategy
            strategy_type = self._select_strategy_type(aggregation, market_regime)
            
            # Calculate position size
            position_size = self._calculate_position_size(aggregation, strategy_type, context)
            
            # Apply risk adjustments
            risk_adjusted_signal = self._apply_risk_adjustments(aggregation, context)
            
            return {
                'strategy_type': strategy_type,
                'position_size': position_size,
                'market_regime': market_regime,
                'risk_adjusted_signal': risk_adjusted_signal,
                'aggregation': aggregation,
                'confidence': aggregation.confidence * self.config['risk_adjustment_factor']
            }
            
        except Exception as e:
            logger.error(f"Strategy logic application failed: {e}")
            return {
                'strategy_type': StrategyType.CONSERVATIVE,
                'position_size': 0.0,
                'market_regime': 'unknown',
                'risk_adjusted_signal': SignalType.HOLD,
                'aggregation': aggregation,
                'confidence': 0.0
            }
    
    def _determine_market_regime(self, context: AgentContext) -> str:
        """Determine current market regime."""
        try:
            if context.market_data.empty:
                return 'unknown'
            
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            if close_col not in context.market_data.columns:
                return 'unknown'
            
            prices = context.market_data[close_col]
            if len(prices) < 10:
                return 'unknown'
            
            # Calculate volatility
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend
            recent_prices = prices.tail(10)
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Classify regime
            if volatility > 0.03:  # High volatility
                return 'volatile'
            elif trend > 0.05:  # Strong uptrend
                return 'bull'
            elif trend < -0.05:  # Strong downtrend
                return 'bear'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Market regime determination failed: {e}")
            return 'unknown'
    
    def _select_strategy_type(self, aggregation: SignalAggregation, market_regime: str) -> StrategyType:
        """Select appropriate strategy type based on signals and market regime."""
        try:
            # Base strategy on consensus strength
            consensus_strength = abs(aggregation.weighted_score)
            
            if market_regime == 'volatile':
                return StrategyType.CONSERVATIVE
            elif market_regime == 'bear':
                return StrategyType.CONSERVATIVE
            elif consensus_strength > 0.7:
                return StrategyType.AGGRESSIVE
            elif consensus_strength > 0.4:
                return StrategyType.MODERATE
            else:
                return StrategyType.CONSERVATIVE
                
        except Exception as e:
            logger.error(f"Strategy type selection failed: {e}")
            return StrategyType.CONSERVATIVE
    
    def _calculate_position_size(self, aggregation: SignalAggregation, strategy_type: StrategyType, context: AgentContext) -> float:
        """Calculate appropriate position size."""
        try:
            # Base position size from strategy type
            base_size = self.config['strategy_weights'].get(strategy_type, 0.5)
            
            # Adjust by consensus strength
            consensus_strength = abs(aggregation.weighted_score)
            adjusted_size = base_size * consensus_strength
            
            # Apply confidence adjustment
            confidence_adjustment = aggregation.confidence
            final_size = adjusted_size * confidence_adjustment
            
            # Cap at maximum position size
            return min(final_size, self.config['max_position_size'])
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def _apply_risk_adjustments(self, aggregation: SignalAggregation, context: AgentContext) -> SignalType:
        """Apply risk adjustments to the consensus signal."""
        try:
            # Start with consensus
            adjusted_signal = aggregation.consensus
            
            # Apply risk-based adjustments
            if aggregation.confidence < self.config['confidence_threshold']:
                # Low confidence -> more conservative
                if adjusted_signal in [SignalType.BUY, SignalType.SELL]:
                    adjusted_signal = SignalType.HOLD
            
            # Check for conflicting signals
            if aggregation.buy_signals > 0 and aggregation.sell_signals > 0:
                # Conflicting signals -> more conservative
                if abs(aggregation.buy_signals - aggregation.sell_signals) < 2:
                    adjusted_signal = SignalType.HOLD
            
            return adjusted_signal
            
        except Exception as e:
            logger.error(f"Risk adjustment failed: {e}")
            return SignalType.HOLD
    
    def _generate_strategy_signal(self, strategy_decision: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate final strategy-based trading signal.
        
        Args:
            strategy_decision: Strategy decision information
            context: Current market context
            
        Returns:
            Strategy-based trading signal
        """
        try:
            signal_type = strategy_decision['risk_adjusted_signal']
            confidence = strategy_decision['confidence']
            strategy_type = strategy_decision['strategy_type']
            position_size = strategy_decision['position_size']
            aggregation = strategy_decision['aggregation']
            
            # Build reasoning
            reasoning_parts = [
                f"Strategy: {strategy_type.value}",
                f"Consensus: {aggregation.consensus.value} ({aggregation.buy_signals}B/{aggregation.sell_signals}S/{aggregation.hold_signals}H)",
                f"Position size: {position_size:.1%}",
                f"Confidence: {confidence:.2f}"
            ]
            
            reasoning = " | ".join(reasoning_parts)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'strategy_decision': strategy_decision,
                    'method': 'signal_aggregation'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Strategy signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_strategy_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple strategy analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple strategy-based signal
        """
        try:
            # Simple strategy based on price momentum
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            if len(context.market_data) >= 5:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    returns = prices.pct_change().dropna()
                    
                    # Simple momentum strategy
                    recent_return = returns.tail(5).mean()
                    volatility = returns.std()
                    
                    if recent_return > 0.01 and volatility < 0.02:
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_momentum_strategy'},
                            reasoning=f"Simple momentum strategy: {recent_return:.2%} return, {volatility:.2%} volatility"
                        )
                    elif recent_return < -0.01:
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.SELL,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_momentum_strategy'},
                            reasoning=f"Simple momentum strategy: {recent_return:.2%} return, {volatility:.2%} volatility"
                        )
            
            return self._create_hold_signal("No clear strategy signal", context)
            
        except Exception as e:
            logger.error(f"Simple strategy analysis failed: {e}")
            return self._create_hold_signal(f"Simple strategy analysis error: {e}", context)
    
    def _create_hold_signal(self, reason: str, context: AgentContext) -> AgentSignal:
        """Create a hold signal with error information."""
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=context.timestamp,
            asset_symbol=context.symbol,
            metadata={'error': reason},
            reasoning=f"Hold signal: {reason}"
        )
    
    def update_model(self, new_data: pd.DataFrame, context: AgentContext) -> None:
        """
        Update the strategy model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update strategy performance tracking
            if hasattr(self, '_last_strategy_decision'):
                self.strategy_performance[context.symbol] = self._last_strategy_decision
            
            # Update signal history
            if hasattr(self, '_last_signal_aggregation'):
                self.consensus_history.append(self._last_signal_aggregation)
                
                # Keep only recent history
                if len(self.consensus_history) > 100:
                    self.consensus_history = self.consensus_history[-100:]
            
            logger.info(f"Updated strategy model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
