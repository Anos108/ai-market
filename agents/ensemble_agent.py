"""
Ensemble Signal Blender Agent

This agent combines outputs from all other agents using weighted voting based on
confidence scores and regime-specific weights to produce high-quality ensemble signals.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import random
from collections import defaultdict

from ..core.agent_base import AgentBase
from ..core.signal_types import SignalType, AgentSignal
from ..core.context import TimeContext, EventContext, RegimeContext

logger = logging.getLogger(__name__)

class BlendMode(Enum):
    """Different blending modes for ensemble signals."""
    AVERAGE = "average"
    MAJORITY = "majority"
    MAX_CONFIDENCE = "max_confidence"
    WEIGHTED_AVERAGE = "weighted_average"

@dataclass
class AgentWeight:
    """Weight configuration for an agent in the ensemble."""
    agent_name: str
    base_weight: float
    regime_weights: Dict[str, float]  # regime -> weight multiplier
    confidence_multiplier: float
    performance_multiplier: float
    last_updated: datetime

@dataclass
class EnsembleSignal:
    """Ensemble signal combining multiple agent signals."""
    symbol: str
    signal_type: SignalType
    confidence: float
    blended_confidence: float
    contributing_agents: List[str]
    agent_signals: Dict[str, AgentSignal]
    blend_mode: BlendMode
    regime: str
    quality_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class SignalQualityMetrics:
    """Metrics for assessing signal quality."""
    consistency_score: float
    agreement_score: float
    confidence_variance: float
    regime_alignment: float
    historical_accuracy: float
    overall_quality: float

class EnsembleAgent(AgentBase):
    """
    Ensemble Signal Blender Agent that combines outputs from all agents.
    
    This agent uses weighted voting based on confidence scores and regime-specific
    weights to produce high-quality ensemble signals with reduced false positives.
    """
    
    def __init__(self, name: str = "EnsembleAgent"):
        super().__init__(name)
        
        # Agent weights configuration
        self.agent_weights: Dict[str, AgentWeight] = {}
        self.blend_mode = BlendMode.WEIGHTED_AVERAGE
        
        # Performance tracking
        self.signal_history: List[EnsembleSignal] = []
        self.quality_metrics: Dict[str, SignalQualityMetrics] = {}
        
        # Dynamic weight adjustment
        self.performance_window = 100  # Number of signals to consider for performance
        self.weight_adjustment_rate = 0.1  # How quickly to adjust weights
        
        # Regime detection
        self.current_regime = "sideways"
        self.regime_history: List[Tuple[datetime, str]] = []
        
        # Initialize agent weights
        self._initialize_agent_weights()
        
        logger.info(f"Initialized {self.name} with {len(self.agent_weights)} agents")
    
    def _initialize_agent_weights(self):
        """Initialize weights for all agents in the ensemble."""
        agent_names = [
            "MomentumAgent", "SentimentAgent", "CorrelationAgent", "RiskAgent",
            "VolatilityAgent", "VolumeAgent", "EventImpactAgent", "ForecastAgent",
            "StrategyAgent", "MetaAgent"
        ]
        
        # Base weights (can be adjusted based on historical performance)
        base_weights = {
            "MomentumAgent": 0.12,
            "SentimentAgent": 0.10,
            "CorrelationAgent": 0.11,
            "RiskAgent": 0.13,
            "VolatilityAgent": 0.10,
            "VolumeAgent": 0.09,
            "EventImpactAgent": 0.12,
            "ForecastAgent": 0.11,
            "StrategyAgent": 0.10,
            "MetaAgent": 0.12
        }
        
        # Regime-specific weight multipliers
        regime_multipliers = {
            "bull": {
                "MomentumAgent": 1.2,
                "SentimentAgent": 1.1,
                "CorrelationAgent": 0.9,
                "RiskAgent": 0.8,
                "VolatilityAgent": 0.9,
                "VolumeAgent": 1.1,
                "EventImpactAgent": 1.0,
                "ForecastAgent": 1.1,
                "StrategyAgent": 1.2,
                "MetaAgent": 1.0
            },
            "bear": {
                "MomentumAgent": 0.8,
                "SentimentAgent": 1.2,
                "CorrelationAgent": 1.1,
                "RiskAgent": 1.3,
                "VolatilityAgent": 1.2,
                "VolumeAgent": 0.9,
                "EventImpactAgent": 1.1,
                "ForecastAgent": 0.9,
                "StrategyAgent": 0.8,
                "MetaAgent": 1.1
            },
            "sideways": {
                "MomentumAgent": 0.9,
                "SentimentAgent": 1.0,
                "CorrelationAgent": 1.2,
                "RiskAgent": 1.0,
                "VolatilityAgent": 1.1,
                "VolumeAgent": 1.0,
                "EventImpactAgent": 1.0,
                "ForecastAgent": 1.0,
                "StrategyAgent": 1.0,
                "MetaAgent": 1.1
            },
            "volatile": {
                "MomentumAgent": 0.7,
                "SentimentAgent": 0.8,
                "CorrelationAgent": 0.9,
                "RiskAgent": 1.4,
                "VolatilityAgent": 1.3,
                "VolumeAgent": 1.2,
                "EventImpactAgent": 1.2,
                "ForecastAgent": 0.8,
                "StrategyAgent": 0.7,
                "MetaAgent": 1.0
            },
            "trending": {
                "MomentumAgent": 1.3,
                "SentimentAgent": 1.0,
                "CorrelationAgent": 1.0,
                "RiskAgent": 0.9,
                "VolatilityAgent": 0.8,
                "VolumeAgent": 1.1,
                "EventImpactAgent": 1.0,
                "ForecastAgent": 1.2,
                "StrategyAgent": 1.3,
                "MetaAgent": 1.0
            }
        }
        
        for agent_name in agent_names:
            self.agent_weights[agent_name] = AgentWeight(
                agent_name=agent_name,
                base_weight=base_weights.get(agent_name, 0.10),
                regime_weights=regime_multipliers.get("sideways", {}).get(agent_name, 1.0),
                confidence_multiplier=1.0,
                performance_multiplier=1.0,
                last_updated=datetime.now()
            )
    
    async def predict(self, symbol: str, context: Dict[str, Any]) -> AgentSignal:
        """
        Generate ensemble signal by blending outputs from all agents.
        
        Args:
            symbol: Trading symbol
            context: Market context including other agent signals
            
        Returns:
            Ensemble signal combining all agent outputs
        """
        try:
            # Extract agent signals from context
            agent_signals = context.get('agent_signals', {})
            if not agent_signals:
                logger.warning(f"No agent signals available for {symbol}")
                return self._create_default_signal(symbol)
            
            # Update current regime
            self._update_regime(context)
            
            # Blend signals using selected mode
            ensemble_signal = await self._blend_signals(symbol, agent_signals, context)
            
            # Assess signal quality
            quality_metrics = self._assess_signal_quality(ensemble_signal, agent_signals)
            ensemble_signal.quality_score = quality_metrics.overall_quality
            
            # Store signal for performance tracking
            self.signal_history.append(ensemble_signal)
            self.quality_metrics[symbol] = quality_metrics
            
            # Update agent weights based on performance
            await self._update_agent_weights(ensemble_signal, agent_signals)
            
            # Create final agent signal
            return AgentSignal(
                agent_name=self.name,
                symbol=symbol,
                signal_type=ensemble_signal.signal_type,
                confidence=ensemble_signal.blended_confidence,
                timestamp=datetime.now(),
                metadata={
                    'blend_mode': ensemble_signal.blend_mode.value,
                    'contributing_agents': ensemble_signal.contributing_agents,
                    'quality_score': ensemble_signal.quality_score,
                    'regime': ensemble_signal.regime,
                    'consistency_score': quality_metrics.consistency_score,
                    'agreement_score': quality_metrics.agreement_score
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction for {symbol}: {e}")
            return self._create_default_signal(symbol)
    
    async def _blend_signals(self, symbol: str, agent_signals: Dict[str, AgentSignal], 
                           context: Dict[str, Any]) -> EnsembleSignal:
        """Blend signals from all agents using the selected blend mode."""
        
        # Filter valid signals
        valid_signals = {
            name: signal for name, signal in agent_signals.items()
            if signal and signal.confidence > 0.1  # Minimum confidence threshold
        }
        
        if not valid_signals:
            return self._create_default_ensemble_signal(symbol)
        
        # Calculate weights for each agent
        agent_weights = {}
        for agent_name, signal in valid_signals.items():
            if agent_name in self.agent_weights:
                weight_config = self.agent_weights[agent_name]
                regime_multiplier = weight_config.regime_weights.get(self.current_regime, 1.0)
                
                # Calculate final weight
                final_weight = (
                    weight_config.base_weight *
                    regime_multiplier *
                    weight_config.confidence_multiplier *
                    weight_config.performance_multiplier *
                    signal.confidence
                )
                agent_weights[agent_name] = final_weight
        
        # Normalize weights
        total_weight = sum(agent_weights.values())
        if total_weight > 0:
            agent_weights = {name: weight / total_weight for name, weight in agent_weights.items()}
        
        # Apply blend mode
        if self.blend_mode == BlendMode.AVERAGE:
            blended_signal = self._blend_average(valid_signals, agent_weights)
        elif self.blend_mode == BlendMode.MAJORITY:
            blended_signal = self._blend_majority(valid_signals, agent_weights)
        elif self.blend_mode == BlendMode.MAX_CONFIDENCE:
            blended_signal = self._blend_max_confidence(valid_signals, agent_weights)
        else:  # WEIGHTED_AVERAGE
            blended_signal = self._blend_weighted_average(valid_signals, agent_weights)
        
        return EnsembleSignal(
            symbol=symbol,
            signal_type=blended_signal['signal_type'],
            confidence=blended_signal['confidence'],
            blended_confidence=blended_signal['blended_confidence'],
            contributing_agents=list(valid_signals.keys()),
            agent_signals=valid_signals,
            blend_mode=self.blend_mode,
            regime=self.current_regime,
            quality_score=0.0,  # Will be calculated later
            timestamp=datetime.now(),
            metadata={
                'agent_weights': agent_weights,
                'total_contributors': len(valid_signals),
                'blend_mode': self.blend_mode.value
            }
        )
    
    def _blend_average(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> Dict[str, Any]:
        """Blend signals using simple average."""
        signal_values = []
        confidences = []
        
        for signal in signals.values():
            signal_values.append(signal.signal_type.value)
            confidences.append(signal.confidence)
        
        avg_signal_value = np.mean(signal_values)
        avg_confidence = np.mean(confidences)
        
        # Map back to signal type
        signal_type = self._value_to_signal_type(avg_signal_value)
        
        return {
            'signal_type': signal_type,
            'confidence': avg_confidence,
            'blended_confidence': avg_confidence
        }
    
    def _blend_majority(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> Dict[str, Any]:
        """Blend signals using majority voting."""
        signal_counts = defaultdict(int)
        total_confidence = 0
        
        for signal in signals.values():
            signal_counts[signal.signal_type] += 1
            total_confidence += signal.confidence
        
        # Find majority signal
        majority_signal = max(signal_counts, key=signal_counts.get)
        avg_confidence = total_confidence / len(signals) if signals else 0.5
        
        return {
            'signal_type': majority_signal,
            'confidence': avg_confidence,
            'blended_confidence': avg_confidence
        }
    
    def _blend_max_confidence(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> Dict[str, Any]:
        """Blend signals using max confidence selection."""
        if not signals:
            return {'signal_type': SignalType.HOLD, 'confidence': 0.5, 'blended_confidence': 0.5}
        
        max_confidence_signal = max(signals.values(), key=lambda s: s.confidence)
        
        return {
            'signal_type': max_confidence_signal.signal_type,
            'confidence': max_confidence_signal.confidence,
            'blended_confidence': max_confidence_signal.confidence
        }
    
    def _blend_weighted_average(self, signals: Dict[str, AgentSignal], weights: Dict[str, float]) -> Dict[str, Any]:
        """Blend signals using weighted average."""
        weighted_signal_sum = 0
        weighted_confidence_sum = 0
        total_weight = 0
        
        for agent_name, signal in signals.items():
            weight = weights.get(agent_name, 0)
            weighted_signal_sum += signal.signal_type.value * weight
            weighted_confidence_sum += signal.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return {'signal_type': SignalType.HOLD, 'confidence': 0.5, 'blended_confidence': 0.5}
        
        avg_signal_value = weighted_signal_sum / total_weight
        avg_confidence = weighted_confidence_sum / total_weight
        
        signal_type = self._value_to_signal_type(avg_signal_value)
        
        return {
            'signal_type': signal_type,
            'confidence': avg_confidence,
            'blended_confidence': avg_confidence
        }
    
    def _value_to_signal_type(self, value: float) -> SignalType:
        """Convert numeric value to signal type."""
        if value <= 0.2:
            return SignalType.STRONG_SELL
        elif value <= 0.4:
            return SignalType.SELL
        elif value <= 0.6:
            return SignalType.HOLD
        elif value <= 0.8:
            return SignalType.BUY
        else:
            return SignalType.STRONG_BUY
    
    def _assess_signal_quality(self, ensemble_signal: EnsembleSignal, 
                             agent_signals: Dict[str, AgentSignal]) -> SignalQualityMetrics:
        """Assess the quality of the ensemble signal."""
        
        # Consistency score: how consistent are the contributing signals
        signal_types = [signal.signal_type for signal in agent_signals.values()]
        consistency_score = self._calculate_consistency(signal_types)
        
        # Agreement score: how much do agents agree on the signal
        agreement_score = self._calculate_agreement(agent_signals, ensemble_signal.signal_type)
        
        # Confidence variance: how much do confidence scores vary
        confidences = [signal.confidence for signal in agent_signals.values()]
        confidence_variance = 1.0 - np.var(confidences) if confidences else 0.5
        
        # Regime alignment: how well does the signal align with current regime
        regime_alignment = self._calculate_regime_alignment(ensemble_signal)
        
        # Historical accuracy: based on past performance
        historical_accuracy = self._calculate_historical_accuracy(ensemble_signal.symbol)
        
        # Overall quality score
        overall_quality = (
            consistency_score * 0.25 +
            agreement_score * 0.25 +
            confidence_variance * 0.20 +
            regime_alignment * 0.15 +
            historical_accuracy * 0.15
        )
        
        return SignalQualityMetrics(
            consistency_score=consistency_score,
            agreement_score=agreement_score,
            confidence_variance=confidence_variance,
            regime_alignment=regime_alignment,
            historical_accuracy=historical_accuracy,
            overall_quality=overall_quality
        )
    
    def _calculate_consistency(self, signal_types: List[SignalType]) -> float:
        """Calculate consistency score based on signal type distribution."""
        if not signal_types:
            return 0.5
        
        # Count signal types
        type_counts = defaultdict(int)
        for signal_type in signal_types:
            type_counts[signal_type] += 1
        
        # Calculate consistency (higher when more signals agree)
        max_count = max(type_counts.values())
        total_signals = len(signal_types)
        consistency = max_count / total_signals
        
        return consistency
    
    def _calculate_agreement(self, agent_signals: Dict[str, AgentSignal], 
                           ensemble_signal_type: SignalType) -> float:
        """Calculate agreement score between agents and ensemble signal."""
        if not agent_signals:
            return 0.5
        
        agreeing_agents = sum(
            1 for signal in agent_signals.values()
            if signal.signal_type == ensemble_signal_type
        )
        
        return agreeing_agents / len(agent_signals)
    
    def _calculate_regime_alignment(self, ensemble_signal: EnsembleSignal) -> float:
        """Calculate how well the signal aligns with current market regime."""
        regime = ensemble_signal.regime
        signal_type = ensemble_signal.signal_type
        
        # Define regime-signal alignment scores
        alignment_scores = {
            "bull": {
                SignalType.STRONG_BUY: 1.0,
                SignalType.BUY: 0.8,
                SignalType.HOLD: 0.6,
                SignalType.SELL: 0.3,
                SignalType.STRONG_SELL: 0.1
            },
            "bear": {
                SignalType.STRONG_SELL: 1.0,
                SignalType.SELL: 0.8,
                SignalType.HOLD: 0.6,
                SignalType.BUY: 0.3,
                SignalType.STRONG_BUY: 0.1
            },
            "sideways": {
                SignalType.HOLD: 1.0,
                SignalType.BUY: 0.7,
                SignalType.SELL: 0.7,
                SignalType.STRONG_BUY: 0.4,
                SignalType.STRONG_SELL: 0.4
            },
            "volatile": {
                SignalType.HOLD: 0.9,
                SignalType.SELL: 0.7,
                SignalType.BUY: 0.7,
                SignalType.STRONG_SELL: 0.5,
                SignalType.STRONG_BUY: 0.5
            },
            "trending": {
                SignalType.STRONG_BUY: 0.9,
                SignalType.BUY: 0.8,
                SignalType.STRONG_SELL: 0.9,
                SignalType.SELL: 0.8,
                SignalType.HOLD: 0.5
            }
        }
        
        return alignment_scores.get(regime, {}).get(signal_type, 0.5)
    
    def _calculate_historical_accuracy(self, symbol: str) -> float:
        """Calculate historical accuracy for the symbol."""
        # For now, return a base accuracy score
        # In a real implementation, this would look at historical performance
        return 0.75
    
    def _update_regime(self, context: Dict[str, Any]):
        """Update current market regime based on context."""
        # Extract regime from context or use default
        regime_context = context.get('regime_context')
        if regime_context and hasattr(regime_context, 'current_regime'):
            new_regime = regime_context.current_regime
        else:
            # Simple regime detection based on market conditions
            new_regime = self._detect_regime(context)
        
        if new_regime != self.current_regime:
            self.current_regime = new_regime
            self.regime_history.append((datetime.now(), new_regime))
            logger.info(f"Regime changed to: {new_regime}")
    
    def _detect_regime(self, context: Dict[str, Any]) -> str:
        """Simple regime detection based on market context."""
        # This is a simplified regime detection
        # In a real implementation, this would use more sophisticated methods
        
        volatility = context.get('volatility', 0.2)
        trend_strength = context.get('trend_strength', 0.0)
        sentiment = context.get('sentiment', 0.0)
        
        if volatility > 0.3:
            return "volatile"
        elif abs(trend_strength) > 0.7:
            return "trending"
        elif sentiment > 0.3:
            return "bull"
        elif sentiment < -0.3:
            return "bear"
        else:
            return "sideways"
    
    async def _update_agent_weights(self, ensemble_signal: EnsembleSignal, 
                                  agent_signals: Dict[str, AgentSignal]):
        """Update agent weights based on performance and signal quality."""
        # This is a simplified weight update mechanism
        # In a real implementation, this would use more sophisticated methods
        
        for agent_name, weight_config in self.agent_weights.items():
            if agent_name in agent_signals:
                signal = agent_signals[agent_name]
                
                # Adjust performance multiplier based on signal quality
                if ensemble_signal.quality_score > 0.8:
                    # Good ensemble signal, slightly increase weight
                    weight_config.performance_multiplier = min(
                        1.2, weight_config.performance_multiplier + 0.01
                    )
                elif ensemble_signal.quality_score < 0.5:
                    # Poor ensemble signal, slightly decrease weight
                    weight_config.performance_multiplier = max(
                        0.8, weight_config.performance_multiplier - 0.01
                    )
                
                weight_config.last_updated = datetime.now()
    
    def _create_default_signal(self, symbol: str) -> AgentSignal:
        """Create a default signal when no agent signals are available."""
        return AgentSignal(
            agent_name=self.name,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=datetime.now(),
            metadata={'error': 'No agent signals available'}
        )
    
    def _create_default_ensemble_signal(self, symbol: str) -> EnsembleSignal:
        """Create a default ensemble signal."""
        return EnsembleSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            blended_confidence=0.5,
            contributing_agents=[],
            agent_signals={},
            blend_mode=self.blend_mode,
            regime=self.current_regime,
            quality_score=0.5,
            timestamp=datetime.now(),
            metadata={'error': 'No valid signals to blend'}
        )
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble performance and configuration."""
        recent_signals = self.signal_history[-10:] if self.signal_history else []
        
        return {
            'agent_name': self.name,
            'blend_mode': self.blend_mode.value,
            'current_regime': self.current_regime,
            'total_signals_generated': len(self.signal_history),
            'recent_quality_scores': [s.quality_score for s in recent_signals],
            'avg_quality_score': np.mean([s.quality_score for s in recent_signals]) if recent_signals else 0.0,
            'agent_weights': {
                name: {
                    'base_weight': weight.base_weight,
                    'performance_multiplier': weight.performance_multiplier,
                    'regime_multiplier': weight.regime_weights.get(self.current_regime, 1.0)
                }
                for name, weight in self.agent_weights.items()
            },
            'regime_history': self.regime_history[-5:],  # Last 5 regime changes
            'last_updated': datetime.now().isoformat()
        }
