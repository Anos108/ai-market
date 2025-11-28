"""
Meta-Evaluation Agent for AI Market Analysis System

This agent tracks which agents perform best under which regimes and rotates dynamically
to optimize overall system performance and resource allocation.
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics for agent evaluation."""
    ACCURACY = "accuracy"
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_RETURN = "total_return"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    CONFIDENCE = "confidence"
    RESPONSE_TIME = "response_time"


class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"


@dataclass
class AgentPerformance:
    """Agent performance data structure."""
    agent_name: str
    regime: str
    accuracy: float
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    confidence: float
    response_time: float
    prediction_count: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRanking:
    """Agent ranking data structure."""
    agent_name: str
    overall_score: float
    regime_scores: Dict[str, float]
    performance_trend: str  # "improving", "stable", "declining"
    last_updated: datetime
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RotationDecision:
    """Agent rotation decision data structure."""
    action: str  # "activate", "deactivate", "maintain"
    agent_name: str
    reason: str
    confidence: float
    expected_impact: float
    timestamp: datetime


class MetaEvaluationAgent(BaseAgent):
    """
    Meta-Evaluation Agent for dynamic agent optimization.
    
    This agent tracks agent performance across different market regimes,
    maintains lifecycle history, and makes dynamic rotation decisions
    to optimize overall system performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Meta-Evaluation Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'evaluation_window': 100,  # Number of predictions to evaluate
            'regime_detection_window': 50,  # Window for regime detection
            'performance_thresholds': {
                'accuracy': 0.6,  # Minimum accuracy threshold
                'sharpe_ratio': 0.5,  # Minimum Sharpe ratio
                'max_drawdown': 0.2,  # Maximum drawdown threshold
                'win_rate': 0.55,  # Minimum win rate
                'response_time': 5.0  # Maximum response time in seconds
            },
            'rotation_thresholds': {
                'performance_decline': 0.1,  # 10% performance decline triggers review
                'consistency_threshold': 0.8,  # Consistency threshold for activation
                'improvement_threshold': 0.05  # 5% improvement threshold
            },
            'regime_weights': {
                'bull': 1.0,
                'bear': 1.0,
                'sideways': 1.0,
                'volatile': 1.2,  # Higher weight for volatile periods
                'trending': 1.1
            },
            'update_frequency': 10,  # Update every N predictions
            'history_retention': 1000,  # Keep last N performance records
            'enable_auto_rotation': True,
            'rotation_cooldown': 300  # 5 minutes cooldown between rotations
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="MetaEvaluationAgent",
            version="1.0.0",
            config=default_config
        )
        
        # Performance tracking
        self.agent_performances: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config['history_retention'])
        )
        self.agent_rankings: Dict[str, AgentRanking] = {}
        self.rotation_decisions: deque = deque(maxlen=100)
        self.regime_history: deque = deque(maxlen=self.config['regime_detection_window'])
        
        # Current state
        self.current_regime = RegimeType.SIDEWAYS
        self.last_rotation_time = datetime.min
        self.evaluation_count = 0
        
        # Performance metrics
        self.overall_system_performance = 0.0
        self.agent_utilization = {}
        self.regime_performance = defaultdict(list)
        
        logger.info(f"Initialized MetaEvaluationAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the Meta-Evaluation Agent.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting Meta-Evaluation training for {self.name}")
            
            # Initialize agent rankings
            self._initialize_agent_rankings()
            
            # Analyze historical performance if available
            if not training_data.empty:
                self._analyze_historical_performance(training_data, context)
            
            self.is_trained = True
            
            logger.info(f"{self.name}: Meta-Evaluation training completed")
            return {
                "status": "training_completed",
                "agents_evaluated": len(self.agent_rankings),
                "regimes_analyzed": len(set(self.regime_history)),
                "performance_baseline": self.overall_system_performance
            }
            
        except Exception as e:
            logger.error(f"Meta-Evaluation training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate meta-evaluation signal and rotation decisions.
        
        Args:
            context: Current market context
            
        Returns:
            Meta-evaluation signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # Update regime detection
            self._update_regime_detection(context)
            
            # Evaluate agent performance
            if self.evaluation_count % self.config['update_frequency'] == 0:
                self._evaluate_agent_performance(context)
                self._update_agent_rankings()
                
                # Make rotation decisions
                if self.config['enable_auto_rotation']:
                    self._make_rotation_decisions()
            
            self.evaluation_count += 1
            
            # Generate meta-evaluation signal
            signal = self._generate_meta_evaluation_signal(context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Meta-Evaluation prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Meta-Evaluation error: {e}", context)
    
    def _initialize_agent_rankings(self) -> None:
        """Initialize agent rankings."""
        try:
            # Default agent list (should be populated from system)
            default_agents = [
                "MomentumAgent", "SentimentAgent", "CorrelationAgent",
                "RiskAgent", "VolatilityAgent", "VolumeAgent",
                "EventImpactAgent", "ForecastAgent", "StrategyAgent", "MetaAgent"
            ]
            
            for agent_name in default_agents:
                self.agent_rankings[agent_name] = AgentRanking(
                    agent_name=agent_name,
                    overall_score=0.5,  # Neutral starting score
                    regime_scores={regime.value: 0.5 for regime in RegimeType},
                    performance_trend="stable",
                    last_updated=datetime.now(),
                    recommendations=[]
                )
            
            logger.info(f"Initialized rankings for {len(self.agent_rankings)} agents")
            
        except Exception as e:
            logger.error(f"Agent ranking initialization failed: {e}")
    
    def _update_regime_detection(self, context: AgentContext) -> None:
        """Update market regime detection."""
        try:
            # Simple regime detection based on market data
            if not context.market_data.empty and len(context.market_data) >= 20:
                prices = context.market_data['Close'] if 'Close' in context.market_data.columns else context.market_data['close']
                
                # Calculate regime indicators
                returns = prices.pct_change().dropna()
                volatility = returns.std()
                trend = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] if len(prices) >= 20 else 0
                
                # Determine regime
                if volatility > 0.03:  # High volatility
                    regime = RegimeType.VOLATILE
                elif abs(trend) > 0.05:  # Strong trend
                    regime = RegimeType.TRENDING
                elif trend > 0.02:  # Bullish
                    regime = RegimeType.BULL
                elif trend < -0.02:  # Bearish
                    regime = RegimeType.BEAR
                else:  # Sideways
                    regime = RegimeType.SIDEWAYS
                
                self.current_regime = regime
                self.regime_history.append(regime.value)
            
            logger.debug(f"Current regime: {self.current_regime.value}")
            
        except Exception as e:
            logger.error(f"Regime detection update failed: {e}")
    
    def _evaluate_agent_performance(self, context: AgentContext) -> None:
        """Evaluate agent performance."""
        try:
            # Simulate agent performance evaluation
            # In a real implementation, this would collect actual performance data
            
            for agent_name in self.agent_rankings.keys():
                # Generate realistic performance metrics
                performance = AgentPerformance(
                    agent_name=agent_name,
                    regime=self.current_regime.value,
                    accuracy=random.uniform(0.55, 0.85),
                    sharpe_ratio=random.uniform(0.3, 2.0),
                    total_return=random.uniform(-0.05, 0.15),
                    max_drawdown=random.uniform(0.05, 0.25),
                    win_rate=random.uniform(0.50, 0.80),
                    confidence=random.uniform(0.60, 0.90),
                    response_time=random.uniform(0.5, 3.0),
                    prediction_count=random.randint(10, 100),
                    timestamp=context.timestamp,
                    metadata={'evaluation_method': 'simulated'}
                )
                
                # Add to performance history
                self.agent_performances[agent_name].append(performance)
                
                # Update regime performance
                self.regime_performance[self.current_regime.value].append(performance)
            
            logger.debug(f"Evaluated performance for {len(self.agent_rankings)} agents")
            
        except Exception as e:
            logger.error(f"Agent performance evaluation failed: {e}")
    
    def _update_agent_rankings(self) -> None:
        """Update agent rankings based on recent performance."""
        try:
            for agent_name, ranking in self.agent_rankings.items():
                if agent_name in self.agent_performances:
                    recent_performances = list(self.agent_performances[agent_name])[-10:]  # Last 10 evaluations
                    
                    if recent_performances:
                        # Calculate overall score
                        overall_score = self._calculate_overall_score(recent_performances)
                        
                        # Calculate regime-specific scores
                        regime_scores = self._calculate_regime_scores(agent_name)
                        
                        # Determine performance trend
                        performance_trend = self._calculate_performance_trend(agent_name)
                        
                        # Generate recommendations
                        recommendations = self._generate_recommendations(agent_name, overall_score, performance_trend)
                        
                        # Update ranking
                        ranking.overall_score = overall_score
                        ranking.regime_scores = regime_scores
                        ranking.performance_trend = performance_trend
                        ranking.last_updated = datetime.now()
                        ranking.recommendations = recommendations
            
            logger.debug("Updated agent rankings")
            
        except Exception as e:
            logger.error(f"Agent ranking update failed: {e}")
    
    def _calculate_overall_score(self, performances: List[AgentPerformance]) -> float:
        """Calculate overall performance score."""
        try:
            if not performances:
                return 0.5
            
            # Weighted average of key metrics
            weights = {
                'accuracy': 0.3,
                'sharpe_ratio': 0.25,
                'win_rate': 0.2,
                'confidence': 0.15,
                'response_time': 0.1  # Lower is better, so invert
            }
            
            total_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric == 'response_time':
                    # Invert response time (lower is better)
                    values = [1.0 / max(p.response_time, 0.1) for p in performances]
                else:
                    values = [getattr(p, metric) for p in performances]
                
                avg_value = np.mean(values)
                total_score += avg_value * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}")
            return 0.5
    
    def _calculate_regime_scores(self, agent_name: str) -> Dict[str, float]:
        """Calculate regime-specific performance scores."""
        try:
            regime_scores = {}
            
            for regime in RegimeType:
                regime_performances = [
                    p for p in self.agent_performances[agent_name]
                    if p.regime == regime.value
                ]
                
                if regime_performances:
                    regime_scores[regime.value] = self._calculate_overall_score(regime_performances)
                else:
                    regime_scores[regime.value] = 0.5  # Default neutral score
            
            return regime_scores
            
        except Exception as e:
            logger.error(f"Regime score calculation failed: {e}")
            return {regime.value: 0.5 for regime in RegimeType}
    
    def _calculate_performance_trend(self, agent_name: str) -> str:
        """Calculate performance trend."""
        try:
            performances = list(self.agent_performances[agent_name])
            
            if len(performances) < 5:
                return "stable"
            
            # Compare recent vs older performance
            recent_scores = [self._calculate_overall_score([p]) for p in performances[-3:]]
            older_scores = [self._calculate_overall_score([p]) for p in performances[-6:-3]]
            
            recent_avg = np.mean(recent_scores)
            older_avg = np.mean(older_scores)
            
            change = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            
            if change > 0.05:
                return "improving"
            elif change < -0.05:
                return "declining"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Performance trend calculation failed: {e}")
            return "stable"
    
    def _generate_recommendations(self, agent_name: str, overall_score: float, trend: str) -> List[str]:
        """Generate recommendations for agent optimization."""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if overall_score < 0.6:
                recommendations.append("Consider retraining or parameter adjustment")
            
            if trend == "declining":
                recommendations.append("Performance declining - investigate root cause")
            
            # Regime-specific recommendations
            regime_scores = self.agent_rankings[agent_name].regime_scores
            worst_regime = min(regime_scores, key=regime_scores.get)
            if regime_scores[worst_regime] < 0.5:
                recommendations.append(f"Poor performance in {worst_regime} regime - regime-specific optimization needed")
            
            # Resource allocation recommendations
            if overall_score > 0.8:
                recommendations.append("High performance - consider increasing allocation")
            elif overall_score < 0.4:
                recommendations.append("Low performance - consider reducing allocation or deactivation")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _make_rotation_decisions(self) -> None:
        """Make agent rotation decisions."""
        try:
            current_time = datetime.now()
            
            # Check cooldown period
            if (current_time - self.last_rotation_time).total_seconds() < self.config['rotation_cooldown']:
                return
            
            # Analyze current regime performance
            current_regime = self.current_regime.value
            regime_rankings = sorted(
                self.agent_rankings.items(),
                key=lambda x: x[1].regime_scores.get(current_regime, 0.5),
                reverse=True
            )
            
            # Make rotation decisions
            for agent_name, ranking in self.agent_rankings.items():
                regime_score = ranking.regime_scores.get(current_regime, 0.5)
                overall_score = ranking.overall_score
                
                # Decision logic
                if overall_score < self.config['performance_thresholds']['accuracy']:
                    decision = RotationDecision(
                        action="deactivate",
                        agent_name=agent_name,
                        reason=f"Overall performance below threshold ({overall_score:.3f})",
                        confidence=0.8,
                        expected_impact=0.1,
                        timestamp=current_time
                    )
                elif regime_score > 0.8 and ranking.performance_trend == "improving":
                    decision = RotationDecision(
                        action="activate",
                        agent_name=agent_name,
                        reason=f"Excellent performance in current regime ({regime_score:.3f})",
                        confidence=0.9,
                        expected_impact=0.15,
                        timestamp=current_time
                    )
                else:
                    decision = RotationDecision(
                        action="maintain",
                        agent_name=agent_name,
                        reason="Performance within acceptable range",
                        confidence=0.7,
                        expected_impact=0.0,
                        timestamp=current_time
                    )
                
                self.rotation_decisions.append(decision)
            
            self.last_rotation_time = current_time
            logger.info(f"Made {len(self.rotation_decisions)} rotation decisions")
            
        except Exception as e:
            logger.error(f"Rotation decision making failed: {e}")
    
    def _generate_meta_evaluation_signal(self, context: AgentContext) -> AgentSignal:
        """Generate meta-evaluation signal."""
        try:
            # Calculate system health score
            system_health = self._calculate_system_health()
            
            # Determine signal type based on system health
            if system_health > 0.8:
                signal_type = SignalType.STRONG_BUY
                confidence = 0.9
                reasoning = "System performing excellently with optimal agent allocation"
            elif system_health > 0.6:
                signal_type = SignalType.BUY
                confidence = 0.7
                reasoning = "System performing well with good agent coordination"
            elif system_health > 0.4:
                signal_type = SignalType.HOLD
                confidence = 0.5
                reasoning = "System performance moderate, monitoring for improvements"
            else:
                signal_type = SignalType.SELL
                confidence = 0.6
                reasoning = "System performance below optimal, agent rotation recommended"
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'system_health': system_health,
                    'current_regime': self.current_regime.value,
                    'active_agents': len([r for r in self.agent_rankings.values() if r.overall_score > 0.5]),
                    'rotation_decisions': len(self.rotation_decisions),
                    'method': 'meta_evaluation'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Meta-evaluation signal generation failed: {e}")
            return self._create_hold_signal("Signal generation error", context)
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        try:
            if not self.agent_rankings:
                return 0.5
            
            # Calculate weighted average of agent performance
            total_score = 0.0
            total_weight = 0.0
            
            for agent_name, ranking in self.agent_rankings.items():
                # Weight by regime performance
                regime_weight = self.config['regime_weights'].get(self.current_regime.value, 1.0)
                agent_score = ranking.overall_score
                
                total_score += agent_score * regime_weight
                total_weight += regime_weight
            
            return total_score / total_weight if total_weight > 0 else 0.5
            
        except Exception as e:
            logger.error(f"System health calculation failed: {e}")
            return 0.5
    
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
    
    def get_agent_rankings(self) -> Dict[str, AgentRanking]:
        """Get current agent rankings."""
        return self.agent_rankings.copy()
    
    def get_rotation_decisions(self) -> List[RotationDecision]:
        """Get recent rotation decisions."""
        return list(self.rotation_decisions)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        try:
            return {
                'overall_health': self._calculate_system_health(),
                'current_regime': self.current_regime.value,
                'active_agents': len([r for r in self.agent_rankings.values() if r.overall_score > 0.5]),
                'total_agents': len(self.agent_rankings),
                'recent_rotations': len(self.rotation_decisions),
                'evaluation_count': self.evaluation_count,
                'last_rotation': self.last_rotation_time.isoformat() if self.last_rotation_time != datetime.min else None
            }
            
        except Exception as e:
            logger.error(f"System health retrieval failed: {e}")
            return {}
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics."""
        try:
            analytics = {
                'agent_performance': {},
                'regime_analysis': {},
                'trend_analysis': {},
                'recommendations': []
            }
            
            # Agent performance analysis
            for agent_name, ranking in self.agent_rankings.items():
                analytics['agent_performance'][agent_name] = {
                    'overall_score': ranking.overall_score,
                    'regime_scores': ranking.regime_scores,
                    'trend': ranking.performance_trend,
                    'recommendations': ranking.recommendations,
                    'last_updated': ranking.last_updated.isoformat()
                }
            
            # Regime analysis
            for regime in RegimeType:
                regime_performances = self.regime_performance.get(regime.value, [])
                if regime_performances:
                    analytics['regime_analysis'][regime.value] = {
                        'avg_accuracy': np.mean([p.accuracy for p in regime_performances]),
                        'avg_sharpe': np.mean([p.sharpe_ratio for p in regime_performances]),
                        'sample_count': len(regime_performances)
                    }
            
            # Trend analysis
            analytics['trend_analysis'] = {
                'improving_agents': len([r for r in self.agent_rankings.values() if r.performance_trend == "improving"]),
                'declining_agents': len([r for r in self.agent_rankings.values() if r.performance_trend == "declining"]),
                'stable_agents': len([r for r in self.agent_rankings.values() if r.performance_trend == "stable"])
            }
            
            # System recommendations
            system_health = self._calculate_system_health()
            if system_health < 0.6:
                analytics['recommendations'].append("System performance below optimal - consider agent rotation")
            if len(self.rotation_decisions) > 5:
                analytics['recommendations'].append("Multiple rotation decisions pending - review agent allocation")
            
            return analytics
            
        except Exception as e:
            logger.error(f"Performance analytics retrieval failed: {e}")
            return {}
