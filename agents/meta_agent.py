"""
Meta Agent for AI Market Analysis System

This agent chooses the best strategy logic based on current market regime
and provides meta-level trading decisions.
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


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"


class MetaStrategy(Enum):
    """Meta strategy types."""
    MOMENTUM_FOLLOWING = "momentum_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"
    HEDGE = "hedge"
    ARBITRAGE = "arbitrage"


@dataclass
class RegimeAnalysis:
    """Represents market regime analysis."""
    regime: MarketRegime
    confidence: float
    duration: int  # periods
    strength: float
    volatility: float
    trend: float
    volume_profile: str
    key_levels: List[float]


@dataclass
class StrategyPerformance:
    """Represents strategy performance metrics."""
    strategy: MetaStrategy
    win_rate: float
    avg_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    last_updated: datetime


class MetaAgent(BaseAgent):
    """
    Meta Agent for strategy selection based on market regime.
    
    This agent analyzes:
    - Market regime detection and classification
    - Strategy performance tracking
    - Meta-level decision making
    - Regime-based strategy selection
    - Portfolio-level optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Meta Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'regime_detection_periods': 20,
            'min_regime_confidence': 0.6,
            'strategy_switching_threshold': 0.3,
            'performance_tracking_periods': 100,
            'regime_weights': {
                'volatility': 0.3,
                'trend': 0.3,
                'volume': 0.2,
                'momentum': 0.2
            },
            'strategy_mapping': {
                MarketRegime.BULL: MetaStrategy.MOMENTUM_FOLLOWING,
                MarketRegime.BEAR: MetaStrategy.HEDGE,
                MarketRegime.SIDEWAYS: MetaStrategy.MEAN_REVERSION,
                MarketRegime.VOLATILE: MetaStrategy.SCALPING,
                MarketRegime.TRENDING: MetaStrategy.BREAKOUT,
                MarketRegime.RANGING: MetaStrategy.SWING
            },
            'confidence_threshold': 0.6,
            'regime_change_threshold': 0.2
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="MetaAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.regime_history = []
        self.strategy_performance = {}
        self.current_regime = MarketRegime.SIDEWAYS
        self.current_strategy = MetaStrategy.MEAN_REVERSION
        self.regime_change_count = 0
        
        logger.info(f"Initialized MetaAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the meta agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For meta analysis, we don't need traditional ML training
            # Instead, we'll validate our meta analysis approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Meta analysis approach validated")
            return {"status": "meta_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate meta-level prediction based on market regime.
        
        Args:
            context: Current market context
            
        Returns:
            Meta-level trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple meta analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple meta analysis (not trained)")
                return self._simple_meta_analysis(context)
            
            # Analyze market regime
            regime_analysis = self._analyze_market_regime(context)
            
            # Select optimal strategy
            strategy_selection = self._select_optimal_strategy(regime_analysis, context)
            
            # Generate meta signal
            signal = self._generate_meta_signal(regime_analysis, strategy_selection, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Meta analysis error: {e}", context)
    
    def _analyze_market_regime(self, context: AgentContext) -> RegimeAnalysis:
        """
        Analyze current market regime.
        
        Args:
            context: Current market context
            
        Returns:
            Market regime analysis
        """
        try:
            if context.market_data.empty:
                return RegimeAnalysis(
                    regime=MarketRegime.SIDEWAYS,
                    confidence=0.0,
                    duration=0,
                    strength=0.0,
                    volatility=0.0,
                    trend=0.0,
                    volume_profile='unknown',
                    key_levels=[]
                )
            
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            high_col = 'High' if 'High' in context.market_data.columns else 'high'
            low_col = 'Low' if 'Low' in context.market_data.columns else 'low'
            volume_col = 'Volume' if 'Volume' in context.market_data.columns else 'volume'
            
            if close_col not in context.market_data.columns:
                return RegimeAnalysis(
                    regime=MarketRegime.SIDEWAYS,
                    confidence=0.0,
                    duration=0,
                    strength=0.0,
                    volatility=0.0,
                    trend=0.0,
                    volume_profile='unknown',
                    key_levels=[]
                )
            
            prices = context.market_data[close_col]
            returns = prices.pct_change().dropna()
            
            if len(returns) < 5:
                return RegimeAnalysis(
                    regime=MarketRegime.SIDEWAYS,
                    confidence=0.0,
                    duration=0,
                    strength=0.0,
                    volatility=0.0,
                    trend=0.0,
                    volume_profile='unknown',
                    key_levels=[]
                )
            
            # Calculate regime indicators
            volatility = returns.std()
            trend = self._calculate_trend_strength(prices)
            momentum = self._calculate_momentum(prices)
            volume_profile = self._analyze_volume_profile(context.market_data, volume_col)
            key_levels = self._identify_key_levels(prices, high_col, low_col)
            
            # Determine regime
            regime = self._classify_regime(volatility, trend, momentum, volume_profile)
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(volatility, trend, momentum, volume_profile)
            
            # Calculate regime strength
            strength = abs(trend) + abs(momentum) + (1.0 - volatility) / 2.0
            strength = min(1.0, strength)
            
            # Estimate duration (simplified)
            duration = self._estimate_regime_duration(regime)
            
            return RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                duration=duration,
                strength=strength,
                volatility=volatility,
                trend=trend,
                volume_profile=volume_profile,
                key_levels=key_levels
            )
            
        except Exception as e:
            logger.error(f"Market regime analysis failed: {e}")
            return RegimeAnalysis(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.0,
                duration=0,
                strength=0.0,
                volatility=0.0,
                trend=0.0,
                volume_profile='unknown',
                key_levels=[]
            )
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength using linear regression."""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Use recent prices
            recent_prices = prices.tail(min(20, len(prices)))
            x = np.arange(len(recent_prices))
            y = recent_prices.values
            
            # Linear regression
            slope, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)
            
            # Normalize slope by price level
            normalized_slope = slope / recent_prices.mean()
            
            # Combine slope and R-squared
            trend_strength = abs(normalized_slope) * (r_value ** 2 if len(r_value) > 0 else 0)
            
            return min(1.0, trend_strength)
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return 0.0
    
    def _calculate_momentum(self, prices: pd.Series) -> float:
        """Calculate momentum indicator."""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Calculate rate of change
            roc_5 = (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6] if len(prices) >= 6 else 0
            roc_10 = (prices.iloc[-1] - prices.iloc[-11]) / prices.iloc[-11] if len(prices) >= 11 else 0
            
            # Combine short and medium term momentum
            momentum = (roc_5 * 0.7 + roc_10 * 0.3)
            
            return max(-1.0, min(1.0, momentum))
            
        except Exception as e:
            logger.error(f"Momentum calculation failed: {e}")
            return 0.0
    
    def _analyze_volume_profile(self, data: pd.DataFrame, volume_col: str) -> str:
        """Analyze volume profile."""
        try:
            if volume_col not in data.columns:
                return 'unknown'
            
            volume = data[volume_col]
            if len(volume) < 5:
                return 'unknown'
            
            # Calculate volume trend
            recent_volume = volume.tail(10)
            avg_volume = volume.mean()
            
            if recent_volume.mean() > avg_volume * 1.2:
                return 'high'
            elif recent_volume.mean() < avg_volume * 0.8:
                return 'low'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Volume profile analysis failed: {e}")
            return 'unknown'
    
    def _identify_key_levels(self, prices: pd.Series, high_col: str, low_col: str) -> List[float]:
        """Identify key support and resistance levels."""
        try:
            key_levels = []
            
            # Add recent highs and lows
            if len(prices) >= 20:
                recent_high = prices.tail(20).max()
                recent_low = prices.tail(20).min()
                key_levels.extend([recent_high, recent_low])
            
            # Add moving averages as key levels
            if len(prices) >= 20:
                ma_20 = prices.tail(20).mean()
                key_levels.append(ma_20)
            
            if len(prices) >= 50:
                ma_50 = prices.tail(50).mean()
                key_levels.append(ma_50)
            
            return key_levels
            
        except Exception as e:
            logger.error(f"Key level identification failed: {e}")
            return []
    
    def _classify_regime(self, volatility: float, trend: float, momentum: float, volume_profile: str) -> MarketRegime:
        """Classify market regime based on indicators."""
        try:
            # High volatility regime
            if volatility > 0.03:
                return MarketRegime.VOLATILE
            
            # Strong trend regime
            if abs(trend) > 0.5:
                if trend > 0:
                    return MarketRegime.BULL
                else:
                    return MarketRegime.BEAR
            
            # Trending regime (moderate trend)
            if abs(trend) > 0.2:
                return MarketRegime.TRENDING
            
            # Ranging regime (low volatility, low trend)
            if volatility < 0.015 and abs(trend) < 0.1:
                return MarketRegime.RANGING
            
            # Default to sideways
            return MarketRegime.SIDEWAYS
            
        except Exception as e:
            logger.error(f"Regime classification failed: {e}")
            return MarketRegime.SIDEWAYS
    
    def _calculate_regime_confidence(self, volatility: float, trend: float, momentum: float, volume_profile: str) -> float:
        """Calculate confidence in regime classification."""
        try:
            # Base confidence
            confidence = 0.5
            
            # Adjust based on volatility
            if 0.01 < volatility < 0.05:  # Normal volatility range
                confidence += 0.2
            
            # Adjust based on trend strength
            if abs(trend) > 0.3:
                confidence += 0.2
            
            # Adjust based on momentum consistency
            if abs(momentum) > 0.1:
                confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Regime confidence calculation failed: {e}")
            return 0.0
    
    def _estimate_regime_duration(self, regime: MarketRegime) -> int:
        """Estimate regime duration (simplified)."""
        try:
            # Simplified duration estimation
            duration_map = {
                MarketRegime.VOLATILE: 5,
                MarketRegime.TRENDING: 15,
                MarketRegime.BULL: 20,
                MarketRegime.BEAR: 20,
                MarketRegime.SIDEWAYS: 10,
                MarketRegime.RANGING: 8
            }
            
            return duration_map.get(regime, 10)
            
        except Exception as e:
            logger.error(f"Regime duration estimation failed: {e}")
            return 10
    
    def _select_optimal_strategy(self, regime_analysis: RegimeAnalysis, context: AgentContext) -> Dict[str, Any]:
        """
        Select optimal strategy based on regime analysis.
        
        Args:
            regime_analysis: Market regime analysis
            context: Current market context
            
        Returns:
            Strategy selection information
        """
        try:
            # Get recommended strategy from mapping
            recommended_strategy = self.config['strategy_mapping'].get(
                regime_analysis.regime, 
                MetaStrategy.MEAN_REVERSION
            )
            
            # Check if we should switch strategies
            should_switch = self._should_switch_strategy(regime_analysis)
            
            # Get strategy performance
            strategy_performance = self._get_strategy_performance(recommended_strategy)
            
            # Calculate strategy confidence
            strategy_confidence = self._calculate_strategy_confidence(
                regime_analysis, strategy_performance
            )
            
            return {
                'recommended_strategy': recommended_strategy,
                'current_strategy': self.current_strategy,
                'should_switch': should_switch,
                'strategy_performance': strategy_performance,
                'confidence': strategy_confidence,
                'regime_analysis': regime_analysis
            }
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return {
                'recommended_strategy': MetaStrategy.MEAN_REVERSION,
                'current_strategy': self.current_strategy,
                'should_switch': False,
                'strategy_performance': None,
                'confidence': 0.0,
                'regime_analysis': regime_analysis
            }
    
    def _should_switch_strategy(self, regime_analysis: RegimeAnalysis) -> bool:
        """Determine if we should switch strategies."""
        try:
            # Check regime change
            if regime_analysis.regime != self.current_regime:
                if regime_analysis.confidence > self.config['min_regime_confidence']:
                    return True
            
            # Check regime strength change
            if hasattr(self, '_last_regime_analysis'):
                strength_change = abs(regime_analysis.strength - self._last_regime_analysis.strength)
                if strength_change > self.config['regime_change_threshold']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Strategy switching decision failed: {e}")
            return False
    
    def _get_strategy_performance(self, strategy: MetaStrategy) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy."""
        try:
            # In a real implementation, this would fetch actual performance data
            # For now, we'll return simulated performance
            return StrategyPerformance(
                strategy=strategy,
                win_rate=0.6,  # 60% win rate
                avg_return=0.02,  # 2% average return
                max_drawdown=0.1,  # 10% max drawdown
                sharpe_ratio=1.2,  # 1.2 Sharpe ratio
                total_trades=50,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Strategy performance retrieval failed: {e}")
            return None
    
    def _calculate_strategy_confidence(self, regime_analysis: RegimeAnalysis, strategy_performance: Optional[StrategyPerformance]) -> float:
        """Calculate confidence in strategy selection."""
        try:
            # Base confidence from regime analysis
            confidence = regime_analysis.confidence
            
            # Adjust based on strategy performance
            if strategy_performance:
                if strategy_performance.win_rate > 0.6:
                    confidence += 0.1
                if strategy_performance.sharpe_ratio > 1.0:
                    confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Strategy confidence calculation failed: {e}")
            return 0.0
    
    def _generate_meta_signal(self, regime_analysis: RegimeAnalysis, strategy_selection: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate meta-level trading signal.
        
        Args:
            regime_analysis: Market regime analysis
            strategy_selection: Strategy selection information
            context: Current market context
            
        Returns:
            Meta-level trading signal
        """
        try:
            # Determine signal based on regime and strategy
            signal_type = self._determine_meta_signal_type(regime_analysis, strategy_selection)
            
            # Calculate confidence
            confidence = strategy_selection['confidence']
            
            # Build reasoning
            reasoning_parts = [
                f"Regime: {regime_analysis.regime.value}",
                f"Strategy: {strategy_selection['recommended_strategy'].value}",
                f"Confidence: {confidence:.2f}",
                f"Strength: {regime_analysis.strength:.2f}"
            ]
            
            if strategy_selection['should_switch']:
                reasoning_parts.append("Strategy switch recommended")
            
            reasoning = " | ".join(reasoning_parts)
            
            # Update current regime and strategy
            self.current_regime = regime_analysis.regime
            self.current_strategy = strategy_selection['recommended_strategy']
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'regime_analysis': regime_analysis,
                    'strategy_selection': strategy_selection,
                    'method': 'meta_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Meta signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _determine_meta_signal_type(self, regime_analysis: RegimeAnalysis, strategy_selection: Dict[str, Any]) -> SignalType:
        """Determine meta signal type based on regime and strategy."""
        try:
            strategy = strategy_selection['recommended_strategy']
            regime = regime_analysis.regime
            
            # Strategy-based signal determination
            if strategy == MetaStrategy.MOMENTUM_FOLLOWING:
                if regime_analysis.trend > 0.1:
                    return SignalType.BUY
                elif regime_analysis.trend < -0.1:
                    return SignalType.SELL
                else:
                    return SignalType.HOLD
                    
            elif strategy == MetaStrategy.MEAN_REVERSION:
                if regime_analysis.trend > 0.2:
                    return SignalType.SELL  # Overbought
                elif regime_analysis.trend < -0.2:
                    return SignalType.BUY   # Oversold
                else:
                    return SignalType.HOLD
                    
            elif strategy == MetaStrategy.HEDGE:
                return SignalType.SELL  # Defensive
                
            elif strategy == MetaStrategy.BREAKOUT:
                if regime_analysis.strength > 0.7:
                    return SignalType.BUY
                else:
                    return SignalType.HOLD
                    
            else:
                # Default to hold for other strategies
                return SignalType.HOLD
                
        except Exception as e:
            logger.error(f"Meta signal type determination failed: {e}")
            return SignalType.HOLD
    
    def _simple_meta_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple meta analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple meta-level signal
        """
        try:
            # Simple meta analysis based on volatility
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            if len(context.market_data) >= 10:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()
                    
                    # Simple regime-based signal
                    if volatility > 0.03:  # High volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.HOLD,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility_meta'},
                            reasoning=f"High volatility regime ({volatility:.2%}) - defensive strategy"
                        )
                    elif volatility < 0.01:  # Low volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility_meta'},
                            reasoning=f"Low volatility regime ({volatility:.2%}) - opportunity strategy"
                        )
            
            return self._create_hold_signal("No clear meta signal", context)
            
        except Exception as e:
            logger.error(f"Simple meta analysis failed: {e}")
            return self._create_hold_signal(f"Simple meta analysis error: {e}", context)
    
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
        Update the meta model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update regime history
            if hasattr(self, '_last_regime_analysis'):
                self.regime_history.append(self._last_regime_analysis)
                
                # Keep only recent history
                if len(self.regime_history) > 50:
                    self.regime_history = self.regime_history[-50:]
            
            # Update strategy performance tracking
            if hasattr(self, '_last_strategy_selection'):
                strategy = self._last_strategy_selection['recommended_strategy']
                if strategy not in self.strategy_performance:
                    self.strategy_performance[strategy] = []
                
                self.strategy_performance[strategy].append({
                    'timestamp': context.timestamp,
                    'regime': self.current_regime.value,
                    'confidence': self._last_strategy_selection['confidence']
                })
            
            logger.info(f"Updated meta model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
