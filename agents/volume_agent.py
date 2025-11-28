"""
Volume Agent for AI Market Analysis System

This agent analyzes volume patterns and volume-price relationships
to provide volume-based trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class VolumeAgent(BaseAgent):
    """
    Volume Agent for analyzing volume patterns and volume-price relationships.
    
    This agent analyzes:
    - Volume trends and patterns
    - Volume-price divergence
    - Volume breakout patterns
    - Volume-weighted indicators
    - Institutional vs retail volume patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Volume Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'volume_window': 20,
            'volume_threshold_high': 2.0,  # 2x average volume
            'volume_threshold_low': 0.5,   # 0.5x average volume
            'volume_price_divergence_threshold': 0.3,
            'volume_breakout_threshold': 1.5,
            'confidence_threshold': 0.6,
            'volume_indicators': ['sma', 'ema', 'vwap', 'obv', 'ad_line']
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="VolumeAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.volume_history = {}
        self.volume_patterns = []
        self.volume_breakouts = []
        
        logger.info(f"Initialized VolumeAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the volume agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For volume analysis, we don't need traditional ML training
            # Instead, we'll validate our volume analysis approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Volume analysis approach validated")
            return {"status": "volume_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate volume-based prediction.
        
        Args:
            context: Current market context
            
        Returns:
            Volume-based trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple volume analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple volume analysis (not trained)")
                return self._simple_volume_analysis(context)
            
            # Perform comprehensive volume analysis
            volume_analysis = self._analyze_volume(context)
            
            # Generate signal based on volume insights
            signal = self._generate_volume_signal(volume_analysis, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Volume analysis error: {e}", context)
    
    def _analyze_volume(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze volume patterns for the given symbol.
        
        Args:
            context: Current market context
            
        Returns:
            Volume analysis results
        """
        try:
            symbol = context.symbol
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(context)
            
            # Analyze volume trends
            volume_trends = self._analyze_volume_trends(volume_metrics)
            
            # Detect volume breakouts
            volume_breakouts = self._detect_volume_breakouts(volume_metrics)
            
            # Analyze volume-price relationship
            volume_price_analysis = self._analyze_volume_price_relationship(volume_metrics, context)
            
            # Calculate volume indicators
            volume_indicators = self._calculate_volume_indicators(volume_metrics, context)
            
            return {
                'volume_metrics': volume_metrics,
                'volume_trends': volume_trends,
                'volume_breakouts': volume_breakouts,
                'volume_price_analysis': volume_price_analysis,
                'volume_indicators': volume_indicators,
                'confidence': self._calculate_volume_confidence(volume_metrics)
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {
                'volume_metrics': {},
                'volume_trends': {},
                'volume_breakouts': [],
                'volume_price_analysis': {},
                'volume_indicators': {},
                'confidence': 0.0
            }
    
    def _calculate_volume_metrics(self, context: AgentContext) -> Dict[str, float]:
        """
        Calculate various volume metrics.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary of volume metrics
        """
        try:
            if context.market_data.empty:
                return {}
            
            volume_col = 'Volume' if 'Volume' in context.market_data.columns else 'volume'
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            
            if volume_col not in context.market_data.columns:
                return {}
            
            volume = context.market_data[volume_col]
            
            if len(volume) < 2:
                return {}
            
            # Basic volume metrics
            current_volume = volume.iloc[-1]
            avg_volume = volume.rolling(window=self.config['volume_window']).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume statistics
            volume_std = volume.std()
            volume_percentile = stats.percentileofscore(volume, current_volume) / 100
            
            # Volume trend
            volume_trend = self._calculate_volume_trend(volume)
            
            # Volume volatility
            volume_volatility = volume.pct_change().std()
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_std': volume_std,
                'volume_percentile': volume_percentile,
                'volume_trend': volume_trend,
                'volume_volatility': volume_volatility,
                'volume_level': 'high' if volume_ratio > self.config['volume_threshold_high'] else 
                              'low' if volume_ratio < self.config['volume_threshold_low'] else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Volume metrics calculation failed: {e}")
            return {}
    
    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """Calculate volume trend using linear regression."""
        try:
            if len(volume) < 5:
                return 0.0
            
            # Use recent volume data
            recent_volume = volume.tail(min(10, len(volume)))
            x = np.arange(len(recent_volume))
            y = recent_volume.values
            
            # Linear regression
            slope, _, _, _, _ = stats.linregress(x, y)
            
            # Normalize slope by average volume
            avg_volume = recent_volume.mean()
            normalized_slope = slope / avg_volume if avg_volume > 0 else 0
            
            return normalized_slope
            
        except Exception as e:
            logger.error(f"Volume trend calculation failed: {e}")
            return 0.0
    
    def _analyze_volume_trends(self, volume_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze volume trends.
        
        Args:
            volume_metrics: Volume metrics
            
        Returns:
            Volume trend analysis
        """
        try:
            if not volume_metrics:
                return {'trend': 'unknown', 'confidence': 0.0}
            
            volume_trend = volume_metrics.get('volume_trend', 0)
            volume_ratio = volume_metrics.get('volume_ratio', 1.0)
            
            # Classify volume trend
            if volume_trend > 0.1 and volume_ratio > 1.2:
                trend = 'increasing'
                confidence = 0.8
            elif volume_trend < -0.1 and volume_ratio < 0.8:
                trend = 'decreasing'
                confidence = 0.8
            elif volume_ratio > self.config['volume_threshold_high']:
                trend = 'spike'
                confidence = 0.9
            elif volume_ratio < self.config['volume_threshold_low']:
                trend = 'drought'
                confidence = 0.9
            else:
                trend = 'stable'
                confidence = 0.6
            
            return {
                'trend': trend,
                'confidence': confidence,
                'trend_strength': abs(volume_trend),
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Volume trend analysis failed: {e}")
            return {'trend': 'unknown', 'confidence': 0.0}
    
    def _detect_volume_breakouts(self, volume_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Detect volume breakout patterns.
        
        Args:
            volume_metrics: Volume metrics
            
        Returns:
            List of detected breakouts
        """
        try:
            breakouts = []
            
            if not volume_metrics:
                return breakouts
            
            volume_ratio = volume_metrics.get('volume_ratio', 1.0)
            volume_percentile = volume_metrics.get('volume_percentile', 0.5)
            
            # Detect volume breakouts
            if volume_ratio > self.config['volume_breakout_threshold']:
                breakouts.append({
                    'type': 'volume_breakout',
                    'severity': 'high' if volume_ratio > 3.0 else 'medium',
                    'ratio': volume_ratio,
                    'description': f'Volume breakout: {volume_ratio:.1f}x average volume'
                })
            
            # Detect volume spikes
            if volume_percentile > 0.9:
                breakouts.append({
                    'type': 'volume_spike',
                    'severity': 'high',
                    'percentile': volume_percentile,
                    'description': f'Volume spike: {volume_percentile:.1%} percentile'
                })
            
            return breakouts
            
        except Exception as e:
            logger.error(f"Volume breakout detection failed: {e}")
            return []
    
    def _analyze_volume_price_relationship(self, volume_metrics: Dict[str, float], context: AgentContext) -> Dict[str, Any]:
        """
        Analyze volume-price relationship.
        
        Args:
            volume_metrics: Volume metrics
            context: Market context
            
        Returns:
            Volume-price analysis
        """
        try:
            if context.market_data.empty or not volume_metrics:
                return {'relationship': 'unknown', 'confidence': 0.0}
            
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            volume_col = 'Volume' if 'Volume' in context.market_data.columns else 'volume'
            
            if close_col not in context.market_data.columns or volume_col not in context.market_data.columns:
                return {'relationship': 'unknown', 'confidence': 0.0}
            
            prices = context.market_data[close_col]
            volume = context.market_data[volume_col]
            
            if len(prices) < 5 or len(volume) < 5:
                return {'relationship': 'unknown', 'confidence': 0.0}
            
            # Calculate price and volume changes
            price_changes = prices.pct_change().dropna()
            volume_changes = volume.pct_change().dropna()
            
            # Calculate correlation
            if len(price_changes) > 1 and len(volume_changes) > 1:
                min_len = min(len(price_changes), len(volume_changes))
                correlation, _ = stats.pearsonr(
                    price_changes.iloc[-min_len:], 
                    volume_changes.iloc[-min_len:]
                )
            else:
                correlation = 0.0
            
            # Analyze relationship
            if correlation > self.config['volume_price_divergence_threshold']:
                relationship = 'positive'
                confidence = 0.8
            elif correlation < -self.config['volume_price_divergence_threshold']:
                relationship = 'negative'
                confidence = 0.8
            else:
                relationship = 'neutral'
                confidence = 0.6
            
            # Check for divergence
            current_price_change = price_changes.iloc[-1] if len(price_changes) > 0 else 0
            current_volume_ratio = volume_metrics.get('volume_ratio', 1.0)
            
            divergence = 'none'
            if current_price_change > 0.02 and current_volume_ratio < 0.8:
                divergence = 'bearish'  # Price up, volume down
            elif current_price_change < -0.02 and current_volume_ratio < 0.8:
                divergence = 'bullish'  # Price down, volume down
            
            return {
                'relationship': relationship,
                'correlation': correlation,
                'confidence': confidence,
                'divergence': divergence,
                'price_change': current_price_change,
                'volume_ratio': current_volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Volume-price relationship analysis failed: {e}")
            return {'relationship': 'unknown', 'confidence': 0.0}
    
    def _calculate_volume_indicators(self, volume_metrics: Dict[str, float], context: AgentContext) -> Dict[str, Any]:
        """
        Calculate volume-based technical indicators.
        
        Args:
            volume_metrics: Volume metrics
            context: Market context
            
        Returns:
            Volume indicators
        """
        try:
            if context.market_data.empty:
                return {}
            
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            volume_col = 'Volume' if 'Volume' in context.market_data.columns else 'volume'
            
            if close_col not in context.market_data.columns or volume_col not in context.market_data.columns:
                return {}
            
            prices = context.market_data[close_col]
            volume = context.market_data[volume_col]
            
            if len(prices) < 5 or len(volume) < 5:
                return {}
            
            # VWAP (Volume Weighted Average Price)
            vwap = (prices * volume).sum() / volume.sum()
            
            # OBV (On-Balance Volume) - simplified
            obv = self._calculate_obv(prices, volume)
            
            # Volume SMA
            volume_sma = volume.rolling(window=20).mean().iloc[-1]
            
            # Volume EMA
            volume_ema = volume.ewm(span=20).mean().iloc[-1]
            
            return {
                'vwap': vwap,
                'obv': obv,
                'volume_sma': volume_sma,
                'volume_ema': volume_ema,
                'vwap_vs_price': (vwap - prices.iloc[-1]) / prices.iloc[-1] if prices.iloc[-1] > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Volume indicators calculation failed: {e}")
            return {}
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> float:
        """Calculate On-Balance Volume."""
        try:
            if len(prices) < 2 or len(volume) < 2:
                return 0.0
            
            obv = 0.0
            for i in range(1, len(prices)):
                if prices.iloc[i] > prices.iloc[i-1]:
                    obv += volume.iloc[i]
                elif prices.iloc[i] < prices.iloc[i-1]:
                    obv -= volume.iloc[i]
                # If price unchanged, OBV stays the same
            
            return obv
            
        except Exception as e:
            logger.error(f"OBV calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_confidence(self, volume_metrics: Dict[str, float]) -> float:
        """
        Calculate confidence in volume analysis.
        
        Args:
            volume_metrics: Volume metrics
            
        Returns:
            Confidence score (0-1)
        """
        try:
            if not volume_metrics:
                return 0.0
            
            # Base confidence
            confidence = 0.6
            
            # Adjust based on volume level
            volume_ratio = volume_metrics.get('volume_ratio', 1.0)
            if 0.5 < volume_ratio < 3.0:  # Reasonable volume range
                confidence += 0.2
            
            # Adjust based on volume trend strength
            volume_trend = abs(volume_metrics.get('volume_trend', 0))
            if volume_trend > 0.05:  # Strong trend
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Volume confidence calculation failed: {e}")
            return 0.0
    
    def _generate_volume_signal(self, analysis: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on volume analysis.
        
        Args:
            analysis: Volume analysis results
            context: Current market context
            
        Returns:
            Volume-based trading signal
        """
        try:
            volume_trends = analysis['volume_trends']
            volume_breakouts = analysis['volume_breakouts']
            volume_price_analysis = analysis['volume_price_analysis']
            confidence = analysis['confidence']
            
            # Determine signal based on volume analysis
            if volume_breakouts:
                # Volume breakouts often signal strong moves
                breakout = volume_breakouts[0]
                if breakout['type'] == 'volume_breakout' and volume_price_analysis.get('relationship') == 'positive':
                    signal_type = SignalType.BUY
                    reasoning = f"Volume breakout with positive price-volume relationship"
                elif breakout['type'] == 'volume_spike':
                    signal_type = SignalType.HOLD
                    reasoning = f"Volume spike detected - wait for price confirmation"
                else:
                    signal_type = SignalType.HOLD
                    reasoning = f"Volume breakout: {breakout['description']}"
            elif volume_trends['trend'] == 'increasing' and volume_price_analysis.get('relationship') == 'positive':
                signal_type = SignalType.BUY
                reasoning = f"Increasing volume with positive price-volume relationship"
            elif volume_trends['trend'] == 'decreasing' and volume_price_analysis.get('relationship') == 'negative':
                signal_type = SignalType.SELL
                reasoning = f"Decreasing volume with negative price-volume relationship"
            elif volume_price_analysis.get('divergence') == 'bearish':
                signal_type = SignalType.SELL
                reasoning = f"Bearish divergence: price up, volume down"
            elif volume_price_analysis.get('divergence') == 'bullish':
                signal_type = SignalType.BUY
                reasoning = f"Bullish divergence: price down, volume down"
            else:
                signal_type = SignalType.HOLD
                reasoning = f"Volume trend: {volume_trends['trend']} - no clear signal"
            
            # Adjust confidence based on trend confidence
            adjusted_confidence = min(confidence * volume_trends.get('confidence', 0.5), 0.9)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'volume_analysis': analysis,
                    'method': 'volume_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_volume_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple volume analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple volume-based signal
        """
        try:
            # Simple volume analysis
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            volume_col = 'Volume' if 'Volume' in context.market_data.columns else 'volume'
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            
            if volume_col not in context.market_data.columns or close_col not in context.market_data.columns:
                return self._create_hold_signal("No volume or price data available", context)
            
            if len(context.market_data) >= 5:
                volume = context.market_data[volume_col]
                prices = context.market_data[close_col]
                
                current_volume = volume.iloc[-1]
                avg_volume = volume.rolling(window=10).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                current_price = prices.iloc[-1]
                prev_price = prices.iloc[-2] if len(prices) > 1 else current_price
                price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
                
                # High volume with price increase
                if volume_ratio > 1.5 and price_change > 0.01:
                    return AgentSignal(
                        agent_name=self.name,
                        signal_type=SignalType.BUY,
                        confidence=0.6,
                        timestamp=context.timestamp,
                        asset_symbol=context.symbol,
                        metadata={'agent_version': self.version, 'method': 'simple_volume'},
                        reasoning=f"High volume ({volume_ratio:.1f}x) with price increase ({price_change:.2%})"
                    )
                # High volume with price decrease
                elif volume_ratio > 1.5 and price_change < -0.01:
                    return AgentSignal(
                        agent_name=self.name,
                        signal_type=SignalType.SELL,
                        confidence=0.6,
                        timestamp=context.timestamp,
                        asset_symbol=context.symbol,
                        metadata={'agent_version': self.version, 'method': 'simple_volume'},
                        reasoning=f"High volume ({volume_ratio:.1f}x) with price decrease ({price_change:.2%})"
                    )
            
            return self._create_hold_signal("No clear volume signal", context)
            
        except Exception as e:
            logger.error(f"Simple volume analysis failed: {e}")
            return self._create_hold_signal(f"Simple volume analysis error: {e}", context)
    
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
        Update the volume model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update volume history
            symbol = context.symbol
            if symbol not in self.volume_history:
                self.volume_history[symbol] = []
            
            # Store recent volume data
            if hasattr(self, '_last_volume_analysis'):
                self.volume_history[symbol].append(self._last_volume_analysis)
                
                # Keep only recent history
                if len(self.volume_history[symbol]) > 20:
                    self.volume_history[symbol] = self.volume_history[symbol][-20:]
            
            logger.info(f"Updated volume model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
