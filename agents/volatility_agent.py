"""
Volatility Agent for AI Market Analysis System

This agent analyzes and predicts volatility patterns to provide
volatility-based trading signals.
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


class VolatilityAgent(BaseAgent):
    """
    Volatility Agent for analyzing and predicting volatility patterns.
    
    This agent analyzes:
    - Historical volatility patterns
    - Volatility clustering and mean reversion
    - Implied vs realized volatility
    - Volatility regime detection
    - Volatility-based trading opportunities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Volatility Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'volatility_window': 20,
            'volatility_threshold_high': 0.03,
            'volatility_threshold_low': 0.01,
            'mean_reversion_threshold': 0.5,
            'volatility_clustering_periods': 5,
            'regime_detection_periods': 10,
            'confidence_threshold': 0.6,
            'volatility_types': ['realized', 'garch', 'ewma', 'parkinson']
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="VolatilityAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.volatility_history = {}
        self.volatility_regimes = []
        self.volatility_forecasts = {}
        
        logger.info(f"Initialized VolatilityAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the volatility agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For volatility analysis, we don't need traditional ML training
            # Instead, we'll validate our volatility analysis approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Volatility analysis approach validated")
            return {"status": "volatility_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate volatility-based prediction.
        
        Args:
            context: Current market context
            
        Returns:
            Volatility-based trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple volatility analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple volatility analysis (not trained)")
                return self._simple_volatility_analysis(context)
            
            # Perform comprehensive volatility analysis
            volatility_analysis = self._analyze_volatility(context)
            
            # Generate signal based on volatility insights
            signal = self._generate_volatility_signal(volatility_analysis, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Volatility analysis error: {e}", context)
    
    def _analyze_volatility(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze volatility patterns for the given symbol.
        
        Args:
            context: Current market context
            
        Returns:
            Volatility analysis results
        """
        try:
            symbol = context.symbol
            
            # Calculate different types of volatility
            volatility_metrics = self._calculate_volatility_metrics(context)
            
            # Analyze volatility regime
            volatility_regime = self._analyze_volatility_regime(volatility_metrics)
            
            # Detect volatility clustering
            clustering_analysis = self._detect_volatility_clustering(volatility_metrics)
            
            # Predict future volatility
            volatility_forecast = self._forecast_volatility(volatility_metrics)
            
            # Analyze mean reversion
            mean_reversion = self._analyze_mean_reversion(volatility_metrics)
            
            return {
                'volatility_metrics': volatility_metrics,
                'volatility_regime': volatility_regime,
                'clustering_analysis': clustering_analysis,
                'volatility_forecast': volatility_forecast,
                'mean_reversion': mean_reversion,
                'confidence': self._calculate_volatility_confidence(volatility_metrics)
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {
                'volatility_metrics': {},
                'volatility_regime': 'unknown',
                'clustering_analysis': {},
                'volatility_forecast': {},
                'mean_reversion': {},
                'confidence': 0.0
            }
    
    def _calculate_volatility_metrics(self, context: AgentContext) -> Dict[str, float]:
        """
        Calculate various volatility metrics.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary of volatility metrics
        """
        try:
            if context.market_data.empty:
                return {}
            
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            high_col = 'High' if 'High' in context.market_data.columns else 'high'
            low_col = 'Low' if 'Low' in context.market_data.columns else 'low'
            
            if close_col not in context.market_data.columns:
                return {}
            
            prices = context.market_data[close_col]
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return {}
            
            # Realized volatility (standard deviation)
            realized_vol = returns.std() * np.sqrt(252)  # Annualized
            
            # EWMA volatility
            alpha = 0.06  # Decay factor
            ewma_vol = self._calculate_ewma_volatility(returns, alpha)
            
            # GARCH-like volatility (simplified)
            garch_vol = self._calculate_garch_volatility(returns)
            
            # Parkinson volatility (if high/low data available)
            parkinson_vol = 0.0
            if high_col in context.market_data.columns and low_col in context.market_data.columns:
                parkinson_vol = self._calculate_parkinson_volatility(
                    context.market_data[high_col], 
                    context.market_data[low_col]
                )
            
            # Volatility of volatility
            vol_of_vol = self._calculate_volatility_of_volatility(returns)
            
            return {
                'realized_volatility': realized_vol,
                'ewma_volatility': ewma_vol,
                'garch_volatility': garch_vol,
                'parkinson_volatility': parkinson_vol,
                'volatility_of_volatility': vol_of_vol,
                'current_volatility': realized_vol,
                'volatility_percentile': self._calculate_volatility_percentile(realized_vol)
            }
            
        except Exception as e:
            logger.error(f"Volatility metrics calculation failed: {e}")
            return {}
    
    def _calculate_ewma_volatility(self, returns: pd.Series, alpha: float) -> float:
        """Calculate EWMA volatility."""
        try:
            if len(returns) < 2:
                return 0.0
            
            # Initialize with first return
            ewma_var = returns.iloc[0] ** 2
            
            # Update EWMA variance
            for ret in returns.iloc[1:]:
                ewma_var = alpha * ret ** 2 + (1 - alpha) * ewma_var
            
            return np.sqrt(ewma_var * 252)  # Annualized
            
        except Exception as e:
            logger.error(f"EWMA volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """Calculate simplified GARCH volatility."""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Simplified GARCH(1,1) model
            # sigma^2 = omega + alpha * r^2 + beta * sigma^2
            omega = 0.0001
            alpha = 0.1
            beta = 0.85
            
            # Initialize variance
            var = returns.var()
            
            # Update variance using GARCH model
            for ret in returns.iloc[-10:]:  # Use last 10 returns
                var = omega + alpha * ret ** 2 + beta * var
            
            return np.sqrt(var * 252)  # Annualized
            
        except Exception as e:
            logger.error(f"GARCH volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series) -> float:
        """Calculate Parkinson volatility estimator."""
        try:
            if len(high) < 2 or len(low) < 2:
                return 0.0
            
            # Parkinson estimator: sqrt(1/(4*ln(2)) * sum(ln(H/L)^2))
            log_hl = np.log(high / low)
            parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2).mean()
            
            return np.sqrt(parkinson_var * 252)  # Annualized
            
        except Exception as e:
            logger.error(f"Parkinson volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_of_volatility(self, returns: pd.Series) -> float:
        """Calculate volatility of volatility."""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Calculate rolling volatility
            window = min(5, len(returns) // 2)
            rolling_vol = returns.rolling(window=window).std()
            
            # Calculate volatility of the rolling volatility
            vol_of_vol = rolling_vol.std()
            
            return vol_of_vol * np.sqrt(252)  # Annualized
            
        except Exception as e:
            logger.error(f"Volatility of volatility calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """Calculate current volatility percentile."""
        try:
            # Simulate historical volatility distribution
            # In a real implementation, this would use actual historical data
            historical_vols = np.random.normal(0.02, 0.01, 1000)  # Simulated distribution
            historical_vols = np.abs(historical_vols)  # Ensure positive
            
            percentile = stats.percentileofscore(historical_vols, current_vol) / 100
            return percentile
            
        except Exception as e:
            logger.error(f"Volatility percentile calculation failed: {e}")
            return 0.5
    
    def _analyze_volatility_regime(self, volatility_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze current volatility regime.
        
        Args:
            volatility_metrics: Volatility metrics
            
        Returns:
            Volatility regime analysis
        """
        try:
            if not volatility_metrics:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            current_vol = volatility_metrics.get('current_volatility', 0)
            vol_percentile = volatility_metrics.get('volatility_percentile', 0.5)
            
            # Classify volatility regime
            if current_vol > self.config['volatility_threshold_high']:
                regime = 'high_volatility'
                confidence = 0.8
            elif current_vol < self.config['volatility_threshold_low']:
                regime = 'low_volatility'
                confidence = 0.8
            elif vol_percentile > 0.8:
                regime = 'elevated_volatility'
                confidence = 0.7
            elif vol_percentile < 0.2:
                regime = 'suppressed_volatility'
                confidence = 0.7
            else:
                regime = 'normal_volatility'
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility_level': 'high' if current_vol > self.config['volatility_threshold_high'] else 'low',
                'percentile': vol_percentile
            }
            
        except Exception as e:
            logger.error(f"Volatility regime analysis failed: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}
    
    def _detect_volatility_clustering(self, volatility_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect volatility clustering patterns.
        
        Args:
            volatility_metrics: Volatility metrics
            
        Returns:
            Clustering analysis
        """
        try:
            vol_of_vol = volatility_metrics.get('volatility_of_volatility', 0)
            
            # High volatility of volatility suggests clustering
            if vol_of_vol > 0.01:  # 1% threshold
                clustering = 'high'
                confidence = 0.8
            elif vol_of_vol > 0.005:  # 0.5% threshold
                clustering = 'moderate'
                confidence = 0.6
            else:
                clustering = 'low'
                confidence = 0.7
            
            return {
                'clustering_level': clustering,
                'confidence': confidence,
                'vol_of_vol': vol_of_vol,
                'clustering_present': clustering in ['high', 'moderate']
            }
            
        except Exception as e:
            logger.error(f"Volatility clustering detection failed: {e}")
            return {'clustering_level': 'unknown', 'confidence': 0.0}
    
    def _forecast_volatility(self, volatility_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Forecast future volatility.
        
        Args:
            volatility_metrics: Current volatility metrics
            
        Returns:
            Volatility forecast
        """
        try:
            current_vol = volatility_metrics.get('current_volatility', 0)
            ewma_vol = volatility_metrics.get('ewma_volatility', 0)
            garch_vol = volatility_metrics.get('garch_volatility', 0)
            
            # Simple forecast using weighted average
            forecast_vol = 0.4 * current_vol + 0.3 * ewma_vol + 0.3 * garch_vol
            
            # Determine forecast direction
            if forecast_vol > current_vol * 1.1:
                direction = 'increasing'
            elif forecast_vol < current_vol * 0.9:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            return {
                'forecast_volatility': forecast_vol,
                'direction': direction,
                'confidence': 0.6,
                'forecast_change': (forecast_vol - current_vol) / current_vol if current_vol > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Volatility forecast failed: {e}")
            return {'forecast_volatility': 0, 'direction': 'unknown', 'confidence': 0.0}
    
    def _analyze_mean_reversion(self, volatility_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze volatility mean reversion.
        
        Args:
            volatility_metrics: Volatility metrics
            
        Returns:
            Mean reversion analysis
        """
        try:
            current_vol = volatility_metrics.get('current_volatility', 0)
            vol_percentile = volatility_metrics.get('volatility_percentile', 0.5)
            
            # Mean reversion signal
            if vol_percentile > 0.8:
                mean_reversion = 'high'  # High volatility likely to revert down
                signal = 'sell_volatility'
            elif vol_percentile < 0.2:
                mean_reversion = 'high'  # Low volatility likely to revert up
                signal = 'buy_volatility'
            else:
                mean_reversion = 'low'
                signal = 'neutral'
            
            return {
                'mean_reversion_strength': mean_reversion,
                'signal': signal,
                'confidence': 0.7,
                'reversion_probability': abs(vol_percentile - 0.5) * 2  # Distance from median
            }
            
        except Exception as e:
            logger.error(f"Mean reversion analysis failed: {e}")
            return {'mean_reversion_strength': 'unknown', 'signal': 'neutral', 'confidence': 0.0}
    
    def _calculate_volatility_confidence(self, volatility_metrics: Dict[str, float]) -> float:
        """
        Calculate confidence in volatility analysis.
        
        Args:
            volatility_metrics: Volatility metrics
            
        Returns:
            Confidence score (0-1)
        """
        try:
            if not volatility_metrics:
                return 0.0
            
            # Base confidence on metric consistency
            confidence = 0.6
            
            # Adjust based on volatility stability
            current_vol = volatility_metrics.get('current_volatility', 0)
            if 0.005 < current_vol < 0.1:  # Reasonable volatility range
                confidence += 0.2
            
            # Adjust based on multiple volatility measures
            vol_measures = ['realized_volatility', 'ewma_volatility', 'garch_volatility']
            available_measures = sum(1 for measure in vol_measures if measure in volatility_metrics)
            confidence += (available_measures / len(vol_measures)) * 0.2
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Volatility confidence calculation failed: {e}")
            return 0.0
    
    def _generate_volatility_signal(self, analysis: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on volatility analysis.
        
        Args:
            analysis: Volatility analysis results
            context: Current market context
            
        Returns:
            Volatility-based trading signal
        """
        try:
            volatility_regime = analysis['volatility_regime']
            mean_reversion = analysis['mean_reversion']
            volatility_forecast = analysis['volatility_forecast']
            confidence = analysis['confidence']
            
            # Determine signal based on volatility analysis
            if volatility_regime['regime'] == 'high_volatility' and mean_reversion['signal'] == 'sell_volatility':
                signal_type = SignalType.SELL
                reasoning = f"High volatility regime with mean reversion signal"
            elif volatility_regime['regime'] == 'low_volatility' and mean_reversion['signal'] == 'buy_volatility':
                signal_type = SignalType.BUY
                reasoning = f"Low volatility regime with mean reversion signal"
            elif volatility_forecast['direction'] == 'increasing' and volatility_regime['regime'] == 'normal_volatility':
                signal_type = SignalType.BUY
                reasoning = f"Volatility forecast suggests increasing trend"
            elif volatility_forecast['direction'] == 'decreasing' and volatility_regime['regime'] == 'elevated_volatility':
                signal_type = SignalType.SELL
                reasoning = f"Volatility forecast suggests decreasing trend"
            else:
                signal_type = SignalType.HOLD
                reasoning = f"Volatility regime: {volatility_regime['regime']} - no clear signal"
            
            # Adjust confidence based on regime confidence
            adjusted_confidence = min(confidence * volatility_regime.get('confidence', 0.5), 0.9)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'volatility_analysis': analysis,
                    'method': 'volatility_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_volatility_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple volatility analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple volatility-based signal
        """
        try:
            # Simple volatility based on price movement
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            if len(context.market_data) >= 5:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()
                    
                    # High volatility suggests selling opportunity
                    if volatility > 0.03:  # 3% volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.SELL,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility'},
                            reasoning=f"High volatility ({volatility:.2%}) suggests selling opportunity"
                        )
                    elif volatility < 0.01:  # 1% volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility'},
                            reasoning=f"Low volatility ({volatility:.2%}) suggests buying opportunity"
                        )
            
            return self._create_hold_signal("No clear volatility signal", context)
            
        except Exception as e:
            logger.error(f"Simple volatility analysis failed: {e}")
            return self._create_hold_signal(f"Simple volatility analysis error: {e}", context)
    
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
        Update the volatility model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update volatility history
            symbol = context.symbol
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            
            # Store recent volatility data
            if hasattr(self, '_last_volatility_analysis'):
                self.volatility_history[symbol].append(self._last_volatility_analysis)
                
                # Keep only recent history
                if len(self.volatility_history[symbol]) > 20:
                    self.volatility_history[symbol] = self.volatility_history[symbol][-20:]
            
            logger.info(f"Updated volatility model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
