"""
Real Market Regime Detection Service

This service provides real market regime detection using actual market data,
volatility analysis, correlation patterns, and economic indicators.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import ta

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"
    NEUTRAL = "neutral"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"

@dataclass
class RegimeMetrics:
    """Market regime metrics."""
    volatility: float
    trend_strength: float
    correlation_clustering: float
    volume_profile: float
    momentum_score: float
    risk_level: float
    market_breadth: float

@dataclass
class MarketRegimeAnalysis:
    """Comprehensive market regime analysis."""
    current_regime: MarketRegime
    regime_confidence: float
    regime_duration_days: int
    regime_metrics: RegimeMetrics
    regime_probabilities: Dict[MarketRegime, float]
    regime_transition_probability: float
    recommended_strategy: str
    risk_assessment: str
    analysis_timestamp: datetime

class RealMarketRegimeService:
    """Real market regime detection service using actual market data."""
    
    def __init__(self):
        self.lookback_days = 252  # 1 year of trading days
        self.regime_thresholds = {
            'volatility': {'low': 0.15, 'high': 0.35},
            'trend_strength': {'weak': 0.3, 'strong': 0.7},
            'correlation': {'low': 0.3, 'high': 0.7},
            'momentum': {'negative': -0.2, 'positive': 0.2}
        }
        
        # Market indices for regime detection
        self.market_indices = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'IWM': 'Russell 2000',
            'DIA': 'Dow Jones',
            'VIX': 'Volatility Index'
        }
        
        # Regime characteristics
        self.regime_characteristics = {
            MarketRegime.BULL: {
                'volatility_range': (0.1, 0.25),
                'trend_strength_min': 0.6,
                'momentum_min': 0.1,
                'correlation_range': (0.4, 0.8)
            },
            MarketRegime.BEAR: {
                'volatility_range': (0.2, 0.5),
                'trend_strength_min': 0.6,
                'momentum_max': -0.1,
                'correlation_range': (0.5, 0.9)
            },
            MarketRegime.VOLATILE: {
                'volatility_range': (0.3, 1.0),
                'trend_strength_max': 0.4,
                'momentum_range': (-0.3, 0.3),
                'correlation_range': (0.6, 1.0)
            },
            MarketRegime.NEUTRAL: {
                'volatility_range': (0.15, 0.3),
                'trend_strength_max': 0.5,
                'momentum_range': (-0.2, 0.2),
                'correlation_range': (0.3, 0.7)
            }
        }
    
    async def analyze_market_regime(self, symbol: str = None) -> MarketRegimeAnalysis:
        """Analyze current market regime."""
        try:
            logger.info(f"📊 Analyzing market regime")
            
            # Get market data for regime detection
            market_data = await self._fetch_market_data()
            
            # Calculate regime metrics
            regime_metrics = await self._calculate_regime_metrics(market_data)
            
            # Determine current regime
            current_regime = self._determine_current_regime(regime_metrics)
            
            # Calculate regime confidence
            regime_confidence = self._calculate_regime_confidence(regime_metrics, current_regime)
            
            # Calculate regime probabilities
            regime_probabilities = self._calculate_regime_probabilities(regime_metrics)
            
            # Estimate regime duration
            regime_duration = await self._estimate_regime_duration(market_data, current_regime)
            
            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(market_data, current_regime)
            
            # Generate recommendations
            recommendations = self._generate_regime_recommendations(current_regime, regime_metrics)
            
            return MarketRegimeAnalysis(
                current_regime=current_regime,
                regime_confidence=regime_confidence,
                regime_duration_days=regime_duration,
                regime_metrics=regime_metrics,
                regime_probabilities=regime_probabilities,
                regime_transition_probability=transition_prob,
                recommended_strategy=recommendations['strategy'],
                risk_assessment=recommendations['risk'],
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return self._get_fallback_analysis()
    
    async def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data for regime detection."""
        market_data = {}
        
        for symbol, name in self.market_indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                
                if len(hist) > 50:
                    # Calculate additional indicators
                    hist['Returns'] = hist['Close'].pct_change()
                    hist['Volatility'] = hist['Returns'].rolling(window=20).std() * np.sqrt(252)
                    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                    
                    # Technical indicators
                    hist['RSI'] = ta.momentum.RSIIndicator(hist['Close']).rsi()
                    hist['MACD'] = ta.trend.MACD(hist['Close']).macd()
                    hist['BB_Upper'] = ta.volatility.BollingerBands(hist['Close']).bollinger_hband()
                    hist['BB_Lower'] = ta.volatility.BollingerBands(hist['Close']).bollinger_lband()
                    
                    market_data[symbol] = hist
                    
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return market_data
    
    async def _calculate_regime_metrics(self, market_data: Dict[str, pd.DataFrame]) -> RegimeMetrics:
        """Calculate comprehensive regime metrics."""
        if not market_data:
            return self._get_default_metrics()
        
        # Calculate volatility (using SPY as primary indicator)
        spy_data = market_data.get('SPY')
        if spy_data is not None and len(spy_data) > 20:
            volatility = spy_data['Volatility'].iloc[-1]
        else:
            volatility = 0.2  # Default volatility
        
        # Calculate trend strength
        trend_strength = await self._calculate_trend_strength(market_data)
        
        # Calculate correlation clustering
        correlation_clustering = await self._calculate_correlation_clustering(market_data)
        
        # Calculate volume profile
        volume_profile = await self._calculate_volume_profile(market_data)
        
        # Calculate momentum score
        momentum_score = await self._calculate_momentum_score(market_data)
        
        # Calculate risk level
        risk_level = await self._calculate_risk_level(market_data)
        
        # Calculate market breadth
        market_breadth = await self._calculate_market_breadth(market_data)
        
        return RegimeMetrics(
            volatility=volatility,
            trend_strength=trend_strength,
            correlation_clustering=correlation_clustering,
            volume_profile=volume_profile,
            momentum_score=momentum_score,
            risk_level=risk_level,
            market_breadth=market_breadth
        )
    
    async def _calculate_trend_strength(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate overall trend strength across indices."""
        trend_scores = []
        
        for symbol, data in market_data.items():
            if len(data) < 50:
                continue
            
            try:
                # Calculate trend strength using multiple methods
                sma_20 = data['SMA_20'].iloc[-1]
                sma_50 = data['SMA_50'].iloc[-1]
                current_price = data['Close'].iloc[-1]
                
                # Price above/below moving averages
                price_above_sma20 = 1 if current_price > sma_20 else 0
                price_above_sma50 = 1 if current_price > sma_50 else 0
                
                # Trend consistency over time
                recent_prices = data['Close'].tail(20)
                trend_consistency = stats.linregress(range(len(recent_prices)), recent_prices)[0]
                trend_consistency = 1 if trend_consistency > 0 else 0
                
                # Combined trend score
                trend_score = (price_above_sma20 + price_above_sma50 + trend_consistency) / 3
                trend_scores.append(trend_score)
                
            except Exception as e:
                logger.warning(f"Error calculating trend strength for {symbol}: {e}")
                continue
        
        return np.mean(trend_scores) if trend_scores else 0.5
    
    async def _calculate_correlation_clustering(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate correlation clustering between indices."""
        if len(market_data) < 2:
            return 0.5
        
        try:
            # Get returns for correlation analysis
            returns_data = {}
            for symbol, data in market_data.items():
                if len(data) > 20:
                    returns_data[symbol] = data['Returns'].tail(60).dropna()
            
            if len(returns_data) < 2:
                return 0.5
            
            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Calculate average correlation (excluding diagonal)
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    correlations.append(correlation_matrix.iloc[i, j])
            
            return np.mean(correlations) if correlations else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating correlation clustering: {e}")
            return 0.5
    
    async def _calculate_volume_profile(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate volume profile across indices."""
        volume_scores = []
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                continue
            
            try:
                # Compare recent volume to average
                recent_volume = data['Volume'].tail(5).mean()
                avg_volume = data['Volume'].tail(20).mean()
                
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    # Normalize to 0-1 range
                    volume_score = min(1.0, volume_ratio / 2.0)
                    volume_scores.append(volume_score)
                
            except Exception as e:
                logger.warning(f"Error calculating volume profile for {symbol}: {e}")
                continue
        
        return np.mean(volume_scores) if volume_scores else 0.5
    
    async def _calculate_momentum_score(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate overall momentum score."""
        momentum_scores = []
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                continue
            
            try:
                # Calculate multiple momentum indicators
                rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
                macd = data['MACD'].iloc[-1] if 'MACD' in data.columns else 0
                
                # RSI momentum (-1 to 1)
                rsi_momentum = (rsi - 50) / 50
                
                # MACD momentum (normalized)
                macd_momentum = np.tanh(macd) if not np.isnan(macd) else 0
                
                # Price momentum (20-day return)
                price_momentum = data['Close'].pct_change(20).iloc[-1] if len(data) >= 20 else 0
                price_momentum = np.tanh(price_momentum * 10) if not np.isnan(price_momentum) else 0
                
                # Combined momentum score
                combined_momentum = (rsi_momentum + macd_momentum + price_momentum) / 3
                momentum_scores.append(combined_momentum)
                
            except Exception as e:
                logger.warning(f"Error calculating momentum for {symbol}: {e}")
                continue
        
        return np.mean(momentum_scores) if momentum_scores else 0.0
    
    async def _calculate_risk_level(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate overall market risk level."""
        risk_factors = []
        
        # VIX as primary risk indicator
        vix_data = market_data.get('VIX')
        if vix_data is not None and len(vix_data) > 0:
            vix_level = vix_data['Close'].iloc[-1]
            # Normalize VIX to 0-1 range (10-50 VIX range)
            vix_risk = min(1.0, max(0.0, (vix_level - 10) / 40))
            risk_factors.append(vix_risk)
        
        # Volatility across indices
        volatilities = []
        for symbol, data in market_data.items():
            if symbol != 'VIX' and len(data) > 20:
                vol = data['Volatility'].iloc[-1]
                if not np.isnan(vol):
                    volatilities.append(vol)
        
        if volatilities:
            avg_volatility = np.mean(volatilities)
            vol_risk = min(1.0, max(0.0, (avg_volatility - 0.15) / 0.35))
            risk_factors.append(vol_risk)
        
        # Correlation clustering (high correlation = higher risk)
        correlation_risk = await self._calculate_correlation_clustering(market_data)
        risk_factors.append(correlation_risk)
        
        return np.mean(risk_factors) if risk_factors else 0.5
    
    async def _calculate_market_breadth(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate market breadth (how many indices are trending in same direction)."""
        if len(market_data) < 2:
            return 0.5
        
        breadth_scores = []
        
        for symbol, data in market_data.items():
            if symbol == 'VIX' or len(data) < 20:
                continue
            
            try:
                # Calculate if index is above/below moving averages
                current_price = data['Close'].iloc[-1]
                sma_20 = data['SMA_20'].iloc[-1]
                
                # Positive breadth if above SMA
                breadth_score = 1 if current_price > sma_20 else 0
                breadth_scores.append(breadth_score)
                
            except Exception as e:
                logger.warning(f"Error calculating breadth for {symbol}: {e}")
                continue
        
        return np.mean(breadth_scores) if breadth_scores else 0.5
    
    def _determine_current_regime(self, metrics: RegimeMetrics) -> MarketRegime:
        """Determine current market regime based on metrics."""
        # Bull market conditions
        if (metrics.volatility < 0.25 and 
            metrics.trend_strength > 0.6 and 
            metrics.momentum_score > 0.2 and
            metrics.market_breadth > 0.6):
            return MarketRegime.BULL
        
        # Bear market conditions
        elif (metrics.volatility > 0.2 and 
              metrics.trend_strength > 0.6 and 
              metrics.momentum_score < -0.2 and
              metrics.market_breadth < 0.4):
            return MarketRegime.BEAR
        
        # Volatile market conditions
        elif (metrics.volatility > 0.3 and 
              metrics.correlation_clustering > 0.7 and
              metrics.risk_level > 0.6):
            return MarketRegime.VOLATILE
        
        # Trending markets
        elif metrics.trend_strength > 0.7:
            if metrics.momentum_score > 0.1:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # Sideways market
        elif (metrics.trend_strength < 0.4 and 
              metrics.volatility < 0.3):
            return MarketRegime.SIDEWAYS
        
        # Default to neutral
        else:
            return MarketRegime.NEUTRAL
    
    def _calculate_regime_confidence(self, metrics: RegimeMetrics, regime: MarketRegime) -> float:
        """Calculate confidence in regime classification."""
        characteristics = self.regime_characteristics.get(regime, {})
        
        if not characteristics:
            return 0.5
        
        confidence_factors = []
        
        # Volatility confidence
        vol_range = characteristics.get('volatility_range', (0.1, 0.4))
        if vol_range[0] <= metrics.volatility <= vol_range[1]:
            confidence_factors.append(1.0)
        else:
            distance = min(abs(metrics.volatility - vol_range[0]), 
                          abs(metrics.volatility - vol_range[1]))
            confidence_factors.append(max(0.0, 1.0 - distance))
        
        # Trend strength confidence
        trend_min = characteristics.get('trend_strength_min', 0.0)
        if metrics.trend_strength >= trend_min:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(metrics.trend_strength / trend_min)
        
        # Momentum confidence
        momentum_min = characteristics.get('momentum_min', -1.0)
        momentum_max = characteristics.get('momentum_max', 1.0)
        if momentum_min <= metrics.momentum_score <= momentum_max:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.5)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_regime_probabilities(self, metrics: RegimeMetrics) -> Dict[MarketRegime, float]:
        """Calculate probabilities for each regime."""
        probabilities = {}
        
        for regime in MarketRegime:
            characteristics = self.regime_characteristics.get(regime, {})
            probability = 0.5  # Base probability
            
            # Adjust based on metrics alignment
            if characteristics:
                # Volatility alignment
                vol_range = characteristics.get('volatility_range', (0.1, 0.4))
                if vol_range[0] <= metrics.volatility <= vol_range[1]:
                    probability += 0.2
                
                # Trend alignment
                trend_min = characteristics.get('trend_strength_min', 0.0)
                if metrics.trend_strength >= trend_min:
                    probability += 0.2
                
                # Momentum alignment
                momentum_min = characteristics.get('momentum_min', -1.0)
                momentum_max = characteristics.get('momentum_max', 1.0)
                if momentum_min <= metrics.momentum_score <= momentum_max:
                    probability += 0.1
            
            probabilities[regime] = min(1.0, probability)
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    async def _estimate_regime_duration(self, market_data: Dict[str, pd.DataFrame], 
                                       current_regime: MarketRegime) -> int:
        """Estimate how long current regime has been active."""
        # This is a simplified estimation
        # In practice, you'd analyze regime transitions over historical data
        
        spy_data = market_data.get('SPY')
        if spy_data is None or len(spy_data) < 50:
            return 30  # Default estimate
        
        try:
            # Analyze trend consistency over time
            recent_data = spy_data.tail(50)
            trend_changes = 0
            
            for i in range(1, len(recent_data)):
                prev_trend = 1 if recent_data['Close'].iloc[i-1] > recent_data['SMA_20'].iloc[i-1] else 0
                curr_trend = 1 if recent_data['Close'].iloc[i] > recent_data['SMA_20'].iloc[i] else 0
                
                if prev_trend != curr_trend:
                    trend_changes += 1
            
            # Estimate duration based on trend stability
            if trend_changes < 5:  # Stable trend
                return 60  # ~3 months
            elif trend_changes < 15:  # Moderate stability
                return 30  # ~1.5 months
            else:  # Unstable
                return 15  # ~3 weeks
                
        except Exception as e:
            logger.warning(f"Error estimating regime duration: {e}")
            return 30
    
    def _calculate_transition_probability(self, market_data: Dict[str, pd.DataFrame], 
                                        current_regime: MarketRegime) -> float:
        """Calculate probability of regime transition."""
        # Simplified transition probability calculation
        # In practice, you'd use historical transition patterns
        
        spy_data = market_data.get('SPY')
        if spy_data is None or len(spy_data) < 20:
            return 0.2  # Default transition probability
        
        try:
            # Analyze recent volatility and trend changes
            recent_vol = spy_data['Volatility'].tail(5).mean()
            avg_vol = spy_data['Volatility'].tail(20).mean()
            
            vol_change = abs(recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0
            
            # Higher volatility change = higher transition probability
            transition_prob = min(0.8, 0.2 + vol_change * 2)
            
            return transition_prob
            
        except Exception as e:
            logger.warning(f"Error calculating transition probability: {e}")
            return 0.2
    
    def _generate_regime_recommendations(self, regime: MarketRegime, 
                                       metrics: RegimeMetrics) -> Dict[str, str]:
        """Generate strategy and risk recommendations based on regime."""
        recommendations = {
            'strategy': '',
            'risk': ''
        }
        
        if regime == MarketRegime.BULL:
            recommendations['strategy'] = 'Momentum and growth strategies favored'
            recommendations['risk'] = 'Moderate risk - focus on trend following'
            
        elif regime == MarketRegime.BEAR:
            recommendations['strategy'] = 'Defensive and value strategies recommended'
            recommendations['risk'] = 'High risk - consider hedging and position sizing'
            
        elif regime == MarketRegime.VOLATILE:
            recommendations['strategy'] = 'Short-term trading and volatility strategies'
            recommendations['risk'] = 'Very high risk - use strict risk management'
            
        elif regime == MarketRegime.TRENDING_UP:
            recommendations['strategy'] = 'Trend following and breakout strategies'
            recommendations['risk'] = 'Moderate risk - monitor trend strength'
            
        elif regime == MarketRegime.TRENDING_DOWN:
            recommendations['strategy'] = 'Short selling and mean reversion strategies'
            recommendations['risk'] = 'High risk - consider defensive positioning'
            
        elif regime == MarketRegime.SIDEWAYS:
            recommendations['strategy'] = 'Range trading and mean reversion strategies'
            recommendations['risk'] = 'Low risk - range-bound market'
            
        else:  # NEUTRAL
            recommendations['strategy'] = 'Balanced approach with diversification'
            recommendations['risk'] = 'Moderate risk - monitor for regime changes'
        
        return recommendations
    
    def _get_default_metrics(self) -> RegimeMetrics:
        """Get default metrics when data is unavailable."""
        return RegimeMetrics(
            volatility=0.2,
            trend_strength=0.5,
            correlation_clustering=0.5,
            volume_profile=0.5,
            momentum_score=0.0,
            risk_level=0.5,
            market_breadth=0.5
        )
    
    def _get_fallback_analysis(self) -> MarketRegimeAnalysis:
        """Get fallback analysis when real analysis fails."""
        return MarketRegimeAnalysis(
            current_regime=MarketRegime.NEUTRAL,
            regime_confidence=0.5,
            regime_duration_days=30,
            regime_metrics=self._get_default_metrics(),
            regime_probabilities={regime: 0.125 for regime in MarketRegime},
            regime_transition_probability=0.2,
            recommended_strategy='Monitor market conditions',
            risk_assessment='Moderate risk',
            analysis_timestamp=datetime.now()
        )
