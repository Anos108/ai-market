"""
Enhanced Forecasting Service

This service creates intelligent forecasting by leveraging outputs from all specialized agents
to improve accuracy and robustness of predictions.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import asyncpg
import yfinance as yf
import ta
from dataclasses import dataclass
import json
import random
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class AgentInsight:
    """Represents an insight from a specialized agent."""
    agent_name: str
    signal_type: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class EnhancedForecast:
    """Enhanced forecast that combines multiple agent insights."""
    symbol: str
    horizon: str
    predicted_price: float
    confidence: float
    direction: str
    signal_strength: str
    market_regime: str
    
    # Agent insights used
    momentum_insight: Optional[AgentInsight]
    sentiment_insight: Optional[AgentInsight]
    volatility_insight: Optional[AgentInsight]
    risk_insight: Optional[AgentInsight]
    correlation_insight: Optional[AgentInsight]
    volume_insight: Optional[AgentInsight]
    
    # Technical analysis from forecasting agents
    technical_indicators: List[Dict[str, Any]]
    
    # Risk and confidence metrics
    risk_score: float
    ensemble_confidence: float
    
    # Metadata
    created_at: datetime
    valid_until: datetime

class EnhancedForecastingService:
    """Service that creates enhanced forecasts by leveraging all agent outputs."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        
        # Agent weights for ensemble forecasting
        self.agent_weights = {
            'MomentumAgent': 0.20,      # Trend analysis
            'SentimentAgent': 0.15,     # Market sentiment
            'VolatilityAgent': 0.15,    # Volatility patterns
            'RiskAgent': 0.15,          # Risk assessment
            'CorrelationAgent': 0.15,   # Market correlations
            'VolumeAgent': 0.10,        # Volume analysis
            'ForecastAgent': 0.10       # Technical forecasting
        }
        
        # Regime-specific agent adjustments
        self.regime_adjustments = {
            'bull': {
                'MomentumAgent': 1.2,   # Higher weight in bull markets
                'SentimentAgent': 1.1,
                'RiskAgent': 0.8        # Lower risk weight in bull markets
            },
            'bear': {
                'RiskAgent': 1.3,       # Higher risk weight in bear markets
                'VolatilityAgent': 1.2,
                'MomentumAgent': 0.7    # Lower momentum weight in bear markets
            },
            'volatile': {
                'VolatilityAgent': 1.4, # Higher volatility weight in volatile markets
                'RiskAgent': 1.2,
                'SentimentAgent': 0.8   # Lower sentiment weight in volatile markets
            },
            'neutral': {
                # Default weights
            }
        }
    
    async def generate_enhanced_day_forecast(self, symbol: str, horizon: str = "end_of_day") -> EnhancedForecast:
        """Generate enhanced day forecast using all agent insights."""
        try:
            # Get insights from all agents for this symbol
            agent_insights = await self._get_agent_insights(symbol)
            
            # Determine market regime based on agent insights
            market_regime = await self._determine_market_regime(agent_insights)
            
            # Get current market data
            current_price = await self._get_current_price(symbol)
            
            # Calculate ensemble prediction
            ensemble_prediction = self._calculate_ensemble_prediction(
                symbol, current_price, agent_insights, market_regime, "day"
            )
            
            # Generate technical indicators from forecasting agents
            technical_indicators = self._generate_technical_indicators(symbol, agent_insights)
            
            # Calculate risk and confidence metrics
            risk_score = self._calculate_risk_score(agent_insights)
            ensemble_confidence = self._calculate_ensemble_confidence(agent_insights, market_regime)
            
            # Determine signal strength and direction
            direction, signal_strength = self._determine_signal_strength(ensemble_confidence, ensemble_prediction, current_price)
            
            return EnhancedForecast(
                symbol=symbol,
                horizon=horizon,
                predicted_price=ensemble_prediction,
                confidence=ensemble_confidence,
                direction=direction,
                signal_strength=signal_strength,
                market_regime=market_regime,
                momentum_insight=agent_insights.get('MomentumAgent'),
                sentiment_insight=agent_insights.get('SentimentAgent'),
                volatility_insight=agent_insights.get('VolatilityAgent'),
                risk_insight=agent_insights.get('RiskAgent'),
                correlation_insight=agent_insights.get('CorrelationAgent'),
                volume_insight=agent_insights.get('VolumeAgent'),
                technical_indicators=technical_indicators,
                risk_score=risk_score,
                ensemble_confidence=ensemble_confidence,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=1)
            )
            
        except Exception as e:
            logger.error(f"Error generating enhanced day forecast for {symbol}: {e}")
            return await self._get_fallback_forecast(symbol, horizon, "day")
    
    async def generate_enhanced_swing_forecast(self, symbol: str, horizon: str = "1_week") -> EnhancedForecast:
        """Generate enhanced swing forecast using all agent insights."""
        try:
            # Get insights from all agents for this symbol
            agent_insights = await self._get_agent_insights(symbol)
            
            # Determine market regime based on agent insights
            market_regime = await self._determine_market_regime(agent_insights)
            
            # Get current market data
            current_price = await self._get_current_price(symbol)
            
            # Calculate ensemble prediction (with swing-specific adjustments)
            ensemble_prediction = self._calculate_ensemble_prediction(
                symbol, current_price, agent_insights, market_regime, "swing"
            )
            
            # Generate technical indicators from forecasting agents
            technical_indicators = self._generate_technical_indicators(symbol, agent_insights)
            
            # Calculate risk and confidence metrics
            risk_score = self._calculate_risk_score(agent_insights)
            ensemble_confidence = self._calculate_ensemble_confidence(agent_insights, market_regime)
            
            # Determine signal strength and direction
            direction, signal_strength = self._determine_signal_strength(ensemble_confidence, ensemble_prediction, current_price)
            
            return EnhancedForecast(
                symbol=symbol,
                horizon=horizon,
                predicted_price=ensemble_prediction,
                confidence=ensemble_confidence,
                direction=direction,
                signal_strength=signal_strength,
                market_regime=market_regime,
                momentum_insight=agent_insights.get('MomentumAgent'),
                sentiment_insight=agent_insights.get('SentimentAgent'),
                volatility_insight=agent_insights.get('VolatilityAgent'),
                risk_insight=agent_insights.get('RiskAgent'),
                correlation_insight=agent_insights.get('CorrelationAgent'),
                volume_insight=agent_insights.get('VolumeAgent'),
                technical_indicators=technical_indicators,
                risk_score=risk_score,
                ensemble_confidence=ensemble_confidence,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(days=7)
            )
            
        except Exception as e:
            logger.error(f"Error generating enhanced swing forecast for {symbol}: {e}")
            return await self._get_fallback_forecast(symbol, horizon, "swing")
    
    async def _get_agent_insights(self, symbol: str) -> Dict[str, AgentInsight]:
        """Get insights from all agents for a specific symbol."""
        insights = {}
        
        try:
            async with self.db_pool.acquire() as conn:
                for agent_name in self.agent_weights.keys():
                    # Get the most recent insight from this agent
                    result = await conn.fetchrow("""
                        SELECT agent_name, symbol, signal_type, confidence, reasoning, metadata, timestamp
                        FROM agent_signals 
                        WHERE agent_name = $1 
                        AND symbol = $2
                        AND timestamp >= NOW() - INTERVAL '1 hour'
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """, agent_name, symbol)
                    
                    if result:
                        # Parse metadata if it's a string
                        metadata = result['metadata'] or {}
                        if isinstance(metadata, str):
                            try:
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                        
                        insights[agent_name] = AgentInsight(
                            agent_name=result['agent_name'],
                            signal_type=result['signal_type'],
                            confidence=float(result['confidence']),
                            reasoning=result['reasoning'],
                            metadata=metadata,
                            timestamp=result['timestamp']
                        )
                        
        except Exception as e:
            logger.error(f"Error getting agent insights for {symbol}: {e}")
        
        return insights
    
    async def _determine_market_regime(self, agent_insights: Dict[str, AgentInsight]) -> str:
        """Determine market regime based on agent insights (simplified for now)."""
        try:
            # Simple agent-based regime detection
            regime_scores = {'bull': 0, 'bear': 0, 'volatile': 0, 'neutral': 0}
            
            for agent_name, insight in agent_insights.items():
                weight = self.agent_weights.get(agent_name, 0)
                
                if insight.signal_type == 'buy':
                    regime_scores['bull'] += weight * insight.confidence
                elif insight.signal_type == 'sell':
                    regime_scores['bear'] += weight * insight.confidence
                
                # Check volatility indicators
                if agent_name == 'VolatilityAgent':
                    volatility = insight.metadata.get('volatility_ratio', 1.0)
                    if volatility > 1.3:
                        regime_scores['volatile'] += weight * insight.confidence
                
                # Check risk indicators
                if agent_name == 'RiskAgent':
                    risk_score = 1.0 - insight.confidence
                    if risk_score > 0.7:
                        regime_scores['volatile'] += weight * insight.confidence
            
            # Determine regime with highest score
            if max(regime_scores.values()) > 0:
                return max(regime_scores, key=regime_scores.get)
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return 'neutral'
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            return 100.0  # Fallback price
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 100.0
    
    def _calculate_ensemble_prediction(self, symbol: str, current_price: float, 
                                     agent_insights: Dict[str, AgentInsight], 
                                     market_regime: str, forecast_type: str) -> float:
        """Calculate ensemble price prediction using weighted agent insights."""
        try:
            total_weight = 0
            weighted_prediction = 0
            
            # Get regime-specific weight adjustments
            regime_weights = self.regime_adjustments.get(market_regime, {})
            
            for agent_name, insight in agent_insights.items():
                base_weight = self.agent_weights.get(agent_name, 0)
                regime_multiplier = regime_weights.get(agent_name, 1.0)
                adjusted_weight = base_weight * regime_multiplier
                
                # Calculate price change based on agent signal and confidence
                if insight.signal_type == 'buy':
                    price_change = 0.02 * insight.confidence  # 2% max change for buy
                elif insight.signal_type == 'sell':
                    price_change = -0.02 * insight.confidence  # -2% max change for sell
                else:
                    price_change = 0  # Hold signal
                
                # Adjust for forecast type
                if forecast_type == "swing":
                    price_change *= 2  # Larger changes for swing trading
                
                # Apply agent-specific adjustments
                if agent_name == 'VolatilityAgent':
                    volatility = insight.metadata.get('volatility_ratio', 1.0)
                    price_change *= volatility  # Adjust for volatility
                
                if agent_name == 'RiskAgent':
                    risk_score = 1.0 - insight.confidence
                    price_change *= (1 - risk_score * 0.5)  # Reduce change based on risk
                
                predicted_price = current_price * (1 + price_change)
                
                weighted_prediction += predicted_price * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                return weighted_prediction / total_weight
            else:
                return current_price
                
        except Exception as e:
            logger.error(f"Error calculating ensemble prediction: {e}")
            return current_price
    
    def _generate_technical_indicators(self, symbol: str, agent_insights: Dict[str, AgentInsight]) -> List[Dict[str, Any]]:
        """Generate technical indicators from forecasting agents."""
        indicators = []
        
        try:
            # Get technical indicators from momentum agent
            if 'MomentumAgent' in agent_insights:
                momentum = agent_insights['MomentumAgent']
                indicators.append({
                    "name": "RSI",
                    "value": momentum.metadata.get('rsi', 50.0),
                    "signal": "buy" if momentum.metadata.get('rsi', 50.0) < 30 else "sell" if momentum.metadata.get('rsi', 50.0) > 70 else "hold",
                    "strength": abs(momentum.metadata.get('rsi', 50.0) - 50) / 50,
                    "timestamp": momentum.timestamp.isoformat()
                })
                
                indicators.append({
                    "name": "Price Momentum",
                    "value": momentum.metadata.get('price_change_5d', 0.0) * 100,
                    "signal": momentum.signal_type,
                    "strength": momentum.confidence,
                    "timestamp": momentum.timestamp.isoformat()
                })
            
            # Get volatility indicators
            if 'VolatilityAgent' in agent_insights:
                volatility = agent_insights['VolatilityAgent']
                indicators.append({
                    "name": "Volatility Ratio",
                    "value": volatility.metadata.get('volatility_ratio', 1.0),
                    "signal": "sell" if volatility.metadata.get('volatility_ratio', 1.0) > 1.5 else "buy" if volatility.metadata.get('volatility_ratio', 1.0) < 0.7 else "hold",
                    "strength": abs(volatility.metadata.get('volatility_ratio', 1.0) - 1.0),
                    "timestamp": volatility.timestamp.isoformat()
                })
            
            # Get volume indicators
            if 'VolumeAgent' in agent_insights:
                volume = agent_insights['VolumeAgent']
                indicators.append({
                    "name": "Volume Ratio",
                    "value": volume.metadata.get('volume_ratio', 1.0),
                    "signal": volume.signal_type,
                    "strength": volume.confidence,
                    "timestamp": volume.timestamp.isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error generating technical indicators: {e}")
        
        return indicators
    
    def _calculate_risk_score(self, agent_insights: Dict[str, AgentInsight]) -> float:
        """Calculate overall risk score from agent insights."""
        try:
            risk_scores = []
            weights = []
            
            for agent_name, insight in agent_insights.items():
                weight = self.agent_weights.get(agent_name, 0)
                
                if agent_name == 'RiskAgent':
                    # Direct risk score from risk agent
                    risk_score = 1.0 - insight.confidence
                elif agent_name == 'VolatilityAgent':
                    # Volatility-based risk
                    volatility = insight.metadata.get('volatility_ratio', 1.0)
                    risk_score = min(1.0, (volatility - 1.0) * 2)
                else:
                    # Inverse of confidence as risk proxy
                    risk_score = 1.0 - insight.confidence
                
                risk_scores.append(risk_score)
                weights.append(weight)
            
            if weights and sum(weights) > 0:
                return np.average(risk_scores, weights=weights)
            else:
                return 0.5  # Default risk score
                
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def _calculate_ensemble_confidence(self, agent_insights: Dict[str, AgentInsight], market_regime: str) -> float:
        """Calculate ensemble confidence from agent insights."""
        try:
            confidences = []
            weights = []
            
            for agent_name, insight in agent_insights.items():
                base_weight = self.agent_weights.get(agent_name, 0)
                regime_multiplier = self.regime_adjustments.get(market_regime, {}).get(agent_name, 1.0)
                adjusted_weight = base_weight * regime_multiplier
                
                confidences.append(insight.confidence)
                weights.append(adjusted_weight)
            
            if weights and sum(weights) > 0:
                base_confidence = np.average(confidences, weights=weights)
                
                # Boost confidence if agents agree on direction
                directions = [insight.signal_type for insight in agent_insights.values()]
                if len(set(directions)) == 1:  # All agents agree
                    base_confidence *= 1.2
                
                return min(0.95, base_confidence)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating ensemble confidence: {e}")
            return 0.5
    
    def _determine_signal_strength(self, confidence: float, predicted_price: float, current_price: float) -> Tuple[str, str]:
        """Determine signal direction and strength."""
        price_change = (predicted_price - current_price) / current_price
        
        if price_change > 0.02:
            direction = 'buy'
            strength = 'strong' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'weak'
        elif price_change < -0.02:
            direction = 'sell'
            strength = 'strong' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'weak'
        else:
            direction = 'hold'
            strength = 'moderate' if confidence > 0.7 else 'weak'
        
        return direction, strength
    
    async def _get_fallback_forecast(self, symbol: str, horizon: str, forecast_type: str) -> EnhancedForecast:
        """Get fallback forecast when agent insights are not available."""
        current_price = await self._get_current_price(symbol)
        
        return EnhancedForecast(
            symbol=symbol,
            horizon=horizon,
            predicted_price=current_price * 1.01,  # 1% increase as fallback
            confidence=0.5,
            direction='hold',
            signal_strength='weak',
            market_regime='neutral',
            momentum_insight=None,
            sentiment_insight=None,
            volatility_insight=None,
            risk_insight=None,
            correlation_insight=None,
            volume_insight=None,
            technical_indicators=[],
            risk_score=0.5,
            ensemble_confidence=0.5,
            created_at=datetime.now(),
            valid_until=datetime.now() + (timedelta(hours=1) if forecast_type == "day" else timedelta(days=7))
        )
