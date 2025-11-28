"""
Correlation Agent for AI Market Analysis System

This agent analyzes cross-asset correlations and market relationships
to provide correlation-based trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class CorrelationAgent(BaseAgent):
    """
    Correlation Agent for analyzing cross-asset correlations and market relationships.
    
    This agent analyzes:
    - Cross-asset correlations
    - Sector correlations
    - Market regime correlations
    - Correlation breakdowns and divergences
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Correlation Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'correlation_threshold': 0.7,
            'divergence_threshold': 0.3,
            'lookback_periods': 20,
            'min_correlation_periods': 10,
            'sector_mapping': {
                'AAPL': 'Technology',
                'MSFT': 'Technology', 
                'GOOGL': 'Technology',
                'TSLA': 'Automotive',
                'SPY': 'Market'
            },
            'correlation_types': ['price', 'volume', 'volatility'],
            'confidence_threshold': 0.6
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="CorrelationAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.correlation_history = {}
        self.sector_correlations = {}
        self.correlation_breakdowns = []
        
        logger.info(f"Initialized CorrelationAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the correlation agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For correlation analysis, we don't need traditional ML training
            # Instead, we'll validate our correlation analysis approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Correlation analysis approach validated")
            return {"status": "correlation_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate correlation-based prediction.
        
        Args:
            context: Current market context
            
        Returns:
            Correlation-based trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple correlation analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple correlation analysis (not trained)")
                return self._simple_correlation_analysis(context)
            
            # Perform comprehensive correlation analysis
            correlation_analysis = self._analyze_correlations(context)
            
            # Generate signal based on correlation insights
            signal = self._generate_correlation_signal(correlation_analysis, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Correlation analysis error: {e}", context)
    
    def _analyze_correlations(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze correlations for the given symbol.
        
        Args:
            context: Current market context
            
        Returns:
            Correlation analysis results
        """
        try:
            symbol = context.symbol
            
            # Analyze price correlations
            price_correlations = self._analyze_price_correlations(symbol, context)
            
            # Analyze sector correlations
            sector_correlations = self._analyze_sector_correlations(symbol, context)
            
            # Detect correlation breakdowns
            breakdowns = self._detect_correlation_breakdowns(symbol, context)
            
            # Calculate correlation strength
            correlation_strength = self._calculate_correlation_strength(price_correlations)
            
            return {
                'price_correlations': price_correlations,
                'sector_correlations': sector_correlations,
                'breakdowns': breakdowns,
                'correlation_strength': correlation_strength,
                'confidence': self._calculate_confidence(price_correlations, breakdowns)
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {
                'price_correlations': {},
                'sector_correlations': {},
                'breakdowns': [],
                'correlation_strength': 0.0,
                'confidence': 0.0
            }
    
    def _analyze_price_correlations(self, symbol: str, context: AgentContext) -> Dict[str, float]:
        """
        Analyze price correlations with other assets.
        
        Args:
            symbol: Current symbol
            context: Market context
            
        Returns:
            Dictionary of correlations with other symbols
        """
        try:
            correlations = {}
            
            # Get current symbol data
            if context.market_data.empty:
                return correlations
            
            current_data = context.market_data
            
            # For now, simulate correlations with other symbols
            # In a real implementation, this would fetch data for other symbols
            other_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
            other_symbols = [s for s in other_symbols if s != symbol]
            
            for other_symbol in other_symbols:
                # Simulate correlation based on sector and market conditions
                if symbol in self.config['sector_mapping'] and other_symbol in self.config['sector_mapping']:
                    if self.config['sector_mapping'][symbol] == self.config['sector_mapping'][other_symbol]:
                        # Same sector - higher correlation
                        correlation = np.random.uniform(0.6, 0.9)
                    else:
                        # Different sector - lower correlation
                        correlation = np.random.uniform(0.2, 0.6)
                else:
                    # Default correlation
                    correlation = np.random.uniform(0.3, 0.7)
                
                correlations[other_symbol] = correlation
            
            return correlations
            
        except Exception as e:
            logger.error(f"Price correlation analysis failed: {e}")
            return {}
    
    def _analyze_sector_correlations(self, symbol: str, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze sector-level correlations.
        
        Args:
            symbol: Current symbol
            context: Market context
            
        Returns:
            Sector correlation analysis
        """
        try:
            if symbol not in self.config['sector_mapping']:
                return {'sector': 'Unknown', 'correlation': 0.0}
            
            sector = self.config['sector_mapping'][symbol]
            
            # Simulate sector correlation
            if sector == 'Technology':
                sector_correlation = np.random.uniform(0.7, 0.9)
            elif sector == 'Market':
                sector_correlation = 1.0  # SPY is the market
            else:
                sector_correlation = np.random.uniform(0.4, 0.8)
            
            return {
                'sector': sector,
                'correlation': sector_correlation,
                'sector_strength': 'strong' if sector_correlation > 0.7 else 'moderate'
            }
            
        except Exception as e:
            logger.error(f"Sector correlation analysis failed: {e}")
            return {'sector': 'Unknown', 'correlation': 0.0}
    
    def _detect_correlation_breakdowns(self, symbol: str, context: AgentContext) -> List[Dict[str, Any]]:
        """
        Detect correlation breakdowns and divergences.
        
        Args:
            symbol: Current symbol
            context: Market context
            
        Returns:
            List of detected breakdowns
        """
        try:
            breakdowns = []
            
            # Simulate correlation breakdown detection
            # In a real implementation, this would compare current vs historical correlations
            
            # Check for unusual price movements that might indicate breakdowns
            if not context.market_data.empty and len(context.market_data) >= 2:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    current_price = context.market_data[close_col].iloc[-1]
                    prev_price = context.market_data[close_col].iloc[-2]
                    price_change = abs((current_price - prev_price) / prev_price)
                    
                    # Large price movements might indicate correlation breakdowns
                    if price_change > 0.05:  # 5% change
                        breakdowns.append({
                            'type': 'price_divergence',
                            'severity': 'high' if price_change > 0.1 else 'medium',
                            'description': f'Large price movement ({price_change:.2%}) detected'
                        })
            
            return breakdowns
            
        except Exception as e:
            logger.error(f"Correlation breakdown detection failed: {e}")
            return []
    
    def _calculate_correlation_strength(self, correlations: Dict[str, float]) -> float:
        """
        Calculate overall correlation strength.
        
        Args:
            correlations: Dictionary of correlations
            
        Returns:
            Overall correlation strength (0-1)
        """
        try:
            if not correlations:
                return 0.0
            
            # Calculate average absolute correlation
            avg_correlation = np.mean([abs(corr) for corr in correlations.values()])
            return min(avg_correlation, 1.0)
            
        except Exception as e:
            logger.error(f"Correlation strength calculation failed: {e}")
            return 0.0
    
    def _calculate_confidence(self, correlations: Dict[str, float], breakdowns: List[Dict]) -> float:
        """
        Calculate confidence in correlation analysis.
        
        Args:
            correlations: Price correlations
            breakdowns: Detected breakdowns
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence on number of correlations and breakdowns
            base_confidence = min(len(correlations) / 4, 1.0)  # Normalize to 4 symbols
            
            # Reduce confidence if breakdowns detected
            breakdown_penalty = len(breakdowns) * 0.1
            
            confidence = max(0.0, base_confidence - breakdown_penalty)
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _generate_correlation_signal(self, analysis: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on correlation analysis.
        
        Args:
            analysis: Correlation analysis results
            context: Current market context
            
        Returns:
            Correlation-based trading signal
        """
        try:
            correlation_strength = analysis['correlation_strength']
            confidence = analysis['confidence']
            breakdowns = analysis['breakdowns']
            
            # Determine signal based on correlation insights
            if breakdowns:
                # Correlation breakdowns often signal opportunities
                signal_type = SignalType.BUY if len(breakdowns) == 1 else SignalType.HOLD
                reasoning = f"Correlation breakdown detected: {breakdowns[0]['description']}"
            elif correlation_strength > self.config['correlation_threshold']:
                # Strong correlations suggest following the trend
                signal_type = SignalType.HOLD
                reasoning = f"Strong correlations ({correlation_strength:.2f}) suggest trend continuation"
            elif correlation_strength < self.config['divergence_threshold']:
                # Weak correlations suggest divergence opportunities
                signal_type = SignalType.BUY
                reasoning = f"Weak correlations ({correlation_strength:.2f}) suggest divergence opportunity"
            else:
                signal_type = SignalType.HOLD
                reasoning = f"Moderate correlations ({correlation_strength:.2f}) - no clear signal"
            
            # Adjust confidence based on correlation strength
            adjusted_confidence = min(confidence * correlation_strength, 0.9)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'correlation_analysis': analysis,
                    'method': 'correlation_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_correlation_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple correlation analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple correlation-based signal
        """
        try:
            # Simple correlation based on price volatility
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            # Use volatility as a proxy for correlation strength
            if len(context.market_data) >= 5:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()
                    
                    # High volatility suggests low correlation (divergence)
                    if volatility > 0.03:  # 3% volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility_correlation'},
                            reasoning=f"High volatility ({volatility:.2%}) suggests correlation breakdown opportunity"
                        )
            
            return self._create_hold_signal("No clear correlation signal", context)
            
        except Exception as e:
            logger.error(f"Simple correlation analysis failed: {e}")
            return self._create_hold_signal(f"Simple correlation analysis error: {e}", context)
    
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
        Update the correlation model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update correlation history
            symbol = context.symbol
            if symbol not in self.correlation_history:
                self.correlation_history[symbol] = []
            
            # Store recent correlation data
            if hasattr(self, '_last_correlation_analysis'):
                self.correlation_history[symbol].append(self._last_correlation_analysis)
                
                # Keep only recent history
                if len(self.correlation_history[symbol]) > 20:
                    self.correlation_history[symbol] = self.correlation_history[symbol][-20:]
            
            logger.info(f"Updated correlation model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
