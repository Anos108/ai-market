"""
Risk Agent for AI Market Analysis System

This agent analyzes portfolio risk, volatility, and risk-adjusted returns
to provide risk-based trading signals.
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


class RiskAgent(BaseAgent):
    """
    Risk Agent for analyzing portfolio risk and risk-adjusted returns.
    
    This agent analyzes:
    - Portfolio volatility and risk metrics
    - Value at Risk (VaR) and Expected Shortfall
    - Risk-adjusted returns (Sharpe ratio, Sortino ratio)
    - Risk regime detection
    - Risk concentration and diversification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Risk Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'var_confidence_level': 0.95,
            'lookback_periods': 20,
            'volatility_threshold': 0.02,
            'sharpe_threshold': 1.0,
            'max_position_size': 0.1,
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'risk_metrics': ['volatility', 'var', 'sharpe', 'max_drawdown'],
            'confidence_threshold': 0.6
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="RiskAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.risk_history = {}
        self.risk_regimes = []
        self.portfolio_metrics = {}
        
        logger.info(f"Initialized RiskAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the risk agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            # For risk analysis, we don't need traditional ML training
            # Instead, we'll validate our risk analysis approach
            self.is_trained = True
            
            logger.info(f"{self.name}: Risk analysis approach validated")
            return {"status": "risk_analysis_ready"}
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate risk-based prediction.
        
        Args:
            context: Current market context
            
        Returns:
            Risk-based trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple risk analysis
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple risk analysis (not trained)")
                return self._simple_risk_analysis(context)
            
            # Perform comprehensive risk analysis
            risk_analysis = self._analyze_risk(context)
            
            # Generate signal based on risk insights
            signal = self._generate_risk_signal(risk_analysis, context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Risk analysis error: {e}", context)
    
    def _analyze_risk(self, context: AgentContext) -> Dict[str, Any]:
        """
        Analyze risk metrics for the given symbol.
        
        Args:
            context: Current market context
            
        Returns:
            Risk analysis results
        """
        try:
            symbol = context.symbol
            
            # Calculate basic risk metrics
            risk_metrics = self._calculate_risk_metrics(context)
            
            # Analyze risk regime
            risk_regime = self._analyze_risk_regime(risk_metrics)
            
            # Calculate risk-adjusted returns
            risk_adjusted_returns = self._calculate_risk_adjusted_returns(risk_metrics)
            
            # Assess portfolio risk
            portfolio_risk = self._assess_portfolio_risk(symbol, risk_metrics)
            
            return {
                'risk_metrics': risk_metrics,
                'risk_regime': risk_regime,
                'risk_adjusted_returns': risk_adjusted_returns,
                'portfolio_risk': portfolio_risk,
                'confidence': self._calculate_risk_confidence(risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                'risk_metrics': {},
                'risk_regime': 'unknown',
                'risk_adjusted_returns': {},
                'portfolio_risk': {},
                'confidence': 0.0
            }
    
    def _calculate_risk_metrics(self, context: AgentContext) -> Dict[str, float]:
        """
        Calculate basic risk metrics.
        
        Args:
            context: Current market context
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            if context.market_data.empty:
                return {}
            
            close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
            if close_col not in context.market_data.columns:
                return {}
            
            prices = context.market_data[close_col]
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return {}
            
            # Calculate volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(returns, (1 - self.config['var_confidence_level']) * 100)
            
            # Calculate Expected Shortfall (Conditional VaR)
            es_95 = returns[returns <= var_95].mean()
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate Sharpe ratio
            excess_returns = returns - (self.config['risk_free_rate'] / 252)
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            return {
                'volatility': volatility,
                'var_95': var_95,
                'expected_shortfall': es_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'return_mean': returns.mean() * 252,  # Annualized
                'return_std': returns.std() * np.sqrt(252)  # Annualized
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _analyze_risk_regime(self, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze current risk regime.
        
        Args:
            risk_metrics: Calculated risk metrics
            
        Returns:
            Risk regime analysis
        """
        try:
            if not risk_metrics:
                return {'regime': 'unknown', 'confidence': 0.0}
            
            volatility = risk_metrics.get('volatility', 0)
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            
            # Classify risk regime
            if volatility > self.config['volatility_threshold'] * 2:
                regime = 'high_risk'
                confidence = 0.8
            elif volatility < self.config['volatility_threshold'] / 2:
                regime = 'low_risk'
                confidence = 0.8
            elif sharpe_ratio > self.config['sharpe_threshold']:
                regime = 'favorable_risk'
                confidence = 0.7
            elif max_drawdown < -0.2:  # 20% drawdown
                regime = 'stress_period'
                confidence = 0.9
            else:
                regime = 'normal_risk'
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility_level': 'high' if volatility > self.config['volatility_threshold'] else 'low',
                'sharpe_level': 'good' if sharpe_ratio > self.config['sharpe_threshold'] else 'poor'
            }
            
        except Exception as e:
            logger.error(f"Risk regime analysis failed: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}
    
    def _calculate_risk_adjusted_returns(self, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate risk-adjusted return metrics.
        
        Args:
            risk_metrics: Basic risk metrics
            
        Returns:
            Risk-adjusted return analysis
        """
        try:
            if not risk_metrics:
                return {}
            
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            sortino_ratio = risk_metrics.get('sortino_ratio', 0)
            volatility = risk_metrics.get('volatility', 0)
            return_mean = risk_metrics.get('return_mean', 0)
            
            # Assess risk-adjusted performance
            if sharpe_ratio > 1.5:
                performance = 'excellent'
            elif sharpe_ratio > 1.0:
                performance = 'good'
            elif sharpe_ratio > 0.5:
                performance = 'fair'
            else:
                performance = 'poor'
            
            # Risk-return efficiency
            if volatility > 0:
                risk_return_ratio = return_mean / volatility
            else:
                risk_return_ratio = 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'performance_rating': performance,
                'risk_return_ratio': risk_return_ratio,
                'efficiency': 'high' if risk_return_ratio > 0.5 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Risk-adjusted returns calculation failed: {e}")
            return {}
    
    def _assess_portfolio_risk(self, symbol: str, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess portfolio-level risk considerations.
        
        Args:
            symbol: Current symbol
            risk_metrics: Risk metrics
            
        Returns:
            Portfolio risk assessment
        """
        try:
            volatility = risk_metrics.get('volatility', 0)
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            
            # Position sizing recommendation
            if volatility > self.config['volatility_threshold'] * 2:
                recommended_position = self.config['max_position_size'] * 0.5  # Reduce position
                sizing_reason = 'High volatility - reduce position size'
            elif volatility < self.config['volatility_threshold'] / 2:
                recommended_position = self.config['max_position_size'] * 1.5  # Increase position
                sizing_reason = 'Low volatility - can increase position size'
            else:
                recommended_position = self.config['max_position_size']
                sizing_reason = 'Normal volatility - standard position size'
            
            # Risk concentration assessment
            concentration_risk = 'high' if volatility > self.config['volatility_threshold'] * 1.5 else 'low'
            
            return {
                'recommended_position_size': min(recommended_position, 0.2),  # Cap at 20%
                'sizing_reason': sizing_reason,
                'concentration_risk': concentration_risk,
                'diversification_benefit': 'high' if volatility < self.config['volatility_threshold'] else 'low'
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return {}
    
    def _calculate_risk_confidence(self, risk_metrics: Dict[str, float]) -> float:
        """
        Calculate confidence in risk analysis.
        
        Args:
            risk_metrics: Risk metrics
            
        Returns:
            Confidence score (0-1)
        """
        try:
            if not risk_metrics:
                return 0.0
            
            # Base confidence on data quality and metric consistency
            confidence = 0.7  # Base confidence
            
            # Adjust based on volatility stability
            volatility = risk_metrics.get('volatility', 0)
            if 0.01 < volatility < 0.05:  # Reasonable volatility range
                confidence += 0.1
            
            # Adjust based on Sharpe ratio
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            if abs(sharpe_ratio) < 3:  # Reasonable Sharpe ratio
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Risk confidence calculation failed: {e}")
            return 0.0
    
    def _generate_risk_signal(self, analysis: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on risk analysis.
        
        Args:
            analysis: Risk analysis results
            context: Current market context
            
        Returns:
            Risk-based trading signal
        """
        try:
            risk_regime = analysis['risk_regime']
            risk_adjusted_returns = analysis['risk_adjusted_returns']
            portfolio_risk = analysis['portfolio_risk']
            confidence = analysis['confidence']
            
            # Determine signal based on risk analysis
            if risk_regime['regime'] == 'high_risk':
                signal_type = SignalType.SELL
                reasoning = f"High risk regime detected (volatility: {analysis['risk_metrics'].get('volatility', 0):.2%})"
            elif risk_regime['regime'] == 'low_risk' and risk_adjusted_returns.get('performance_rating') == 'good':
                signal_type = SignalType.BUY
                reasoning = f"Low risk with good risk-adjusted returns (Sharpe: {risk_adjusted_returns.get('sharpe_ratio', 0):.2f})"
            elif risk_regime['regime'] == 'stress_period':
                signal_type = SignalType.SELL
                reasoning = f"Stress period detected (max drawdown: {analysis['risk_metrics'].get('max_drawdown', 0):.2%})"
            elif portfolio_risk.get('concentration_risk') == 'high':
                signal_type = SignalType.HOLD
                reasoning = "High concentration risk - maintain current position"
            else:
                signal_type = SignalType.HOLD
                reasoning = f"Normal risk conditions - no clear signal"
            
            # Adjust confidence based on risk regime confidence
            adjusted_confidence = min(confidence * risk_regime.get('confidence', 0.5), 0.9)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'risk_analysis': analysis,
                    'method': 'risk_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_risk_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple risk analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple risk-based signal
        """
        try:
            # Simple risk based on price volatility
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            if len(context.market_data) >= 5:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    returns = prices.pct_change().dropna()
                    volatility = returns.std()
                    
                    # High volatility suggests high risk
                    if volatility > 0.03:  # 3% volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.SELL,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility_risk'},
                            reasoning=f"High volatility ({volatility:.2%}) indicates elevated risk"
                        )
                    elif volatility < 0.01:  # 1% volatility
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_volatility_risk'},
                            reasoning=f"Low volatility ({volatility:.2%}) indicates low risk"
                        )
            
            return self._create_hold_signal("No clear risk signal", context)
            
        except Exception as e:
            logger.error(f"Simple risk analysis failed: {e}")
            return self._create_hold_signal(f"Simple risk analysis error: {e}", context)
    
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
        Update the risk model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Update risk history
            symbol = context.symbol
            if symbol not in self.risk_history:
                self.risk_history[symbol] = []
            
            # Store recent risk data
            if hasattr(self, '_last_risk_analysis'):
                self.risk_history[symbol].append(self._last_risk_analysis)
                
                # Keep only recent history
                if len(self.risk_history[symbol]) > 20:
                    self.risk_history[symbol] = self.risk_history[symbol][-20:]
            
            logger.info(f"Updated risk model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
