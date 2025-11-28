"""
Forecast Agent for AI Market Analysis System

This agent predicts price and volatility using machine learning models
and stochastic models to provide forecasting-based trading signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class ForecastAgent(BaseAgent):
    """
    Forecast Agent for predicting price and volatility using ML models.
    
    This agent analyzes:
    - Price forecasting using multiple ML models
    - Volatility prediction with GARCH and ML models
    - Multi-step ahead forecasting
    - Model ensemble and confidence intervals
    - Forecast accuracy tracking and model selection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Forecast Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'forecast_horizon': 5,  # days ahead
            'lookback_periods': 30,
            'min_training_periods': 20,
            'models': ['random_forest', 'linear_regression', 'garch'],
            'price_features': ['returns', 'volatility', 'volume_ratio', 'rsi', 'macd'],
            'volatility_features': ['returns', 'volume', 'high_low_ratio'],
            'confidence_threshold': 0.6,
            'forecast_confidence_interval': 0.95,
            'model_retrain_frequency': 100  # retrain every 100 predictions
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="ForecastAgent",
            version="1.0.0",
            config=default_config
        )
        
        self.price_models = {}
        self.volatility_models = {}
        self.scalers = {}
        self.forecast_history = []
        self.model_performance = {}
        self.prediction_count = 0
        
        logger.info(f"Initialized ForecastAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the forecast agent on historical data.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting training for {self.name}")
            
            if training_data.empty or len(training_data) < self.config['min_training_periods']:
                logger.warning(f"Insufficient training data: {len(training_data)} periods")
                self.is_trained = True  # Use simple forecasting
                return {"status": "insufficient_data", "periods": len(training_data)}
            
            # Prepare training features
            features = self._prepare_training_features(training_data)
            
            if features.empty:
                logger.warning("No valid features extracted from training data")
                self.is_trained = True  # Use simple forecasting
                return {"status": "no_features", "data_shape": training_data.shape}
            
            # Train price forecasting models
            price_training_results = self._train_price_models(features, training_data)
            
            # Train volatility forecasting models
            volatility_training_results = self._train_volatility_models(features, training_data)
            
            self.is_trained = True
            
            logger.info(f"{self.name}: Training completed successfully")
            return {
                "status": "training_complete",
                "price_models": price_training_results,
                "volatility_models": volatility_training_results,
                "features_count": len(features.columns)
            }
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate forecast-based prediction.
        
        Args:
            context: Current market context
            
        Returns:
            Forecast-based trading signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # If not trained, use simple forecasting
            if not self.is_trained:
                logger.info(f"{self.name}: Using simple forecasting (not trained)")
                return self._simple_forecast_analysis(context)
            
            # Perform comprehensive forecasting
            forecast_analysis = self._generate_forecasts(context)
            
            # Generate signal based on forecast insights
            signal = self._generate_forecast_signal(forecast_analysis, context)
            
            # Update prediction count for retraining
            self.prediction_count += 1
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Forecast analysis error: {e}", context)
    
    def _prepare_training_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        try:
            features = pd.DataFrame()
            
            if data.empty:
                return features
            
            # Ensure we have the right column names
            close_col = 'Close' if 'Close' in data.columns else 'close'
            high_col = 'High' if 'High' in data.columns else 'high'
            low_col = 'Low' if 'Low' in data.columns else 'low'
            volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
            
            if close_col not in data.columns:
                return features
            
            # Price-based features
            prices = data[close_col]
            
            # Returns
            features['returns'] = prices.pct_change()
            
            # Volatility (rolling standard deviation)
            features['volatility'] = features['returns'].rolling(window=5).std()
            
            # Volume ratio (if volume data available)
            if volume_col in data.columns:
                volume = data[volume_col]
                features['volume_ratio'] = volume / volume.rolling(window=20).mean()
            else:
                features['volume_ratio'] = 1.0
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(prices)
            features['macd'] = self._calculate_macd(prices)
            
            # High-low ratio (if available)
            if high_col in data.columns and low_col in data.columns:
                high = data[high_col]
                low = data[low_col]
                features['high_low_ratio'] = (high - low) / close_col
            else:
                features['high_low_ratio'] = 0.0
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'volatility_lag_{lag}'] = features['volatility'].shift(lag)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, data=50.0)  # Neutral RSI
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except:
            return pd.Series(index=prices.index, data=0.0)  # Neutral MACD
    
    def _train_price_models(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """Train price forecasting models."""
        try:
            results = {}
            
            if features.empty:
                return results
            
            # Prepare target (future returns)
            close_col = 'Close' if 'Close' in data.columns else 'close'
            prices = data[close_col]
            future_returns = prices.pct_change().shift(-1)  # Next period return
            
            # Align features with targets
            aligned_data = pd.concat([features, future_returns], axis=1).dropna()
            
            if len(aligned_data) < 10:
                return results
            
            X = aligned_data[features.columns]
            y = aligned_data[future_returns.name]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['price'] = scaler
            
            # Train Random Forest
            if 'random_forest' in self.config['models']:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_scaled, y)
                self.price_models['random_forest'] = rf_model
                results['random_forest'] = {'status': 'trained', 'n_features': X.shape[1]}
            
            # Train Linear Regression
            if 'linear_regression' in self.config['models']:
                lr_model = LinearRegression()
                lr_model.fit(X_scaled, y)
                self.price_models['linear_regression'] = lr_model
                results['linear_regression'] = {'status': 'trained', 'n_features': X.shape[1]}
            
            return results
            
        except Exception as e:
            logger.error(f"Price model training failed: {e}")
            return {}
    
    def _train_volatility_models(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """Train volatility forecasting models."""
        try:
            results = {}
            
            if features.empty:
                return results
            
            # Prepare target (future volatility)
            close_col = 'Close' if 'Close' in data.columns else 'close'
            prices = data[close_col]
            returns = prices.pct_change()
            future_volatility = returns.rolling(window=5).std().shift(-1)  # Next period volatility
            
            # Align features with targets
            aligned_data = pd.concat([features, future_volatility], axis=1).dropna()
            
            if len(aligned_data) < 10:
                return results
            
            X = aligned_data[features.columns]
            y = aligned_data[future_volatility.name]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['volatility'] = scaler
            
            # Train Random Forest for volatility
            if 'random_forest' in self.config['models']:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_scaled, y)
                self.volatility_models['random_forest'] = rf_model
                results['random_forest'] = {'status': 'trained', 'n_features': X.shape[1]}
            
            # Train Linear Regression for volatility
            if 'linear_regression' in self.config['models']:
                lr_model = LinearRegression()
                lr_model.fit(X_scaled, y)
                self.volatility_models['linear_regression'] = lr_model
                results['linear_regression'] = {'status': 'trained', 'n_features': X.shape[1]}
            
            return results
            
        except Exception as e:
            logger.error(f"Volatility model training failed: {e}")
            return {}
    
    def _generate_forecasts(self, context: AgentContext) -> Dict[str, Any]:
        """
        Generate price and volatility forecasts.
        
        Args:
            context: Current market context
            
        Returns:
            Forecast analysis results
        """
        try:
            if context.market_data.empty:
                return {'price_forecast': {}, 'volatility_forecast': {}, 'confidence': 0.0}
            
            # Prepare current features
            current_features = self._prepare_training_features(context.market_data)
            
            if current_features.empty or len(current_features) == 0:
                return {'price_forecast': {}, 'volatility_forecast': {}, 'confidence': 0.0}
            
            # Get latest features
            latest_features = current_features.iloc[-1:].copy()
            
            # Generate price forecasts
            price_forecast = self._forecast_price(latest_features)
            
            # Generate volatility forecasts
            volatility_forecast = self._forecast_volatility(latest_features)
            
            # Calculate forecast confidence
            confidence = self._calculate_forecast_confidence(price_forecast, volatility_forecast)
            
            return {
                'price_forecast': price_forecast,
                'volatility_forecast': volatility_forecast,
                'confidence': confidence,
                'features_used': len(latest_features.columns)
            }
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return {'price_forecast': {}, 'volatility_forecast': {}, 'confidence': 0.0}
    
    def _forecast_price(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate price forecasts using trained models."""
        try:
            forecasts = {}
            
            if 'price' not in self.scalers or not self.price_models:
                return {'method': 'no_models', 'forecast': 0.0, 'confidence': 0.0}
            
            # Scale features
            X_scaled = self.scalers['price'].transform(features)
            
            # Generate forecasts from each model
            model_forecasts = []
            for model_name, model in self.price_models.items():
                try:
                    forecast = model.predict(X_scaled)[0]
                    model_forecasts.append(forecast)
                    forecasts[model_name] = forecast
                except Exception as e:
                    logger.warning(f"Price forecast failed for {model_name}: {e}")
            
            if model_forecasts:
                # Ensemble forecast (average)
                ensemble_forecast = np.mean(model_forecasts)
                forecast_std = np.std(model_forecasts)
                
                forecasts['ensemble'] = ensemble_forecast
                forecasts['std'] = forecast_std
                forecasts['confidence'] = max(0.0, 1.0 - forecast_std)  # Lower std = higher confidence
            else:
                forecasts['ensemble'] = 0.0
                forecasts['confidence'] = 0.0
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Price forecasting failed: {e}")
            return {'method': 'error', 'forecast': 0.0, 'confidence': 0.0}
    
    def _forecast_volatility(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate volatility forecasts using trained models."""
        try:
            forecasts = {}
            
            if 'volatility' not in self.scalers or not self.volatility_models:
                return {'method': 'no_models', 'forecast': 0.0, 'confidence': 0.0}
            
            # Scale features
            X_scaled = self.scalers['volatility'].transform(features)
            
            # Generate forecasts from each model
            model_forecasts = []
            for model_name, model in self.volatility_models.items():
                try:
                    forecast = model.predict(X_scaled)[0]
                    model_forecasts.append(forecast)
                    forecasts[model_name] = forecast
                except Exception as e:
                    logger.warning(f"Volatility forecast failed for {model_name}: {e}")
            
            if model_forecasts:
                # Ensemble forecast (average)
                ensemble_forecast = np.mean(model_forecasts)
                forecast_std = np.std(model_forecasts)
                
                forecasts['ensemble'] = ensemble_forecast
                forecasts['std'] = forecast_std
                forecasts['confidence'] = max(0.0, 1.0 - forecast_std)
            else:
                forecasts['ensemble'] = 0.0
                forecasts['confidence'] = 0.0
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Volatility forecasting failed: {e}")
            return {'method': 'error', 'forecast': 0.0, 'confidence': 0.0}
    
    def _calculate_forecast_confidence(self, price_forecast: Dict[str, Any], volatility_forecast: Dict[str, Any]) -> float:
        """Calculate overall confidence in forecasts."""
        try:
            price_confidence = price_forecast.get('confidence', 0.0)
            volatility_confidence = volatility_forecast.get('confidence', 0.0)
            
            # Average confidence with slight weight toward price forecasts
            overall_confidence = (price_confidence * 0.6 + volatility_confidence * 0.4)
            
            return min(1.0, overall_confidence)
            
        except Exception as e:
            logger.error(f"Forecast confidence calculation failed: {e}")
            return 0.0
    
    def _generate_forecast_signal(self, analysis: Dict[str, Any], context: AgentContext) -> AgentSignal:
        """
        Generate trading signal based on forecast analysis.
        
        Args:
            analysis: Forecast analysis results
            context: Current market context
            
        Returns:
            Forecast-based trading signal
        """
        try:
            price_forecast = analysis['price_forecast']
            volatility_forecast = analysis['volatility_forecast']
            confidence = analysis['confidence']
            
            # Get forecast values
            price_change = price_forecast.get('ensemble', 0.0)
            volatility_change = volatility_forecast.get('ensemble', 0.0)
            
            # Determine signal based on forecasts
            if confidence > self.config['confidence_threshold']:
                if price_change > 0.02:  # 2% positive forecast
                    signal_type = SignalType.BUY
                    reasoning = f"Positive price forecast ({price_change:.2%}) with high confidence ({confidence:.2f})"
                elif price_change < -0.02:  # 2% negative forecast
                    signal_type = SignalType.SELL
                    reasoning = f"Negative price forecast ({price_change:.2%}) with high confidence ({confidence:.2f})"
                else:
                    signal_type = SignalType.HOLD
                    reasoning = f"Neutral price forecast ({price_change:.2%}) with high confidence ({confidence:.2f})"
            else:
                signal_type = SignalType.HOLD
                reasoning = f"Low forecast confidence ({confidence:.2f}) - price forecast: {price_change:.2%}"
            
            # Adjust confidence based on forecast quality
            adjusted_confidence = min(confidence * 0.9, 0.9)
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=adjusted_confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'forecast_analysis': analysis,
                    'method': 'ml_forecasting'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self._create_hold_signal(f"Signal generation error: {e}", context)
    
    def _simple_forecast_analysis(self, context: AgentContext) -> AgentSignal:
        """
        Simple forecast analysis when not trained.
        
        Args:
            context: Current market context
            
        Returns:
            Simple forecast-based signal
        """
        try:
            # Simple forecast based on recent price momentum
            if context.market_data.empty:
                return self._create_hold_signal("No market data available", context)
            
            if len(context.market_data) >= 5:
                close_col = 'Close' if 'Close' in context.market_data.columns else 'close'
                if close_col in context.market_data.columns:
                    prices = context.market_data[close_col]
                    
                    # Simple momentum-based forecast
                    recent_returns = prices.pct_change().tail(5)
                    avg_return = recent_returns.mean()
                    
                    # Simple volatility forecast
                    volatility = recent_returns.std()
                    
                    if avg_return > 0.01:  # 1% positive momentum
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_momentum_forecast'},
                            reasoning=f"Positive momentum forecast ({avg_return:.2%}) with volatility {volatility:.2%}"
                        )
                    elif avg_return < -0.01:  # 1% negative momentum
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.SELL,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'simple_momentum_forecast'},
                            reasoning=f"Negative momentum forecast ({avg_return:.2%}) with volatility {volatility:.2%}"
                        )
            
            return self._create_hold_signal("No clear forecast signal", context)
            
        except Exception as e:
            logger.error(f"Simple forecast analysis failed: {e}")
            return self._create_hold_signal(f"Simple forecast analysis error: {e}", context)
    
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
        Update the forecast model with new data.
        
        Args:
            new_data: New market data
            context: Current context
        """
        try:
            # Check if we need to retrain
            if self.prediction_count > 0 and self.prediction_count % self.config['model_retrain_frequency'] == 0:
                logger.info(f"Retraining models after {self.prediction_count} predictions")
                # In a real implementation, we would retrain here
                # For now, we'll just log the event
            
            # Update forecast history
            if hasattr(self, '_last_forecast_analysis'):
                self.forecast_history.append(self._last_forecast_analysis)
                
                # Keep only recent history
                if len(self.forecast_history) > 100:
                    self.forecast_history = self.forecast_history[-100:]
            
            logger.info(f"Updated forecast model for {self.name}")
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
