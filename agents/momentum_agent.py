"""
Momentum Agent for AI Market Analysis System

This module implements the MomentumAgent class that detects price trend continuation
or reversal using LSTM networks and ARIMA models. It analyzes momentum patterns
and provides trading signals based on trend strength and direction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import base agent and context
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentSignal, SignalType, AgentContext
from context.context_managers import TimeContext, EventContext, RegimeContext

logger = logging.getLogger(__name__)


class MomentumAgent(BaseAgent):
    """
    Momentum Agent for detecting price trend continuation or reversal.
    
    This agent uses multiple technical indicators and machine learning models
    to identify momentum patterns and predict future price movements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Momentum Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'lookback_periods': [5, 10, 20, 50],
            'momentum_threshold': 0.02,
            'trend_strength_threshold': 0.5,
            'volume_confirmation': True,
            'use_lstm': True,
            'lstm_sequence_length': 60,
            'lstm_units': [50, 25],
            'lstm_dropout': 0.2,
            'use_arima': True,
            'arima_order': (1, 1, 1),
            'min_confidence': 0.6,
            'max_position_size': 0.1
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="MomentumAgent",
            version="1.0.0",
            config=default_config
        )
        
        # Initialize components
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.lstm_model = None
        self.arima_model = None
        self.feature_columns = []
        self.is_trained = False
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
        logger.info(f"Initialized MomentumAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the momentum agent's models on historical data.
        
        Args:
            training_data: Historical market data for training
            context: Shared context information
            
        Returns:
            Dictionary containing training metrics and results
        """
        try:
            self.status = "training"
            logger.info(f"Starting training for {self.name} with {len(training_data)} records")
            
            # For minimal data scenarios, use simple rule-based approach
            if len(training_data) < 10:
                logger.info(f"Insufficient data for ML training ({len(training_data)} records). Using rule-based approach.")
                self.is_trained = True
                self.status = "idle"
                return {
                    'training_samples': len(training_data),
                    'feature_count': 0,
                    'training_method': 'rule_based',
                    'training_completed': datetime.now()
                }
            
            # Prepare training data
            processed_data = self._prepare_training_data(training_data)
            
            if processed_data.empty:
                logger.warning("No valid training data after preprocessing. Using rule-based approach.")
                self.is_trained = True
                self.status = "idle"
                return {
                    'training_samples': 0,
                    'feature_count': 0,
                    'training_method': 'rule_based',
                    'training_completed': datetime.now()
                }
            
            # Extract features
            features_df = self._extract_momentum_features(processed_data)
            
            if features_df.empty:
                logger.warning("No features extracted from training data. Using rule-based approach.")
                self.is_trained = True
                self.status = "idle"
                return {
                    'training_samples': len(processed_data),
                    'feature_count': 0,
                    'training_method': 'rule_based',
                    'training_completed': datetime.now()
                }
            
            # Prepare targets (future returns)
            targets = self._prepare_targets(processed_data)
            
            # Train LSTM model if enabled
            lstm_metrics = {}
            if self.config['use_lstm']:
                lstm_metrics = self._train_lstm_model(features_df, targets)
            
            # Train ARIMA model if enabled
            arima_metrics = {}
            if self.config['use_arima']:
                arima_metrics = self._train_arima_model(processed_data)
            
            # Calculate overall training metrics
            training_metrics = {
                'training_samples': len(processed_data),
                'feature_count': len(self.feature_columns),
                'lstm_metrics': lstm_metrics,
                'arima_metrics': arima_metrics,
                'training_method': 'ml_models',
                'training_completed': datetime.now()
            }
            
            self.is_trained = True
            self.status = "idle"
            
            logger.info(f"Training completed for {self.name}")
            return training_metrics
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Training failed for {self.name}: {e}")
            # Even if training fails, mark as trained for rule-based predictions
            self.is_trained = True
            self.status = "idle"
            return {
                'training_samples': 0,
                'feature_count': 0,
                'training_method': 'rule_based_fallback',
                'training_completed': datetime.now(),
                'error': str(e)
            }
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate a momentum-based prediction/signal.
        
        Args:
            context: Current market context and data
            
        Returns:
            AgentSignal object containing the prediction
        """
        try:
            logger.info(f"{self.name}: Starting prediction with context data shape: {context.market_data.shape}")
            self.status = "predicting"
            
            # If not trained, use simple rule-based approach
            if not self.is_trained:
                logger.info(f"{self.name}: Using rule-based prediction (not trained)")
                return self._simple_rule_based_prediction(context)
            
            # Try ML-based prediction first
            try:
                # Extract current features
                current_features = self._extract_momentum_features(context.market_data)
                
                if not current_features.empty:
                    # Get latest features
                    latest_features = current_features.iloc[-1:].copy()
                    
                    # Make predictions using both models
                    lstm_prediction = None
                    arima_prediction = None
                    
                    if self.config['use_lstm'] and self.lstm_model is not None:
                        lstm_prediction = self._predict_lstm(latest_features)
                    
                    if self.config['use_arima'] and self.arima_model is not None:
                        arima_prediction = self._predict_arima(context.market_data)
                    
                    # Combine predictions
                    final_prediction = self._combine_predictions(lstm_prediction, arima_prediction)
                    
                    # Generate signal based on prediction
                    signal = self._generate_signal(final_prediction, latest_features, context)
                    
                    # Store prediction for performance tracking
                    self.prediction_history.append({
                        'timestamp': context.timestamp,
                        'prediction': final_prediction,
                        'signal': signal,
                        'features': latest_features.to_dict('records')[0]
                    })
                    
                    self.status = "idle"
                    return signal
                    
            except Exception as e:
                logger.warning(f"ML prediction failed, falling back to rule-based: {e}")
            
            # Fallback to simple rule-based prediction
            signal = self._simple_rule_based_prediction(context)
            
            # Store prediction for performance tracking
            self.prediction_history.append({
                'timestamp': context.timestamp,
                'prediction': 'rule_based',
                'signal': signal,
                'features': {}
            })
            
            self.status = "idle"
            return signal
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Prediction failed for {self.name}: {e}")
            return self._create_hold_signal(f"Prediction error: {e}", context)
    
    def update_model(self, new_data: pd.DataFrame, context: AgentContext) -> None:
        """
        Update the agent's model with new data (online learning).
        
        Args:
            new_data: New market data to incorporate
            context: Current context information
        """
        try:
            logger.info(f"Updating model for {self.name} with {len(new_data)} new records")
            
            # For now, implement simple retraining with recent data
            # In production, this could use online learning algorithms
            
            # Combine with recent historical data for retraining
            if hasattr(self, 'recent_data'):
                combined_data = pd.concat([self.recent_data, new_data], ignore_index=True)
            else:
                combined_data = new_data
            
            # Keep only recent data (e.g., last 1000 records)
            if len(combined_data) > 1000:
                combined_data = combined_data.tail(1000)
            
            self.recent_data = combined_data
            
            # Retrain models periodically (e.g., every 100 new records)
            if len(new_data) % 100 == 0:
                logger.info("Triggering model retraining...")
                self.train(combined_data, context)
            
        except Exception as e:
            logger.error(f"Model update failed for {self.name}: {e}")
    
    def _prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean training data."""
        if data.empty:
            return data
        
        # Make a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Ensure we have required columns (handle different naming conventions)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        column_mapping = {}
        
        # Check for exact matches first
        for col in required_columns:
            if col in processed_data.columns:
                column_mapping[col] = col
            else:
                # Try lowercase version
                lower_col = col.lower()
                if lower_col in processed_data.columns:
                    column_mapping[col] = lower_col
                else:
                    logger.warning(f"Missing column: {col} (tried {col} and {lower_col})")
        
        if len(column_mapping) < len(required_columns):
            logger.warning(f"Missing required columns. Available columns: {list(processed_data.columns)}")
            return pd.DataFrame()
        
        # Rename columns to standard names
        processed_data = processed_data.rename(columns={v: k for k, v in column_mapping.items()})
        
        # Remove rows with missing values
        processed_data = processed_data.dropna()
        
        # Sort by date if index is datetime
        if isinstance(processed_data.index, pd.DatetimeIndex):
            processed_data = processed_data.sort_index()
        
        return processed_data
    
    def _extract_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum-related features from market data."""
        if data.empty:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(index=data.index)
        
        # Price-based momentum indicators
        for period in self.config['lookback_periods']:
            if len(data) >= period:
                # Price momentum
                features_df[f'price_momentum_{period}'] = data['Close'].pct_change(period)
                
                # Rate of change
                features_df[f'roc_{period}'] = (data['Close'] / data['Close'].shift(period) - 1) * 100
                
                # Moving average convergence/divergence
                ema_fast = data['Close'].ewm(span=period//2).mean()
                ema_slow = data['Close'].ewm(span=period).mean()
                features_df[f'macd_{period}'] = ema_fast - ema_slow
                features_df[f'macd_signal_{period}'] = features_df[f'macd_{period}'].ewm(span=9).mean()
                features_df[f'macd_histogram_{period}'] = features_df[f'macd_{period}'] - features_df[f'macd_signal_{period}']
        
        # Volatility indicators
        features_df['volatility_5d'] = data['Close'].pct_change().rolling(window=5).std()
        features_df['volatility_20d'] = data['Close'].pct_change().rolling(window=20).std()
        features_df['volatility_ratio'] = features_df['volatility_5d'] / features_df['volatility_20d']
        
        # Volume indicators
        features_df['volume_sma_20'] = data['Volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = data['Volume'] / features_df['volume_sma_20']
        features_df['volume_momentum'] = data['Volume'].pct_change(5)
        
        # Price position indicators
        features_df['price_position_20'] = (data['Close'] - data['Close'].rolling(window=20).min()) / (data['Close'].rolling(window=20).max() - data['Close'].rolling(window=20).min())
        features_df['price_position_50'] = (data['Close'] - data['Close'].rolling(window=50).min()) / (data['Close'].rolling(window=50).max() - data['Close'].rolling(window=50).min())
        
        # Trend strength indicators
        features_df['trend_strength_20'] = abs(data['Close'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        features_df['trend_strength_50'] = abs(data['Close'].rolling(window=50).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        # RSI
        features_df['rsi_14'] = self._calculate_rsi(data['Close'], 14)
        features_df['rsi_30'] = self._calculate_rsi(data['Close'], 30)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'])
        features_df['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Time-based features
        if isinstance(data.index, pd.DatetimeIndex):
            time_context = TimeContext()
            time_features = time_context.get_time_features()
            for key, value in time_features.items():
                if isinstance(value, (int, float)):
                    features_df[f'time_{key}'] = value
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        # Store feature columns for later use
        self.feature_columns = features_df.columns.tolist()
        
        return features_df
    
    def _prepare_targets(self, data: pd.DataFrame) -> pd.Series:
        """Prepare target variables for training (future returns)."""
        # Use 1-day, 3-day, and 5-day forward returns as targets
        targets = pd.Series(index=data.index, dtype=float)
        
        # Primary target: 1-day forward return
        targets['return_1d'] = data['Close'].shift(-1) / data['Close'] - 1
        
        # Additional targets for multi-task learning
        targets['return_3d'] = data['Close'].shift(-3) / data['Close'] - 1
        targets['return_5d'] = data['Close'].shift(-5) / data['Close'] - 1
        
        return targets['return_1d'].dropna()
    
    def _train_lstm_model(self, features_df: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Train LSTM model for momentum prediction."""
        try:
            # For now, implement a simple linear model as placeholder
            # In production, this would use TensorFlow/Keras LSTM
            
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            
            # Align features and targets
            common_index = features_df.index.intersection(targets.index)
            X = features_df.loc[common_index]
            y = targets.loc[common_index]
            
            if len(X) < 100:  # Need minimum data for training
                logger.warning("Insufficient data for LSTM training")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model (placeholder for LSTM)
            self.lstm_model = LinearRegression()
            self.lstm_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.lstm_model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = self.lstm_model.score(X_test_scaled, y_test)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"LSTM model trained with R² = {r2:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return {}
    
    def _train_arima_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train ARIMA model for momentum prediction."""
        try:
            # For now, implement a simple moving average as placeholder
            # In production, this would use statsmodels ARIMA
            
            prices = data['Close'].dropna()
            
            if len(prices) < 50:  # Need minimum data for ARIMA
                logger.warning("Insufficient data for ARIMA training")
                return {}
            
            # Simple moving average model as placeholder
            self.arima_model = {
                'sma_short': prices.rolling(window=10).mean(),
                'sma_long': prices.rolling(window=30).mean(),
                'last_price': prices.iloc[-1]
            }
            
            metrics = {
                'model_type': 'moving_average_placeholder',
                'training_samples': len(prices),
                'short_window': 10,
                'long_window': 30
            }
            
            logger.info("ARIMA model trained (placeholder)")
            return metrics
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return {}
    
    def _predict_lstm(self, features: pd.DataFrame) -> Optional[float]:
        """Make prediction using LSTM model."""
        if self.lstm_model is None:
            return None
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.lstm_model.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return None
    
    def _predict_arima(self, data: pd.DataFrame) -> Optional[float]:
        """Make prediction using ARIMA model."""
        if self.arima_model is None:
            return None
        
        try:
            # Simple moving average prediction as placeholder
            current_price = data['Close'].iloc[-1]
            sma_short = data['Close'].rolling(window=10).mean().iloc[-1]
            sma_long = data['Close'].rolling(window=30).mean().iloc[-1]
            
            # Simple momentum signal
            if sma_short > sma_long:
                prediction = 0.01  # Positive momentum
            elif sma_short < sma_long:
                prediction = -0.01  # Negative momentum
            else:
                prediction = 0.0  # Neutral
            
            return prediction
            
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return None
    
    def _combine_predictions(self, lstm_pred: Optional[float], arima_pred: Optional[float]) -> float:
        """Combine predictions from different models."""
        predictions = []
        weights = []
        
        if lstm_pred is not None:
            predictions.append(lstm_pred)
            weights.append(0.6)  # Higher weight for LSTM
        
        if arima_pred is not None:
            predictions.append(arima_pred)
            weights.append(0.4)  # Lower weight for ARIMA
        
        if not predictions:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            combined_pred = sum(p * w for p, w in zip(predictions, weights)) / total_weight
        else:
            combined_pred = 0.0
        
        return combined_pred
    
    def _generate_signal(self, prediction: float, features: pd.DataFrame, context: AgentContext) -> AgentSignal:
        """Generate trading signal based on prediction and features."""
        # Determine signal type based on prediction
        if prediction > self.config['momentum_threshold']:
            signal_type = SignalType.BUY
            confidence = min(0.95, abs(prediction) * 10)
        elif prediction < -self.config['momentum_threshold']:
            signal_type = SignalType.SELL
            confidence = min(0.95, abs(prediction) * 10)
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
        
        # Adjust confidence based on trend strength
        if 'trend_strength_20' in features.columns:
            trend_strength = features['trend_strength_20'].iloc[0]
            if trend_strength > self.config['trend_strength_threshold']:
                confidence = min(0.95, confidence * 1.2)
        
        # Volume confirmation
        if self.config['volume_confirmation'] and 'volume_ratio' in features.columns:
            volume_ratio = features['volume_ratio'].iloc[0]
            if volume_ratio > 1.5:  # High volume
                confidence = min(0.95, confidence * 1.1)
            elif volume_ratio < 0.5:  # Low volume
                confidence = max(0.3, confidence * 0.8)
        
        # Ensure minimum confidence
        confidence = max(self.config['min_confidence'], confidence)
        
        # Create reasoning
        reasoning = f"Momentum prediction: {prediction:.4f}, Signal: {signal_type.value}, Confidence: {confidence:.2f}"
        
        return AgentSignal(
            agent_name=self.name,
            signal_type=signal_type,
            confidence=confidence,
            timestamp=context.timestamp,
            asset_symbol=context.market_data.get('symbol', 'UNKNOWN').iloc[-1] if 'symbol' in context.market_data.columns else 'UNKNOWN',
            metadata={
                'prediction': prediction,
                'lstm_prediction': self._predict_lstm(features),
                'arima_prediction': self._predict_arima(context.market_data),
                'features_used': len(self.feature_columns),
                'trend_strength': features.get('trend_strength_20', [0]).iloc[0] if 'trend_strength_20' in features.columns else 0,
                'volume_ratio': features.get('volume_ratio', [1]).iloc[0] if 'volume_ratio' in features.columns else 1
            },
            reasoning=reasoning
        )
    
    def _simple_rule_based_prediction(self, context: AgentContext) -> AgentSignal:
        """
        Simple rule-based prediction when ML models are not available.
        
        Args:
            context: Current market context and data
            
        Returns:
            AgentSignal object containing the prediction
        """
        try:
            logger.info(f"Rule-based prediction called for {context.symbol} with {len(context.market_data)} data points")
            
            # Get basic price data
            if context.market_data.empty:
                logger.warning("No market data available for rule-based prediction")
                return self._create_hold_signal("No market data available", context)
            
            # Use the last available price data
            latest_data = context.market_data.iloc[-1]
            logger.info(f"Latest data columns: {list(latest_data.index)}")
            
            # Simple momentum rules
            if 'Close' in latest_data and 'Volume' in latest_data:
                current_price = latest_data['Close']
                current_volume = latest_data['Volume']
                
                # Simple price momentum (if we have at least 2 data points)
                if len(context.market_data) >= 2:
                    prev_price = context.market_data.iloc[-2]['Close']
                    price_change = (current_price - prev_price) / prev_price
                    
                    logger.info(f"Price change: {price_change:.2%}")
                    
                    # Simple rules based on price change
                    if price_change > 0.02:  # 2% increase
                        logger.info(f"Generating BUY signal for {context.symbol}")
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.BUY,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'rule_based'},
                            reasoning=f"Price increased by {price_change:.2%}"
                        )
                    elif price_change < -0.02:  # 2% decrease
                        logger.info(f"Generating SELL signal for {context.symbol}")
                        return AgentSignal(
                            agent_name=self.name,
                            signal_type=SignalType.SELL,
                            confidence=0.6,
                            timestamp=context.timestamp,
                            asset_symbol=context.symbol,
                            metadata={'agent_version': self.version, 'method': 'rule_based'},
                            reasoning=f"Price decreased by {price_change:.2%}"
                        )
            
            # Default to hold if no clear signal
            logger.info(f"Generating HOLD signal for {context.symbol}")
            return self._create_hold_signal("No clear momentum signal", context)
            
        except Exception as e:
            logger.error(f"Rule-based prediction failed: {e}")
            return self._create_hold_signal(f"Rule-based prediction error: {e}", context)
    
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
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the agent."""
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}
        
        recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
        
        signal_counts = {}
        confidence_scores = []
        
        for pred in recent_predictions:
            signal_type = pred['signal'].signal_type.value
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            confidence_scores.append(pred['signal'].confidence)
        
        return {
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len(recent_predictions),
            'signal_distribution': signal_counts,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_columns),
            'last_prediction': self.prediction_history[-1]['timestamp'] if self.prediction_history else None
        }
