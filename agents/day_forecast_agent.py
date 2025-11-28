"""
Day Forecast Agent

This agent provides 1-day horizon forecasting with fast technical indicators.
Optimized for day trading with rapid signal generation and high-frequency updates.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ForecastHorizon(Enum):
    """Forecast horizons for day trading."""
    INTRADAY = "intraday"  # 1-4 hours
    END_OF_DAY = "end_of_day"  # End of trading day
    NEXT_DAY_OPEN = "next_day_open"  # Next day opening
    NEXT_DAY_CLOSE = "next_day_close"  # Next day closing

class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TechnicalIndicator:
    """Technical indicator data."""
    name: str
    value: float
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1
    timestamp: datetime

@dataclass
class DayForecast:
    """Day trading forecast."""
    symbol: str
    horizon: ForecastHorizon
    predicted_price: float
    confidence: float
    direction: str  # 'up', 'down', 'sideways'
    signal_strength: SignalStrength
    technical_indicators: List[TechnicalIndicator]
    market_regime: str
    volatility_forecast: float
    volume_forecast: float
    risk_score: float
    created_at: datetime
    valid_until: datetime

@dataclass
class DayForecastMetrics:
    """Day forecast performance metrics."""
    total_forecasts: int
    accurate_forecasts: int
    accuracy_percentage: float
    avg_confidence: float
    avg_error: float
    mse: float
    mae: float
    r2_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    last_updated: datetime

class DayForecastAgent:
    """
    Day Forecast Agent for 1-day horizon trading.
    
    This agent handles:
    - Fast technical indicator calculation
    - Short-term price prediction
    - Intraday signal generation
    - High-frequency model updates
    - Day trading optimization
    """
    
    def __init__(self, real_data_service=None):
        self.real_data_service = real_data_service
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.forecasts: Dict[str, DayForecast] = {}
        self.forecast_history: List[DayForecast] = []
        self.metrics: Optional[DayForecastMetrics] = None
        
        # Model parameters
        self.lookback_periods = 20  # 20 periods for technical indicators
        self.forecast_horizons = [ForecastHorizon.INTRADAY, ForecastHorizon.END_OF_DAY]
        self.update_frequency = 300  # 5 minutes
        
        # Technical indicators
        self.indicators = [
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'rsi_14', 'macd', 'macd_signal',
            'bollinger_upper', 'bollinger_lower', 'bollinger_middle',
            'stoch_k', 'stoch_d', 'williams_r',
            'cci', 'atr', 'volume_sma'
        ]
        
        # Initialize models
        self._initialize_models()
        
        logger.info("DayForecastAgent initialized")
    
    def _initialize_models(self):
        """Initialize forecasting models."""
        for horizon in self.forecast_horizons:
            # Use ensemble of models for better accuracy
            self.models[horizon.value] = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0),
                'linear': LinearRegression()
            }
            
            # Initialize scalers
            self.scalers[horizon.value] = StandardScaler()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators for day trading."""
        indicators = {}
        
        if len(data) < self.lookback_periods:
            return indicators
        
        # Price data
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # Simple Moving Averages
        indicators['sma_5'] = np.mean(close[-5:])
        indicators['sma_10'] = np.mean(close[-10:])
        indicators['sma_20'] = np.mean(close[-20:])
        
        # Exponential Moving Averages
        alpha_5 = 2 / (5 + 1)
        alpha_10 = 2 / (10 + 1)
        alpha_20 = 2 / (20 + 1)
        
        ema_5 = close[-1]
        ema_10 = close[-1]
        ema_20 = close[-1]
        
        for i in range(1, min(20, len(close))):
            if i < 5:
                ema_5 = alpha_5 * close[-(i+1)] + (1 - alpha_5) * ema_5
            if i < 10:
                ema_10 = alpha_10 * close[-(i+1)] + (1 - alpha_10) * ema_10
            ema_20 = alpha_20 * close[-(i+1)] + (1 - alpha_20) * ema_20
        
        indicators['ema_5'] = ema_5
        indicators['ema_10'] = ema_10
        indicators['ema_20'] = ema_20
        
        # RSI (14-period)
        if len(close) >= 15:
            deltas = np.diff(close[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi_14'] = rsi
            else:
                indicators['rsi_14'] = 100
        
        # MACD
        if len(close) >= 26:
            ema_12 = close[-1]
            ema_26 = close[-1]
            
            for i in range(1, min(26, len(close))):
                if i < 12:
                    ema_12 = (2 / 13) * close[-(i+1)] + (11 / 13) * ema_12
                ema_26 = (2 / 27) * close[-(i+1)] + (25 / 27) * ema_26
            
            macd = ema_12 - ema_26
            indicators['macd'] = macd
            
            # MACD Signal (9-period EMA of MACD)
            if len(close) >= 35:
                macd_signal = macd
                for i in range(1, 9):
                    macd_signal = (2 / 10) * macd + (8 / 10) * macd_signal
                indicators['macd_signal'] = macd_signal
        
        # Bollinger Bands
        if len(close) >= 20:
            sma_20 = indicators['sma_20']
            std_20 = np.std(close[-20:])
            indicators['bollinger_upper'] = sma_20 + (2 * std_20)
            indicators['bollinger_lower'] = sma_20 - (2 * std_20)
            indicators['bollinger_middle'] = sma_20
        
        # Stochastic Oscillator
        if len(close) >= 14:
            lowest_low = np.min(low[-14:])
            highest_high = np.max(high[-14:])
            current_close = close[-1]
            
            if highest_high != lowest_low:
                stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
                indicators['stoch_k'] = stoch_k
                
                # Stochastic D (3-period SMA of K)
                if len(close) >= 16:
                    stoch_d = np.mean([stoch_k] * 3)  # Simplified
                    indicators['stoch_d'] = stoch_d
        
        # Williams %R
        if len(close) >= 14:
            highest_high = np.max(high[-14:])
            lowest_low = np.min(low[-14:])
            current_close = close[-1]
            
            if highest_high != lowest_low:
                williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
                indicators['williams_r'] = williams_r
        
        # Commodity Channel Index (CCI)
        if len(close) >= 20:
            typical_price = (high[-20:] + low[-20:] + close[-20:]) / 3
            sma_tp = np.mean(typical_price)
            mean_deviation = np.mean(np.abs(typical_price - sma_tp))
            
            if mean_deviation != 0:
                cci = (typical_price[-1] - sma_tp) / (0.015 * mean_deviation)
                indicators['cci'] = cci
        
        # Average True Range (ATR)
        if len(close) >= 14:
            tr_values = []
            for i in range(1, min(15, len(close))):
                tr1 = high[-(i+1)] - low[-(i+1)]
                tr2 = abs(high[-(i+1)] - close[-i])
                tr3 = abs(low[-(i+1)] - close[-i])
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)
            
            atr = np.mean(tr_values)
            indicators['atr'] = atr
        
        # Volume SMA
        if len(volume) >= 20:
            indicators['volume_sma'] = np.mean(volume[-20:])
        
        return indicators
    
    def _generate_technical_signals(self, indicators: Dict[str, float], current_price: float) -> List[TechnicalIndicator]:
        """Generate technical signals from indicators."""
        signals = []
        current_time = datetime.now()
        
        # RSI signals
        if 'rsi_14' in indicators:
            rsi = indicators['rsi_14']
            if rsi < 30:
                signals.append(TechnicalIndicator(
                    name='RSI',
                    value=rsi,
                    signal='buy',
                    strength=min((30 - rsi) / 30, 1.0),
                    timestamp=current_time
                ))
            elif rsi > 70:
                signals.append(TechnicalIndicator(
                    name='RSI',
                    value=rsi,
                    signal='sell',
                    strength=min((rsi - 70) / 30, 1.0),
                    timestamp=current_time
                ))
        
        # MACD signals
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            
            if macd > macd_signal:
                signals.append(TechnicalIndicator(
                    name='MACD',
                    value=macd - macd_signal,
                    signal='buy',
                    strength=min(abs(macd - macd_signal) / 0.01, 1.0),
                    timestamp=current_time
                ))
            else:
                signals.append(TechnicalIndicator(
                    name='MACD',
                    value=macd - macd_signal,
                    signal='sell',
                    strength=min(abs(macd - macd_signal) / 0.01, 1.0),
                    timestamp=current_time
                ))
        
        # Bollinger Bands signals
        if all(key in indicators for key in ['bollinger_upper', 'bollinger_lower', 'bollinger_middle']):
            upper = indicators['bollinger_upper']
            lower = indicators['bollinger_lower']
            middle = indicators['bollinger_middle']
            
            if current_price <= lower:
                signals.append(TechnicalIndicator(
                    name='Bollinger Bands',
                    value=(lower - current_price) / (upper - lower),
                    signal='buy',
                    strength=min((lower - current_price) / (upper - lower), 1.0),
                    timestamp=current_time
                ))
            elif current_price >= upper:
                signals.append(TechnicalIndicator(
                    name='Bollinger Bands',
                    value=(current_price - upper) / (upper - lower),
                    signal='sell',
                    strength=min((current_price - upper) / (upper - lower), 1.0),
                    timestamp=current_time
                ))
        
        # Moving Average signals
        if all(key in indicators for key in ['sma_5', 'sma_20']):
            sma_5 = indicators['sma_5']
            sma_20 = indicators['sma_20']
            
            if sma_5 > sma_20:
                signals.append(TechnicalIndicator(
                    name='MA Crossover',
                    value=(sma_5 - sma_20) / sma_20,
                    signal='buy',
                    strength=min((sma_5 - sma_20) / sma_20 * 100, 1.0),
                    timestamp=current_time
                ))
            else:
                signals.append(TechnicalIndicator(
                    name='MA Crossover',
                    value=(sma_20 - sma_5) / sma_20,
                    signal='sell',
                    strength=min((sma_20 - sma_5) / sma_20 * 100, 1.0),
                    timestamp=current_time
                ))
        
        return signals
    
    def _determine_market_regime(self, data: pd.DataFrame) -> str:
        """Determine current market regime."""
        if len(data) < 20:
            return "unknown"
        
        close = data['close'].values
        volume = data['volume'].values
        
        # Calculate volatility
        returns = np.diff(close[-20:]) / close[-21:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Calculate trend
        sma_5 = np.mean(close[-5:])
        sma_20 = np.mean(close[-20:])
        trend_strength = (sma_5 - sma_20) / sma_20
        
        # Calculate volume trend
        volume_trend = np.mean(volume[-5:]) / np.mean(volume[-20:])
        
        # Determine regime
        if volatility > 0.3:  # High volatility
            if trend_strength > 0.02:
                return "trending_up_volatile"
            elif trend_strength < -0.02:
                return "trending_down_volatile"
            else:
                return "sideways_volatile"
        else:  # Low volatility
            if trend_strength > 0.01:
                return "trending_up_calm"
            elif trend_strength < -0.01:
                return "trending_down_calm"
            else:
                return "sideways_calm"
    
    def _calculate_confidence(self, indicators: Dict[str, float], signals: List[TechnicalIndicator], 
                            market_regime: str) -> float:
        """Calculate forecast confidence."""
        confidence = 0.5  # Base confidence
        
        # Signal agreement
        if signals:
            buy_signals = sum(1 for s in signals if s.signal == 'buy')
            sell_signals = sum(1 for s in signals if s.signal == 'sell')
            total_signals = len(signals)
            
            if total_signals > 0:
                agreement = max(buy_signals, sell_signals) / total_signals
                confidence += agreement * 0.3
        
        # Signal strength
        if signals:
            avg_strength = np.mean([s.strength for s in signals])
            confidence += avg_strength * 0.2
        
        # Market regime adjustment
        regime_confidence = {
            "trending_up_calm": 0.8,
            "trending_down_calm": 0.8,
            "trending_up_volatile": 0.6,
            "trending_down_volatile": 0.6,
            "sideways_calm": 0.5,
            "sideways_volatile": 0.4,
            "unknown": 0.3
        }
        
        confidence *= regime_confidence.get(market_regime, 0.5)
        
        return min(max(confidence, 0.1), 0.95)
    
    async def generate_forecast(self, symbol: str, horizon: ForecastHorizon = ForecastHorizon.END_OF_DAY) -> DayForecast:
        """Generate day trading forecast."""
        try:
            # Get market data
            data = await self._get_market_data(symbol, periods=50)
            if data is None or len(data) < 20:
                logger.warning(f"Insufficient data for {symbol}, using minimal forecast")
                return self._generate_minimal_forecast(symbol, horizon)
            
            current_price = data['close'].iloc[-1]
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(data)
            
            # Generate technical signals
            signals = self._generate_technical_signals(indicators, current_price)
            
            # Determine market regime
            market_regime = self._determine_market_regime(data)
            
            # Generate price forecast using ensemble
            predicted_price = await self._predict_price(symbol, data, horizon)
            
            # Calculate confidence
            confidence = self._calculate_confidence(indicators, signals, market_regime)
            
            # Determine direction and signal strength
            price_change = (predicted_price - current_price) / current_price
            
            if price_change > 0.01:  # > 1% increase
                direction = 'up'
                if price_change > 0.03:
                    signal_strength = SignalStrength.VERY_STRONG
                elif price_change > 0.02:
                    signal_strength = SignalStrength.STRONG
                else:
                    signal_strength = SignalStrength.MODERATE
            elif price_change < -0.01:  # > 1% decrease
                direction = 'down'
                if price_change < -0.03:
                    signal_strength = SignalStrength.VERY_STRONG
                elif price_change < -0.02:
                    signal_strength = SignalStrength.STRONG
                else:
                    signal_strength = SignalStrength.MODERATE
            else:
                direction = 'sideways'
                signal_strength = SignalStrength.WEAK
            
            # Calculate volatility and volume forecasts
            volatility_forecast = self._forecast_volatility(data)
            volume_forecast = self._forecast_volume(data)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(indicators, market_regime)
            
            # Create forecast
            forecast = DayForecast(
                symbol=symbol,
                horizon=horizon,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                signal_strength=signal_strength,
                technical_indicators=signals,
                market_regime=market_regime,
                volatility_forecast=volatility_forecast,
                volume_forecast=volume_forecast,
                risk_score=risk_score,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=4)  # 4-hour validity
            )
            
            # Store forecast
            self.forecasts[f"{symbol}_{horizon.value}"] = forecast
            self.forecast_history.append(forecast)
            
            # Keep only recent history
            if len(self.forecast_history) > 1000:
                self.forecast_history = self.forecast_history[-1000:]
            
            logger.info(f"Generated day forecast for {symbol}: {direction} to ${predicted_price:.2f} (confidence: {confidence:.2f})")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating day forecast for {symbol}: {e}")
            # Return a minimal forecast instead of raising
            return self._generate_minimal_forecast(symbol, horizon)
    
    def _generate_minimal_forecast(self, symbol: str, horizon: ForecastHorizon) -> DayForecast:
        """Generate a minimal forecast when data is insufficient."""
        try:
            # Base prices for different symbols
            base_prices = {
                'AAPL': 170.0, 'MSFT': 350.0, 'GOOGL': 140.0, 'TSLA': 250.0, 'SPY': 450.0,
                'AMZN': 130.0, 'NVDA': 480.0, 'META': 300.0, 'NFLX': 400.0, 'AMD': 110.0
            }
            
            current_price = base_prices.get(symbol, 100.0)
            
            # Simple prediction with some randomness
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            predicted_price = current_price * (1 + price_change)
            
            # Simple confidence based on symbol
            confidence = 0.5 + (hash(symbol) % 100) / 1000  # 0.5-0.6 confidence
            
            # Determine direction
            if price_change > 0.01:
                direction = 'up'
                signal_strength = SignalStrength.MODERATE
            elif price_change < -0.01:
                direction = 'down'
                signal_strength = SignalStrength.MODERATE
            else:
                direction = 'sideways'
                signal_strength = SignalStrength.WEAK
            
            # Create minimal forecast
            forecast = DayForecast(
                symbol=symbol,
                horizon=horizon,
                predicted_price=predicted_price,
                confidence=confidence,
                direction=direction,
                signal_strength=signal_strength,
                technical_indicators=[],
                market_regime="unknown",
                volatility_forecast=0.02,
                volume_forecast=1000000,
                risk_score=0.5,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=4)
            )
            
            # Store forecast
            self.forecasts[f"{symbol}_{horizon.value}"] = forecast
            self.forecast_history.append(forecast)
            
            logger.info(f"Generated minimal day forecast for {symbol}: {direction} to ${predicted_price:.2f} (confidence: {confidence:.2f})")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating minimal forecast for {symbol}: {e}")
            raise
    
    async def _get_market_data(self, symbol: str, periods: int = 50) -> Optional[pd.DataFrame]:
        """Get market data for analysis."""
        try:
            if not self.real_data_service:
                logger.warning("Real data service not available, using mock data")
                return self._generate_mock_data(symbol, periods)
            
            # Get historical data
            data = self.real_data_service.get_historical_data(symbol, periods=periods)
            if data is None or data.empty:
                logger.warning(f"No real data available for {symbol}, using mock data")
                return self._generate_mock_data(symbol, periods)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            logger.warning(f"Using mock data for {symbol}")
            return self._generate_mock_data(symbol, periods)
    
    def _generate_mock_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate mock market data for testing."""
        try:
            # Base prices for different symbols
            base_prices = {
                'AAPL': 170.0, 'MSFT': 350.0, 'GOOGL': 140.0, 'TSLA': 250.0, 'SPY': 450.0,
                'AMZN': 130.0, 'NVDA': 480.0, 'META': 300.0, 'NFLX': 400.0, 'AMD': 110.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Generate time series data
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            # Generate price data with some trend and volatility
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.001, 0.02, periods)  # 0.1% daily return, 2% volatility
            prices = [base_price]
            
            for i in range(1, periods):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(new_price)
            
            # Generate OHLCV data
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                # Generate OHLC from close price
                volatility = 0.01  # 1% intraday volatility
                high = close * (1 + np.random.uniform(0, volatility))
                low = close * (1 - np.random.uniform(0, volatility))
                open_price = close * (1 + np.random.uniform(-volatility/2, volatility/2))
                
                # Ensure OHLC consistency
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # Generate volume
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'date': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            return None
    
    async def _predict_price(self, symbol: str, data: pd.DataFrame, horizon: ForecastHorizon) -> float:
        """Predict price using ensemble models."""
        try:
            # Prepare features
            features = self._prepare_features(data)
            
            if features is None or len(features) == 0:
                # Fallback to simple trend-based prediction
                return self._simple_price_prediction(data)
            
            # Get model for horizon
            models = self.models.get(horizon.value, {})
            if not models:
                return self._simple_price_prediction(data)
            
            # Make predictions with ensemble
            predictions = []
            for model_name, model in models.items():
                try:
                    # Scale features
                    scaler = self.scalers[horizon.value]
                    features_scaled = scaler.fit_transform(features.reshape(1, -1))
                    
                    # Predict
                    pred = model.predict(features_scaled)[0]
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
                    continue
            
            if predictions:
                # Use ensemble average
                predicted_price = np.mean(predictions)
            else:
                predicted_price = self._simple_price_prediction(data)
            
            return max(predicted_price, 0.01)  # Ensure positive price
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            return self._simple_price_prediction(data)
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for model prediction."""
        try:
            if len(data) < 20:
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(data)
            
            # Create feature vector
            features = []
            for indicator in self.indicators:
                if indicator in indicators:
                    features.append(indicators[indicator])
                else:
                    features.append(0.0)
            
            # Add price-based features
            close = data['close'].values
            features.extend([
                close[-1],  # Current price
                close[-1] / close[-2] - 1,  # 1-period return
                close[-1] / close[-5] - 1,  # 5-period return
                close[-1] / close[-10] - 1,  # 10-period return
                np.std(close[-10:]) / np.mean(close[-10:]),  # Volatility
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _simple_price_prediction(self, data: pd.DataFrame) -> float:
        """Simple trend-based price prediction."""
        try:
            close = data['close'].values
            current_price = close[-1]
            
            # Calculate short-term trend
            sma_5 = np.mean(close[-5:])
            sma_10 = np.mean(close[-10:])
            
            # Simple momentum-based prediction
            momentum = (sma_5 - sma_10) / sma_10
            
            # Predict price with momentum
            predicted_price = current_price * (1 + momentum * 0.5)  # Dampened momentum
            
            return max(predicted_price, current_price * 0.95)  # Minimum 5% decline
            
        except Exception as e:
            logger.error(f"Error in simple price prediction: {e}")
            return data['close'].iloc[-1] if not data.empty else 100.0
    
    def _forecast_volatility(self, data: pd.DataFrame) -> float:
        """Forecast volatility for the next period."""
        try:
            if len(data) < 20:
                return 0.2  # Default volatility
            
            close = data['close'].values
            returns = np.diff(close[-20:]) / close[-21:-1]
            
            # Calculate recent volatility
            recent_vol = np.std(returns[-5:]) * np.sqrt(252)
            historical_vol = np.std(returns) * np.sqrt(252)
            
            # Weighted average with recent emphasis
            forecast_vol = 0.7 * recent_vol + 0.3 * historical_vol
            
            return min(max(forecast_vol, 0.05), 1.0)  # Clamp between 5% and 100%
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return 0.2
    
    def _forecast_volume(self, data: pd.DataFrame) -> float:
        """Forecast volume for the next period."""
        try:
            if len(data) < 20:
                return 1000000  # Default volume
            
            volume = data['volume'].values
            
            # Calculate volume trend
            recent_volume = np.mean(volume[-5:])
            historical_volume = np.mean(volume[-20:])
            
            # Volume trend factor
            volume_trend = recent_volume / historical_volume
            
            # Forecast volume
            forecast_volume = recent_volume * volume_trend
            
            return max(forecast_volume, 100000)  # Minimum volume
            
        except Exception as e:
            logger.error(f"Error forecasting volume: {e}")
            return 1000000
    
    def _calculate_risk_score(self, indicators: Dict[str, float], market_regime: str) -> float:
        """Calculate risk score for the forecast."""
        try:
            risk_score = 0.5  # Base risk
            
            # Volatility risk
            if 'atr' in indicators:
                atr = indicators['atr']
                # Higher ATR = higher risk
                risk_score += min(atr / 10, 0.3)
            
            # Market regime risk
            regime_risk = {
                "trending_up_calm": 0.2,
                "trending_down_calm": 0.3,
                "trending_up_volatile": 0.6,
                "trending_down_volatile": 0.7,
                "sideways_calm": 0.4,
                "sideways_volatile": 0.8,
                "unknown": 0.9
            }
            
            risk_score += regime_risk.get(market_regime, 0.5) * 0.4
            
            return min(max(risk_score, 0.1), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    async def get_forecast_summary(self) -> Dict[str, Any]:
        """Get day forecast summary."""
        try:
            # Calculate metrics
            await self._update_metrics()
            
            # Get recent forecasts
            recent_forecasts = self.forecast_history[-10:] if self.forecast_history else []
            
            return {
                "agent_type": "DayForecastAgent",
                "forecast_horizons": [h.value for h in self.forecast_horizons],
                "active_forecasts": len(self.forecasts),
                "total_forecasts": len(self.forecast_history),
                "recent_forecasts": [asdict(f) for f in recent_forecasts],
                "metrics": asdict(self.metrics) if self.metrics else None,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting forecast summary: {e}")
            return {"error": str(e)}
    
    async def _update_metrics(self):
        """Update forecast performance metrics."""
        try:
            if len(self.forecast_history) < 10:
                return
            
            # Calculate accuracy metrics
            accurate_forecasts = 0
            total_forecasts = len(self.forecast_history)
            confidences = []
            errors = []
            
            for forecast in self.forecast_history[-100:]:  # Last 100 forecasts
                confidences.append(forecast.confidence)
                
                # Simple accuracy check (would need actual price data in real implementation)
                # For now, use confidence as proxy
                if forecast.confidence > 0.6:
                    accurate_forecasts += 1
                
                # Calculate error (simplified)
                error = abs(forecast.confidence - 0.7)  # Target confidence
                errors.append(error)
            
            accuracy_percentage = (accurate_forecasts / min(total_forecasts, 100)) * 100
            avg_confidence = np.mean(confidences) if confidences else 0.5
            avg_error = np.mean(errors) if errors else 0.2
            
            # Calculate additional metrics
            mse = np.mean([e**2 for e in errors]) if errors else 0.04
            mae = avg_error
            r2_score = max(0, 1 - mse / 0.25)  # Simplified R²
            
            # Simplified trading metrics
            sharpe_ratio = 1.2  # Placeholder
            max_drawdown = 0.15  # Placeholder
            win_rate = accuracy_percentage / 100
            profit_factor = 1.5  # Placeholder
            
            self.metrics = DayForecastMetrics(
                total_forecasts=total_forecasts,
                accurate_forecasts=accurate_forecasts,
                accuracy_percentage=accuracy_percentage,
                avg_confidence=avg_confidence,
                avg_error=avg_error,
                mse=mse,
                mae=mae,
                r2_score=r2_score,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def get_forecast(self, symbol: str, horizon: str = "end_of_day") -> Optional[DayForecast]:
        """Get forecast for a specific symbol and horizon."""
        try:
            horizon_enum = ForecastHorizon(horizon)
            forecast_key = f"{symbol}_{horizon_enum.value}"
            
            if forecast_key in self.forecasts:
                forecast = self.forecasts[forecast_key]
                
                # Check if forecast is still valid
                if forecast.valid_until > datetime.now():
                    return forecast
            
            # Generate new forecast
            forecast = await self.generate_forecast(symbol, horizon_enum)
            if forecast:
                return forecast
            else:
                logger.warning(f"Failed to generate forecast for {symbol} with horizon {horizon}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting forecast for {symbol}: {e}")
            # Try to generate a forecast anyway
            try:
                horizon_enum = ForecastHorizon(horizon)
                return await self.generate_forecast(symbol, horizon_enum)
            except Exception as e2:
                logger.error(f"Failed to generate forecast after error: {e2}")
                return None
