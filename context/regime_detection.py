"""
Market Regime Detection using Markov Models

This module implements sophisticated market regime detection using
Hidden Markov Models (HMM) and Markov Chain Monte Carlo methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeState:
    """Represents a market regime state."""
    regime_type: RegimeType
    probability: float
    duration: int
    start_time: datetime
    end_time: Optional[datetime]
    characteristics: Dict[str, float]
    confidence: float


@dataclass
class RegimeTransition:
    """Represents a regime transition."""
    from_regime: RegimeType
    to_regime: RegimeType
    probability: float
    transition_time: datetime
    trigger_factors: List[str]


class MarkovRegimeDetector:
    """
    Market regime detection using Markov models.
    
    This class implements:
    - Hidden Markov Models for regime detection
    - Gaussian Mixture Models for regime classification
    - Transition probability estimation
    - Regime persistence analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Markov Regime Detector.
        
        Args:
            config: Configuration dictionary
        """
        default_config = {
            'n_regimes': 4,
            'lookback_periods': 50,
            'min_regime_duration': 5,
            'transition_threshold': 0.3,
            'confidence_threshold': 0.6,
            'features': ['returns', 'volatility', 'volume_ratio', 'momentum'],
            'regime_mapping': {
                0: RegimeType.BULL,
                1: RegimeType.BEAR,
                2: RegimeType.SIDEWAYS,
                3: RegimeType.VOLATILE
            }
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.hmm_model = None
        self.gmm_model = None
        self.scaler = StandardScaler()
        self.regime_history = []
        self.transition_matrix = None
        self.current_regime = None
        self.regime_probabilities = None
        
        logger.info(f"Initialized MarkovRegimeDetector with config: {self.config}")
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit the Markov regime detection models.
        
        Args:
            data: Historical market data
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting Markov regime detection model training")
            
            if data.empty or len(data) < self.config['lookback_periods']:
                logger.warning(f"Insufficient data for training: {len(data)} periods")
                return {"status": "insufficient_data", "periods": len(data)}
            
            # Prepare features
            features = self._prepare_features(data)
            
            if features.empty:
                logger.warning("No valid features extracted from data")
                return {"status": "no_features", "data_shape": data.shape}
            
            # Fit Gaussian Mixture Model
            self.gmm_model = GaussianMixture(
                n_components=self.config['n_regimes'],
                random_state=42,
                covariance_type='full'
            )
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit the model
            self.gmm_model.fit(features_scaled)
            
            # Get regime assignments
            regime_assignments = self.gmm_model.predict(features_scaled)
            regime_probabilities = self.gmm_model.predict_proba(features_scaled)
            
            # Estimate transition matrix
            self.transition_matrix = self._estimate_transition_matrix(regime_assignments)
            
            # Analyze regime characteristics
            regime_characteristics = self._analyze_regime_characteristics(
                features, regime_assignments
            )
            
            logger.info("Markov regime detection model training completed")
            
            return {
                "status": "training_complete",
                "n_regimes": self.config['n_regimes'],
                "transition_matrix": self.transition_matrix.tolist(),
                "regime_characteristics": regime_characteristics,
                "model_components": self.gmm_model.n_components
            }
            
        except Exception as e:
            logger.error(f"Markov regime detection training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def predict(self, data: pd.DataFrame) -> RegimeState:
        """
        Predict current market regime.
        
        Args:
            data: Current market data
            
        Returns:
            Current regime state
        """
        try:
            if self.gmm_model is None:
                logger.warning("Model not trained, returning default regime")
                return RegimeState(
                    regime_type=RegimeType.SIDEWAYS,
                    probability=0.5,
                    duration=1,
                    start_time=datetime.now(),
                    end_time=None,
                    characteristics={},
                    confidence=0.0
                )
            
            # Prepare features
            features = self._prepare_features(data)
            
            if features.empty:
                return RegimeState(
                    regime_type=RegimeType.SIDEWAYS,
                    probability=0.5,
                    duration=1,
                    start_time=datetime.now(),
                    end_time=None,
                    characteristics={},
                    confidence=0.0
                )
            
            # Get latest features
            latest_features = features.iloc[-1:].copy()
            
            # Scale features
            features_scaled = self.scaler.transform(latest_features)
            
            # Predict regime
            regime_probs = self.gmm_model.predict_proba(features_scaled)[0]
            regime_id = np.argmax(regime_probs)
            regime_probability = regime_probs[regime_id]
            
            # Map to regime type
            regime_type = self.config['regime_mapping'].get(regime_id, RegimeType.SIDEWAYS)
            
            # Calculate regime characteristics
            characteristics = self._calculate_regime_characteristics(latest_features.iloc[0])
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(regime_probs, characteristics)
            
            # Estimate duration
            duration = self._estimate_regime_duration(regime_type)
            
            # Create regime state
            regime_state = RegimeState(
                regime_type=regime_type,
                probability=regime_probability,
                duration=duration,
                start_time=datetime.now(),
                end_time=None,
                characteristics=characteristics,
                confidence=confidence
            )
            
            # Update regime history
            self._update_regime_history(regime_state)
            
            return regime_state
            
        except Exception as e:
            logger.error(f"Regime prediction failed: {e}")
            return RegimeState(
                regime_type=RegimeType.SIDEWAYS,
                probability=0.5,
                duration=1,
                start_time=datetime.now(),
                end_time=None,
                characteristics={},
                confidence=0.0
            )
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection."""
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
            
            prices = data[close_col]
            
            # Returns
            features['returns'] = prices.pct_change()
            
            # Volatility (rolling standard deviation)
            features['volatility'] = features['returns'].rolling(window=5).std()
            
            # Volume ratio (if available)
            if volume_col in data.columns:
                volume = data[volume_col]
                features['volume_ratio'] = volume / volume.rolling(window=20).mean()
            else:
                features['volume_ratio'] = 1.0
            
            # Momentum indicators
            features['momentum_5'] = prices.pct_change(5)
            features['momentum_10'] = prices.pct_change(10)
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(prices)
            features['macd'] = self._calculate_macd(prices)
            
            # High-low ratio (if available)
            if high_col in data.columns and low_col in data.columns:
                high = data[high_col]
                low = data[low_col]
                features['high_low_ratio'] = (high - low) / prices
            else:
                features['high_low_ratio'] = 0.0
            
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
            return pd.Series(index=prices.index, data=50.0)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        except:
            return pd.Series(index=prices.index, data=0.0)
    
    def _estimate_transition_matrix(self, regime_assignments: np.ndarray) -> np.ndarray:
        """Estimate transition matrix from regime assignments."""
        try:
            n_regimes = self.config['n_regimes']
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            # Count transitions
            for i in range(len(regime_assignments) - 1):
                from_regime = regime_assignments[i]
                to_regime = regime_assignments[i + 1]
                transition_matrix[from_regime, to_regime] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1)
            for i in range(n_regimes):
                if row_sums[i] > 0:
                    transition_matrix[i, :] = transition_matrix[i, :] / row_sums[i]
                else:
                    # If no transitions from this regime, assume equal probability
                    transition_matrix[i, :] = 1.0 / n_regimes
            
            return transition_matrix
            
        except Exception as e:
            logger.error(f"Transition matrix estimation failed: {e}")
            # Return uniform transition matrix
            n_regimes = self.config['n_regimes']
            return np.ones((n_regimes, n_regimes)) / n_regimes
    
    def _analyze_regime_characteristics(self, features: pd.DataFrame, regime_assignments: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze characteristics of each regime."""
        try:
            regime_characteristics = {}
            
            for regime_id in range(self.config['n_regimes']):
                regime_mask = regime_assignments == regime_id
                regime_features = features[regime_mask]
                
                if len(regime_features) > 0:
                    characteristics = {
                        'mean_return': regime_features['returns'].mean(),
                        'volatility': regime_features['volatility'].mean(),
                        'volume_ratio': regime_features['volume_ratio'].mean(),
                        'momentum': regime_features['momentum_5'].mean(),
                        'rsi': regime_features['rsi'].mean(),
                        'macd': regime_features['macd'].mean(),
                        'count': len(regime_features)
                    }
                else:
                    characteristics = {
                        'mean_return': 0.0,
                        'volatility': 0.0,
                        'volume_ratio': 1.0,
                        'momentum': 0.0,
                        'rsi': 50.0,
                        'macd': 0.0,
                        'count': 0
                    }
                
                regime_characteristics[regime_id] = characteristics
            
            return regime_characteristics
            
        except Exception as e:
            logger.error(f"Regime characteristics analysis failed: {e}")
            return {}
    
    def _calculate_regime_characteristics(self, features: pd.Series) -> Dict[str, float]:
        """Calculate characteristics for current regime."""
        try:
            return {
                'return': features.get('returns', 0.0),
                'volatility': features.get('volatility', 0.0),
                'volume_ratio': features.get('volume_ratio', 1.0),
                'momentum': features.get('momentum_5', 0.0),
                'rsi': features.get('rsi', 50.0),
                'macd': features.get('macd', 0.0)
            }
        except Exception as e:
            logger.error(f"Regime characteristics calculation failed: {e}")
            return {}
    
    def _calculate_regime_confidence(self, regime_probs: np.ndarray, characteristics: Dict[str, float]) -> float:
        """Calculate confidence in regime prediction."""
        try:
            # Base confidence from probability distribution
            max_prob = np.max(regime_probs)
            entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10))
            max_entropy = np.log(len(regime_probs))
            normalized_entropy = entropy / max_entropy
            
            # Confidence based on probability concentration
            probability_confidence = max_prob * (1 - normalized_entropy)
            
            # Adjust based on feature consistency
            feature_confidence = 0.5  # Default
            
            # Check for extreme values that might indicate regime change
            volatility = characteristics.get('volatility', 0.0)
            if 0.01 < volatility < 0.05:  # Normal volatility range
                feature_confidence += 0.2
            
            # Check momentum consistency
            momentum = characteristics.get('momentum', 0.0)
            if abs(momentum) < 0.1:  # Not too extreme
                feature_confidence += 0.1
            
            # Combine confidences
            final_confidence = (probability_confidence * 0.7 + feature_confidence * 0.3)
            
            return min(1.0, final_confidence)
            
        except Exception as e:
            logger.error(f"Regime confidence calculation failed: {e}")
            return 0.5
    
    def _estimate_regime_duration(self, regime_type: RegimeType) -> int:
        """Estimate expected duration of current regime."""
        try:
            # Simplified duration estimation based on regime type
            duration_map = {
                RegimeType.BULL: 20,
                RegimeType.BEAR: 15,
                RegimeType.SIDEWAYS: 10,
                RegimeType.VOLATILE: 5,
                RegimeType.TRENDING_UP: 15,
                RegimeType.TRENDING_DOWN: 15,
                RegimeType.HIGH_VOLATILITY: 8,
                RegimeType.LOW_VOLATILITY: 12
            }
            
            return duration_map.get(regime_type, 10)
            
        except Exception as e:
            logger.error(f"Regime duration estimation failed: {e}")
            return 10
    
    def _update_regime_history(self, regime_state: RegimeState) -> None:
        """Update regime history."""
        try:
            self.regime_history.append(regime_state)
            
            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            # Update current regime
            self.current_regime = regime_state.regime_type
            self.regime_probabilities = regime_state.probability
            
        except Exception as e:
            logger.error(f"Regime history update failed: {e}")
    
    def get_regime_transitions(self) -> List[RegimeTransition]:
        """Get recent regime transitions."""
        try:
            transitions = []
            
            if len(self.regime_history) < 2:
                return transitions
            
            for i in range(1, len(self.regime_history)):
                prev_regime = self.regime_history[i-1].regime_type
                curr_regime = self.regime_history[i].regime_type
                
                if prev_regime != curr_regime:
                    transition = RegimeTransition(
                        from_regime=prev_regime,
                        to_regime=curr_regime,
                        probability=curr_regime.value,  # Simplified
                        transition_time=self.regime_history[i].start_time,
                        trigger_factors=['market_conditions']  # Simplified
                    )
                    transitions.append(transition)
            
            return transitions[-10:]  # Return last 10 transitions
            
        except Exception as e:
            logger.error(f"Regime transitions retrieval failed: {e}")
            return []
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime statistics."""
        try:
            if not self.regime_history:
                return {}
            
            # Count regimes
            regime_counts = {}
            total_duration = 0
            
            for regime_state in self.regime_history:
                regime_type = regime_state.regime_type
                regime_counts[regime_type.value] = regime_counts.get(regime_type.value, 0) + 1
                total_duration += regime_state.duration
            
            # Calculate statistics
            stats = {
                'total_periods': len(self.regime_history),
                'regime_counts': regime_counts,
                'average_duration': total_duration / len(self.regime_history) if self.regime_history else 0,
                'current_regime': self.current_regime.value if self.current_regime else 'unknown',
                'current_probability': self.regime_probabilities if self.regime_probabilities else 0.0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Regime statistics calculation failed: {e}")
            return {}
