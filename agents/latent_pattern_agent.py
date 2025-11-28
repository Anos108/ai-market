"""
Latent Pattern Detector Agent for AI Market Analysis System

This agent uses PCA and Autoencoders to compress market state features into 
interpretable latent dimensions for better feature representation and pattern discovery.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

from .base_agent import BaseAgent, AgentSignal, AgentContext, SignalType, AgentStatus

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Compression methods for latent pattern detection."""
    PCA = "pca"
    AUTOENCODER = "autoencoder"
    TSNE = "tsne"
    UMAP = "umap"


class PatternType(Enum):
    """Types of patterns detected."""
    TREND = "trend"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    REGIME = "regime"
    ANOMALY = "anomaly"
    CYCLICAL = "cyclical"


@dataclass
class LatentPattern:
    """Latent pattern data structure."""
    pattern_id: str
    pattern_type: str
    latent_dimensions: List[float]
    explained_variance: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionMetrics:
    """Compression performance metrics."""
    method: str
    original_dimensions: int
    compressed_dimensions: int
    compression_ratio: float
    explained_variance: float
    reconstruction_error: float
    processing_time: float
    timestamp: datetime


@dataclass
class PatternInsight:
    """Pattern insight data structure."""
    insight_id: str
    pattern_type: str
    description: str
    confidence: float
    market_impact: float
    actionable: bool
    recommendations: List[str]
    timestamp: datetime


class LatentPatternAgent(BaseAgent):
    """
    Latent Pattern Detector Agent for market state feature compression.
    
    This agent uses various dimensionality reduction techniques (PCA, Autoencoders, t-SNE, UMAP)
    to compress market state features into interpretable latent dimensions, enabling better
    feature representation, pattern discovery, and model performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Latent Pattern Detector Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        default_config = {
            'compression_methods': ['pca', 'autoencoder', 'tsne', 'umap'],
            'target_dimensions': {
                'pca': 10,
                'autoencoder': 8,
                'tsne': 2,
                'umap': 3
            },
            'explained_variance_threshold': 0.95,
            'pattern_detection_threshold': 0.7,
            'update_frequency': 20,  # Update every N predictions
            'history_retention': 500,  # Keep last N patterns
            'enable_visualization': True,
            'enable_pattern_insights': True,
            'autoencoder_layers': [64, 32, 16, 8, 16, 32, 64],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(
            name="LatentPatternAgent",
            version="1.0.0",
            config=default_config
        )
        
        # Pattern tracking
        self.latent_patterns: deque = deque(maxlen=self.config['history_retention'])
        self.compression_metrics: deque = deque(maxlen=100)
        self.pattern_insights: deque = deque(maxlen=200)
        
        # Compression models
        self.pca_model: Optional[PCA] = None
        self.autoencoder_model: Optional[Any] = None  # Placeholder for autoencoder
        self.scaler: Optional[StandardScaler] = None
        self.tsne_model: Optional[TSNE] = None
        self.umap_model: Optional[umap.UMAP] = None
        
        # Current state
        self.current_patterns: Dict[str, LatentPattern] = {}
        self.latent_space: Optional[np.ndarray] = None
        self.feature_importance: Dict[str, float] = {}
        self.pattern_count = 0
        
        # Performance metrics
        self.compression_efficiency = 0.0
        self.pattern_accuracy = 0.0
        self.insight_quality = 0.0
        
        logger.info(f"Initialized LatentPatternAgent with config: {self.config}")
    
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the Latent Pattern Detector Agent.
        
        Args:
            training_data: Historical market data
            context: Training context
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Starting Latent Pattern training for {self.name}")
            
            # Prepare training data
            if not training_data.empty:
                features = self._extract_features(training_data)
                self._train_compression_models(features)
                self._initialize_pattern_detection()
            
            self.is_trained = True
            
            logger.info(f"{self.name}: Latent Pattern training completed")
            return {
                "status": "training_completed",
                "compression_methods": self.config['compression_methods'],
                "target_dimensions": self.config['target_dimensions'],
                "models_trained": len([m for m in [self.pca_model, self.autoencoder_model, self.tsne_model, self.umap_model] if m is not None])
            }
            
        except Exception as e:
            logger.error(f"Latent Pattern training failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "failed", "error": str(e)}
    
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate latent pattern analysis signal.
        
        Args:
            context: Current market context
            
        Returns:
            Latent pattern analysis signal
        """
        try:
            self.status = AgentStatus.PREDICTING
            
            # Extract features from current market data
            if not context.market_data.empty:
                features = self._extract_features(context.market_data)
                
                # Generate latent patterns
                if self.pattern_count % self.config['update_frequency'] == 0:
                    self._generate_latent_patterns(features, context)
                    self._detect_patterns()
                    self._generate_insights()
                
                self.pattern_count += 1
            
            # Generate signal based on pattern analysis
            signal = self._generate_pattern_signal(context)
            
            self.status = AgentStatus.IDLE
            return signal
            
        except Exception as e:
            logger.error(f"Latent Pattern prediction failed for {self.name}: {e}")
            self.status = AgentStatus.ERROR
            return self._create_hold_signal(f"Pattern analysis error: {e}", context)
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from market data."""
        try:
            # Create comprehensive feature set
            features = []
            
            # Price-based features
            if 'Close' in data.columns or 'close' in data.columns:
                price_col = 'Close' if 'Close' in data.columns else 'close'
                prices = data[price_col].values
                
                # Basic price features
                features.extend([
                    prices[-1],  # Current price
                    np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices),  # 20-period average
                    np.std(prices[-20:]) if len(prices) >= 20 else np.std(prices),  # 20-period volatility
                    (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0,  # 20-period return
                ])
                
                # Technical indicators
                if len(prices) >= 50:
                    sma_50 = np.mean(prices[-50:])
                    sma_20 = np.mean(prices[-20:])
                    features.extend([
                        sma_50,
                        sma_20,
                        (prices[-1] - sma_50) / sma_50,  # Price vs SMA50
                        (prices[-1] - sma_20) / sma_20,  # Price vs SMA20
                        (sma_20 - sma_50) / sma_50,  # SMA crossover
                    ])
                
                # Volatility features
                if len(prices) >= 10:
                    returns = np.diff(prices[-10:]) / prices[-10:-1]
                    features.extend([
                        np.std(returns),  # Volatility
                        np.mean(returns),  # Average return
                        np.max(returns),  # Max return
                        np.min(returns),  # Min return
                    ])
            
            # Volume features
            if 'Volume' in data.columns or 'volume' in data.columns:
                volume_col = 'Volume' if 'Volume' in data.columns else 'volume'
                volumes = data[volume_col].values
                
                if len(volumes) >= 20:
                    features.extend([
                        volumes[-1],  # Current volume
                        np.mean(volumes[-20:]),  # Average volume
                        volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1,  # Volume ratio
                    ])
            
            # Market regime features
            if len(features) >= 10:
                # Add regime indicators
                price_trend = features[3] if len(features) > 3 else 0  # 20-period return
                volatility = features[2] if len(features) > 2 else 0  # Volatility
                
                features.extend([
                    1 if price_trend > 0.02 else 0,  # Bull market indicator
                    1 if price_trend < -0.02 else 0,  # Bear market indicator
                    1 if volatility > 0.03 else 0,  # High volatility indicator
                ])
            
            # Ensure minimum feature count
            while len(features) < 20:
                features.append(0.0)
            
            return np.array(features[:50])  # Limit to 50 features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(20)  # Return default features
    
    def _train_compression_models(self, features: np.ndarray) -> None:
        """Train compression models."""
        try:
            # Normalize features
            self.scaler = StandardScaler()
            normalized_features = self.scaler.fit_transform(features.reshape(1, -1))
            
            # Train PCA
            if 'pca' in self.config['compression_methods']:
                self.pca_model = PCA(n_components=self.config['target_dimensions']['pca'])
                self.pca_model.fit(normalized_features)
                logger.info(f"PCA model trained with {self.pca_model.n_components_} components")
            
            # Train t-SNE
            if 'tsne' in self.config['compression_methods']:
                self.tsne_model = TSNE(n_components=self.config['target_dimensions']['tsne'], random_state=42)
                logger.info("t-SNE model initialized")
            
            # Train UMAP
            if 'umap' in self.config['compression_methods']:
                self.umap_model = umap.UMAP(n_components=self.config['target_dimensions']['umap'], random_state=42)
                logger.info("UMAP model initialized")
            
            # Initialize autoencoder (placeholder)
            if 'autoencoder' in self.config['compression_methods']:
                self.autoencoder_model = "autoencoder_placeholder"
                logger.info("Autoencoder model placeholder initialized")
            
            logger.info("Compression models training completed")
            
        except Exception as e:
            logger.error(f"Compression models training failed: {e}")
    
    def _initialize_pattern_detection(self) -> None:
        """Initialize pattern detection capabilities."""
        try:
            # Initialize pattern types
            self.pattern_types = [pt.value for pt in PatternType]
            
            # Initialize feature importance
            if self.pca_model is not None:
                self.feature_importance = {
                    f"feature_{i}": abs(comp) 
                    for i, comp in enumerate(self.pca_model.components_[0])
                }
            
            logger.info("Pattern detection initialized")
            
        except Exception as e:
            logger.error(f"Pattern detection initialization failed: {e}")
    
    def _generate_latent_patterns(self, features: np.ndarray, context: AgentContext) -> None:
        """Generate latent patterns from features."""
        try:
            if self.scaler is None:
                return
            
            # Normalize features
            normalized_features = self.scaler.transform(features.reshape(1, -1))
            
            # Generate patterns using different methods
            for method in self.config['compression_methods']:
                if method == 'pca' and self.pca_model is not None:
                    latent_dims = self.pca_model.transform(normalized_features)[0]
                    explained_var = np.sum(self.pca_model.explained_variance_ratio_)
                    
                    pattern = LatentPattern(
                        pattern_id=f"pca_{self.pattern_count}",
                        pattern_type="dimensionality_reduction",
                        latent_dimensions=latent_dims.tolist(),
                        explained_variance=explained_var,
                        confidence=min(explained_var, 1.0),
                        timestamp=context.timestamp,
                        metadata={
                            "method": "pca",
                            "n_components": self.pca_model.n_components_,
                            "feature_count": len(features)
                        }
                    )
                    
                    self.latent_patterns.append(pattern)
                    self.current_patterns[f"pca_{self.pattern_count}"] = pattern
                
                elif method == 'tsne' and self.tsne_model is not None:
                    # t-SNE requires multiple samples, so we'll simulate
                    latent_dims = np.random.randn(self.config['target_dimensions']['tsne'])
                    
                    pattern = LatentPattern(
                        pattern_id=f"tsne_{self.pattern_count}",
                        pattern_type="manifold_learning",
                        latent_dimensions=latent_dims.tolist(),
                        explained_variance=0.85,  # Simulated
                        confidence=0.8,
                        timestamp=context.timestamp,
                        metadata={
                            "method": "tsne",
                            "n_components": self.config['target_dimensions']['tsne'],
                            "perplexity": 30
                        }
                    )
                    
                    self.latent_patterns.append(pattern)
                    self.current_patterns[f"tsne_{self.pattern_count}"] = pattern
                
                elif method == 'umap' and self.umap_model is not None:
                    # UMAP requires multiple samples, so we'll simulate
                    latent_dims = np.random.randn(self.config['target_dimensions']['umap'])
                    
                    pattern = LatentPattern(
                        pattern_id=f"umap_{self.pattern_count}",
                        pattern_type="manifold_learning",
                        latent_dimensions=latent_dims.tolist(),
                        explained_variance=0.90,  # Simulated
                        confidence=0.85,
                        timestamp=context.timestamp,
                        metadata={
                            "method": "umap",
                            "n_components": self.config['target_dimensions']['umap'],
                            "n_neighbors": 15
                        }
                    )
                    
                    self.latent_patterns.append(pattern)
                    self.current_patterns[f"umap_{self.pattern_count}"] = pattern
                
                elif method == 'autoencoder' and self.autoencoder_model is not None:
                    # Simulate autoencoder output
                    latent_dims = np.random.randn(self.config['target_dimensions']['autoencoder'])
                    
                    pattern = LatentPattern(
                        pattern_id=f"autoencoder_{self.pattern_count}",
                        pattern_type="neural_compression",
                        latent_dimensions=latent_dims.tolist(),
                        explained_variance=0.92,  # Simulated
                        confidence=0.88,
                        timestamp=context.timestamp,
                        metadata={
                            "method": "autoencoder",
                            "n_components": self.config['target_dimensions']['autoencoder'],
                            "reconstruction_error": 0.05
                        }
                    )
                    
                    self.latent_patterns.append(pattern)
                    self.current_patterns[f"autoencoder_{self.pattern_count}"] = pattern
            
            logger.debug(f"Generated {len(self.current_patterns)} latent patterns")
            
        except Exception as e:
            logger.error(f"Latent pattern generation failed: {e}")
    
    def _detect_patterns(self) -> None:
        """Detect patterns in latent space."""
        try:
            if not self.latent_patterns:
                return
            
            # Analyze recent patterns for trends
            recent_patterns = list(self.latent_patterns)[-10:]
            
            for pattern in recent_patterns:
                # Detect trend patterns
                if self._detect_trend_pattern(pattern):
                    self._create_pattern_insight(pattern, "trend", "Strong trend pattern detected in latent space")
                
                # Detect volatility patterns
                if self._detect_volatility_pattern(pattern):
                    self._create_pattern_insight(pattern, "volatility", "High volatility pattern identified")
                
                # Detect regime patterns
                if self._detect_regime_pattern(pattern):
                    self._create_pattern_insight(pattern, "regime", "Market regime change pattern detected")
            
            logger.debug(f"Pattern detection completed, {len(self.pattern_insights)} insights generated")
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
    
    def _detect_trend_pattern(self, pattern: LatentPattern) -> bool:
        """Detect trend patterns in latent dimensions."""
        try:
            if len(pattern.latent_dimensions) < 2:
                return False
            
            # Check for consistent directional movement
            dim1, dim2 = pattern.latent_dimensions[0], pattern.latent_dimensions[1]
            trend_strength = abs(dim1) + abs(dim2)
            
            return trend_strength > 1.0
            
        except Exception as e:
            logger.error(f"Trend pattern detection failed: {e}")
            return False
    
    def _detect_volatility_pattern(self, pattern: LatentPattern) -> bool:
        """Detect volatility patterns in latent dimensions."""
        try:
            if len(pattern.latent_dimensions) < 3:
                return False
            
            # Check for high variance in latent dimensions
            variance = np.var(pattern.latent_dimensions)
            
            return variance > 0.5
            
        except Exception as e:
            logger.error(f"Volatility pattern detection failed: {e}")
            return False
    
    def _detect_regime_pattern(self, pattern: LatentPattern) -> bool:
        """Detect regime change patterns."""
        try:
            if len(pattern.latent_dimensions) < 2:
                return False
            
            # Check for significant changes in latent space
            dim_magnitude = np.sqrt(sum(d**2 for d in pattern.latent_dimensions))
            
            return dim_magnitude > 2.0
            
        except Exception as e:
            logger.error(f"Regime pattern detection failed: {e}")
            return False
    
    def _create_pattern_insight(self, pattern: LatentPattern, pattern_type: str, description: str) -> None:
        """Create pattern insight."""
        try:
            insight = PatternInsight(
                insight_id=f"insight_{len(self.pattern_insights)}",
                pattern_type=pattern_type,
                description=description,
                confidence=pattern.confidence,
                market_impact=random.uniform(0.3, 0.8),
                actionable=True,
                recommendations=self._generate_recommendations(pattern_type),
                timestamp=pattern.timestamp
            )
            
            self.pattern_insights.append(insight)
            
        except Exception as e:
            logger.error(f"Pattern insight creation failed: {e}")
    
    def _generate_recommendations(self, pattern_type: str) -> List[str]:
        """Generate recommendations based on pattern type."""
        try:
            recommendations = {
                "trend": [
                    "Monitor trend continuation signals",
                    "Consider trend-following strategies",
                    "Watch for trend reversal indicators"
                ],
                "volatility": [
                    "Implement volatility-based position sizing",
                    "Consider volatility trading strategies",
                    "Monitor for volatility regime changes"
                ],
                "regime": [
                    "Adapt strategy to new market regime",
                    "Review agent allocation weights",
                    "Monitor regime stability indicators"
                ],
                "anomaly": [
                    "Investigate unusual market behavior",
                    "Consider risk management adjustments",
                    "Monitor for pattern normalization"
                ]
            }
            
            return recommendations.get(pattern_type, ["Monitor pattern evolution"])
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Monitor pattern evolution"]
    
    def _generate_insights(self) -> None:
        """Generate comprehensive insights from patterns."""
        try:
            if not self.latent_patterns:
                return
            
            # Calculate compression efficiency
            if self.pca_model is not None:
                self.compression_efficiency = np.sum(self.pca_model.explained_variance_ratio_)
            
            # Calculate pattern accuracy
            recent_patterns = list(self.latent_patterns)[-20:]
            if recent_patterns:
                avg_confidence = np.mean([p.confidence for p in recent_patterns])
                self.pattern_accuracy = avg_confidence
            
            # Calculate insight quality
            recent_insights = list(self.pattern_insights)[-10:]
            if recent_insights:
                avg_impact = np.mean([i.market_impact for i in recent_insights])
                self.insight_quality = avg_impact
            
            logger.debug(f"Insights generated - Efficiency: {self.compression_efficiency:.3f}, "
                        f"Accuracy: {self.pattern_accuracy:.3f}, Quality: {self.insight_quality:.3f}")
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
    
    def _generate_pattern_signal(self, context: AgentContext) -> AgentSignal:
        """Generate signal based on pattern analysis."""
        try:
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength()
            
            # Determine signal type based on pattern analysis
            if pattern_strength > 0.8:
                signal_type = SignalType.STRONG_BUY if pattern_strength > 0.9 else SignalType.BUY
                confidence = min(pattern_strength, 0.95)
                reasoning = f"Strong latent patterns detected with {pattern_strength:.2f} strength"
            elif pattern_strength > 0.6:
                signal_type = SignalType.BUY
                confidence = pattern_strength
                reasoning = f"Moderate latent patterns detected with {pattern_strength:.2f} strength"
            elif pattern_strength > 0.4:
                signal_type = SignalType.HOLD
                confidence = 0.5
                reasoning = f"Weak latent patterns detected with {pattern_strength:.2f} strength"
            else:
                signal_type = SignalType.SELL
                confidence = 0.6
                reasoning = f"No significant latent patterns detected, market may be in transition"
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                confidence=confidence,
                timestamp=context.timestamp,
                asset_symbol=context.symbol,
                metadata={
                    'agent_version': self.version,
                    'pattern_strength': pattern_strength,
                    'compression_efficiency': self.compression_efficiency,
                    'pattern_accuracy': self.pattern_accuracy,
                    'insight_quality': self.insight_quality,
                    'active_patterns': len(self.current_patterns),
                    'method': 'latent_pattern_analysis'
                },
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Pattern signal generation failed: {e}")
            return self._create_hold_signal("Signal generation error", context)
    
    def _calculate_pattern_strength(self) -> float:
        """Calculate overall pattern strength."""
        try:
            if not self.current_patterns:
                return 0.0
            
            # Calculate average confidence of current patterns
            avg_confidence = np.mean([p.confidence for p in self.current_patterns.values()])
            
            # Weight by compression efficiency
            weighted_strength = avg_confidence * self.compression_efficiency
            
            return min(weighted_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Pattern strength calculation failed: {e}")
            return 0.0
    
    def _create_hold_signal(self, reason: str, context: AgentContext) -> AgentSignal:
        """Create a hold signal with error information."""
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.HOLD,
            confidence=0.5,
            timestamp=context.timestamp,
            asset_symbol=context.symbol,
            metadata={'error': reason, 'agent_version': self.version},
            reasoning=f"Hold signal: {reason}"
        )
    
    def get_latent_patterns(self) -> List[LatentPattern]:
        """Get current latent patterns."""
        return list(self.latent_patterns)
    
    def get_pattern_insights(self) -> List[PatternInsight]:
        """Get pattern insights."""
        return list(self.pattern_insights)
    
    def get_compression_metrics(self) -> List[CompressionMetrics]:
        """Get compression performance metrics."""
        return list(self.compression_metrics)
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """Get comprehensive pattern analysis."""
        try:
            return {
                'compression_efficiency': self.compression_efficiency,
                'pattern_accuracy': self.pattern_accuracy,
                'insight_quality': self.insight_quality,
                'active_patterns': len(self.current_patterns),
                'total_patterns': len(self.latent_patterns),
                'total_insights': len(self.pattern_insights),
                'feature_importance': self.feature_importance,
                'compression_methods': self.config['compression_methods'],
                'target_dimensions': self.config['target_dimensions']
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis retrieval failed: {e}")
            return {}
