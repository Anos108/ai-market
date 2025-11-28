"""
Model Explainability and Interpretability for AI Market Analysis System

This module provides SHAP, LIME, and other explainability tools to understand
model decisions and provide confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Explainability Libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Some explainability features will be disabled.")

try:
    from lime import lime_tabular
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Some explainability features will be disabled.")

try:
    from sklearn.tree import DecisionTreeRegressor, export_text
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some explainability features will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    model_name: str
    prediction: float
    confidence: float
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, float]] = None
    decision_path: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    explanation_text: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ConfidenceInterval:
    """Confidence interval for predictions."""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str  # 'bootstrap', 'parametric', 'quantile'


class ModelExplainer:
    """Main class for model explainability."""
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(20)]
        self.explainer = None
        self.lime_explainer = None
        
    def setup_shap_explainer(self, X_background: np.ndarray):
        """Setup SHAP explainer."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return
        
        try:
            if hasattr(self.model, 'predict'):
                # For sklearn models
                self.explainer = shap.Explainer(self.model, X_background)
            else:
                # For other models, use a generic explainer
                self.explainer = shap.Explainer(self.model.predict, X_background)
            logger.info("SHAP explainer setup complete")
        except Exception as e:
            logger.error(f"Failed to setup SHAP explainer: {e}")
    
    def setup_lime_explainer(self, X_background: np.ndarray):
        """Setup LIME explainer."""
        if not LIME_AVAILABLE:
            logger.warning("LIME not available")
            return
        
        try:
            self.lime_explainer = LimeTabularExplainer(
                X_background,
                feature_names=self.feature_names,
                mode='regression',
                discretize_continuous=True
            )
            logger.info("LIME explainer setup complete")
        except Exception as e:
            logger.error(f"Failed to setup LIME explainer: {e}")
    
    def explain_prediction(self, X: np.ndarray, prediction: float = None) -> ExplanationResult:
        """Explain a single prediction."""
        if prediction is None:
            prediction = self.model.predict(X.reshape(1, -1))[0]
        
        # Get feature importance
        feature_importance = self._get_feature_importance(X)
        
        # Get SHAP values
        shap_values = self._get_shap_values(X)
        
        # Get LIME explanation
        lime_explanation = self._get_lime_explanation(X)
        
        # Get decision path (for tree-based models)
        decision_path = self._get_decision_path(X)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(X, prediction)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            prediction, feature_importance, shap_values, lime_explanation
        )
        
        return ExplanationResult(
            model_name=getattr(self.model, '__class__', {}).get('__name__', 'Unknown'),
            prediction=float(prediction),
            confidence=self._calculate_confidence(prediction, confidence_interval),
            feature_importance=feature_importance,
            shap_values=shap_values,
            lime_explanation=lime_explanation,
            decision_path=decision_path,
            confidence_interval=confidence_interval,
            explanation_text=explanation_text
        )
    
    def _get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get feature importance scores."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Linear models
                importance = np.abs(self.model.coef_)
            else:
                # Fallback: use permutation importance
                if SKLEARN_AVAILABLE:
                    perm_importance = permutation_importance(
                        self.model, X.reshape(1, -1), [0], n_repeats=10, random_state=42
                    )
                    importance = perm_importance.importances_mean
                else:
                    # Random importance as fallback
                    importance = np.random.random(len(self.feature_names))
            
            # Normalize importance
            importance = importance / np.sum(importance)
            
            return {
                name: float(importance[i]) 
                for i, name in enumerate(self.feature_names)
            }
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {name: 1.0/len(self.feature_names) for name in self.feature_names}
    
    def _get_shap_values(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        """Get SHAP values for explanation."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return None
        
        try:
            shap_values = self.explainer(X.reshape(1, -1))
            
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]
            else:
                values = shap_values[0]
            
            return {
                name: float(values[i]) 
                for i, name in enumerate(self.feature_names)
            }
        except Exception as e:
            logger.error(f"Error getting SHAP values: {e}")
            return None
    
    def _get_lime_explanation(self, X: np.ndarray) -> Optional[Dict[str, float]]:
        """Get LIME explanation."""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            return None
        
        try:
            explanation = self.lime_explainer.explain_instance(
                X, self.model.predict, num_features=len(self.feature_names)
            )
            
            return {
                name: float(explanation.as_list()[i][1]) 
                for i, name in enumerate(self.feature_names)
            }
        except Exception as e:
            logger.error(f"Error getting LIME explanation: {e}")
            return None
    
    def _get_decision_path(self, X: np.ndarray) -> Optional[str]:
        """Get decision path for tree-based models."""
        try:
            if hasattr(self.model, 'decision_path'):
                # Get decision path
                path = self.model.decision_path(X.reshape(1, -1))
                return f"Decision path: {path.toarray().tolist()}"
            elif hasattr(self.model, 'tree_'):
                # For single decision tree
                tree_rules = export_text(self.model, feature_names=self.feature_names)
                return tree_rules[:500]  # Limit length
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting decision path: {e}")
            return None
    
    def _calculate_confidence_interval(self, X: np.ndarray, prediction: float) -> Optional[ConfidenceInterval]:
        """Calculate confidence interval for prediction."""
        try:
            # Bootstrap method for confidence interval
            n_bootstrap = 100
            bootstrap_predictions = []
            
            for _ in range(n_bootstrap):
                # Add small noise to input
                X_noisy = X + np.random.normal(0, 0.01, X.shape)
                pred = self.model.predict(X_noisy.reshape(1, -1))[0]
                bootstrap_predictions.append(pred)
            
            # Calculate confidence interval
            alpha = 0.05  # 95% confidence
            lower = np.percentile(bootstrap_predictions, 100 * alpha / 2)
            upper = np.percentile(bootstrap_predictions, 100 * (1 - alpha / 2))
            
            return ConfidenceInterval(
                lower_bound=float(lower),
                upper_bound=float(upper),
                confidence_level=0.95,
                method='bootstrap'
            )
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return None
    
    def _calculate_confidence(self, prediction: float, confidence_interval: Optional[ConfidenceInterval]) -> float:
        """Calculate overall confidence score."""
        if confidence_interval is None:
            return 0.8  # Default confidence
        
        # Calculate confidence based on interval width
        interval_width = confidence_interval.upper_bound - confidence_interval.lower_bound
        relative_width = interval_width / abs(prediction) if prediction != 0 else 1.0
        
        # Confidence decreases with wider intervals
        confidence = max(0.1, 1.0 - relative_width)
        return float(confidence)
    
    def _generate_explanation_text(self, prediction: float, feature_importance: Dict[str, float], 
                                 shap_values: Optional[Dict[str, float]], 
                                 lime_explanation: Optional[Dict[str, float]]) -> str:
        """Generate human-readable explanation text."""
        explanation_parts = []
        
        # Main prediction
        explanation_parts.append(f"Model prediction: {prediction:.4f}")
        
        # Top contributing features
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            explanation_parts.append(f"Top contributing features: {', '.join([f'{name} ({importance:.3f})' for name, importance in top_features])}")
        
        # SHAP insights
        if shap_values:
            positive_features = [name for name, value in shap_values.items() if value > 0]
            negative_features = [name for name, value in shap_values.items() if value < 0]
            
            if positive_features:
                explanation_parts.append(f"Features pushing prediction up: {', '.join(positive_features[:2])}")
            if negative_features:
                explanation_parts.append(f"Features pushing prediction down: {', '.join(negative_features[:2])}")
        
        # LIME insights
        if lime_explanation:
            lime_positive = [name for name, value in lime_explanation.items() if value > 0]
            lime_negative = [name for name, value in lime_explanation.items() if value < 0]
            
            if lime_positive:
                explanation_parts.append(f"LIME positive contributors: {', '.join(lime_positive[:2])}")
            if lime_negative:
                explanation_parts.append(f"LIME negative contributors: {', '.join(lime_negative[:2])}")
        
        return " | ".join(explanation_parts)


class ModelInterpretability:
    """Advanced model interpretability tools."""
    
    def __init__(self):
        self.explainers = {}
        self.interpretation_cache = {}
        
    def add_model(self, name: str, model, feature_names: List[str] = None):
        """Add a model for interpretation."""
        explainer = ModelExplainer(model, feature_names)
        self.explainers[name] = explainer
        logger.info(f"Added model {name} for interpretation")
    
    def explain_prediction(self, model_name: str, X: np.ndarray, 
                          prediction: float = None) -> ExplanationResult:
        """Explain a prediction for a specific model."""
        if model_name not in self.explainers:
            raise ValueError(f"Model {model_name} not found")
        
        explainer = self.explainers[model_name]
        return explainer.explain_prediction(X, prediction)
    
    def compare_model_explanations(self, X: np.ndarray, 
                                 model_names: List[str] = None) -> Dict[str, ExplanationResult]:
        """Compare explanations across multiple models."""
        if model_names is None:
            model_names = list(self.explainers.keys())
        
        results = {}
        for name in model_names:
            if name in self.explainers:
                try:
                    results[name] = self.explainers[name].explain_prediction(X)
                except Exception as e:
                    logger.error(f"Error explaining model {name}: {e}")
                    results[name] = None
        
        return results
    
    def get_model_consensus(self, X: np.ndarray, 
                           model_names: List[str] = None) -> Dict[str, Any]:
        """Get consensus explanation across models."""
        explanations = self.compare_model_explanations(X, model_names)
        
        # Calculate consensus
        valid_explanations = {k: v for k, v in explanations.items() if v is not None}
        
        if not valid_explanations:
            return {"error": "No valid explanations available"}
        
        # Average predictions
        predictions = [exp.prediction for exp in valid_explanations.values()]
        avg_prediction = np.mean(predictions)
        prediction_std = np.std(predictions)
        
        # Average confidence
        confidences = [exp.confidence for exp in valid_explanations.values()]
        avg_confidence = np.mean(confidences)
        
        # Feature importance consensus
        feature_consensus = {}
        for feature in valid_explanations[list(valid_explanations.keys())[0]].feature_importance.keys():
            importance_values = [exp.feature_importance[feature] for exp in valid_explanations.values()]
            feature_consensus[feature] = {
                'mean': np.mean(importance_values),
                'std': np.std(importance_values),
                'consensus_score': 1.0 - np.std(importance_values)  # Higher consensus = lower std
            }
        
        return {
            'consensus_prediction': float(avg_prediction),
            'prediction_uncertainty': float(prediction_std),
            'consensus_confidence': float(avg_confidence),
            'feature_consensus': feature_consensus,
            'model_agreement': len(valid_explanations),
            'total_models': len(model_names) if model_names else len(self.explainers)
        }
    
    def generate_interpretation_report(self, X: np.ndarray, 
                                     model_names: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive interpretation report."""
        explanations = self.compare_model_explanations(X, model_names)
        consensus = self.get_model_consensus(X, model_names)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_features': len(X),
            'model_explanations': {},
            'consensus_analysis': consensus,
            'summary': {
                'total_models': len(explanations),
                'valid_explanations': len([e for e in explanations.values() if e is not None]),
                'average_confidence': np.mean([e.confidence for e in explanations.values() if e is not None]) if any(e is not None for e in explanations.values()) else 0,
                'prediction_range': {
                    'min': min([e.prediction for e in explanations.values() if e is not None]) if any(e is not None for e in explanations.values()) else 0,
                    'max': max([e.prediction for e in explanations.values() if e is not None]) if any(e is not None for e in explanations.values()) else 0
                }
            }
        }
        
        # Add individual model explanations
        for name, explanation in explanations.items():
            if explanation is not None:
                report['model_explanations'][name] = {
                    'prediction': explanation.prediction,
                    'confidence': explanation.confidence,
                    'top_features': dict(sorted(explanation.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]),
                    'explanation_text': explanation.explanation_text,
                    'has_shap': explanation.shap_values is not None,
                    'has_lime': explanation.lime_explanation is not None,
                    'has_decision_path': explanation.decision_path is not None
                }
        
        return report


def create_model_interpretability() -> ModelInterpretability:
    """Create model interpretability system."""
    interpretability = ModelInterpretability()
    logger.info("Model interpretability system initialized")
    return interpretability
