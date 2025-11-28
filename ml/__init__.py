"""
ML Module for AI Market Analysis System

This module provides advanced machine learning capabilities including:
- Advanced models (Transformers, Ensemble, Reinforcement Learning)
- Model explainability (SHAP, LIME, confidence intervals)
- Real-time learning and adaptation
- Model management and versioning
"""

try:
    from .advanced_models import (
        ModelConfig,
        ModelMetrics,
        BaseMLModel,
        TransformerModel,
        EnsembleModel,
        ReinforcementLearningModel,
        ModelManager,
        create_advanced_models
    )
except ImportError as e:
    print(f"Warning: Could not import advanced_models: {e}")
    create_advanced_models = None

try:
    from .model_explainability import (
        ExplanationResult,
        ConfidenceInterval,
        ModelExplainer,
        ModelInterpretability,
        create_model_interpretability
    )
except ImportError as e:
    print(f"Warning: Could not import model_explainability: {e}")
    create_model_interpretability = None

try:
    from .real_time_learning import (
        LearningEvent,
        ModelUpdate,
        OnlineLearner,
        SGDOnlineLearner,
        PassiveAggressiveLearner,
        NeuralNetworkOnlineLearner,
        RealTimeLearningManager,
        create_real_time_learning_system
    )
except ImportError as e:
    print(f"Warning: Could not import real_time_learning: {e}")
    create_real_time_learning_system = None

__all__ = [
    # Advanced Models
    'ModelConfig',
    'ModelMetrics',
    'BaseMLModel',
    'TransformerModel',
    'EnsembleModel',
    'ReinforcementLearningModel',
    'ModelManager',
    'create_advanced_models',
    
    # Model Explainability
    'ExplanationResult',
    'ConfidenceInterval',
    'ModelExplainer',
    'ModelInterpretability',
    'create_model_interpretability',
    
    # Real-time Learning
    'LearningEvent',
    'ModelUpdate',
    'OnlineLearner',
    'SGDOnlineLearner',
    'PassiveAggressiveLearner',
    'NeuralNetworkOnlineLearner',
    'RealTimeLearningManager',
    'create_real_time_learning_system'
]