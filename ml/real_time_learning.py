"""
Real-time Learning and Model Adaptation for AI Market Analysis System

This module provides real-time learning capabilities, online learning,
and adaptive model updates based on new market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timedelta
import asyncio
import threading
import queue
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Some real-time learning features will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Some real-time learning features will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class LearningEvent:
    """Event for real-time learning."""
    timestamp: datetime
    symbol: str
    features: np.ndarray
    target: float
    prediction: float
    error: float
    event_type: str  # 'prediction', 'correction', 'feedback'
    metadata: Dict[str, Any] = None


@dataclass
class ModelUpdate:
    """Model update information."""
    model_name: str
    timestamp: datetime
    update_type: str  # 'incremental', 'full_retrain', 'parameter_update'
    performance_metrics: Dict[str, float]
    update_size: int
    success: bool
    error_message: Optional[str] = None


class OnlineLearner(ABC):
    """Abstract base class for online learning algorithms."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_initialized = False
        self.update_count = 0
        self.last_update = None
        self.performance_history = []
        
    @abstractmethod
    def initialize(self, X: np.ndarray, y: np.ndarray):
        """Initialize the model with initial data."""
        pass
    
    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Update the model with new data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-10:]  # Last 10 updates
        return {
            'avg_mse': np.mean([m['mse'] for m in recent_metrics]),
            'avg_mae': np.mean([m['mae'] for m in recent_metrics]),
            'update_count': self.update_count,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


class SGDOnlineLearner(OnlineLearner):
    """Stochastic Gradient Descent online learner."""
    
    def __init__(self, model_name: str, learning_rate: float = 0.01):
        super().__init__(model_name)
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for SGDOnlineLearner")
        
        self.learning_rate = learning_rate
        self.model = SGDRegressor(
            learning_rate='adaptive',
            eta0=learning_rate,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def initialize(self, X: np.ndarray, y: np.ndarray):
        """Initialize with initial data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.partial_fit(X_scaled, y)
        self.is_initialized = True
        logger.info(f"SGD learner {self.model_name} initialized with {len(X)} samples")
    
    def update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Update model with new data."""
        if not self.is_initialized:
            self.initialize(X, y)
            return {}
        
        X_scaled = self.scaler.transform(X)
        
        # Make predictions before update
        old_predictions = self.model.predict(X_scaled)
        
        # Update model
        self.model.partial_fit(X_scaled, y)
        
        # Calculate performance metrics
        new_predictions = self.model.predict(X_scaled)
        mse = mean_squared_error(y, new_predictions)
        mae = mean_absolute_error(y, new_predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'improvement': mean_squared_error(y, old_predictions) - mse
        }
        
        self.performance_history.append(metrics)
        self.update_count += 1
        self.last_update = datetime.now()
        
        logger.info(f"SGD learner {self.model_name} updated. MSE: {mse:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_initialized:
            raise ValueError("Model must be initialized before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class PassiveAggressiveLearner(OnlineLearner):
    """Passive Aggressive online learner."""
    
    def __init__(self, model_name: str, C: float = 1.0):
        super().__init__(model_name)
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for PassiveAggressiveLearner")
        
        self.C = C
        self.model = PassiveAggressiveRegressor(C=C, random_state=42)
        self.scaler = StandardScaler()
        
    def initialize(self, X: np.ndarray, y: np.ndarray):
        """Initialize with initial data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.partial_fit(X_scaled, y)
        self.is_initialized = True
        logger.info(f"Passive Aggressive learner {self.model_name} initialized with {len(X)} samples")
    
    def update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Update model with new data."""
        if not self.is_initialized:
            self.initialize(X, y)
            return {}
        
        X_scaled = self.scaler.transform(X)
        
        # Make predictions before update
        old_predictions = self.model.predict(X_scaled)
        
        # Update model
        self.model.partial_fit(X_scaled, y)
        
        # Calculate performance metrics
        new_predictions = self.model.predict(X_scaled)
        mse = mean_squared_error(y, new_predictions)
        mae = mean_absolute_error(y, new_predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'improvement': mean_squared_error(y, old_predictions) - mse
        }
        
        self.performance_history.append(metrics)
        self.update_count += 1
        self.last_update = datetime.now()
        
        logger.info(f"Passive Aggressive learner {self.model_name} updated. MSE: {mse:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_initialized:
            raise ValueError("Model must be initialized before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class NeuralNetworkOnlineLearner(OnlineLearner):
    """Neural Network online learner using PyTorch."""
    
    def __init__(self, model_name: str, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 learning_rate: float = 0.001):
        super().__init__(model_name)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralNetworkOnlineLearner")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()
        
    def initialize(self, X: np.ndarray, y: np.ndarray):
        """Initialize with initial data."""
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Initial training
        self.model.train()
        for _ in range(50):  # Initial epochs
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor).squeeze()
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
        self.is_initialized = True
        logger.info(f"Neural Network learner {self.model_name} initialized with {len(X)} samples")
    
    def update(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Update model with new data."""
        if not self.is_initialized:
            self.initialize(X, y)
            return {}
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Make predictions before update
        self.model.eval()
        with torch.no_grad():
            old_predictions = self.model(X_tensor).squeeze().cpu().numpy()
        
        # Update model
        self.model.train()
        for _ in range(5):  # Few epochs for online update
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor).squeeze()
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
        # Calculate performance metrics
        self.model.eval()
        with torch.no_grad():
            new_predictions = self.model(X_tensor).squeeze().cpu().numpy()
        
        mse = mean_squared_error(y, new_predictions)
        mae = mean_absolute_error(y, new_predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'improvement': mean_squared_error(y, old_predictions) - mse
        }
        
        self.performance_history.append(metrics)
        self.update_count += 1
        self.last_update = datetime.now()
        
        logger.info(f"Neural Network learner {self.model_name} updated. MSE: {mse:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_initialized:
            raise ValueError("Model must be initialized before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).squeeze().cpu().numpy()
        
        return predictions


class RealTimeLearningManager:
    """Manager for real-time learning system."""
    
    def __init__(self):
        self.learners = {}
        self.learning_queue = queue.Queue()
        self.is_running = False
        self.learning_thread = None
        self.update_callbacks = []
        self.performance_monitor = {}
        
    def add_learner(self, name: str, learner: OnlineLearner):
        """Add an online learner."""
        self.learners[name] = learner
        self.performance_monitor[name] = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'last_update': None,
            'avg_performance': {}
        }
        logger.info(f"Added online learner: {name}")
    
    def add_update_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for model updates."""
        self.update_callbacks.append(callback)
    
    def start_learning(self):
        """Start the real-time learning system."""
        if self.is_running:
            logger.warning("Real-time learning is already running")
            return
        
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("Real-time learning system started")
    
    def stop_learning(self):
        """Stop the real-time learning system."""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        logger.info("Real-time learning system stopped")
    
    def add_learning_event(self, event: LearningEvent):
        """Add a learning event to the queue."""
        self.learning_queue.put(event)
    
    def _learning_loop(self):
        """Main learning loop."""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = self.learning_queue.get(timeout=1.0)
                self._process_learning_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
    
    def _process_learning_event(self, event: LearningEvent):
        """Process a learning event."""
        try:
            # Determine which learners to update
            learners_to_update = self._select_learners_for_event(event)
            
            for learner_name in learners_to_update:
                if learner_name in self.learners:
                    learner = self.learners[learner_name]
                    
                    # Update learner
                    metrics = learner.update(
                        event.features.reshape(1, -1), 
                        np.array([event.target])
                    )
                    
                    # Update performance monitor
                    self.performance_monitor[learner_name]['total_updates'] += 1
                    self.performance_monitor[learner_name]['successful_updates'] += 1
                    self.performance_monitor[learner_name]['last_update'] = datetime.now()
                    self.performance_monitor[learner_name]['avg_performance'] = metrics
                    
                    # Notify callbacks
                    for callback in self.update_callbacks:
                        try:
                            callback(learner_name, {
                                'event': event,
                                'metrics': metrics,
                                'timestamp': datetime.now()
                            })
                        except Exception as e:
                            logger.error(f"Error in update callback: {e}")
                    
                    logger.info(f"Updated learner {learner_name} with event {event.event_type}")
            
        except Exception as e:
            logger.error(f"Error processing learning event: {e}")
            # Update failure count
            for learner_name in self._select_learners_for_event(event):
                if learner_name in self.performance_monitor:
                    self.performance_monitor[learner_name]['failed_updates'] += 1
    
    def _select_learners_for_event(self, event: LearningEvent) -> List[str]:
        """Select which learners to update based on event."""
        # For now, update all learners
        # In the future, this could be more sophisticated
        return list(self.learners.keys())
    
    def get_learner_predictions(self, X: np.ndarray, learner_names: List[str] = None) -> Dict[str, np.ndarray]:
        """Get predictions from multiple learners."""
        if learner_names is None:
            learner_names = list(self.learners.keys())
        
        predictions = {}
        for name in learner_names:
            if name in self.learners:
                try:
                    predictions[name] = self.learners[name].predict(X)
                except Exception as e:
                    logger.error(f"Error getting predictions from {name}: {e}")
                    predictions[name] = None
        
        return predictions
    
    def get_consensus_prediction(self, X: np.ndarray, learner_names: List[str] = None) -> Dict[str, Any]:
        """Get consensus prediction from multiple learners."""
        predictions = self.get_learner_predictions(X, learner_names)
        
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_predictions:
            return {"error": "No valid predictions available"}
        
        # Calculate consensus
        pred_values = list(valid_predictions.values())
        consensus_prediction = np.mean(pred_values)
        prediction_std = np.std(pred_values)
        confidence = 1.0 / (1.0 + prediction_std)  # Higher confidence for lower std
        
        return {
            'consensus_prediction': float(consensus_prediction),
            'prediction_std': float(prediction_std),
            'confidence': float(confidence),
            'individual_predictions': {k: float(v[0]) if len(v) > 0 else 0.0 for k, v in valid_predictions.items()},
            'learner_count': len(valid_predictions)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get real-time learning system status."""
        return {
            'is_running': self.is_running,
            'queue_size': self.learning_queue.qsize(),
            'active_learners': len(self.learners),
            'performance_monitor': self.performance_monitor,
            'total_events_processed': sum(
                monitor['total_updates'] for monitor in self.performance_monitor.values()
            ),
            'success_rate': self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_updates = sum(monitor['total_updates'] for monitor in self.performance_monitor.values())
        successful_updates = sum(monitor['successful_updates'] for monitor in self.performance_monitor.values())
        
        if total_updates == 0:
            return 0.0
        
        return successful_updates / total_updates


def create_real_time_learning_system() -> RealTimeLearningManager:
    """Create and configure real-time learning system."""
    manager = RealTimeLearningManager()
    
    # Create online learners
    try:
        if SKLEARN_AVAILABLE:
            sgd_learner = SGDOnlineLearner("sgd_learner", learning_rate=0.01)
            manager.add_learner("sgd_learner", sgd_learner)
            
            pa_learner = PassiveAggressiveLearner("pa_learner", C=1.0)
            manager.add_learner("pa_learner", pa_learner)
            
            logger.info("Added sklearn-based online learners")
    except Exception as e:
        logger.warning(f"Failed to create sklearn learners: {e}")
    
    try:
        if TORCH_AVAILABLE:
            nn_learner = NeuralNetworkOnlineLearner("nn_learner", input_dim=20, hidden_dims=[64, 32])
            manager.add_learner("nn_learner", nn_learner)
            logger.info("Added neural network online learner")
    except Exception as e:
        logger.warning(f"Failed to create neural network learner: {e}")
    
    # Start the learning system
    manager.start_learning()
    
    logger.info("Real-time learning system initialized")
    return manager
