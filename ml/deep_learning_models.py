"""
Deep Learning Models for AI Market Analysis System

This module provides advanced deep learning models including LSTM, Transformer,
and other neural network architectures for enhanced market prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass
    logging.warning("PyTorch not available. Deep learning models will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Create dummy Model class for type hints
    class Model:
        pass
    logging.warning("TensorFlow not available. Some deep learning models will be disabled.")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of deep learning models."""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    GRU = "gru"
    ATTENTION_LSTM = "attention_lstm"
    WAVENET = "wavenet"


class PredictionType(Enum):
    """Types of predictions."""
    PRICE = "price"
    VOLATILITY = "volatility"
    DIRECTION = "direction"
    VOLUME = "volume"
    MULTI_TARGET = "multi_target"


@dataclass
class ModelConfig:
    """Configuration for deep learning models."""
    model_type: ModelType
    prediction_type: PredictionType
    sequence_length: int = 60
    features: List[str] = None
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    mse: float
    mae: float
    rmse: float
    r2_score: float
    mape: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    prediction_time: float


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, sequence_length: int):
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(sequence), torch.FloatTensor([target])


class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout_rate: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out


class TransformerModel(nn.Module):
    """Transformer model for time series prediction."""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, output_size: int, dropout_rate: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Take the last output
        x = x[:, -1, :]
        x = self.output_projection(x)
        
        return x


class DeepLearningPredictor:
    """
    Deep learning predictor for market analysis.
    
    This class provides:
    - LSTM models for time series prediction
    - Transformer models for sequence modeling
    - CNN-LSTM hybrid models
    - Model training and evaluation
    - Real-time prediction capabilities
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the deep learning predictor.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False
        self.training_history = None
        self.performance_metrics = None
        
        # Check if deep learning libraries are available
        if not TORCH_AVAILABLE and not TENSORFLOW_AVAILABLE:
            raise ImportError("Neither PyTorch nor TensorFlow is available. Please install one of them.")
        
        logger.info(f"Initialized DeepLearningPredictor with config: {config}")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for deep learning model training.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Tuple of (features, targets)
        """
        try:
            # Select features
            if self.config.features:
                feature_columns = [col for col in self.config.features if col in data.columns]
            else:
                # Default features
                feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                feature_columns = [col for col in feature_columns if col in data.columns]
            
            if not feature_columns:
                raise ValueError("No valid feature columns found")
            
            # Prepare features
            features = data[feature_columns].values
            
            # Prepare targets based on prediction type
            if self.config.prediction_type == PredictionType.PRICE:
                targets = data['Close'].values
            elif self.config.prediction_type == PredictionType.VOLATILITY:
                returns = data['Close'].pct_change().dropna()
                targets = returns.rolling(window=20).std().values
            elif self.config.prediction_type == PredictionType.DIRECTION:
                returns = data['Close'].pct_change()
                targets = (returns > 0).astype(int).values
            elif self.config.prediction_type == PredictionType.VOLUME:
                targets = data['Volume'].values
            else:
                targets = data['Close'].values
            
            # Remove NaN values
            valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(targets))
            features = features[valid_indices]
            targets = targets[valid_indices]
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform(features)
            
            # Scale targets
            targets_scaled = self.scaler.fit_transform(targets.reshape(-1, 1)).flatten()
            
            return features_scaled, targets_scaled
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features: Feature data
            targets: Target data
            
        Returns:
            Tuple of (X, y) sequences
        """
        try:
            X, y = [], []
            
            for i in range(self.config.sequence_length, len(features)):
                X.append(features[i - self.config.sequence_length:i])
                y.append(targets[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Sequence creation failed: {e}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> Union[nn.Module, Model]:
        """
        Build the deep learning model.
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
            
        Returns:
            Built model
        """
        try:
            if self.config.model_type == ModelType.LSTM:
                if TORCH_AVAILABLE:
                    return self._build_lstm_pytorch(input_shape)
                elif TENSORFLOW_AVAILABLE:
                    return self._build_lstm_tensorflow(input_shape)
            elif self.config.model_type == ModelType.TRANSFORMER:
                if TORCH_AVAILABLE:
                    return self._build_transformer_pytorch(input_shape)
                else:
                    raise ImportError("Transformer model requires PyTorch")
            elif self.config.model_type == ModelType.CNN_LSTM:
                if TENSORFLOW_AVAILABLE:
                    return self._build_cnn_lstm_tensorflow(input_shape)
                else:
                    raise ImportError("CNN-LSTM model requires TensorFlow")
            else:
                # Default to LSTM
                if TORCH_AVAILABLE:
                    return self._build_lstm_pytorch(input_shape)
                elif TENSORFLOW_AVAILABLE:
                    return self._build_lstm_tensorflow(input_shape)
                
        except Exception as e:
            logger.error(f"Model building failed: {e}")
            raise
    
    def _build_lstm_pytorch(self, input_shape: Tuple[int, int]) -> nn.Module:
        """Build LSTM model using PyTorch."""
        return LSTMModel(
            input_size=input_shape[1],
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=1,
            dropout_rate=self.config.dropout_rate
        )
    
    def _build_lstm_tensorflow(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model using TensorFlow."""
        model = Sequential([
            LSTM(self.config.hidden_size, return_sequences=True, input_shape=input_shape),
            Dropout(self.config.dropout_rate),
            LSTM(self.config.hidden_size, return_sequences=False),
            Dropout(self.config.dropout_rate),
            Dense(50),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _build_transformer_pytorch(self, input_shape: Tuple[int, int]) -> nn.Module:
        """Build Transformer model using PyTorch."""
        return TransformerModel(
            input_size=input_shape[1],
            d_model=self.config.hidden_size,
            nhead=8,
            num_layers=self.config.num_layers,
            output_size=1,
            dropout_rate=self.config.dropout_rate
        )
    
    def _build_cnn_lstm_tensorflow(self, input_shape: Tuple[int, int]) -> Model:
        """Build CNN-LSTM hybrid model using TensorFlow."""
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(self.config.hidden_size, return_sequences=False),
            Dropout(self.config.dropout_rate),
            Dense(50),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the deep learning model.
        
        Args:
            data: Training data
            
        Returns:
            Training results
        """
        try:
            logger.info(f"Starting training for {self.config.model_type.value} model")
            
            # Prepare data
            features, targets = self.prepare_data(data)
            X, y = self.create_sequences(features, targets)
            
            if len(X) == 0:
                raise ValueError("No valid sequences created from data")
            
            # Build model
            self.model = self.build_model((self.config.sequence_length, features.shape[1]))
            
            # Split data
            split_idx = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
                self.training_history = self._train_pytorch(X_train, y_train, X_val, y_val)
            elif TENSORFLOW_AVAILABLE and isinstance(self.model, Model):
                self.training_history = self._train_tensorflow(X_train, y_train, X_val, y_val)
            else:
                raise ValueError("No suitable training method available")
            
            # Evaluate model
            self.performance_metrics = self._evaluate_model(X_val, y_val)
            
            self.is_trained = True
            logger.info(f"Training completed for {self.config.model_type.value} model")
            
            return {
                "status": "training_complete",
                "model_type": self.config.model_type.value,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "performance_metrics": self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _train_pytorch(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, List[float]]:
        """Train PyTorch model."""
        try:
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, self.config.sequence_length)
            val_dataset = TimeSeriesDataset(X_val, y_val, self.config.sequence_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            train_losses = []
            val_losses = []
            
            for epoch in range(self.config.epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if epoch > self.config.early_stopping_patience:
                    if val_loss > min(val_losses[-self.config.early_stopping_patience:]):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            return {
                "train_loss": train_losses,
                "val_loss": val_losses
            }
            
        except Exception as e:
            logger.error(f"PyTorch training failed: {e}")
            raise
    
    def _train_tensorflow(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train TensorFlow model."""
        try:
            # Callbacks
            callbacks = [
                EarlyStopping(patience=self.config.early_stopping_patience, restore_best_weights=True),
                ReduceLROnPlateau(patience=self.config.reduce_lr_patience, factor=0.5)
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return history.history
            
        except Exception as e:
            logger.error(f"TensorFlow training failed: {e}")
            raise
    
    def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray) -> ModelPerformance:
        """Evaluate model performance."""
        try:
            # Make predictions
            predictions = self.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, predictions)
            mae = mean_absolute_error(y_val, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, predictions)
            mape = np.mean(np.abs((y_val - predictions) / y_val)) * 100
            
            # For classification tasks
            if self.config.prediction_type == PredictionType.DIRECTION:
                y_pred_binary = (predictions > 0.5).astype(int)
                y_val_binary = (y_val > 0.5).astype(int)
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                accuracy = accuracy_score(y_val_binary, y_pred_binary)
                precision = precision_score(y_val_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_val_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_val_binary, y_pred_binary, zero_division=0)
            else:
                accuracy = precision = recall = f1 = 0.0
            
            return ModelPerformance(
                model_name=f"{self.config.model_type.value}_{self.config.prediction_type.value}",
                mse=mse,
                mae=mae,
                rmse=rmse,
                r2_score=r2,
                mape=mape,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_time=0.0,  # Would need to track this
                prediction_time=0.0  # Would need to track this
            )
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model not trained")
            
            if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    predictions = self.model(X_tensor)
                    return predictions.numpy().flatten()
            elif TENSORFLOW_AVAILABLE and isinstance(self.model, Model):
                predictions = self.model.predict(X, verbose=0)
                return predictions.flatten()
            else:
                raise ValueError("No suitable prediction method available")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_next(self, data: pd.DataFrame) -> float:
        """
        Predict the next value using the most recent data.
        
        Args:
            data: Recent market data
            
        Returns:
            Next prediction
        """
        try:
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            # Prepare features
            features, _ = self.prepare_data(data)
            
            if len(features) < self.config.sequence_length:
                raise ValueError(f"Insufficient data. Need at least {self.config.sequence_length} periods")
            
            # Get the last sequence
            last_sequence = features[-self.config.sequence_length:].reshape(1, -1, features.shape[1])
            
            # Make prediction
            prediction = self.predict(last_sequence)
            
            # Inverse transform if needed
            if self.config.prediction_type != PredictionType.DIRECTION:
                prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
            
            return prediction[0]
            
        except Exception as e:
            logger.error(f"Next prediction failed: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        try:
            if not self.is_trained:
                return {"status": "not_trained"}
            
            summary = {
                "model_type": self.config.model_type.value,
                "prediction_type": self.config.prediction_type.value,
                "sequence_length": self.config.sequence_length,
                "features": self.config.features,
                "is_trained": self.is_trained,
                "performance_metrics": self.performance_metrics.__dict__ if self.performance_metrics else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Model summary failed: {e}")
            return {"error": str(e)}
