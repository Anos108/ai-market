"""
Base Agent Class for AI Market Analysis System

This module provides the abstract base class that all specialized agents inherit from.
It defines the common interface and shared functionality for all market analysis agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status enumeration."""
    IDLE = "idle"
    TRAINING = "training"
    PREDICTING = "predicting"
    ERROR = "error"


class SignalType(Enum):
    """Signal type enumeration for agent outputs."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class AgentSignal:
    """Standardized signal output from agents."""
    agent_name: str
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    asset_symbol: str
    metadata: Dict[str, Any]
    reasoning: str


@dataclass
class AgentContext:
    """Shared context information for all agents."""
    time_context: Dict[str, Any]
    event_context: Dict[str, Any]
    regime_context: Dict[str, Any]
    market_data: pd.DataFrame
    timestamp: datetime
    symbol: str


class BaseAgent(ABC):
    """
    Abstract base class for all market analysis agents.
    
    This class defines the common interface and shared functionality that all
    specialized agents must implement. It provides a standardized way for agents
    to interact with the shared context and produce signals.
    """
    
    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name for the agent
            version: Agent version string
            config: Configuration dictionary for the agent
        """
        self.name = name
        self.version = version
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.last_update = None
        self.performance_metrics = {}
        self.model = None
        
        logger.info(f"Initialized {self.name} agent v{self.version}")
    
    @abstractmethod
    def train(self, training_data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """
        Train the agent's model on historical data.
        
        Args:
            training_data: Historical market data for training
            context: Shared context information
            
        Returns:
            Dictionary containing training metrics and results
        """
        pass
    
    @abstractmethod
    def predict(self, context: AgentContext) -> AgentSignal:
        """
        Generate a prediction/signal based on current context.
        
        Args:
            context: Current market context and data
            
        Returns:
            AgentSignal object containing the prediction
        """
        pass
    
    @abstractmethod
    def update_model(self, new_data: pd.DataFrame, context: AgentContext) -> None:
        """
        Update the agent's model with new data (online learning).
        
        Args:
            new_data: New market data to incorporate
            context: Current context information
        """
        pass
    
    def validate_signal(self, signal: AgentSignal) -> bool:
        """
        Validate that a signal meets quality standards.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        if not isinstance(signal, AgentSignal):
            logger.error(f"Invalid signal type: {type(signal)}")
            return False
        
        if not 0.0 <= signal.confidence <= 1.0:
            logger.error(f"Invalid confidence value: {signal.confidence}")
            return False
        
        if signal.agent_name != self.name:
            logger.error(f"Signal agent name mismatch: {signal.agent_name} != {self.name}")
            return False
        
        return True
    
    def extract_time_features(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Extract time-based features from a timestamp.
        
        Args:
            timestamp: Datetime object to extract features from
            
        Returns:
            Dictionary of time features
        """
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'quarter': (timestamp.month - 1) // 3 + 1,
            'is_weekend': timestamp.weekday() >= 5,
            'is_market_hours': self._is_market_hours(timestamp),
            'time_since_open': self._time_since_market_open(timestamp),
            'time_to_close': self._time_to_market_close(timestamp)
        }
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during market hours (9:30 AM - 4:00 PM ET)."""
        # Simplified market hours check (weekdays 9:30-16:00)
        if timestamp.weekday() >= 5:  # Weekend
            return False
        
        hour_minute = timestamp.hour * 100 + timestamp.minute
        return 930 <= hour_minute <= 1600
    
    def _time_since_market_open(self, timestamp: datetime) -> float:
        """Calculate minutes since market open."""
        if not self._is_market_hours(timestamp):
            return 0.0
        
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        return (timestamp - market_open).total_seconds() / 60.0
    
    def _time_to_market_close(self, timestamp: datetime) -> float:
        """Calculate minutes until market close."""
        if not self._is_market_hours(timestamp):
            return 0.0
        
        market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        return (market_close - timestamp).total_seconds() / 60.0
    
    def calculate_performance_metrics(self, predictions: List[AgentSignal], actuals: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics for the agent.
        
        Args:
            predictions: List of agent predictions
            actuals: List of actual market movements
            
        Returns:
            Dictionary of performance metrics
        """
        if len(predictions) != len(actuals):
            logger.error("Mismatch between predictions and actuals length")
            return {}
        
        # Convert signals to numeric values for calculation
        signal_values = []
        for pred in predictions:
            if pred.signal_type == SignalType.BUY or pred.signal_type == SignalType.STRONG_BUY:
                signal_values.append(1.0)
            elif pred.signal_type == SignalType.SELL or pred.signal_type == SignalType.STRONG_SELL:
                signal_values.append(-1.0)
            else:
                signal_values.append(0.0)
        
        # Calculate basic metrics
        accuracy = np.mean([1 if (s > 0 and a > 0) or (s < 0 and a < 0) or (s == 0 and a == 0) else 0 
                           for s, a in zip(signal_values, actuals)])
        
        mse = np.mean([(s - a) ** 2 for s, a in zip(signal_values, actuals)])
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'total_predictions': len(predictions),
            'avg_confidence': np.mean([p.confidence for p in predictions])
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metadata."""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status.value,
            'last_update': self.last_update,
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} Agent v{self.version} ({self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"BaseAgent(name='{self.name}', version='{self.version}', status='{self.status.value}')"
