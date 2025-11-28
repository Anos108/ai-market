"""
Context Managers for AI Market Analysis System

This module provides the shared context classes that all agents use to access
time-based features, event information, and market regime data.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime enumeration."""
    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class EventType(Enum):
    """Event type enumeration."""
    EARNINGS = "earnings"
    FOMC = "fomc"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    NEWS = "news"
    WEATHER = "weather"
    OTHER = "other"


@dataclass
class MarketEvent:
    """Represents a market-impacting event."""
    event_id: str
    event_type: EventType
    title: str
    description: str
    timestamp: datetime
    impact_score: float  # 0.0 to 1.0
    affected_assets: List[str]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeContext:
    """
    Manages time-based features and market timing information.
    
    This class provides time-aware features that agents can use to understand
    temporal patterns in market behavior.
    """
    
    def __init__(self, timezone: str = "US/Eastern"):
        """
        Initialize the time context manager.
        
        Args:
            timezone: Timezone for market hours calculations
        """
        self.timezone = timezone
        self.current_time = datetime.now()
        
    def get_time_features(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Extract comprehensive time-based features.
        
        Args:
            timestamp: Time to extract features from (defaults to current time)
            
        Returns:
            Dictionary of time features
        """
        if timestamp is None:
            timestamp = self.current_time
        
        # Basic time features
        features = {
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'minute': timestamp.minute,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'quarter': (timestamp.month - 1) // 3 + 1,
            'year': timestamp.year,
            'is_weekend': timestamp.weekday() >= 5,
            'is_month_end': self._is_month_end(timestamp),
            'is_quarter_end': self._is_quarter_end(timestamp),
            'is_year_end': self._is_year_end(timestamp)
        }
        
        # Market-specific features
        features.update(self._get_market_timing_features(timestamp))
        
        # Seasonal features
        features.update(self._get_seasonal_features(timestamp))
        
        return features
    
    def _get_market_timing_features(self, timestamp: datetime) -> Dict[str, Any]:
        """Get market timing specific features."""
        is_market_hours = self._is_market_hours(timestamp)
        
        features = {
            'is_market_hours': is_market_hours,
            'is_pre_market': self._is_pre_market(timestamp),
            'is_after_hours': self._is_after_hours(timestamp),
            'is_market_open': self._is_market_open(timestamp),
            'is_market_close': self._is_market_close(timestamp)
        }
        
        if is_market_hours:
            features.update({
                'minutes_since_open': self._minutes_since_market_open(timestamp),
                'minutes_to_close': self._minutes_to_market_close(timestamp),
                'market_session_progress': self._market_session_progress(timestamp)
            })
        
        return features
    
    def _get_seasonal_features(self, timestamp: datetime) -> Dict[str, Any]:
        """Get seasonal and cyclical features."""
        return {
            'day_of_year': timestamp.timetuple().tm_yday,
            'week_of_year': timestamp.isocalendar()[1],
            'is_holiday_season': self._is_holiday_season(timestamp),
            'is_earnings_season': self._is_earnings_season(timestamp),
            'is_options_expiry': self._is_options_expiry(timestamp)
        }
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular market hours."""
        if timestamp.weekday() >= 5:  # Weekend
            return False
        
        hour_minute = timestamp.hour * 100 + timestamp.minute
        return 930 <= hour_minute <= 1600
    
    def _is_pre_market(self, timestamp: datetime) -> bool:
        """Check if timestamp is during pre-market hours."""
        if timestamp.weekday() >= 5:
            return False
        
        hour_minute = timestamp.hour * 100 + timestamp.minute
        return 400 <= hour_minute < 930
    
    def _is_after_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during after-hours."""
        if timestamp.weekday() >= 5:
            return False
        
        hour_minute = timestamp.hour * 100 + timestamp.minute
        return 1600 < hour_minute <= 2000
    
    def _is_market_open(self, timestamp: datetime) -> bool:
        """Check if timestamp is exactly at market open."""
        return (timestamp.hour == 9 and timestamp.minute == 30 and 
                timestamp.weekday() < 5)
    
    def _is_market_close(self, timestamp: datetime) -> bool:
        """Check if timestamp is exactly at market close."""
        return (timestamp.hour == 16 and timestamp.minute == 0 and 
                timestamp.weekday() < 5)
    
    def _minutes_since_market_open(self, timestamp: datetime) -> float:
        """Calculate minutes since market open."""
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        return (timestamp - market_open).total_seconds() / 60.0
    
    def _minutes_to_market_close(self, timestamp: datetime) -> float:
        """Calculate minutes until market close."""
        market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        return (market_close - timestamp).total_seconds() / 60.0
    
    def _market_session_progress(self, timestamp: datetime) -> float:
        """Calculate market session progress (0.0 to 1.0)."""
        minutes_since_open = self._minutes_since_market_open(timestamp)
        total_session_minutes = 6.5 * 60  # 6.5 hours
        return min(1.0, minutes_since_open / total_session_minutes)
    
    def _is_month_end(self, timestamp: datetime) -> bool:
        """Check if timestamp is near month end."""
        return timestamp.day >= 28
    
    def _is_quarter_end(self, timestamp: datetime) -> bool:
        """Check if timestamp is near quarter end."""
        return (timestamp.month in [3, 6, 9, 12] and timestamp.day >= 28)
    
    def _is_year_end(self, timestamp: datetime) -> bool:
        """Check if timestamp is near year end."""
        return timestamp.month == 12 and timestamp.day >= 28
    
    def _is_holiday_season(self, timestamp: datetime) -> bool:
        """Check if timestamp is during holiday season."""
        return timestamp.month == 12 or (timestamp.month == 11 and timestamp.day >= 20)
    
    def _is_earnings_season(self, timestamp: datetime) -> bool:
        """Check if timestamp is during earnings season."""
        # Rough approximation: weeks 2-4 of each quarter
        week_of_month = (timestamp.day - 1) // 7 + 1
        return week_of_month in [2, 3, 4]
    
    def _is_options_expiry(self, timestamp: datetime) -> bool:
        """Check if timestamp is near options expiry (third Friday)."""
        # Third Friday of the month
        first_day = timestamp.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        return abs((timestamp - third_friday).days) <= 1


class EventContext:
    """
    Manages market-impacting events and their temporal effects.
    
    This class tracks events, applies decay weighting based on time,
    and provides event-aware context for agents.
    """
    
    def __init__(self, max_events: int = 1000, decay_half_life_hours: float = 24.0):
        """
        Initialize the event context manager.
        
        Args:
            max_events: Maximum number of events to keep in memory
            decay_half_life_hours: Half-life for event impact decay in hours
        """
        self.max_events = max_events
        self.decay_half_life_hours = decay_half_life_hours
        self.events: deque = deque(maxlen=max_events)
        self.event_index: Dict[str, MarketEvent] = {}
        
    def add_event(self, event: MarketEvent) -> None:
        """
        Add a new market event.
        
        Args:
            event: MarketEvent object to add
        """
        self.events.append(event)
        self.event_index[event.event_id] = event
        logger.info(f"Added event: {event.event_type.value} - {event.title}")
    
    def get_active_events(self, timestamp: Optional[datetime] = None, 
                         min_impact: float = 0.1) -> List[MarketEvent]:
        """
        Get events that are still active (above minimum impact threshold).
        
        Args:
            timestamp: Time to calculate impact at (defaults to now)
            min_impact: Minimum impact threshold
            
        Returns:
            List of active events with decay-weighted impact
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        active_events = []
        for event in self.events:
            decayed_impact = self._calculate_decay_impact(event, timestamp)
            if decayed_impact >= min_impact:
                # Create a copy with decayed impact
                decayed_event = MarketEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    title=event.title,
                    description=event.description,
                    timestamp=event.timestamp,
                    impact_score=decayed_impact,
                    affected_assets=event.affected_assets.copy(),
                    source=event.source,
                    metadata=event.metadata.copy()
                )
                active_events.append(decayed_event)
        
        # Sort by impact score (highest first)
        active_events.sort(key=lambda x: x.impact_score, reverse=True)
        return active_events
    
    def get_events_by_type(self, event_type: EventType, 
                          timestamp: Optional[datetime] = None) -> List[MarketEvent]:
        """
        Get events of a specific type.
        
        Args:
            event_type: Type of events to retrieve
            timestamp: Time to calculate impact at
            
        Returns:
            List of events of the specified type
        """
        all_events = self.get_active_events(timestamp)
        return [event for event in all_events if event.event_type == event_type]
    
    def get_events_for_asset(self, asset_symbol: str, 
                            timestamp: Optional[datetime] = None) -> List[MarketEvent]:
        """
        Get events that affect a specific asset.
        
        Args:
            asset_symbol: Asset symbol to filter by
            timestamp: Time to calculate impact at
            
        Returns:
            List of events affecting the specified asset
        """
        all_events = self.get_active_events(timestamp)
        return [event for event in all_events if asset_symbol in event.affected_assets]
    
    def get_event_features(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Extract event-based features for agents.
        
        Args:
            timestamp: Time to calculate features at
            
        Returns:
            Dictionary of event features
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        active_events = self.get_active_events(timestamp)
        
        features = {
            'total_active_events': len(active_events),
            'max_event_impact': max([e.impact_score for e in active_events], default=0.0),
            'avg_event_impact': np.mean([e.impact_score for e in active_events]) if active_events else 0.0,
            'total_event_impact': sum([e.impact_score for e in active_events])
        }
        
        # Event type breakdown
        event_type_counts = defaultdict(int)
        event_type_impacts = defaultdict(list)
        
        for event in active_events:
            event_type_counts[event.event_type.value] += 1
            event_type_impacts[event.event_type.value].append(event.impact_score)
        
        for event_type in EventType:
            type_name = event_type.value
            features[f'{type_name}_count'] = event_type_counts[type_name]
            features[f'{type_name}_avg_impact'] = (
                np.mean(event_type_impacts[type_name]) if event_type_impacts[type_name] else 0.0
            )
        
        return features
    
    def _calculate_decay_impact(self, event: MarketEvent, timestamp: datetime) -> float:
        """
        Calculate decay-weighted impact for an event.
        
        Args:
            event: MarketEvent to calculate decay for
            timestamp: Current timestamp
            
        Returns:
            Decayed impact score
        """
        time_diff_hours = (timestamp - event.timestamp).total_seconds() / 3600.0
        
        # Exponential decay: impact = original_impact * exp(-ln(2) * time / half_life)
        decay_factor = np.exp(-np.log(2) * time_diff_hours / self.decay_half_life_hours)
        return event.impact_score * decay_factor


class RegimeContext:
    """
    Manages market regime detection and regime-aware features.
    
    This class uses Markov models and other techniques to detect
    current market regime and provide regime-specific context.
    """
    
    def __init__(self, lookback_days: int = 30):
        """
        Initialize the regime context manager.
        
        Args:
            lookback_days: Number of days to look back for regime detection
        """
        self.lookback_days = lookback_days
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history: List[Tuple[datetime, MarketRegime, float]] = []
        self.regime_transition_matrix = self._initialize_transition_matrix()
        
    def update_regime(self, market_data: pd.DataFrame, timestamp: datetime) -> MarketRegime:
        """
        Update the current market regime based on recent data.
        
        Args:
            market_data: Recent market data for regime detection
            timestamp: Current timestamp
            
        Returns:
            Detected market regime
        """
        if market_data.empty:
            return self.current_regime
        
        # Calculate regime indicators
        regime_indicators = self._calculate_regime_indicators(market_data)
        
        # Determine regime based on indicators
        new_regime = self._classify_regime(regime_indicators)
        confidence = self._calculate_regime_confidence(regime_indicators, new_regime)
        
        # Update regime history
        self.regime_history.append((timestamp, new_regime, confidence))
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(days=self.lookback_days)
        self.regime_history = [
            (t, r, c) for t, r, c in self.regime_history 
            if t >= cutoff_time
        ]
        
        self.current_regime = new_regime
        logger.info(f"Updated regime to {new_regime.value} (confidence: {confidence:.2f})")
        
        return new_regime
    
    def get_regime_features(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Extract regime-based features for agents.
        
        Args:
            timestamp: Time to calculate features at
            
        Returns:
            Dictionary of regime features
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        features = {
            'current_regime': self.current_regime.value,
            'regime_duration': self._get_regime_duration(timestamp),
            'regime_stability': self._get_regime_stability(),
            'recent_regime_changes': self._get_recent_regime_changes(timestamp)
        }
        
        # Add regime-specific features
        features.update(self._get_regime_specific_features())
        
        return features
    
    def _calculate_regime_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate indicators for regime classification."""
        if market_data.empty:
            return {}
        
        # Price momentum indicators - handle different column names
        close_col = 'Close' if 'Close' in market_data.columns else 'close'
        if close_col not in market_data.columns:
            logger.warning(f"No close price column found in market data. Available columns: {list(market_data.columns)}")
            return {}
            
        returns = market_data[close_col].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1] if len(returns) >= 20 else 0.0
        
        # Trend indicators
        sma_20 = market_data[close_col].rolling(window=20).mean().iloc[-1] if len(market_data) >= 20 else market_data[close_col].iloc[-1]
        sma_50 = market_data[close_col].rolling(window=50).mean().iloc[-1] if len(market_data) >= 50 else market_data[close_col].iloc[-1]
        
        # Volume indicators - handle different column names
        volume_col = 'Volume' if 'Volume' in market_data.columns else 'volume'
        if volume_col in market_data.columns:
            avg_volume = market_data[volume_col].rolling(window=20).mean().iloc[-1] if len(market_data) >= 20 else market_data[volume_col].iloc[-1]
            current_volume = market_data[volume_col].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0  # Default if no volume data
        
        return {
            'volatility': volatility,
            'price_momentum': (market_data[close_col].iloc[-1] - sma_20) / sma_20 if sma_20 > 0 else 0.0,
            'trend_strength': (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0.0,
            'volume_ratio': volume_ratio,
            'recent_return': returns.iloc[-1] if len(returns) > 0 else 0.0
        }
    
    def _classify_regime(self, indicators: Dict[str, float]) -> MarketRegime:
        """Classify market regime based on indicators."""
        if not indicators:
            return MarketRegime.UNKNOWN
        
        volatility = indicators.get('volatility', 0.0)
        momentum = indicators.get('price_momentum', 0.0)
        trend = indicators.get('trend_strength', 0.0)
        
        # Simple rule-based classification (can be enhanced with ML models)
        if volatility > 0.03:  # High volatility threshold
            return MarketRegime.VOLATILE
        elif momentum > 0.02 and trend > 0.01:  # Strong upward momentum and trend
            return MarketRegime.BULL
        elif momentum < -0.02 and trend < -0.01:  # Strong downward momentum and trend
            return MarketRegime.BEAR
        elif abs(momentum) < 0.01 and abs(trend) < 0.005:  # Low momentum and trend
            return MarketRegime.SIDEWAYS
        else:
            return MarketRegime.UNKNOWN
    
    def _calculate_regime_confidence(self, indicators: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate confidence in regime classification."""
        if not indicators:
            return 0.0
        
        # Simple confidence calculation based on indicator strength
        volatility = abs(indicators.get('volatility', 0.0))
        momentum = abs(indicators.get('price_momentum', 0.0))
        trend = abs(indicators.get('trend_strength', 0.0))
        
        # Higher confidence for stronger signals
        confidence = min(1.0, (volatility * 10 + momentum * 20 + trend * 20))
        return confidence
    
    def _get_regime_duration(self, timestamp: datetime) -> float:
        """Get duration of current regime in hours."""
        if not self.regime_history:
            return 0.0
        
        # Find when current regime started
        current_regime_start = None
        for t, r, c in reversed(self.regime_history):
            if r == self.current_regime:
                current_regime_start = t
            else:
                break
        
        if current_regime_start is None:
            return 0.0
        
        return (timestamp - current_regime_start).total_seconds() / 3600.0
    
    def _get_regime_stability(self) -> float:
        """Calculate regime stability based on recent history."""
        if len(self.regime_history) < 2:
            return 1.0
        
        # Count regime changes in recent history
        recent_changes = 0
        for i in range(1, min(len(self.regime_history), 10)):
            if self.regime_history[-i][1] != self.regime_history[-i-1][1]:
                recent_changes += 1
        
        # Stability is inverse of change frequency
        stability = max(0.0, 1.0 - (recent_changes / min(len(self.regime_history) - 1, 10)))
        return stability
    
    def _get_recent_regime_changes(self, timestamp: datetime) -> int:
        """Get number of regime changes in the last 24 hours."""
        cutoff_time = timestamp - timedelta(hours=24)
        changes = 0
        
        for i in range(1, len(self.regime_history)):
            if (self.regime_history[-i][0] >= cutoff_time and 
                self.regime_history[-i][1] != self.regime_history[-i-1][1]):
                changes += 1
        
        return changes
    
    def _get_regime_specific_features(self) -> Dict[str, Any]:
        """Get features specific to the current regime."""
        features = {}
        
        if self.current_regime == MarketRegime.BULL:
            features['bull_market_indicators'] = {
                'momentum_positive': True,
                'trend_upward': True,
                'volatility_moderate': True
            }
        elif self.current_regime == MarketRegime.BEAR:
            features['bear_market_indicators'] = {
                'momentum_negative': True,
                'trend_downward': True,
                'volatility_elevated': True
            }
        elif self.current_regime == MarketRegime.VOLATILE:
            features['volatile_market_indicators'] = {
                'high_volatility': True,
                'unpredictable_moves': True,
                'elevated_risk': True
            }
        
        return features
    
    def _initialize_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize Markov transition matrix for regime changes."""
        regimes = list(MarketRegime)
        matrix = {}
        
        for from_regime in regimes:
            matrix[from_regime] = {}
            for to_regime in regimes:
                if from_regime == to_regime:
                    matrix[from_regime][to_regime] = 0.8  # High probability of staying
                else:
                    matrix[from_regime][to_regime] = 0.05  # Low probability of changing
        
        return matrix
