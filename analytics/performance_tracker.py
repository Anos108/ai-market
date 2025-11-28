"""
Agent Performance Analytics System

This module provides comprehensive performance tracking and analytics
for all AI agents in the system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..agents.base_agent import AgentSignal, SignalType

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    AVG_RETURN = "avg_return"
    VOLATILITY = "volatility"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class AgentPerformance:
    """Agent performance data."""
    agent_name: str
    timestamp: datetime
    total_signals: int
    correct_signals: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    volatility: float
    calmar_ratio: float
    information_ratio: float
    signal_distribution: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: Dict[str, float] = field(default_factory=dict)
    performance_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SignalOutcome:
    """Signal outcome tracking."""
    signal_id: str
    agent_name: str
    symbol: str
    signal_type: str
    confidence: float
    timestamp: datetime
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_period: Optional[int] = None
    outcome: Optional[str] = None  # 'correct', 'incorrect', 'neutral'
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTracker:
    """
    Agent performance tracking and analytics system.
    
    This class provides:
    - Real-time performance monitoring
    - Historical performance analysis
    - Signal outcome tracking
    - Performance metrics calculation
    - Agent comparison and ranking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Performance Tracker.
        
        Args:
            config: Configuration dictionary
        """
        default_config = {
            'db_path': 'performance_tracker.db',
            'lookback_periods': 252,  # 1 year
            'min_signals_for_analysis': 10,
            'performance_update_frequency': 24,  # hours
            'signal_tracking_window': 5,  # days to track signal outcomes
            'benchmark_symbol': 'SPY',
            'risk_free_rate': 0.02,
            'metrics': [
                PerformanceMetric.ACCURACY,
                PerformanceMetric.PRECISION,
                PerformanceMetric.RECALL,
                PerformanceMetric.F1_SCORE,
                PerformanceMetric.SHARPE_RATIO,
                PerformanceMetric.SORTINO_RATIO,
                PerformanceMetric.MAX_DRAWDOWN,
                PerformanceMetric.WIN_RATE,
                PerformanceMetric.AVG_RETURN,
                PerformanceMetric.VOLATILITY,
                PerformanceMetric.CALMAR_RATIO,
                PerformanceMetric.INFORMATION_RATIO
            ]
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.db_path = Path(self.config['db_path'])
        self.signal_outcomes = []
        self.agent_performance_history = {}
        self.benchmark_data = None
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Initialized PerformanceTracker with config: {self.config}")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for performance tracking."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create signal outcomes table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signal_outcomes (
                        signal_id TEXT PRIMARY KEY,
                        agent_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        exit_timestamp TEXT,
                        pnl REAL,
                        pnl_percent REAL,
                        holding_period INTEGER,
                        outcome TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create agent performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS agent_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        total_signals INTEGER,
                        correct_signals INTEGER,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        sharpe_ratio REAL,
                        sortino_ratio REAL,
                        max_drawdown REAL,
                        win_rate REAL,
                        avg_return REAL,
                        volatility REAL,
                        calmar_ratio REAL,
                        information_ratio REAL,
                        signal_distribution TEXT,
                        confidence_distribution TEXT,
                        performance_by_regime TEXT
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_agent ON signal_outcomes(agent_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON signal_outcomes(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_agent ON agent_performance(agent_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON agent_performance(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def track_signal(self, signal: AgentSignal, entry_price: float, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Track a new signal for performance analysis.
        
        Args:
            signal: Agent signal to track
            entry_price: Entry price for the signal
            metadata: Additional metadata
            
        Returns:
            Signal ID for tracking
        """
        try:
            signal_id = f"{signal.agent_name}_{signal.asset_symbol}_{signal.timestamp.isoformat()}"
            
            signal_outcome = SignalOutcome(
                signal_id=signal_id,
                agent_name=signal.agent_name,
                symbol=signal.asset_symbol,
                signal_type=signal.signal_type.value if hasattr(signal.signal_type, 'value') else signal.signal_type,
                confidence=signal.confidence,
                timestamp=signal.timestamp,
                entry_price=entry_price,
                metadata=metadata or {}
            )
            
            # Store in memory
            self.signal_outcomes.append(signal_outcome)
            
            # Store in database
            self._store_signal_outcome(signal_outcome)
            
            logger.info(f"Tracking signal {signal_id} for {signal.agent_name}")
            return signal_id
            
        except Exception as e:
            logger.error(f"Signal tracking failed: {e}")
            return ""
    
    def update_signal_outcome(self, signal_id: str, exit_price: float, exit_timestamp: datetime) -> None:
        """
        Update signal outcome when position is closed.
        
        Args:
            signal_id: Signal ID to update
            exit_price: Exit price
            exit_timestamp: Exit timestamp
        """
        try:
            # Find signal in memory
            signal_outcome = None
            for i, outcome in enumerate(self.signal_outcomes):
                if outcome.signal_id == signal_id:
                    signal_outcome = outcome
                    break
            
            if signal_outcome is None:
                logger.warning(f"Signal {signal_id} not found for outcome update")
                return
            
            # Calculate P&L
            pnl = exit_price - signal_outcome.entry_price
            pnl_percent = pnl / signal_outcome.entry_price
            
            # Calculate holding period
            holding_period = (exit_timestamp - signal_outcome.timestamp).days
            
            # Determine outcome
            if signal_outcome.signal_type == 'buy':
                outcome = 'correct' if pnl > 0 else 'incorrect'
            elif signal_outcome.signal_type == 'sell':
                outcome = 'correct' if pnl < 0 else 'incorrect'
            else:
                outcome = 'neutral'
            
            # Update signal outcome
            signal_outcome.exit_price = exit_price
            signal_outcome.exit_timestamp = exit_timestamp
            signal_outcome.pnl = pnl
            signal_outcome.pnl_percent = pnl_percent
            signal_outcome.holding_period = holding_period
            signal_outcome.outcome = outcome
            
            # Update in database
            self._update_signal_outcome(signal_outcome)
            
            logger.info(f"Updated signal {signal_id} outcome: {outcome}, P&L: {pnl_percent:.2%}")
            
        except Exception as e:
            logger.error(f"Signal outcome update failed: {e}")
    
    def calculate_agent_performance(self, agent_name: str, lookback_days: Optional[int] = None) -> AgentPerformance:
        """
        Calculate performance metrics for a specific agent.
        
        Args:
            agent_name: Name of the agent
            lookback_days: Number of days to look back (default: config value)
            
        Returns:
            Agent performance metrics
        """
        try:
            lookback_days = lookback_days or self.config['lookback_periods']
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            # Get signals for the agent
            agent_signals = [
                s for s in self.signal_outcomes 
                if s.agent_name == agent_name and s.timestamp >= cutoff_date
            ]
            
            if len(agent_signals) < self.config['min_signals_for_analysis']:
                logger.warning(f"Insufficient signals for {agent_name}: {len(agent_signals)}")
                return self._create_empty_performance(agent_name)
            
            # Calculate basic metrics
            total_signals = len(agent_signals)
            completed_signals = [s for s in agent_signals if s.outcome is not None]
            
            if not completed_signals:
                return self._create_empty_performance(agent_name)
            
            correct_signals = len([s for s in completed_signals if s.outcome == 'correct'])
            accuracy = correct_signals / len(completed_signals) if completed_signals else 0.0
            
            # Calculate precision, recall, F1-score
            precision, recall, f1_score = self._calculate_classification_metrics(completed_signals)
            
            # Calculate financial metrics
            returns = [s.pnl_percent for s in completed_signals if s.pnl_percent is not None]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(returns)
            win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0.0
            avg_return = np.mean(returns) if returns else 0.0
            volatility = np.std(returns) if returns else 0.0
            calmar_ratio = avg_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
            information_ratio = self._calculate_information_ratio(returns)
            
            # Calculate distributions
            signal_distribution = self._calculate_signal_distribution(agent_signals)
            confidence_distribution = self._calculate_confidence_distribution(agent_signals)
            performance_by_regime = self._calculate_performance_by_regime(completed_signals)
            
            performance = AgentPerformance(
                agent_name=agent_name,
                timestamp=datetime.now(),
                total_signals=total_signals,
                correct_signals=correct_signals,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_return=avg_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                information_ratio=information_ratio,
                signal_distribution=signal_distribution,
                confidence_distribution=confidence_distribution,
                performance_by_regime=performance_by_regime
            )
            
            # Store performance
            self._store_agent_performance(performance)
            
            return performance
            
        except Exception as e:
            logger.error(f"Agent performance calculation failed for {agent_name}: {e}")
            return self._create_empty_performance(agent_name)
    
    def _calculate_classification_metrics(self, signals: List[SignalOutcome]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1-score."""
        try:
            if not signals:
                return 0.0, 0.0, 0.0
            
            # Count true positives, false positives, false negatives
            tp = len([s for s in signals if s.outcome == 'correct'])
            fp = len([s for s in signals if s.outcome == 'incorrect'])
            fn = 0  # We don't track false negatives in this context
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return precision, recall, f1_score
            
        except Exception as e:
            logger.error(f"Classification metrics calculation failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        try:
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualize
            annual_return = mean_return * 252
            annual_std = std_return * np.sqrt(252)
            
            sharpe = (annual_return - self.config['risk_free_rate']) / annual_std
            return sharpe
            
        except Exception as e:
            logger.error(f"Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio."""
        try:
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            negative_returns = [r for r in returns if r < 0]
            
            if not negative_returns:
                return float('inf')
            
            downside_std = np.std(negative_returns)
            
            if downside_std == 0:
                return 0.0
            
            # Annualize
            annual_return = mean_return * 252
            annual_downside_std = downside_std * np.sqrt(252)
            
            sortino = (annual_return - self.config['risk_free_rate']) / annual_downside_std
            return sortino
            
        except Exception as e:
            logger.error(f"Sortino ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        try:
            if not returns:
                return 0.0
            
            cumulative_returns = np.cumprod([1 + r for r in returns])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            
            return abs(np.min(drawdowns))
            
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, returns: List[float]) -> float:
        """Calculate information ratio."""
        try:
            if not returns or not self.benchmark_data:
                return 0.0
            
            # This is a simplified calculation
            # In practice, you'd need benchmark returns for the same period
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Assume benchmark return of 0 for simplicity
            excess_return = mean_return
            tracking_error = std_return
            
            return excess_return / tracking_error if tracking_error > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Information ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_signal_distribution(self, signals: List[SignalOutcome]) -> Dict[str, int]:
        """Calculate signal type distribution."""
        try:
            distribution = {}
            for signal in signals:
                signal_type = signal.signal_type
                distribution[signal_type] = distribution.get(signal_type, 0) + 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"Signal distribution calculation failed: {e}")
            return {}
    
    def _calculate_confidence_distribution(self, signals: List[SignalOutcome]) -> Dict[str, float]:
        """Calculate confidence distribution."""
        try:
            if not signals:
                return {}
            
            confidences = [s.confidence for s in signals]
            
            return {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'median': np.median(confidences)
            }
            
        except Exception as e:
            logger.error(f"Confidence distribution calculation failed: {e}")
            return {}
    
    def _calculate_performance_by_regime(self, signals: List[SignalOutcome]) -> Dict[str, Dict[str, float]]:
        """Calculate performance by market regime."""
        try:
            # This is a simplified implementation
            # In practice, you'd need regime data for each signal
            regime_performance = {}
            
            # Group signals by some regime indicator (simplified)
            for signal in signals:
                # For now, use a simple regime classification
                if signal.pnl_percent and signal.pnl_percent > 0.02:
                    regime = 'bull'
                elif signal.pnl_percent and signal.pnl_percent < -0.02:
                    regime = 'bear'
                else:
                    regime = 'sideways'
                
                if regime not in regime_performance:
                    regime_performance[regime] = {
                        'total_signals': 0,
                        'correct_signals': 0,
                        'avg_return': 0.0
                    }
                
                perf = regime_performance[regime]
                perf['total_signals'] += 1
                
                if signal.outcome == 'correct':
                    perf['correct_signals'] += 1
                
                if signal.pnl_percent:
                    perf['avg_return'] += signal.pnl_percent
            
            # Calculate averages
            for regime, perf in regime_performance.items():
                if perf['total_signals'] > 0:
                    perf['win_rate'] = perf['correct_signals'] / perf['total_signals']
                    perf['avg_return'] = perf['avg_return'] / perf['total_signals']
            
            return regime_performance
            
        except Exception as e:
            logger.error(f"Performance by regime calculation failed: {e}")
            return {}
    
    def _create_empty_performance(self, agent_name: str) -> AgentPerformance:
        """Create empty performance for agents with insufficient data."""
        return AgentPerformance(
            agent_name=agent_name,
            timestamp=datetime.now(),
            total_signals=0,
            correct_signals=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_return=0.0,
            volatility=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0
        )
    
    def _store_signal_outcome(self, signal_outcome: SignalOutcome) -> None:
        """Store signal outcome in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO signal_outcomes 
                    (signal_id, agent_name, symbol, signal_type, confidence, timestamp, 
                     entry_price, exit_price, exit_timestamp, pnl, pnl_percent, 
                     holding_period, outcome, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_outcome.signal_id,
                    signal_outcome.agent_name,
                    signal_outcome.symbol,
                    signal_outcome.signal_type,
                    signal_outcome.confidence,
                    signal_outcome.timestamp.isoformat(),
                    signal_outcome.entry_price,
                    signal_outcome.exit_price,
                    signal_outcome.exit_timestamp.isoformat() if signal_outcome.exit_timestamp else None,
                    signal_outcome.pnl,
                    signal_outcome.pnl_percent,
                    signal_outcome.holding_period,
                    signal_outcome.outcome,
                    json.dumps(signal_outcome.metadata)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Signal outcome storage failed: {e}")
    
    def _update_signal_outcome(self, signal_outcome: SignalOutcome) -> None:
        """Update signal outcome in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE signal_outcomes 
                    SET exit_price = ?, exit_timestamp = ?, pnl = ?, pnl_percent = ?, 
                        holding_period = ?, outcome = ?
                    WHERE signal_id = ?
                ''', (
                    signal_outcome.exit_price,
                    signal_outcome.exit_timestamp.isoformat() if signal_outcome.exit_timestamp else None,
                    signal_outcome.pnl,
                    signal_outcome.pnl_percent,
                    signal_outcome.holding_period,
                    signal_outcome.outcome,
                    signal_outcome.signal_id
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Signal outcome update failed: {e}")
    
    def _store_agent_performance(self, performance: AgentPerformance) -> None:
        """Store agent performance in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO agent_performance 
                    (agent_name, timestamp, total_signals, correct_signals, accuracy, 
                     precision, recall, f1_score, sharpe_ratio, sortino_ratio, 
                     max_drawdown, win_rate, avg_return, volatility, calmar_ratio, 
                     information_ratio, signal_distribution, confidence_distribution, 
                     performance_by_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.agent_name,
                    performance.timestamp.isoformat(),
                    performance.total_signals,
                    performance.correct_signals,
                    performance.accuracy,
                    performance.precision,
                    performance.recall,
                    performance.f1_score,
                    performance.sharpe_ratio,
                    performance.sortino_ratio,
                    performance.max_drawdown,
                    performance.win_rate,
                    performance.avg_return,
                    performance.volatility,
                    performance.calmar_ratio,
                    performance.information_ratio,
                    json.dumps(performance.signal_distribution),
                    json.dumps(performance.confidence_distribution),
                    json.dumps(performance.performance_by_regime)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Agent performance storage failed: {e}")
    
    def get_agent_ranking(self, metric: PerformanceMetric = PerformanceMetric.SHARPE_RATIO) -> List[Tuple[str, float]]:
        """
        Get agent ranking by specified metric.
        
        Args:
            metric: Performance metric to rank by
            
        Returns:
            List of (agent_name, metric_value) tuples sorted by metric
        """
        try:
            agent_metrics = {}
            
            # Get all agents
            agent_names = set(s.agent_name for s in self.signal_outcomes)
            
            for agent_name in agent_names:
                performance = self.calculate_agent_performance(agent_name)
                
                if metric == PerformanceMetric.ACCURACY:
                    agent_metrics[agent_name] = performance.accuracy
                elif metric == PerformanceMetric.SHARPE_RATIO:
                    agent_metrics[agent_name] = performance.sharpe_ratio
                elif metric == PerformanceMetric.WIN_RATE:
                    agent_metrics[agent_name] = performance.win_rate
                elif metric == PerformanceMetric.AVG_RETURN:
                    agent_metrics[agent_name] = performance.avg_return
                elif metric == PerformanceMetric.CALMAR_RATIO:
                    agent_metrics[agent_name] = performance.calmar_ratio
                else:
                    agent_metrics[agent_name] = 0.0
            
            # Sort by metric value (descending)
            ranking = sorted(agent_metrics.items(), key=lambda x: x[1], reverse=True)
            
            return ranking
            
        except Exception as e:
            logger.error(f"Agent ranking calculation failed: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        try:
            summary = {
                'total_signals': len(self.signal_outcomes),
                'completed_signals': len([s for s in self.signal_outcomes if s.outcome is not None]),
                'agents_tracked': len(set(s.agent_name for s in self.signal_outcomes)),
                'date_range': {
                    'start': min(s.timestamp for s in self.signal_outcomes).isoformat() if self.signal_outcomes else None,
                    'end': max(s.timestamp for s in self.signal_outcomes).isoformat() if self.signal_outcomes else None
                },
                'agent_rankings': {
                    'sharpe_ratio': self.get_agent_ranking(PerformanceMetric.SHARPE_RATIO),
                    'win_rate': self.get_agent_ranking(PerformanceMetric.WIN_RATE),
                    'accuracy': self.get_agent_ranking(PerformanceMetric.ACCURACY)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance summary calculation failed: {e}")
            return {}
