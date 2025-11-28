import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
import json

from services.real_data_service import RealDataService, RealDataConfig

logger = logging.getLogger(__name__)

class ReportType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TRADE_BASED = "trade_based"
    AGENT_PERFORMANCE = "agent_performance"
    PORTFOLIO_SUMMARY = "portfolio_summary"

class ReportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"

@dataclass
class TradeOutcome:
    symbol: str
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    trade_type: str  # "long" or "short"
    pnl: Optional[float]
    pnl_percent: Optional[float]
    duration_days: Optional[int]
    status: str  # "open", "closed", "cancelled"
    agent_signals: List[str]
    confidence_score: float
    market_regime: str

@dataclass
class AgentPerformance:
    agent_name: str
    total_signals: int
    successful_signals: int
    failed_signals: int
    win_rate: float
    avg_confidence: float
    avg_pnl_percent: float
    best_performing_regime: str
    worst_performing_regime: str
    last_updated: str

@dataclass
class ForecastError:
    symbol: str
    forecast_date: str
    actual_price: float
    predicted_price: float
    error_percent: float
    confidence_score: float
    agent_name: str
    forecast_horizon: str  # "1d", "2w", etc.

@dataclass
class MarketRegimeAnalysis:
    regime: str
    duration_days: int
    total_trades: int
    successful_trades: int
    avg_pnl_percent: float
    best_agent: str
    worst_agent: str
    market_volatility: float

@dataclass
class ReportSummary:
    report_type: ReportType
    report_period: str
    total_trades: int
    successful_trades: int
    failed_trades: int
    open_trades: int
    total_pnl: float
    total_pnl_percent: float
    win_rate: float
    avg_trade_duration: float
    best_performing_agent: str
    worst_performing_agent: str
    market_regimes: List[MarketRegimeAnalysis]
    forecast_errors: List[ForecastError]
    generated_at: str

@dataclass
class ReportConfig:
    include_trade_details: bool = True
    include_agent_performance: bool = True
    include_forecast_analysis: bool = True
    include_market_regime_analysis: bool = True
    include_portfolio_summary: bool = True
    max_trades_per_report: int = 100
    confidence_threshold: float = 0.6

class ReportAgent:
    def __init__(self, real_data_service: RealDataService):
        self.real_data_service = real_data_service
        self.trade_history: List[TradeOutcome] = []
        self.agent_performance: Dict[str, AgentPerformance] = {}
        self.forecast_errors: List[ForecastError] = []
        self.market_regimes: List[MarketRegimeAnalysis] = []
        self.config = ReportConfig()
        logger.info("ReportAgent initialized")

    async def generate_daily_report(self, date: Optional[str] = None) -> ReportSummary:
        """Generate daily performance report."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Generating daily report for {date}")
        
        # Get trades for the day
        daily_trades = await self._get_trades_for_period(date, 1)
        
        # Calculate performance metrics
        summary = await self._calculate_report_summary(
            ReportType.DAILY, 
            f"Daily Report - {date}",
            daily_trades
        )
        
        return summary

    async def generate_weekly_report(self, week_start: Optional[str] = None) -> ReportSummary:
        """Generate weekly performance report."""
        if week_start is None:
            week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        logger.info(f"Generating weekly report for week starting {week_start}")
        
        # Get trades for the week
        weekly_trades = await self._get_trades_for_period(week_start, 7)
        
        # Calculate performance metrics
        summary = await self._calculate_report_summary(
            ReportType.WEEKLY,
            f"Weekly Report - {week_start} to {(datetime.strptime(week_start, '%Y-%m-%d') + timedelta(days=6)).strftime('%Y-%m-%d')}",
            weekly_trades
        )
        
        return summary

    async def generate_trade_based_report(self, trade_ids: List[str]) -> ReportSummary:
        """Generate report for specific trades."""
        logger.info(f"Generating trade-based report for {len(trade_ids)} trades")
        
        # Get specific trades
        specific_trades = [trade for trade in self.trade_history if trade.symbol in trade_ids]
        
        # Calculate performance metrics
        summary = await self._calculate_report_summary(
            ReportType.TRADE_BASED,
            f"Trade-Based Report - {len(trade_ids)} trades",
            specific_trades
        )
        
        return summary

    async def generate_agent_performance_report(self) -> ReportSummary:
        """Generate agent performance report."""
        logger.info("Generating agent performance report")
        
        # Get all trades for performance analysis
        all_trades = self.trade_history
        
        # Calculate performance metrics
        summary = await self._calculate_report_summary(
            ReportType.AGENT_PERFORMANCE,
            "Agent Performance Report",
            all_trades
        )
        
        return summary

    async def _get_trades_for_period(self, start_date: str, days: int) -> List[TradeOutcome]:
        """Get trades for a specific period."""
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=days)
        
        period_trades = []
        for trade in self.trade_history:
            trade_date = datetime.strptime(trade.entry_date, "%Y-%m-%d")
            if start_dt <= trade_date < end_dt:
                period_trades.append(trade)
        
        return period_trades

    async def _calculate_report_summary(
        self, 
        report_type: ReportType, 
        period: str, 
        trades: List[TradeOutcome]
    ) -> ReportSummary:
        """Calculate comprehensive report summary."""
        
        # Basic trade statistics
        total_trades = len(trades)
        successful_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        failed_trades = len([t for t in trades if t.pnl and t.pnl <= 0])
        open_trades = len([t for t in trades if t.status == "open"])
        
        # PnL calculations
        total_pnl = sum([t.pnl for t in trades if t.pnl is not None])
        total_pnl_percent = sum([t.pnl_percent for t in trades if t.pnl_percent is not None])
        
        # Win rate
        win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average trade duration
        closed_trades = [t for t in trades if t.duration_days is not None]
        avg_trade_duration = np.mean([t.duration_days for t in closed_trades]) if closed_trades else 0
        
        # Agent performance analysis
        agent_stats = self._analyze_agent_performance(trades)
        best_agent = max(agent_stats.items(), key=lambda x: x[1]['win_rate'])[0] if agent_stats else "N/A"
        worst_agent = min(agent_stats.items(), key=lambda x: x[1]['win_rate'])[0] if agent_stats else "N/A"
        
        # Market regime analysis
        regime_analysis = self._analyze_market_regimes(trades)
        
        # Forecast error analysis
        forecast_errors = self._get_forecast_errors_for_period(trades)
        
        return ReportSummary(
            report_type=report_type,
            report_period=period,
            total_trades=total_trades,
            successful_trades=successful_trades,
            failed_trades=failed_trades,
            open_trades=open_trades,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            best_performing_agent=best_agent,
            worst_performing_agent=worst_agent,
            market_regimes=regime_analysis,
            forecast_errors=forecast_errors,
            generated_at=datetime.now().isoformat()
        )

    def _analyze_agent_performance(self, trades: List[TradeOutcome]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by agent."""
        agent_stats = {}
        
        for trade in trades:
            for agent in trade.agent_signals:
                if agent not in agent_stats:
                    agent_stats[agent] = {
                        'total_trades': 0,
                        'successful_trades': 0,
                        'total_pnl': 0,
                        'total_confidence': 0
                    }
                
                agent_stats[agent]['total_trades'] += 1
                agent_stats[agent]['total_confidence'] += trade.confidence_score
                
                if trade.pnl and trade.pnl > 0:
                    agent_stats[agent]['successful_trades'] += 1
                
                if trade.pnl:
                    agent_stats[agent]['total_pnl'] += trade.pnl
        
        # Calculate derived metrics
        for agent, stats in agent_stats.items():
            stats['win_rate'] = (stats['successful_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            stats['avg_confidence'] = stats['total_confidence'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            stats['avg_pnl'] = stats['total_pnl'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        
        return agent_stats

    def _analyze_market_regimes(self, trades: List[TradeOutcome]) -> List[MarketRegimeAnalysis]:
        """Analyze performance by market regime."""
        regime_stats = {}
        
        for trade in trades:
            regime = trade.market_regime
            if regime not in regime_stats:
                regime_stats[regime] = {
                    'trades': [],
                    'duration_days': 0
                }
            
            regime_stats[regime]['trades'].append(trade)
            if trade.duration_days:
                regime_stats[regime]['duration_days'] += trade.duration_days
        
        regime_analysis = []
        for regime, stats in regime_stats.items():
            trades = stats['trades']
            successful_trades = len([t for t in trades if t.pnl and t.pnl > 0])
            avg_pnl = np.mean([t.pnl_percent for t in trades if t.pnl_percent is not None]) if trades else 0
            
            # Find best and worst agents for this regime
            agent_performance = self._analyze_agent_performance(trades)
            best_agent = max(agent_performance.items(), key=lambda x: x[1]['win_rate'])[0] if agent_performance else "N/A"
            worst_agent = min(agent_performance.items(), key=lambda x: x[1]['win_rate'])[0] if agent_performance else "N/A"
            
            regime_analysis.append(MarketRegimeAnalysis(
                regime=regime,
                duration_days=stats['duration_days'],
                total_trades=len(trades),
                successful_trades=successful_trades,
                avg_pnl_percent=avg_pnl,
                best_agent=best_agent,
                worst_agent=worst_agent,
                market_volatility=0.15  # Mock volatility
            ))
        
        return regime_analysis

    def _get_forecast_errors_for_period(self, trades: List[TradeOutcome]) -> List[ForecastError]:
        """Get forecast errors for the period."""
        # Filter forecast errors for the period
        period_errors = []
        for error in self.forecast_errors:
            # Check if error is related to any trade in the period
            for trade in trades:
                if error.symbol == trade.symbol:
                    period_errors.append(error)
                    break
        
        return period_errors

    async def add_trade_outcome(self, trade: TradeOutcome):
        """Add a trade outcome to the history."""
        self.trade_history.append(trade)
        logger.info(f"Added trade outcome for {trade.symbol}: {trade.status}")

    async def add_forecast_error(self, error: ForecastError):
        """Add a forecast error to the history."""
        self.forecast_errors.append(error)
        logger.info(f"Added forecast error for {error.symbol}: {error.error_percent:.2f}%")

    async def update_agent_performance(self, agent_name: str, performance: AgentPerformance):
        """Update agent performance metrics."""
        self.agent_performance[agent_name] = performance
        logger.info(f"Updated performance for {agent_name}: {performance.win_rate:.2f}% win rate")

    async def get_report_summary(self) -> Dict[str, Any]:
        """Get overall report summary."""
        return {
            "total_trades": len(self.trade_history),
            "total_forecast_errors": len(self.forecast_errors),
            "agents_tracked": len(self.agent_performance),
            "last_report_generated": datetime.now().isoformat(),
            "report_config": asdict(self.config)
        }

    async def generate_mock_data(self):
        """Generate mock data for testing."""
        logger.info("Generating mock data for ReportAgent")
        
        # Generate mock trades
        symbols = ["BTC-USD", "SOXL", "NVDA", "RIVN", "TSLA", "SPY", "META", "AMD", "TQQQ", "SPXL"]
        agents = ["MomentumAgent", "SentimentAgent", "CorrelationAgent", "RiskAgent", "VolatilityAgent"]
        regimes = ["bull", "bear", "sideways", "volatile", "trending"]
        
        for i in range(50):
            symbol = np.random.choice(symbols)
            entry_date = (datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime("%Y-%m-%d")
            entry_price = np.random.uniform(50, 500)
            quantity = np.random.randint(10, 100)
            trade_type = np.random.choice(["long", "short"])
            status = np.random.choice(["open", "closed", "cancelled"], p=[0.2, 0.7, 0.1])
            
            exit_price = None
            pnl = None
            pnl_percent = None
            duration_days = None
            exit_date = None
            
            if status == "closed":
                exit_date = (datetime.strptime(entry_date, "%Y-%m-%d") + timedelta(days=np.random.randint(1, 10))).strftime("%Y-%m-%d")
                price_change = np.random.uniform(-0.15, 0.15)  # ±15% price change
                exit_price = entry_price * (1 + price_change)
                pnl = (exit_price - entry_price) * quantity * (1 if trade_type == "long" else -1)
                pnl_percent = (exit_price - entry_price) / entry_price * 100 * (1 if trade_type == "long" else -1)
                duration_days = (datetime.strptime(exit_date, "%Y-%m-%d") - datetime.strptime(entry_date, "%Y-%m-%d")).days
            
            trade = TradeOutcome(
                symbol=symbol,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                trade_type=trade_type,
                pnl=pnl,
                pnl_percent=pnl_percent,
                duration_days=duration_days,
                status=status,
                agent_signals=np.random.choice(agents, size=np.random.randint(1, 4), replace=False).tolist(),
                confidence_score=np.random.uniform(0.5, 0.95),
                market_regime=np.random.choice(regimes)
            )
            
            await self.add_trade_outcome(trade)
        
        # Generate mock forecast errors
        for i in range(20):
            symbol = np.random.choice(symbols)
            forecast_date = (datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime("%Y-%m-%d")
            actual_price = np.random.uniform(50, 500)
            predicted_price = actual_price * np.random.uniform(0.9, 1.1)  # ±10% prediction error
            error_percent = abs(actual_price - predicted_price) / actual_price * 100
            
            error = ForecastError(
                symbol=symbol,
                forecast_date=forecast_date,
                actual_price=actual_price,
                predicted_price=predicted_price,
                error_percent=error_percent,
                confidence_score=np.random.uniform(0.6, 0.95),
                agent_name=np.random.choice(agents),
                forecast_horizon=np.random.choice(["1d", "2w", "1m"])
            )
            
            await self.add_forecast_error(error)
        
        # Generate mock agent performance
        for agent in agents:
            performance = AgentPerformance(
                agent_name=agent,
                total_signals=np.random.randint(50, 200),
                successful_signals=np.random.randint(20, 150),
                failed_signals=np.random.randint(10, 100),
                win_rate=np.random.uniform(45, 75),
                avg_confidence=np.random.uniform(0.6, 0.9),
                avg_pnl_percent=np.random.uniform(-5, 15),
                best_performing_regime=np.random.choice(regimes),
                worst_performing_regime=np.random.choice(regimes),
                last_updated=datetime.now().isoformat()
            )
            
            await self.update_agent_performance(agent, performance)
        
        logger.info("Mock data generation completed")
