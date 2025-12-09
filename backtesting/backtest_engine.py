"""
Backtesting Engine for AI Market Analysis System

This module provides comprehensive backtesting capabilities for
testing trading strategies and agent performance on historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from agents.base_agent import AgentSignal, SignalType
from context.regime_detection import MarkovRegimeDetector, RegimeType
from data.mt5_ingestor import MT5Ingestor
from data.data_ingestors import DataIngestionConfig

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for backtesting."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    agent_name: str
    signal_confidence: float
    metadata: Dict[str, Any]


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    agent_name: str
    holding_period: int
    metadata: Dict[str, Any]


@dataclass
class BacktestResults:
    """Results from a backtest."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    monthly_returns: pd.Series
    agent_performance: Dict[str, Dict[str, float]]
    regime_performance: Dict[str, Dict[str, float]]


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    
    This class provides:
    - Historical data simulation
    - Order execution simulation
    - Performance metrics calculation
    - Agent performance analysis
    - Regime-based performance analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Backtest Engine.
        
        Args:
            config: Configuration dictionary
        """
        default_config = {
            'initial_capital': 100000,
            'commission': 0.001,  # 0.1% commission
            'slippage': 0.0005,   # 0.05% slippage
            'max_position_size': 0.2,  # 20% max position
            'min_trade_size': 100,     # Minimum trade size
            'data_source': 'yfinance',
            'lookback_periods': 252,   # 1 year of daily data
            'benchmark_symbol': 'SPY',
            'risk_free_rate': 0.02,    # 2% annual risk-free rate
            'regime_detection': True
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.regime_detector = MarkovRegimeDetector() if config.get('regime_detection', True) else None
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.current_positions = {}
        self.cash = self.config['initial_capital']
        self.portfolio_value = self.config['initial_capital']
        
        logger.info(f"Initialized BacktestEngine with config: {self.config}")
    
    def run_backtest(self, 
                    symbols: List[str], 
                    start_date: datetime, 
                    end_date: datetime,
                    strategy_function: Callable[[pd.DataFrame, str], AgentSignal],
                    agent_name: str = "BacktestAgent") -> BacktestResults:
        """
        Run a backtest for the given strategy.
        
        Args:
            symbols: List of symbols to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_function: Function that generates trading signals
            agent_name: Name of the agent being backtested
            
        Returns:
            Backtest results
        """
        try:
            logger.info(f"Starting backtest for {agent_name} from {start_date} to {end_date}")
            
            # Reset state
            self._reset_backtest_state()
            
            # Fetch historical data
            historical_data = self._fetch_historical_data(symbols, start_date, end_date)
            
            if historical_data.empty:
                logger.error("No historical data available for backtest")
                return self._create_empty_results()
            
            # Train regime detector if enabled
            if self.regime_detector:
                self.regime_detector.fit(historical_data)
            
            # Run backtest simulation
            self._run_simulation(historical_data, strategy_function, agent_name)
            
            # Calculate results
            results = self._calculate_results(historical_data, agent_name)
            
            logger.info(f"Backtest completed for {agent_name}")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return self._create_empty_results()
    
    def _reset_backtest_state(self) -> None:
        """Reset backtest state."""
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.current_positions = {}
        self.cash = self.config['initial_capital']
        self.portfolio_value = self.config['initial_capital']
    
    def _fetch_historical_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        try:
            all_data = {}
            
            # Initialize MT5 ingestor if configured
            mt5_ingestor = None
            if self.config.get('data_source') == 'mt5':
                ingest_config = DataIngestionConfig(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    interval='1d' # Default to daily for backtesting
                )
                mt5_ingestor = MT5Ingestor(ingest_config)

            for symbol in symbols:
                try:
                    if self.config.get('data_source') == 'mt5' and mt5_ingestor:
                        data = mt5_ingestor.fetch_data(symbol, start_date, end_date)
                    else:
                        # Default to yfinance
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(start=start_date, end=end_date)
                    
                    if not data.empty:
                        all_data[symbol] = data
                        logger.info(f"Fetched {len(data)} periods for {symbol}")
                    else:
                        logger.warning(f"No data available for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    continue
            
            if not all_data:
                return pd.DataFrame()
            
            # Combine data from all symbols
            combined_data = pd.concat(all_data.values(), keys=all_data.keys(), names=['symbol', 'date'])
            combined_data = combined_data.reset_index()
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Historical data fetching failed: {e}")
            return pd.DataFrame()
    
    def _run_simulation(self, data: pd.DataFrame, strategy_function: Callable, agent_name: str) -> None:
        """Run the backtest simulation."""
        try:
            # Group data by date for daily simulation
            daily_data = data.groupby('date')
            
            for date, day_data in daily_data:
                # Update portfolio value
                self._update_portfolio_value(day_data)
                
                # Generate signals for each symbol
                for symbol in day_data['symbol'].unique():
                    symbol_data = day_data[day_data['symbol'] == symbol]
                    
                    if len(symbol_data) > 0:
                        # Get historical data up to current date
                        historical_data = data[data['date'] <= date]
                        symbol_historical = historical_data[historical_data['symbol'] == symbol]
                        
                        if len(symbol_historical) >= 20:  # Minimum data requirement
                            # Generate signal
                            signal = strategy_function(symbol_historical, symbol)
                            
                            # Process signal
                            self._process_signal(signal, symbol_data.iloc[0], agent_name)
                
                # Record equity curve
                self.equity_curve.append({
                    'date': date,
                    'portfolio_value': self.portfolio_value,
                    'cash': self.cash,
                    'positions_value': self.portfolio_value - self.cash
                })
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
    
    def _update_portfolio_value(self, day_data: pd.DataFrame) -> None:
        """Update portfolio value based on current positions."""
        try:
            positions_value = 0
            
            for symbol, position in self.current_positions.items():
                symbol_data = day_data[day_data['symbol'] == symbol]
                
                if not symbol_data.empty:
                    current_price = symbol_data.iloc[0]['Close']
                    positions_value += position['quantity'] * current_price
            
            self.portfolio_value = self.cash + positions_value
            
        except Exception as e:
            logger.error(f"Portfolio value update failed: {e}")
    
    def _process_signal(self, signal: AgentSignal, price_data: pd.Series, agent_name: str) -> None:
        """Process a trading signal."""
        try:
            symbol = signal.asset_symbol
            current_price = price_data['Close']
            
            # Apply slippage
            execution_price = current_price * (1 + self.config['slippage'] if signal.signal_type == SignalType.BUY else 1 - self.config['slippage'])
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, current_price)
            
            if position_size <= 0:
                return
            
            # Create order
            if signal.signal_type == SignalType.BUY:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                    price=execution_price,
                    timestamp=price_data.name,
                    agent_name=agent_name,
                    signal_confidence=signal.confidence,
                    metadata=signal.metadata
                )
                
                # Execute buy order
                self._execute_buy_order(order, price_data)
                
            elif signal.signal_type == SignalType.SELL:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position_size,
                    price=execution_price,
                    timestamp=price_data.name,
                    agent_name=agent_name,
                    signal_confidence=signal.confidence,
                    metadata=signal.metadata
                )
                
                # Execute sell order
                self._execute_sell_order(order, price_data)
            
            # Record order
            self.orders.append(order)
            
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
    
    def _calculate_position_size(self, signal: AgentSignal, current_price: float) -> float:
        """Calculate position size based on signal and risk management."""
        try:
            # Base position size from signal confidence
            base_size = signal.confidence * self.config['max_position_size']
            
            # Calculate dollar amount
            dollar_amount = self.portfolio_value * base_size
            
            # Calculate shares
            shares = dollar_amount / current_price
            
            # Apply minimum trade size
            if shares * current_price < self.config['min_trade_size']:
                return 0
            
            return shares
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0
    
    def _execute_buy_order(self, order: Order, price_data: pd.Series) -> None:
        """Execute a buy order."""
        try:
            cost = order.quantity * order.price
            commission = cost * self.config['commission']
            total_cost = cost + commission
            
            if total_cost <= self.cash:
                # Update cash
                self.cash -= total_cost
                
                # Update position
                if order.symbol in self.current_positions:
                    # Add to existing position
                    existing_position = self.current_positions[order.symbol]
                    total_quantity = existing_position['quantity'] + order.quantity
                    total_cost_basis = existing_position['cost_basis'] + cost
                    avg_price = total_cost_basis / total_quantity
                    
                    self.current_positions[order.symbol] = {
                        'quantity': total_quantity,
                        'cost_basis': total_cost_basis,
                        'avg_price': avg_price,
                        'entry_time': existing_position['entry_time']
                    }
                else:
                    # New position
                    self.current_positions[order.symbol] = {
                        'quantity': order.quantity,
                        'cost_basis': cost,
                        'avg_price': order.price,
                        'entry_time': order.timestamp
                    }
            
        except Exception as e:
            logger.error(f"Buy order execution failed: {e}")
    
    def _execute_sell_order(self, order: Order, price_data: pd.Series) -> None:
        """Execute a sell order."""
        try:
            if order.symbol not in self.current_positions:
                return
            
            position = self.current_positions[order.symbol]
            
            # Calculate shares to sell
            shares_to_sell = min(order.quantity, position['quantity'])
            
            if shares_to_sell <= 0:
                return
            
            # Calculate proceeds
            proceeds = shares_to_sell * order.price
            commission = proceeds * self.config['commission']
            net_proceeds = proceeds - commission
            
            # Update cash
            self.cash += net_proceeds
            
            # Calculate P&L
            cost_basis = (shares_to_sell / position['quantity']) * position['cost_basis']
            pnl = proceeds - cost_basis - commission
            pnl_percent = pnl / cost_basis if cost_basis > 0 else 0
            
            # Create trade record
            trade = Trade(
                symbol=order.symbol,
                side=OrderSide.SELL,
                quantity=shares_to_sell,
                entry_price=position['avg_price'],
                exit_price=order.price,
                entry_time=position['entry_time'],
                exit_time=order.timestamp,
                pnl=pnl,
                pnl_percent=pnl_percent,
                agent_name=order.agent_name,
                holding_period=(order.timestamp - position['entry_time']).days,
                metadata=order.metadata
            )
            
            self.trades.append(trade)
            
            # Update position
            remaining_quantity = position['quantity'] - shares_to_sell
            if remaining_quantity > 0:
                remaining_cost_basis = position['cost_basis'] - cost_basis
                self.current_positions[order.symbol] = {
                    'quantity': remaining_quantity,
                    'cost_basis': remaining_cost_basis,
                    'avg_price': remaining_cost_basis / remaining_quantity,
                    'entry_time': position['entry_time']
                }
            else:
                # Close position
                del self.current_positions[order.symbol]
            
        except Exception as e:
            logger.error(f"Sell order execution failed: {e}")
    
    def _calculate_results(self, data: pd.DataFrame, agent_name: str) -> BacktestResults:
        """Calculate backtest results."""
        try:
            if not self.trades:
                return self._create_empty_results()
            
            # Basic statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L statistics
            total_pnl = sum(t.pnl for t in self.trades)
            total_return = total_pnl / self.config['initial_capital']
            
            # Risk metrics
            returns = [t.pnl_percent for t in self.trades]
            max_drawdown = self._calculate_max_drawdown()
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(total_return, max_drawdown)
            
            # Equity curve
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df = equity_df.set_index('date')
            
            # Monthly returns
            monthly_returns = self._calculate_monthly_returns(equity_df)
            
            # Agent performance
            agent_performance = self._calculate_agent_performance()
            
            # Regime performance
            regime_performance = self._calculate_regime_performance(data)
            
            return BacktestResults(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                trades=self.trades,
                equity_curve=equity_df,
                monthly_returns=monthly_returns,
                agent_performance=agent_performance,
                regime_performance=regime_performance
            )
            
        except Exception as e:
            logger.error(f"Results calculation failed: {e}")
            return self._create_empty_results()
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            if not self.equity_curve:
                return 0.0
            
            equity_values = [point['portfolio_value'] for point in self.equity_curve]
            peak = equity_values[0]
            max_dd = 0.0
            
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0
    
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
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        try:
            if max_drawdown == 0:
                return float('inf')
            
            return total_return / max_drawdown
            
        except Exception as e:
            logger.error(f"Calmar ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_monthly_returns(self, equity_df: pd.DataFrame) -> pd.Series:
        """Calculate monthly returns."""
        try:
            if equity_df.empty:
                return pd.Series()
            
            monthly_values = equity_df['portfolio_value'].resample('M').last()
            monthly_returns = monthly_values.pct_change().dropna()
            
            return monthly_returns
            
        except Exception as e:
            logger.error(f"Monthly returns calculation failed: {e}")
            return pd.Series()
    
    def _calculate_agent_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance by agent."""
        try:
            agent_performance = {}
            
            for trade in self.trades:
                agent_name = trade.agent_name
                
                if agent_name not in agent_performance:
                    agent_performance[agent_name] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0.0,
                        'avg_pnl': 0.0,
                        'win_rate': 0.0
                    }
                
                perf = agent_performance[agent_name]
                perf['total_trades'] += 1
                perf['total_pnl'] += trade.pnl
                
                if trade.pnl > 0:
                    perf['winning_trades'] += 1
            
            # Calculate derived metrics
            for agent_name, perf in agent_performance.items():
                if perf['total_trades'] > 0:
                    perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
                    perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
            
            return agent_performance
            
        except Exception as e:
            logger.error(f"Agent performance calculation failed: {e}")
            return {}
    
    def _calculate_regime_performance(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate performance by market regime."""
        try:
            if not self.regime_detector or not self.trades:
                return {}
            
            regime_performance = {}
            
            for trade in self.trades:
                # Get regime for trade date
                trade_date = trade.entry_time.date()
                trade_data = data[data['date'].dt.date == trade_date]
                
                if not trade_data.empty:
                    regime_state = self.regime_detector.predict(trade_data)
                    regime_name = regime_state.regime_type.value
                    
                    if regime_name not in regime_performance:
                        regime_performance[regime_name] = {
                            'total_trades': 0,
                            'winning_trades': 0,
                            'total_pnl': 0.0,
                            'avg_pnl': 0.0,
                            'win_rate': 0.0
                        }
                    
                    perf = regime_performance[regime_name]
                    perf['total_trades'] += 1
                    perf['total_pnl'] += trade.pnl
                    
                    if trade.pnl > 0:
                        perf['winning_trades'] += 1
            
            # Calculate derived metrics
            for regime_name, perf in regime_performance.items():
                if perf['total_trades'] > 0:
                    perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
                    perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
            
            return regime_performance
            
        except Exception as e:
            logger.error(f"Regime performance calculation failed: {e}")
            return {}
    
    def _create_empty_results(self) -> BacktestResults:
        """Create empty results for failed backtests."""
        return BacktestResults(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            trades=[],
            equity_curve=pd.DataFrame(),
            monthly_returns=pd.Series(),
            agent_performance={},
            regime_performance={}
        )
