"""
Script to run a backtest using MetaTrader 5 historical data.
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
from backtesting.backtest_engine import BacktestEngine
from agents.base_agent import AgentSignal, SignalType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_moving_average_strategy(data: pd.DataFrame, symbol: str) -> AgentSignal:
    """
    A simple SMA crossover strategy for testing.
    Buys when SMA_20 > SMA_50, Sells otherwise.
    """
    # Calculate indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    current_row = data.iloc[-1]

    # Generate signal
    if current_row['SMA_20'] > current_row['SMA_50']:
        signal_type = SignalType.BUY
        confidence = 0.8
    else:
        signal_type = SignalType.SELL
        confidence = 0.8

    return AgentSignal(
        source="SMA_Strategy",
        signal_type=signal_type,
        confidence=confidence,
        asset_symbol=symbol,
        timestamp=current_row.name, # Index is Date
        metadata={"sma_20": current_row['SMA_20'], "sma_50": current_row['SMA_50']}
    )

def main():
    print("="*50)
    print("MT5 Backtesting Demo")
    print("="*50)

    # Configuration
    config = {
        'initial_capital': 10000,
        'data_source': 'mt5', # Use MT5 data
        'regime_detection': False # Disable for simple test
    }

    engine = BacktestEngine(config)

    # Parameters
    symbols = ["EURUSD", "GBPUSD"] # Make sure these are in your MT5 Market Watch
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    logger.info(f"Running backtest on {symbols} from {start_date.date()} to {end_date.date()}...")

    try:
        results = engine.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            strategy_function=simple_moving_average_strategy,
            agent_name="SMA_Agent"
        )

        print("\nBacktest Results:")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")

    except Exception as e:
        logger.error(f"Backtest run failed: {e}")
        print("\nNote: Ensure MetaTrader 5 is running and symbols are visible in Market Watch.")

if __name__ == "__main__":
    main()
