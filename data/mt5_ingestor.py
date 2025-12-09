"""
MT5 Data Ingestor
Ingests data from MetaTrader 5 via MT5Service.
"""

from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import logging
from .data_ingestors import BaseDataIngestor, DataIngestionConfig
from services.mt5_service import MT5Service

# Import MetaTrader5 for constants if available
try:
    import MetaTrader5 as Mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    class Mt5:
        TIMEFRAME_D1 = 16408  # Fallback constant

logger = logging.getLogger(__name__)

class MT5Ingestor(BaseDataIngestor):
    """
    Data ingestor for MetaTrader 5.
    """

    def __init__(self, config: DataIngestionConfig):
        super().__init__(config)
        self.mt5_service = MT5Service()

    def fetch_data(self, symbol: str, start_date: datetime,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data from MT5.

        Args:
            symbol: Asset symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
        """
        if not MT5_AVAILABLE:
            logger.error("MT5 not available")
            return pd.DataFrame()

        # Map interval string to MT5 constant
        interval_map = {
            "1m": Mt5.TIMEFRAME_M1,
            "5m": Mt5.TIMEFRAME_M5,
            "15m": Mt5.TIMEFRAME_M15,
            "30m": Mt5.TIMEFRAME_M30,
            "1h": Mt5.TIMEFRAME_H1,
            "4h": Mt5.TIMEFRAME_H4,
            "1d": Mt5.TIMEFRAME_D1,
            "1wk": Mt5.TIMEFRAME_W1,
            "1mo": Mt5.TIMEFRAME_MN1
        }

        timeframe = interval_map.get(self.config.interval, Mt5.TIMEFRAME_D1)

        # Calculate number of bars to fetch based on dates
        # This is an approximation. A robust implementation would use copy_rates_range
        # But Mt5.copy_rates_range takes datetime objects directly which is better.

        # Let's use copy_rates_range functionality if we can access it via the service,
        # but the service currently implements fetch_data via pos.
        # I'll stick to what I implemented in MT5Service for now or use the service to get raw access.
        # Wait, I should probably extend MT5Service to support range-based fetch or implement it here.

        # Actually, let's just use a large enough count for now or improve MT5Service.
        # Improving MT5Service locally here by calling Mt5 functions directly since we are in the ingestor
        # that knows about MT5.

        # However, to be clean, let's use the service connection.

        # We need to initialize the service first
        import asyncio
        # BaseDataIngestor is synchronous, but our service init is async.
        # This presents a challenge. We might need to run loop here or assume it's initialized.
        # For simplicity in this plan, I'll run the async init synchronously.

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we are already in an event loop (e.g. running from main), we can't run_until_complete.
            # But fetch_data is usually called in a thread pool executor in BaseDataIngestor.
            # So creating a new loop might work or we just use `asyncio.run` if appropriate.
            # Given the context, this fetch_data might be called from a sync context.
            pass

        # NOTE: For now, I'll rely on the service to be initialized at startup.

        try:
            # Ensure MT5 is initialized in this thread
            if not Mt5.initialize():
                logger.error("Failed to initialize MT5 in ingestor thread")
                return pd.DataFrame()

            # Login is persistent across threads usually if initialized, but we rely on service for login details
            # Ideally we would re-login if needed, but copy_rates doesn't strictly require login for some brokers if connected.
            # However, to be safe, we assume terminal is running and logged in.

            end = end_date or datetime.now()
            rates = Mt5.copy_rates_range(symbol, timeframe, start_date, end)

            if rates is None or len(rates) == 0:
                 logger.warning(f"No data found for {symbol} in MT5")
                 return pd.DataFrame()

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Normalize columns
            df.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume',
                'real_volume': 'RealVolume',
                'spread': 'Spread'
            }, inplace=True)

            df.set_index('Date', inplace=True)

            # Add technical indicators using the base method (if available in YahooFinanceIngestor but not base)
            # BaseDataIngestor doesn't have _add_technical_indicators, YahooFinanceIngestor does.
            # I should duplicate that logic or move it to base.
            # For now, I will copy the logic from YahooFinanceIngestor to ensure consistency.

            df = self._add_technical_indicators(df)

            # Add metadata
            df['symbol'] = symbol
            df['data_source'] = 'mt5'
            df['ingestion_timestamp'] = datetime.now()

            logger.info(f"Fetched {len(df)} records for {symbol} from MT5")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from MT5 for {symbol}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data."""
        import numpy as np

        if data.empty:
            return data

        # Calculate returns
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # Calculate volatility (rolling standard deviation)
        data['volatility_20d'] = data['returns'].rolling(window=20).std()
        data['volatility_5d'] = data['returns'].rolling(window=5).std()

        # Calculate moving averages
        data['sma_20'] = data['Close'].rolling(window=20).mean()
        data['sma_50'] = data['Close'].rolling(window=50).mean()
        data['ema_12'] = data['Close'].ewm(span=12).mean()
        data['ema_26'] = data['Close'].ewm(span=26).mean()

        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        std = data['Close'].rolling(window=20).std()
        data['bb_middle'] = data['sma_20']
        data['bb_upper'] = data['bb_middle'] + (std * 2)
        data['bb_lower'] = data['bb_middle'] - (std * 2)

        # Calculate volume indicators
        data['volume_sma_20'] = data['Volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_20']

        return data
