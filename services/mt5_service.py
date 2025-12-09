"""
MetaTrader 5 Service
Provides integration with MetaTrader 5 terminal for data and execution.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import asyncio

# Import MetaTrader5 conditionally to avoid import errors on non-Windows systems during dev/CI
try:
    import MetaTrader5 as Mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

logger = logging.getLogger(__name__)

class MT5Service:
    """Service for interacting with MetaTrader 5."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MT5Service, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self.login = int(os.getenv('MT5_LOGIN', '0'))
        self.password = os.getenv('MT5_PASSWORD', '')
        self.server = os.getenv('MT5_SERVER', '')
        self.path = os.getenv('MT5_PATH', None) # Path to terminal.exe if needed
        self.initialized = True
        self.connected = False

    async def initialize(self) -> bool:
        """Initialize connection to MT5 terminal."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not installed or not supported on this OS.")
            return False

        if self.connected:
            return True

        try:
            # Initialize MT5
            if self.path:
                init_result = Mt5.initialize(path=self.path)
            else:
                init_result = Mt5.initialize()

            if not init_result:
                logger.error(f"MT5 initialize failed, error code = {Mt5.last_error()}")
                return False

            # Login if credentials provided
            if self.login and self.password and self.server:
                authorized = Mt5.login(self.login, password=self.password, server=self.server)
                if not authorized:
                    logger.error(f"MT5 login failed, error code = {Mt5.last_error()}")
                    return False
                logger.info(f"Connected to MT5 account #{self.login} on {self.server}")
            else:
                logger.info("MT5 initialized without login (using existing terminal state)")

            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            return False

    def shutdown(self):
        """Shutdown MT5 connection."""
        if MT5_AVAILABLE and self.connected:
            Mt5.shutdown()
            self.connected = False
            logger.info("MT5 connection closed")

    async def fetch_data(self, symbol: str, timeframe: int, start_pos: int = 0, count: int = 1000) -> pd.DataFrame:
        """
        Fetch historical data from MT5.

        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant (e.g. Mt5.TIMEFRAME_D1)
            start_pos: Starting position (0 is latest)
            count: Number of bars to fetch
        """
        if not self.connected:
            if not await self.initialize():
                return pd.DataFrame()

        try:
            rates = Mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

            if rates is None:
                logger.error(f"Failed to get rates for {symbol}, error: {Mt5.last_error()}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Rename columns to match system standard
            df.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume',
                'spread': 'Spread',
                'real_volume': 'RealVolume'
            }, inplace=True)

            # Set index
            df.set_index('Date', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching data from MT5 for {symbol}: {e}")
            return pd.DataFrame()

    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            if not await self.initialize():
                return {}

        info = Mt5.account_info()
        if info is None:
            return {}

        return info._asdict()

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions."""
        if not self.connected:
            if not await self.initialize():
                return []

        positions = Mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append(pos._asdict())
        return result

    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get active orders."""
        if not self.connected:
            if not await self.initialize():
                return []

        orders = Mt5.orders_get()
        if orders is None:
            return []

        result = []
        for order in orders:
            result.append(order._asdict())
        return result

    async def execute_order(self, symbol: str, action_type: str, volume: float,
                           price: Optional[float] = None, sl: Optional[float] = None,
                           tp: Optional[float] = None, comment: str = "") -> Dict[str, Any]:
        """
        Execute an order on MT5.

        Args:
            symbol: Symbol name
            action_type: 'buy', 'sell', 'buy_limit', 'sell_limit', etc.
            volume: Lot size
            price: Price for pending orders
            sl: Stop Loss
            tp: Take Profit
            comment: Order comment
        """
        if not self.connected:
            if not await self.initialize():
                return {"retcode": -1, "comment": "Not connected"}

        # Determine order type and price
        tick = Mt5.symbol_info_tick(symbol)
        if not tick:
            return {"retcode": -1, "comment": "Symbol not found"}

        order_type_map = {
            'buy': Mt5.ORDER_TYPE_BUY,
            'sell': Mt5.ORDER_TYPE_SELL,
            'buy_limit': Mt5.ORDER_TYPE_BUY_LIMIT,
            'sell_limit': Mt5.ORDER_TYPE_SELL_LIMIT,
            'buy_stop': Mt5.ORDER_TYPE_BUY_STOP,
            'sell_stop': Mt5.ORDER_TYPE_SELL_STOP
        }

        mt5_type = order_type_map.get(action_type.lower())
        if mt5_type is None:
             return {"retcode": -1, "comment": "Invalid order type"}

        # For market orders, use current price
        if action_type == 'buy':
            price = tick.ask
        elif action_type == 'sell':
            price = tick.bid

        request = {
            "action": Mt5.TRADE_ACTION_DEAL if action_type in ['buy', 'sell'] else Mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_type,
            "price": float(price),
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": Mt5.ORDER_TIME_GTC,
            "type_filling": Mt5.ORDER_FILLING_IOC,
        }

        if sl:
            request["sl"] = float(sl)
        if tp:
            request["tp"] = float(tp)

        result = Mt5.order_send(request)
        if result is None:
             return {"retcode": -1, "comment": "Order send failed unknown error"}

        logger.info(f"Order executed: {result.retcode} - {result.comment}")
        return result._asdict()
