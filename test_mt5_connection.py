"""
Script to test MetaTrader 5 connection and data fetching.
Use this to verify your MT5 configuration.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test MT5 connection."""
    logger.info("Testing MT5 Connection...")

    try:
        from services.mt5_service import MT5Service
        import MetaTrader5 as Mt5
    except ImportError:
        logger.error("MetaTrader5 package not installed or not available.")
        return False

    service = MT5Service()

    # Attempt initialization
    # Note: If credentials are not set in env vars, it tries to use existing terminal state
    if not os.getenv('MT5_LOGIN'):
        logger.warning("MT5_LOGIN not set. Attempting to connect to running terminal without specific login.")

    # We need to run async init in sync context for this test script
    # MT5Service.initialize is async, but Mt5.initialize is sync.
    # We can just check Mt5.initialize() directly here for the smoke test

    if not Mt5.initialize():
        logger.error(f"MT5 initialize failed, error code = {Mt5.last_error()}")
        return False

    logger.info("MT5 Initialized successfully")

    # Check terminal info
    terminal_info = Mt5.terminal_info()
    if terminal_info:
        logger.info(f"Terminal: {terminal_info.name} (Build {terminal_info.build})")
        logger.info(f"Connected: {terminal_info.connected}")

    # Check account info
    account_info = Mt5.account_info()
    if account_info:
        logger.info(f"Account: {account_info.login} ({account_info.name})")
        logger.info(f"Server: {account_info.server}")
        logger.info(f"Balance: {account_info.balance} {account_info.currency}")

    return True

def test_data_fetching():
    """Test fetching data from MT5."""
    logger.info("\nTesting Data Fetching...")

    try:
        from data.mt5_ingestor import MT5Ingestor
        from data.data_ingestors import DataIngestionConfig

        symbol = "EURUSD" # Common symbol, usually available
        logger.info(f"Fetching data for {symbol}...")

        config = DataIngestionConfig(
            symbols=[symbol],
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
            interval="1h"
        )

        ingestor = MT5Ingestor(config)
        data = ingestor.fetch_data(symbol, config.start_date, config.end_date)

        if not data.empty:
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            print(data.head())
            return True
        else:
            logger.warning(f"No data returned for {symbol}. Try a different symbol that is available in your Market Watch.")
            return False

    except Exception as e:
        logger.error(f"Data fetching failed: {e}")
        return False

def main():
    print("="*50)
    print("MetaTrader 5 Integration Test")
    print("="*50)

    if test_connection():
        test_data_fetching()
    else:
        print("\nConnection test failed. Please check your MT5 installation and configuration.")

    # Shutdown
    try:
        import MetaTrader5 as Mt5
        Mt5.shutdown()
    except:
        pass

if __name__ == "__main__":
    main()
