"""
Script to run the AI Market Analysis System on Windows with MetaTrader 5.

Prerequisites:
1. MetaTrader 5 Terminal installed and running.
2. PostgreSQL installed and running (or Docker container running).
3. Python 3.11+ installed.
4. Requirements installed: pip install -r requirements.txt

Usage:
    python run_with_mt5.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Set environment variables for MT5
    os.environ['USE_MT5'] = 'true'

    # Check if we should prompt for credentials or if they are already set
    if not os.getenv('MT5_LOGIN'):
        print("Please enter your MetaTrader 5 credentials (or press Enter to try using existing terminal state):")
        login = input("Login (Account #): ").strip()
        if login:
            os.environ['MT5_LOGIN'] = login
            os.environ['MT5_PASSWORD'] = input("Password: ").strip()
            os.environ['MT5_SERVER'] = input("Server: ").strip()
            path = input("Path to terminal64.exe (optional): ").strip()
            if path:
                os.environ['MT5_PATH'] = path

    # Check for Postgres credentials
    if not os.getenv('POSTGRES_PASSWORD'):
        print("\nChecking PostgreSQL configuration...")
        # Default to localhost if not set
        if not os.getenv('POSTGRES_HOST'):
             os.environ['POSTGRES_HOST'] = 'localhost'

        # Check if we can connect (simple check if user wants to override)
        # For simplicity, we just set default dev password if not present
        os.environ['POSTGRES_PASSWORD'] = os.getenv('POSTGRES_PASSWORD', 'password')
        print(f"Using PostgreSQL at {os.environ['POSTGRES_HOST']}:{os.getenv('POSTGRES_PORT', '5432')}")

    print("\nStarting System with MT5 Integration...")

    # Run start_system_final.py
    # We use sys.executable to ensure we use the same python interpreter
    try:
        subprocess.run([sys.executable, "start_system_final.py"], check=True)
    except KeyboardInterrupt:
        print("\nSystem stopped by user.")
    except Exception as e:
        print(f"\nError running system: {e}")

if __name__ == "__main__":
    main()
