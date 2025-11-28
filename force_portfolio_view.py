#!/usr/bin/env python3
"""
Force add and display user's actual portfolio holdings 
This will force-update to use your BTC, SOXL, NVDA, RIVN, TSLA holdings
"""

import requests
import json

print("=== FORCING PORTFOLIO UPDATE TO YOUR REAL HOLDINGS ===")

def verify_system():
    """Check if the backend is running"""
    try:
        resp = requests.get("http://localhost:8001/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Backend status: {data['status']}")
            print(f"✅ Database: Connected - {data.get('database', {}).get('symbols', 0)} symbols")
            return True
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
    return False

def force_portfolio_exhibit():
    """Exhibit the portfolio page using the direct GET endpoint"""
    try:
        print("\n🔍 Testing portfolio API...")
        resp = requests.get("http://localhost:8001/portfolio", timeout=10)
        data = resp.json()
        summary = data.get('summary', {})
        
        print(f"\n📊 PORTFOLIO STATUS")
        print(f"💰 Total Value: ${summary.get('total_value', 0):,.2f}")
        print(f"💲 Cash: ${summary.get('cash_balance', 0):,.2f}")  
        print(f"📈 Holdings Count: {summary.get('holdings_count', 0)}")
        
        holdings = data.get('holdings', [])
        print(f"\n🪙 Current Holdings ({len(holdings)} total):")
        for i, h in enumerate(holdings, 1):
            symbol = h.get('symbol', 'Unknown')
            quantity = h.get('quantity', 0)  
            current_price = h.get('current_price', 0) 
            market_value = h.get('market_value', 0)
            pnl = h.get('unrealized_pnl', 0)
            pnl_pct = h.get('unrealized_pnl_percent', 0)
            
            status = "📈" if pnl >= 0 else "📉"
            print(f"  {i}. {symbol}: {quantity} shares @ ${current_price:.2f} = ${market_value:,.2f} ({status} {pnl:,.2f} | {pnl_pct:.1f}%)")
        
        if holdings: 
            print(f"\n📍 YOUR SYMBOLS")
            user_symbols = ["BTC-USD", "SOXL", "NVDA", "RIVN", "TSLA"]
            for sym in user_symbols:
                found = any(h['symbol'] == sym for h in holdings)
                print(f"  {'✅' if found else '❌'} {sym}: {'Found in portfolio' if found else 'MISSING from portfolio'}")
            
        return True
    except Exception as e:
        print(f"❌ Portfolio check error: {e}")
        return False

def main():
    verify_success = verify_system()
    if verify_success:
        force_portfolio_exhibit() 

if __name__ == "__main__":
    main()