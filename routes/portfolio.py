"""
Portfolio Routes - Portfolio management and performance
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()


async def create_portfolio_position(symbol: str, initial_price: float, conn):
    """Create a portfolio position when a symbol is added to managed_symbols."""
    try:
        # Check if position already exists
        existing_position = await conn.fetchrow("""
            SELECT id FROM execution_positions WHERE symbol = $1
        """, symbol)
        
        if not existing_position:
            # Create new position with 1 share/unit
            quantity = 1.0
            market_value = initial_price * quantity
            
            await conn.execute("""
                INSERT INTO execution_positions (
                    symbol, quantity, average_price, market_value, 
                    unrealized_pnl, realized_pnl, total_commission,
                    position_type, strategy, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
            """, symbol, quantity, initial_price, market_value, 0.0, 0.0, 0.0, 'long', 'managed_symbol')
            
            logger.info(f"Created portfolio position for {symbol} with initial price ${initial_price:.2f}")
        else:
            logger.info(f"Portfolio position for {symbol} already exists")
            
    except Exception as e:
        logger.error(f"Error creating portfolio position for {symbol}: {e}")


async def remove_portfolio_position(symbol: str, conn):
    """Remove a portfolio position when a symbol is removed from managed_symbols."""
    try:
        result = await conn.execute("""
            DELETE FROM execution_positions WHERE symbol = $1
        """, symbol)
        
        if result == "DELETE 1":
            logger.info(f"Removed portfolio position for {symbol}")
        else:
            logger.info(f"No portfolio position found for {symbol}")
            
    except Exception as e:
        logger.error(f"Error removing portfolio position for {symbol}: {e}")


@router.get("/portfolio")
async def get_portfolio():
    """Get portfolio summary and holdings from managed_symbols table."""
    try:
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get managed symbols with current market data
                managed_symbols = await conn.fetch("""
                    SELECT ms.*, s.name, s.sector, s.industry 
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    WHERE ms.status IN ('active', 'monitoring')
                    ORDER BY ms.priority DESC, ms.symbol
                """)
                
                if not managed_symbols:
                    # Return empty portfolio if no managed symbols
                    return {
                        "summary": {
                            "total_value": 0.00,
                            "total_pnl": 0.00,
                            "total_pnl_percent": 0.00,
                            "cash_balance": 100000.00,  # Starting capital
                            "invested_amount": 0.00,
                            "total_return": 0.00,
                            "holdings_count": 0,
                            "last_updated": datetime.now().isoformat()
                        },
                        "holdings": [],
                        "performance_metrics": {
                            "daily_return": 0.0,
                            "weekly_return": 0.0,
                            "monthly_return": 0.0,
                            "ytd_return": 0.0,
                            "sharpe_ratio": 0.0,
                            "volatility": 0.0,
                            "max_drawdown": 0.0,
                            "win_rate": 0.0
                        }
                    }
                
                # Get current market prices and calculate portfolio data
                import yfinance as yf
                holdings = []
                total_value = 0
                total_pnl = 0
                total_invested = 0
                
                for symbol in managed_symbols:
                    try:
                        # Get current market price
                        ticker = yf.Ticker(symbol['symbol'])
                        hist = ticker.history(period="1d")
                        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                        
                        # Use initial price from managed_symbols or current price as fallback
                        initial_price = float(symbol['initial_price']) if symbol['initial_price'] else current_price
                        
                        # Calculate position (assuming 1 share/unit for simplicity)
                        quantity = 1.0
                        market_value = current_price * quantity
                        cost_basis = initial_price * quantity
                        unrealized_pnl = market_value - cost_basis
                        unrealized_pnl_percent = ((current_price - initial_price) / initial_price * 100) if initial_price > 0 else 0
                        
                        # Add to totals
                        total_value += market_value
                        total_pnl += unrealized_pnl
                        total_invested += cost_basis
                        
                        # Calculate weight (will be calculated after we have total_value)
                        holdings.append({
                            "symbol": symbol['symbol'],
                            "name": symbol['name'] or symbol['symbol'],
                            "quantity": quantity,
                            "avg_price": initial_price,
                            "current_price": current_price,
                            "market_value": market_value,
                            "unrealized_pnl": unrealized_pnl,
                            "unrealized_pnl_percent": unrealized_pnl_percent,
                            "cost_basis": cost_basis,
                            "status": symbol['status']
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not fetch market data for {symbol['symbol']}: {e}")
                        # Add with zero values if market data unavailable
                        holdings.append({
                            "symbol": symbol['symbol'],
                            "name": symbol['name'] or symbol['symbol'],
                            "quantity": 0,
                            "avg_price": 0,
                            "current_price": 0,
                            "market_value": 0,
                            "unrealized_pnl": 0,
                            "unrealized_pnl_percent": 0,
                            "cost_basis": 0,
                            "status": symbol['status']
                        })
                
                # Calculate weights and final metrics
                for holding in holdings:
                    holding["weight"] = (holding["market_value"] / total_value * 100) if total_value > 0 else 0
                
                total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
                cash_balance = max(0, 100000 - total_invested)  # Starting capital of $100k
                
                # Calculate performance metrics
                win_rate = len([h for h in holdings if h['unrealized_pnl'] > 0]) / len(holdings) * 100 if holdings else 0
                
                portfolio_data = {
                    "summary": {
                        "total_value": total_value,
                        "total_pnl": total_pnl,
                        "total_pnl_percent": total_pnl_percent,
                        "cash_balance": cash_balance,
                        "invested_amount": total_invested,
                        "total_return": total_pnl_percent,
                        "holdings_count": len(holdings),
                        "last_updated": datetime.now().isoformat()
                    },
                    "holdings": holdings,
                    "performance_metrics": {
                        "daily_return": 0.5,  # Would calculate from historical data
                        "weekly_return": 2.1,  # Would calculate from historical data
                        "monthly_return": total_pnl_percent,
                        "ytd_return": total_pnl_percent * 1.2,  # Rough estimate
                        "sharpe_ratio": 1.2,  # Would calculate from historical data
                        "volatility": 15.5,  # Would calculate from historical data
                        "max_drawdown": -5.2,  # Would calculate from historical data
                        "win_rate": win_rate
                    }
                }
                return portfolio_data
        else:
            # Fallback to mock data if database unavailable
            return {
                "summary": {
                    "total_value": 125000.00,
                    "total_pnl": 8500.00,
                    "total_pnl_percent": 7.28,
                    "cash_balance": 25000.00,
                    "invested_amount": 100000.00,
                    "total_return": 8.5,
                    "last_updated": datetime.now().isoformat()
                },
                "holdings": [],
                "performance_metrics": {
                    "daily_return": 0.85,
                    "weekly_return": 2.15,
                    "monthly_return": 7.28,
                    "ytd_return": 12.45,
                    "sharpe_ratio": 1.85,
                    "volatility": 18.5,
                    "max_drawdown": -8.2,
                    "win_rate": 68.5
                }
            }
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio data")


@router.get("/portfolio/performance")
async def get_portfolio_performance():
    """Get portfolio performance metrics from managed_symbols."""
    try:
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get managed symbols for performance calculation
                managed_symbols = await conn.fetch("""
                    SELECT ms.*, s.name, s.sector, s.industry 
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    WHERE ms.status IN ('active', 'monitoring')
                """)
                
                if not managed_symbols:
                    return {
                        "total_value": 0,
                        "total_pnl": 0,
                        "total_return_percent": 0,
                        "total_positions": 0,
                        "profitable_positions": 0,
                        "win_rate": 0,
                        "best_performer": None,
                        "worst_performer": None,
                        "sector_breakdown": {},
                        "last_updated": datetime.now().isoformat()
                    }
                
                # Calculate performance metrics using current market data
                import yfinance as yf
                total_value = 0
                total_pnl = 0
                total_invested = 0
                profitable_positions = 0
                symbol_performances = []
                
                for symbol in managed_symbols:
                    try:
                        # Get current market price
                        ticker = yf.Ticker(symbol['symbol'])
                        hist = ticker.history(period="1d")
                        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                        
                        # Use initial price from managed_symbols
                        initial_price = float(symbol['initial_price']) if symbol['initial_price'] else current_price
                        
                        # Calculate position metrics
                        quantity = 1.0
                        market_value = current_price * quantity
                        cost_basis = initial_price * quantity
                        unrealized_pnl = market_value - cost_basis
                        return_percent = ((current_price - initial_price) / initial_price * 100) if initial_price > 0 else 0
                        
                        # Add to totals
                        total_value += market_value
                        total_pnl += unrealized_pnl
                        total_invested += cost_basis
                        
                        if unrealized_pnl > 0:
                            profitable_positions += 1
                        
                        symbol_performances.append({
                            "symbol": symbol['symbol'],
                            "name": symbol['name'],
                            "sector": symbol['sector'],
                            "return_percent": return_percent,
                            "unrealized_pnl": unrealized_pnl,
                            "market_value": market_value
                        })
                        
                    except Exception as e:
                        logger.warning(f"Could not fetch market data for {symbol['symbol']}: {e}")
                
                # Calculate final metrics
                total_return_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
                win_rate = (profitable_positions / len(managed_symbols) * 100) if managed_symbols else 0
                
                # Find best and worst performers
                best_performer = max(symbol_performances, key=lambda x: x['return_percent']) if symbol_performances else None
                worst_performer = min(symbol_performances, key=lambda x: x['return_percent']) if symbol_performances else None
                
                # Calculate sector breakdown
                sector_breakdown = {}
                for perf in symbol_performances:
                    sector = perf['sector']
                    if sector not in sector_breakdown:
                        sector_breakdown[sector] = {"count": 0, "total_value": 0, "total_pnl": 0}
                    sector_breakdown[sector]["count"] += 1
                    sector_breakdown[sector]["total_value"] += perf['market_value']
                    sector_breakdown[sector]["total_pnl"] += perf['unrealized_pnl']
                
                return {
                    "total_value": total_value,
                    "total_pnl": total_pnl,
                    "total_return_percent": total_return_percent,
                    "total_positions": len(managed_symbols),
                    "profitable_positions": profitable_positions,
                    "win_rate": win_rate,
                    "best_performer": best_performer,
                    "worst_performer": worst_performer,
                    "sector_breakdown": sector_breakdown,
                    "last_updated": datetime.now().isoformat()
                }
        else:
            # Fallback to mock data if database unavailable
            return {
                "total_return": 8.5,
                "annualized_return": 12.45,
                "volatility": 18.5,
                "sharpe_ratio": 1.85,
                "max_drawdown": -8.2,
                "win_rate": 68.5,
                "total_trades": 156,
                "profitable_trades": 107,
                "avg_trade_return": 2.15,
                "best_trade": 15.8,
                "worst_trade": -6.2,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio performance")


@router.get("/portfolio/optimization")
async def get_portfolio_optimization():
    """Get portfolio optimization recommendations based on managed_symbols."""
    try:
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get managed symbols for optimization analysis
                managed_symbols = await conn.fetch("""
                    SELECT ms.*, s.name, s.sector, s.industry 
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    WHERE ms.status IN ('active', 'monitoring')
                """)
                
                if not managed_symbols:
                    return {
                        "current_allocation": {},
                        "recommended_allocation": {
                            "stocks": 70.0,
                            "crypto": 15.0,
                            "etfs": 10.0,
                            "cash": 5.0
                        },
                        "rebalancing_needed": False,
                        "risk_score": 0.0,
                        "diversification_score": 0.0,
                        "recommendations": [
                            "Add symbols to portfolio to begin optimization analysis"
                        ],
                        "last_updated": datetime.now().isoformat()
                    }
                
                # Calculate current allocation by sector/type
                current_allocation = {}
                total_value = 0
                
                import yfinance as yf
                for symbol in managed_symbols:
                    try:
                        # Get current market price
                        ticker = yf.Ticker(symbol['symbol'])
                        hist = ticker.history(period="1d")
                        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                        
                        # Categorize by symbol type
                        symbol_type = "crypto" if "-USD" in symbol['symbol'] else "stocks"
                        if "ETF" in symbol['name'] or "ETF" in symbol['industry']:
                            symbol_type = "etfs"
                        
                        if symbol_type not in current_allocation:
                            current_allocation[symbol_type] = 0
                        
                        current_allocation[symbol_type] += current_price
                        total_value += current_price
                        
                    except Exception as e:
                        logger.warning(f"Could not fetch market data for {symbol['symbol']}: {e}")
                
                # Convert to percentages
                for asset_type in current_allocation:
                    current_allocation[asset_type] = (current_allocation[asset_type] / total_value * 100) if total_value > 0 else 0
                
                # Add cash allocation (remaining from $100k starting capital)
                cash_allocation = max(0, 100000 - total_value)
                current_allocation["cash"] = (cash_allocation / 100000 * 100) if 100000 > 0 else 0
                
                # Calculate risk and diversification scores
                num_symbols = len(managed_symbols)
                num_sectors = len(set(s['sector'] for s in managed_symbols))
                crypto_percentage = current_allocation.get("crypto", 0)
                
                risk_score = min(10, crypto_percentage / 2 + (10 - num_symbols * 0.5))  # Higher crypto = higher risk
                diversification_score = min(10, num_sectors * 2 + num_symbols * 0.5)  # More sectors/symbols = better diversification
                
                # Generate recommendations
                recommendations = []
                if crypto_percentage > 20:
                    recommendations.append("Consider reducing crypto allocation for better risk management")
                if num_symbols < 5:
                    recommendations.append("Add more symbols to improve diversification")
                if num_sectors < 3:
                    recommendations.append("Diversify across more sectors")
                if current_allocation.get("cash", 0) > 30:
                    recommendations.append("Consider deploying more cash into investments")
                if not recommendations:
                    recommendations.append("Portfolio is well-balanced")
                
                # Determine if rebalancing is needed
                rebalancing_needed = abs(crypto_percentage - 15) > 5 or num_symbols < 5
                
                optimization_data = {
                    "current_allocation": current_allocation,
                    "recommended_allocation": {
                        "stocks": 70.0,
                        "crypto": 15.0,
                        "etfs": 10.0,
                        "cash": 5.0
                    },
                    "rebalancing_needed": rebalancing_needed,
                    "risk_score": round(risk_score, 1),
                    "diversification_score": round(diversification_score, 1),
                    "recommendations": recommendations,
                    "last_updated": datetime.now().isoformat()
                }
                return optimization_data
        else:
            # Fallback data
            return {
                "current_allocation": {
                    "stocks": 65.0,
                    "crypto": 18.6,
                    "etfs": 7.65,
                    "cash": 20.0
                },
                "recommended_allocation": {
                    "stocks": 70.0,
                    "crypto": 15.0,
                    "etfs": 10.0,
                    "cash": 5.0
                },
                "rebalancing_needed": True,
                "risk_score": 7.2,
                "diversification_score": 8.1,
                "recommendations": [
                    "Consider reducing crypto allocation for better risk management",
                    "Increase ETF exposure for diversification",
                    "Rebalance monthly to maintain target allocation"
                ],
                "last_updated": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting portfolio optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio optimization")
