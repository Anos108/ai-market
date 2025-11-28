"""
Risk Analysis Routes - Risk assessment and management
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import asyncio

from routes import dependencies

router = APIRouter()


@router.get("/risk-analysis")
async def get_risk_analysis():
    """Get comprehensive risk analysis."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get portfolio data for risk calculation
                managed_symbols = await conn.fetch("""
                    SELECT ms.symbol, ms.initial_price, ms.status, s.sector, s.industry
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    WHERE ms.status IN ('active', 'monitoring')
                """)
                
                if not managed_symbols:
                    return {
                        "overall_risk_score": 0.0,
                        "market_risk": {"volatility": 0.0},
                        "portfolio_risk": {"total_risk": 0.0},
                        "liquidity_risk": 0.0,
                        "last_updated": datetime.now().isoformat(),
                        "total_risk": 0.0,
                        "risk_metrics": {
                            "var_95": 0.0,
                            "var_99": 0.0,
                            "expected_shortfall": 0.0,
                            "stress_test": 0.0
                        }
                    }
                
                # Calculate portfolio risk based on holdings
                total_symbols = len(managed_symbols)
                crypto_symbols = len([s for s in managed_symbols if "-USD" in s['symbol']])
                sectors = len(set(s['sector'] for s in managed_symbols if s['sector']))
                
                # Risk calculation based on portfolio composition
                crypto_risk = (crypto_symbols / total_symbols) * 0.8  # Crypto is riskier
                concentration_risk = max(0, 0.5 - (sectors / total_symbols))  # Fewer sectors = higher risk
                size_risk = max(0, 0.3 - (total_symbols / 10))  # Fewer holdings = higher risk
                
                portfolio_risk = min(1.0, crypto_risk + concentration_risk + size_risk)
                
                # Market volatility from recent ensemble signals
                market_volatility = await conn.fetchval("""
                    SELECT AVG(blended_confidence) 
                    FROM ensemble_signals 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """) or 0.5
                
                # VaR calculation (simplified)
                var_95 = portfolio_risk * 10  # Simplified VaR calculation
                var_99 = var_95 * 1.5
                
                return {
                    "overall_risk_score": round(portfolio_risk, 2),
                    "market_risk": {"volatility": round(float(market_volatility) * 100, 2)},
                    "portfolio_risk": {"total_risk": round(portfolio_risk * 100, 2)},
                    "liquidity_risk": round(concentration_risk * 100, 2),
                    "last_updated": datetime.now().isoformat(),
                    "total_risk": round(portfolio_risk, 2),
                    "risk_metrics": {
                        "var_95": round(var_95, 1),
                        "var_99": round(var_99, 1),
                        "expected_shortfall": round(var_95 * 1.2, 1),
                        "stress_test": round(portfolio_risk * 15, 1)
                    }
                }
        
        # Fallback if no database
        return {
            "overall_risk_score": 0.0,
            "market_risk": {"volatility": 0.0},
            "portfolio_risk": {"total_risk": 0.0},
            "liquidity_risk": 0.0,
            "last_updated": datetime.now().isoformat(),
            "total_risk": 0.0,
            "risk_metrics": {
                "var_95": 0.0,
                "var_99": 0.0,
                "expected_shortfall": 0.0,
                "stress_test": 0.0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting risk analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-analysis/metrics")
async def get_risk_metrics():
    """Get detailed risk metrics."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get portfolio performance data
                managed_symbols = await conn.fetch("""
                    SELECT ms.symbol, ms.initial_price, ms.status
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    WHERE ms.status IN ('active', 'monitoring')
                """)
                
                if not managed_symbols:
                    return {
                        "var_95": 0.0,
                        "var_99": 0.0,
                        "max_drawdown": 0.0,
                        "sharpe_ratio": 0.0,
                        "beta": 1.0
                    }
                
                # Calculate portfolio composition risk
                total_symbols = len(managed_symbols)
                crypto_symbols = len([s for s in managed_symbols if "-USD" in s['symbol']])
                
                # Risk-adjusted metrics
                portfolio_volatility = (crypto_symbols / total_symbols) * 0.3 + 0.1  # Base volatility
                expected_return = 0.08  # 8% expected annual return
                risk_free_rate = 0.02   # 2% risk-free rate
                
                sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
                max_drawdown = portfolio_volatility * 0.4  # Estimated max drawdown
                beta = 1.0 + (crypto_symbols / total_symbols) * 0.5  # Higher beta for crypto exposure
                
                # VaR calculations
                var_95 = portfolio_volatility * 1.645  # 95% VaR
                var_99 = portfolio_volatility * 2.326  # 99% VaR
                
                return {
                    "var_95": round(var_95 * 100, 1),
                    "var_99": round(var_99 * 100, 1),
                    "max_drawdown": round(max_drawdown * 100, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "beta": round(beta, 2)
                }
        
        # Fallback
        return {
            "var_95": 0.0,
            "var_99": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "beta": 1.0
        }
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-analysis/market")
async def get_market_risk():
    """Get market risk analysis."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get market regime from recent ensemble signals
                market_regime = await conn.fetchval("""
                    SELECT regime 
                    FROM ensemble_signals 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY created_at DESC 
                    LIMIT 1
                """) or "neutral"
                
                # Get volatility from recent signals
                volatility = await conn.fetchval("""
                    SELECT STDDEV(blended_confidence) 
                    FROM ensemble_signals 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """) or 0.15
                
                # Get sector concentration
                sectors = await conn.fetch("""
                    SELECT DISTINCT s.sector 
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    WHERE ms.status IN ('active', 'monitoring') AND s.sector IS NOT NULL
                """)
                
                sector_concentration = len(sectors) / 10.0 if len(sectors) > 0 else 0.1  # Normalize to 10 sectors
                
                return {
                    "volatility": round(float(volatility) * 100, 2),
                    "correlation": 0.65,  # Market correlation
                    "sector_concentration": round(sector_concentration * 100, 1),
                    "geographic_concentration": 75.0,  # US-focused portfolio
                    "market_regime": market_regime or "neutral"
                }
        
        # Fallback
        return {
            "volatility": 15.0,
            "correlation": 0.65,
            "sector_concentration": 45.0,
            "geographic_concentration": 75.0,
            "market_regime": "neutral"
        }
        
    except Exception as e:
        logger.error(f"Error getting market risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-analysis/alerts")
async def get_risk_alerts():
    """Get active risk alerts."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Check for high-risk conditions
                alerts = []
                
                # Check portfolio concentration
                managed_symbols = await conn.fetch("""
                    SELECT COUNT(*) as count, s.sector
                    FROM managed_symbols ms
                    JOIN symbols s ON ms.symbol = s.symbol
                    WHERE ms.status IN ('active', 'monitoring') AND s.sector IS NOT NULL
                    GROUP BY s.sector
                    ORDER BY count DESC
                """)
                
                if managed_symbols:
                    max_concentration = max(row['count'] for row in managed_symbols)
                    total_symbols = sum(row['count'] for row in managed_symbols)
                    
                    if total_symbols > 0 and (max_concentration / total_symbols) > 0.4:
                        alerts.append({
                            "type": "concentration",
                            "message": f"High sector concentration detected ({max_concentration}/{total_symbols} symbols in one sector)",
                            "timestamp": datetime.now().isoformat(),
                            "severity": "warning"
                        })
                
                # Check for high volatility
                recent_volatility = await conn.fetchval("""
                    SELECT STDDEV(blended_confidence) 
                    FROM ensemble_signals 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                
                if recent_volatility and recent_volatility > 0.3:
                    alerts.append({
                        "type": "volatility",
                        "message": f"High market volatility detected ({recent_volatility:.1%})",
                        "timestamp": datetime.now().isoformat(),
                        "severity": "warning"
                    })
                
                return alerts
        
        # No alerts
        return []
        
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        return []
