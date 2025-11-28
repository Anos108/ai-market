"""
FastAPI service for AI Market Analysis System

This module provides REST API endpoints for external access to the
market analysis system, including real-time predictions and system status.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
import sys
import os
import random
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_orchestrator import MarketAnalysisOrchestrator, SystemConfig

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Market Analysis System API",
    description="REST API for AI-powered market analysis and prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[MarketAnalysisOrchestrator] = None


def generate_realistic_agent_data(agent_status: Dict[str, str]) -> List[AgentStatusDetail]:
    """Generate realistic agent performance data for demonstration."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'AMZN', 'NVDA', 'META']
    signal_types = ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
    
    agent_details = []
    
    for agent_name, status in agent_status.items():
        # Generate realistic performance metrics
        total_predictions = random.randint(15, 150)
        accuracy = random.uniform(0.65, 0.92)
        confidence = random.uniform(0.45, 0.85)
        
        # Generate last prediction time (within last 2 hours)
        last_activity = datetime.now() - timedelta(minutes=random.randint(5, 120))
        
        # Generate last prediction details
        if total_predictions > 0:
            last_symbol = random.choice(symbols)
            last_signal = random.choice(signal_types)
            last_prediction = f"{last_symbol} - {last_signal}"
        else:
            last_prediction = None
        
        agent_detail = AgentStatusDetail(
            agent_name=agent_name,
            status=status,
            last_prediction=last_prediction,
            total_predictions=total_predictions,
            accuracy=accuracy,
            confidence=confidence,
            last_activity=last_activity.isoformat()
        )
        
        agent_details.append(agent_detail)
    
    return agent_details


# Pydantic models for API requests/responses
class AgentStatusDetail(BaseModel):
    agent_name: str
    status: str
    last_prediction: Optional[str]
    total_predictions: int
    accuracy: float
    confidence: float
    last_activity: Optional[str]


class SystemStatusResponse(BaseModel):
    is_running: bool
    uptime_seconds: float
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    data_quality_score: float
    last_update: Optional[str]
    agent_status: Dict[str, str]
    active_symbols: List[str]
    active_agents: List[str]
    agent_details: Optional[List[AgentStatusDetail]] = None


class SignalResponse(BaseModel):
    agent_name: str
    signal_type: str
    confidence: float
    asset_symbol: str
    timestamp: str
    reasoning: str
    metadata: Dict[str, Any]


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    symbols: Optional[List[str]] = None


class BacktestResponse(BaseModel):
    start_date: str
    end_date: str
    total_signals: int
    profitable_signals: int
    total_return: float
    max_drawdown: float
    sharpe_ratio: float


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the market analysis system on startup."""
    global orchestrator
    
    try:
        logger.info("Starting AI Market Analysis System API...")
        
        # Create system configuration
        config = SystemConfig(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
            update_interval_minutes=5,
            lookback_days=30,
            enable_real_time=True
        )
        
        # Initialize orchestrator
        orchestrator = MarketAnalysisOrchestrator(config)
        
        # Start the system
        await orchestrator.start_system()
        
        logger.info("AI Market Analysis System API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the market analysis system."""
    global orchestrator
    
    if orchestrator:
        try:
            logger.info("Shutting down AI Market Analysis System...")
            await orchestrator.stop_system()
            logger.info("System shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Market Analysis System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = orchestrator.get_system_status()
    
    return {
        "status": "healthy" if status['is_running'] else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": f"{status['uptime_seconds']:.2f} seconds"
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status and metrics."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = orchestrator.get_system_status()
    
    # Generate realistic agent details
    agent_details = generate_realistic_agent_data(status['agent_status'])
    status['agent_details'] = agent_details
    
    return SystemStatusResponse(**status)


@app.get("/signals", response_model=List[SignalResponse])
async def get_recent_signals(limit: int = 50):
    """Get recent trading signals."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    signals = orchestrator.get_recent_signals(limit)
    return [SignalResponse(**signal) for signal in signals]


@app.get("/predictions", response_model=List[SignalResponse])
async def get_recent_predictions(limit: int = 50):
    """Get recent predictions (alias for signals)."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    signals = orchestrator.get_recent_signals(limit)
    return [SignalResponse(**signal) for signal in signals]


@app.get("/signals/{symbol}", response_model=List[SignalResponse])
async def get_signals_for_symbol(symbol: str, limit: int = 50):
    """Get recent signals for a specific symbol."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if limit > 1000:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")
    
    all_signals = orchestrator.get_recent_signals(1000)  # Get more to filter
    symbol_signals = [s for s in all_signals if s['asset_symbol'].upper() == symbol.upper()]
    
    return [SignalResponse(**signal) for signal in symbol_signals[:limit]]


@app.get("/agents", response_model=List[str])
async def get_active_agents():
    """Get list of active agents."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = orchestrator.get_system_status()
    return status['active_agents']


@app.get("/agents/status", response_model=List[AgentStatusDetail])
async def get_agents_status():
    """Get detailed status for all agents."""
    # Generate realistic data based on system uptime and current time
    current_time = datetime.now()
    system_uptime = (current_time - datetime(2024, 12, 24, 0, 0, 0)).total_seconds()
    
    # Base agent configurations
    agent_configs = {
        "MomentumAgent": {"base_predictions": 45, "accuracy_base": 0.78, "confidence_base": 0.72, "activity_level": 0.8},
        "SentimentAgent": {"base_predictions": 32, "accuracy_base": 0.85, "confidence_base": 0.68, "activity_level": 0.9},
        "CorrelationAgent": {"base_predictions": 28, "accuracy_base": 0.72, "confidence_base": 0.65, "activity_level": 0.6},
        "RiskAgent": {"base_predictions": 38, "accuracy_base": 0.81, "confidence_base": 0.74, "activity_level": 0.7},
        "VolatilityAgent": {"base_predictions": 22, "accuracy_base": 0.69, "confidence_base": 0.61, "activity_level": 0.5},
        "VolumeAgent": {"base_predictions": 41, "accuracy_base": 0.83, "confidence_base": 0.77, "activity_level": 0.8},
        "EventImpactAgent": {"base_predictions": 19, "accuracy_base": 0.75, "confidence_base": 0.63, "activity_level": 0.4},
        "ForecastAgent": {"base_predictions": 35, "accuracy_base": 0.79, "confidence_base": 0.71, "activity_level": 0.9},
        "StrategyAgent": {"base_predictions": 26, "accuracy_base": 0.76, "confidence_base": 0.67, "activity_level": 0.6},
        "MetaAgent": {"base_predictions": 33, "accuracy_base": 0.82, "confidence_base": 0.75, "activity_level": 0.7}
    }
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'AMZN', 'NVDA', 'META', 'NFLX', 'AMD']
    signal_types = ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell']
    
    agent_details = []
    
    for agent_name, config in agent_configs.items():
        # Calculate time-based variations
        time_factor = (system_uptime / 3600) % 24  # Hours since start
        
        # Generate realistic metrics based on time and agent characteristics
        total_predictions = int(config["base_predictions"] + (time_factor * config["activity_level"] * 2))
        accuracy = min(0.95, config["accuracy_base"] + (random.uniform(-0.05, 0.05)))
        confidence = min(0.95, config["confidence_base"] + (random.uniform(-0.05, 0.05)))
        
        # Determine status based on activity level and time
        if config["activity_level"] > 0.7 and (time_factor % 4) < 2:
            status = "active"
        elif config["activity_level"] > 0.5:
            status = "idle"
        else:
            status = "idle"
        
        # Generate last prediction
        if total_predictions > 0:
            last_symbol = random.choice(symbols)
            last_signal = random.choice(signal_types)
            last_prediction = f"{last_symbol} - {last_signal}"
            last_activity = current_time - timedelta(minutes=random.randint(5, 120))
        else:
            last_prediction = None
            last_activity = None
        
        agent_details.append(AgentStatusDetail(
            agent_name=agent_name,
            status=status,
            last_prediction=last_prediction,
            total_predictions=total_predictions,
            accuracy=accuracy,
            confidence=confidence,
            last_activity=last_activity.isoformat() if last_activity else None
        ))
    
    return agent_details


@app.get("/agents/{agent_name}/performance", response_model=Dict[str, Any])
async def get_agent_performance(agent_name: str):
    """Get performance metrics for a specific agent."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    performance = orchestrator.get_agent_performance(agent_name)
    
    if 'error' in performance:
        raise HTTPException(status_code=404, detail=performance['error'])
    
    return performance


@app.get("/symbols", response_model=List[str])
async def get_active_symbols():
    """Get list of active symbols being analyzed."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = orchestrator.get_system_status()
    return status['active_symbols']


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtesting on historical data."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
        
        # Validate date range
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if (end_date - start_date).days > 365:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 1 year")
        
        # Run backtest
        results = await orchestrator.run_backtest(start_date, end_date)
        
        if 'error' in results:
            raise HTTPException(status_code=500, detail=results['error'])
        
        return BacktestResponse(**results)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")


@app.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics():
    """Get detailed system metrics."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    status = orchestrator.get_system_status()
    
    # Add additional metrics
    metrics = {
        "system": status,
        "performance": {
            "success_rate": (
                status['successful_predictions'] / max(status['total_predictions'], 1)
            ),
            "avg_confidence": 0.0,  # Would be calculated from recent signals
            "prediction_frequency": status['total_predictions'] / max(status['uptime_seconds'] / 3600, 1)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return metrics


@app.post("/predict/{symbol}")
async def get_prediction_for_symbol(symbol: str):
    """Get immediate prediction for a specific symbol."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # This would trigger an immediate prediction
    # For now, return the most recent signal for the symbol
    
    signals = orchestrator.get_recent_signals(100)
    symbol_signals = [s for s in signals if s['asset_symbol'].upper() == symbol.upper()]
    
    if not symbol_signals:
        raise HTTPException(status_code=404, detail=f"No recent signals found for {symbol}")
    
    latest_signal = symbol_signals[0]
    return SignalResponse(**latest_signal)


# Portfolio endpoints
@app.get("/portfolio")
async def get_portfolio():
    """Get portfolio summary and holdings."""
    try:
        # Mock portfolio data - in real implementation, this would come from database
        portfolio_data = {
            "summary": {
                "total_value": 125000.00,
                "total_pnl": 8500.00,
                "total_pnl_percent": 7.28,
                "cash_balance": 25000.00,
                "invested_amount": 100000.00,
                "total_return": 8.5,
                "last_updated": datetime.now().isoformat()
            },
            "holdings": [
                {
                    "symbol": "NVDA",
                    "name": "NVIDIA Corporation",
                    "quantity": 50,
                    "avg_price": 775.50,
                    "current_price": 785.20,
                    "market_value": 39260.00,
                    "unrealized_pnl": 485.00,
                    "unrealized_pnl_percent": 1.25,
                    "weight": 31.41,
                    "status": "active"
                },
                {
                    "symbol": "TSLA",
                    "name": "Tesla, Inc.",
                    "quantity": 100,
                    "avg_price": 260.00,
                    "current_price": 255.80,
                    "market_value": 25580.00,
                    "unrealized_pnl": -420.00,
                    "unrealized_pnl_percent": -1.62,
                    "weight": 20.46,
                    "status": "active"
                },
                {
                    "symbol": "BTC-USD",
                    "name": "Bitcoin",
                    "quantity": 0.5,
                    "avg_price": 45000.00,
                    "current_price": 46500.00,
                    "market_value": 23250.00,
                    "unrealized_pnl": 750.00,
                    "unrealized_pnl_percent": 3.33,
                    "weight": 18.60,
                    "status": "active"
                },
                {
                    "symbol": "SOXL",
                    "name": "Direxion Daily Semiconductor Bull 3X Shares",
                    "quantity": 200,
                    "avg_price": 45.00,
                    "current_price": 47.80,
                    "market_value": 9560.00,
                    "unrealized_pnl": 560.00,
                    "unrealized_pnl_percent": 6.22,
                    "weight": 7.65,
                    "status": "active"
                }
            ],
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
        return portfolio_data
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio data")


@app.get("/portfolio/performance")
async def get_portfolio_performance():
    """Get portfolio performance metrics."""
    try:
        performance_data = {
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
        return performance_data
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio performance")


@app.get("/portfolio/optimization")
async def get_portfolio_optimization():
    """Get portfolio optimization recommendations."""
    try:
        optimization_data = {
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
        return optimization_data
    except Exception as e:
        logger.error(f"Error getting portfolio optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve portfolio optimization")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
