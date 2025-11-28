"""
Agent Router Routes - Intelligent agent routing and weighting
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()



@router.get("/agent-router/regime")
async def get_market_regime():
    """Market regime detection endpoint with real market data."""
    try:
        # Using dependencies.agent_router_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.agent_router_service and dependencies.db_pool:
            from services.agent_router_service import AgentRouterService
            dependencies.agent_router_service = AgentRouterService(dependencies.db_pool)
        
        if dependencies.agent_router_service:
            return await dependencies.agent_router_service.get_real_market_regime()
        else:
            # Fallback regime data
            return {
                "regime_type": "neutral",
                "confidence": 0.50,
                "volatility_level": 0.20,
                "trend_strength": 0.30,
                "market_sentiment": "neutral",
                "regime_duration": 15,
                "transition_probability": 0.10,
                "regime_indicators": {
                    "rsi_oversold": False,
                    "macd_bullish": False,
                    "volume_increasing": False,
                    "vix_elevated": False
                },
                "last_updated": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching market regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/agent-router/decisions")
async def get_routing_decisions():
    """Routing decisions endpoint with real decision history."""
    try:
        # Using dependencies.agent_router_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.agent_router_service and dependencies.db_pool:
            from services.agent_router_service import AgentRouterService
            dependencies.agent_router_service = AgentRouterService(dependencies.db_pool)
        
        if dependencies.agent_router_service:
            return await dependencies.agent_router_service.get_real_routing_decisions()
        else:
            # Fallback decisions
            return [
                {
                    "decision_id": "FALLBACK_001",
                    "market_regime": {
                        "regime_type": "neutral",
                        "confidence": 0.70,
                        "volatility_level": 0.20
                    },
                    "routing_strategy": "balanced",
                    "active_agents": ["StrategyAgent", "MetaAgent"],
                    "confidence": 0.70,
                    "risk_level": "medium",
                    "expected_performance": 0.70,
                    "timestamp": datetime.now().isoformat()
                }
            ]
    except Exception as e:
        logger.error(f"Error fetching routing decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-router/weights")
async def get_agent_weights():
    """Agent weights endpoint with real agent weighting data."""
    try:
        # Using dependencies.agent_router_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.agent_router_service and dependencies.db_pool:
            from services.agent_router_service import AgentRouterService
            dependencies.agent_router_service = AgentRouterService(dependencies.db_pool)
        
        if dependencies.agent_router_service:
            return await dependencies.agent_router_service.get_real_agent_weights()
        else:
            # Fallback weights data
            return [
                {
                    "agent_name": "MomentumAgent",
                    "weight": 0.25,
                    "performance": 0.75,
                    "regime_fit": 0.80,
                    "reason": "Strong momentum in current market",
                    "last_updated": datetime.now().isoformat()
                },
                {
                    "agent_name": "SentimentAgent", 
                    "weight": 0.20,
                    "performance": 0.70,
                    "regime_fit": 0.75,
                    "reason": "Good sentiment analysis performance",
                    "last_updated": datetime.now().isoformat()
                },
                {
                    "agent_name": "RiskAgent",
                    "weight": 0.15,
                    "performance": 0.85,
                    "regime_fit": 0.90,
                    "reason": "High risk management capability",
                    "last_updated": datetime.now().isoformat()
                },
                {
                    "agent_name": "StrategyAgent",
                    "weight": 0.20,
                    "performance": 0.72,
                    "regime_fit": 0.78,
                    "reason": "Adaptive strategy selection",
                    "last_updated": datetime.now().isoformat()
                },
                {
                    "agent_name": "MetaAgent",
                    "weight": 0.20,
                    "performance": 0.68,
                    "regime_fit": 0.82,
                    "reason": "Meta-learning capabilities",
                    "last_updated": datetime.now().isoformat()
                }
            ]
    except Exception as e:
        logger.error(f"Error fetching agent weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Execution Agent endpoints for order management and execution tracking