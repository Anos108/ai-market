"""
Execution Agent Routes - Order management and execution tracking
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()



@router.get("/execution-agent/orders")
async def get_execution_orders(limit: int = 50):
    """Get recent orders from execution agent with real data."""
    try:
        # Using dependencies.execution_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.execution_agent_service and dependencies.db_pool:
            from services.execution_agent_service import ExecutionAgentService
            dependencies.execution_agent_service = ExecutionAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.execution_agent_service.create_sample_data()
        
        if dependencies.execution_agent_service:
            return await dependencies.execution_agent_service.get_orders(limit)
        else:
            # Fallback orders
            return []
            
    except Exception as e:
        logger.error(f"Error fetching execution orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/execution-agent/strategies")
async def get_execution_strategies():
    """Get execution strategies from execution agent with real data."""
    try:
        # Using dependencies.execution_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.execution_agent_service and dependencies.db_pool:
            from services.execution_agent_service import ExecutionAgentService
            dependencies.execution_agent_service = ExecutionAgentService(dependencies.db_pool)
            # Create sample data if needed
            await dependencies.execution_agent_service.create_sample_data()
        
        if dependencies.execution_agent_service:
            return await dependencies.execution_agent_service.get_execution_strategies()
        else:
            # Fallback strategies
            return []
            
    except Exception as e:
        logger.error(f"Error fetching execution strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG Event Agent endpoints for LLM-RAG powered event analysis