"""
Agent Monitor Routes - Performance tracking and feedback
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()


@router.get("/agent-monitor")
async def get_agent_monitor_summary():
    """Agent monitor summary endpoint with real data collection."""
    try:
        # Using dependencies.agent_performance_service
        
        # Initialize service if not already done
        if not dependencies.agent_performance_service:
            from services.agent_performance_service import AgentPerformanceService
            dependencies.agent_performance_service = AgentPerformanceService(dependencies.db_pool)
        
        # Get real summary data
        summary = await dependencies.agent_performance_service.get_agent_monitor_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Error fetching agent monitor summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/agent-monitor/performance")
async def get_agent_performance_metrics():
    """Agent performance metrics endpoint with real data collection."""
    try:
        # Using dependencies.agent_performance_service
        
        # Initialize service if not already done
        if not dependencies.agent_performance_service:
            from services.agent_performance_service import AgentPerformanceService
            dependencies.agent_performance_service = AgentPerformanceService(dependencies.db_pool)
        
        # Get real performance data
        performance_data = await dependencies.agent_performance_service.calculate_real_performance_metrics()
        
        # Convert to API format
        metrics = []
        for performance in performance_data:
            metrics.append({
                "agent_name": performance.agent_name,
                "total_predictions": performance.total_predictions,
                "correct_predictions": performance.correct_predictions,
                "accuracy": performance.accuracy,
                "avg_confidence": performance.avg_confidence,
                "sharpe_ratio": performance.sharpe_ratio,
                "win_rate": performance.win_rate,
                "health_score": performance.health_score,
                "performance_trend": performance.performance_trend,
                "last_prediction_time": performance.last_prediction_time.isoformat() if performance.last_prediction_time else None
            })
        
        return metrics
    except Exception as e:
        logger.error(f"Error fetching agent performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/agent-monitor/online-learning")
async def get_online_learning_status():
    """Online learning status endpoint with real data collection."""
    try:
        # Using dependencies.agent_performance_service
        
        # Initialize service if not already done
        if not dependencies.agent_performance_service:
            from services.agent_performance_service import AgentPerformanceService
            dependencies.agent_performance_service = AgentPerformanceService(dependencies.db_pool)
        
        # Get real learning status data
        learning_data = await dependencies.agent_performance_service.collect_real_learning_status()
        
        # Convert to API format
        status = []
        for learning in learning_data:
            status.append({
                "agent_name": learning.agent_name,
                "model_type": learning.model_type,
                "model_accuracy": learning.model_accuracy,
                "training_samples": learning.training_samples,
                "is_training": learning.is_training,
                "last_training": learning.last_training.isoformat(),
                "learning_rate": learning.learning_rate,
                "epochs_completed": learning.epochs_completed
            })
        
        return status
    except Exception as e:
        logger.error(f"Error fetching online learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-monitor/feedback")
async def get_agent_feedback():
    """Agent feedback endpoint with real data collection."""
    try:
        # Using dependencies.agent_performance_service
        
        # Initialize service if not already done
        if not dependencies.agent_performance_service:
            from services.agent_performance_service import AgentPerformanceService
            dependencies.agent_performance_service = AgentPerformanceService(dependencies.db_pool)
        
        # Get real feedback data
        feedback_data = await dependencies.agent_performance_service.collect_real_feedback_data(limit=20)
        
        # Convert to API format
        feedback = []
        for fb in feedback_data:
            feedback.append({
                "agent_name": fb.agent_name,
                "predicted_signal": fb.predicted_signal,
                "actual_outcome": fb.actual_outcome,
                "feedback_score": fb.feedback_score,
                "timestamp": fb.timestamp.isoformat()
            })
        
        # Sort by timestamp (most recent first)
        feedback.sort(key=lambda x: x["timestamp"], reverse=True)
        return feedback
    except Exception as e:
        logger.error(f"Error fetching agent feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent Router endpoints for intelligent agent routing