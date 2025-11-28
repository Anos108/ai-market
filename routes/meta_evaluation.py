"""
Meta-Evaluation Routes - Agent performance evaluation and rotation
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()



@router.get("/meta-evaluation/rankings")
async def get_agent_rankings(regime: str = "neutral"):
    """Get agent rankings for a specific regime."""
    try:
        # Using dependencies.meta_evaluation_service, dependencies.db_pool
        if dependencies.meta_evaluation_service:
            async with dependencies.db_pool.acquire() as conn:
                rankings = await conn.fetch("""
                    SELECT * FROM meta_agent_rankings 
                    WHERE regime = $1 
                    ORDER BY rank ASC
                """, regime)
                
                return [
                    {
                        "agent_name": r['agent_name'],
                        "rank": r['rank'],
                        "composite_score": float(r['composite_score']),
                        "accuracy": float(r['accuracy']),
                        "sharpe_ratio": float(r['sharpe_ratio']),
                        "total_return": float(r['total_return']),
                        "max_drawdown": float(r['max_drawdown']),
                        "win_rate": float(r['win_rate']),
                        "confidence": float(r['confidence']),
                        "response_time": float(r['response_time']),
                        "created_at": r['created_at'].isoformat()
                    } for r in rankings
                ]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting agent rankings: {e}")
        return []



@router.get("/meta-evaluation/regime-analysis")
async def get_regime_analysis():
    """Get current market regime analysis."""
    try:
        # Using dependencies.meta_evaluation_service, dependencies.db_pool
        if dependencies.meta_evaluation_service:
            async with dependencies.db_pool.acquire() as conn:
                analysis = await conn.fetchrow("""
                    SELECT * FROM meta_regime_analysis 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                if analysis:
                    return {
                        "regime": analysis['regime'],
                        "confidence": float(analysis['confidence']),
                        "volatility": float(analysis['volatility']),
                        "trend_strength": float(analysis['trend_strength']),
                        "volume_ratio": float(analysis['volume_ratio']),
                        "trend_direction": analysis['trend_direction'],
                        "market_indicators": analysis['market_indicators'],
                        "created_at": analysis['created_at'].isoformat()
                    }
                else:
                    return {
                        "regime": "neutral",
                        "confidence": 0.6,
                        "volatility": 0.15,
                        "trend_strength": 0.02,
                        "volume_ratio": 1.0,
                        "trend_direction": "neutral",
                        "market_indicators": {},
                        "created_at": datetime.now().isoformat()
                    }
        else:
            return {"error": "Meta-evaluation service not available"}
    except Exception as e:
        logger.error(f"Error getting regime analysis: {e}")
        return {"error": str(e)}

@router.get("/meta-evaluation-agent")
async def get_meta_evaluation_summary():
    """Get comprehensive Meta-Evaluation Agent summary."""
    try:
        # Using dependencies.meta_evaluation_service, dependencies.db_pool
        if dependencies.meta_evaluation_service and dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get performance summary
                performance_summary = await conn.fetchrow("""
                    SELECT 
                        AVG(accuracy) as avg_accuracy,
                        COUNT(DISTINCT agent_name) as total_agents
                    FROM meta_agent_performance 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                
                # Get current regime
                current_regime = await conn.fetchrow("""
                    SELECT regime, confidence 
                    FROM meta_regime_analysis 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                
                # Get recent rotations
                recent_rotations = await conn.fetch("""
                    SELECT * FROM meta_rotation_decisions 
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                    ORDER BY created_at DESC 
                    LIMIT 10
                """)
                
                return {
                    "total_evaluations": performance_summary['total_agents'] or 0,
                    "active_evaluations": 3,
                    "evaluation_accuracy": float(performance_summary['avg_accuracy']) if performance_summary['avg_accuracy'] else 0.0,
                    "last_evaluation": datetime.now().isoformat(),
                    "regime_analysis_enabled": True,
                    "last_updated": datetime.now().isoformat(),
                    "performance_summary": {
                        "avg_accuracy": float(performance_summary['avg_accuracy']) if performance_summary['avg_accuracy'] else 0.0,
                        "total_agents": performance_summary['total_agents'] or 0
                    },
                    "current_regime": current_regime['regime'] if current_regime else 'neutral',
                    "recent_rotations": [
                        {
                            "id": str(r['id']),
                            "from_agent": r['from_agent'],
                            "to_agent": r['to_agent'],
                            "reason": r['reason'],
                            "confidence": float(r['confidence']),
                            "timestamp": r['created_at'].isoformat()
                        } for r in recent_rotations
                    ]
                }
        else:
            # Fallback data
            return {
                "total_evaluations": 0,
                "active_evaluations": 0,
                "evaluation_accuracy": 0.0,
                "last_evaluation": datetime.now().isoformat(),
                "regime_analysis_enabled": True,
                "last_updated": datetime.now().isoformat(),
                "performance_summary": {"avg_accuracy": 0.0, "total_agents": 0},
                "current_regime": "neutral",
                "recent_rotations": []
            }
    except Exception as e:
        logger.error(f"Error getting meta-evaluation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/meta-evaluation-agent/rotations")
async def get_rotation_decisions(limit: int = 10):
    """Get recent rotation decisions."""
    try:
        # Using dependencies.meta_evaluation_service, dependencies.db_pool
        if dependencies.meta_evaluation_service and dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                rotations = await conn.fetch("""
                    SELECT * FROM meta_rotation_decisions 
                    ORDER BY created_at DESC 
                    LIMIT $1
                """, limit)
                
                return [
                    {
                        "id": str(r['id']),
                        "from_agent": r['from_agent'],
                        "to_agent": r['to_agent'],
                        "reason": r['reason'],
                        "confidence": float(r['confidence']),
                        "timestamp": r['created_at'].isoformat()
                    } for r in rotations
                ]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting rotation decisions: {e}")
        return []

# Latent Pattern Detector endpoints