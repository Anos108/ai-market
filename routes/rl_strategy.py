"""
RL Strategy Routes - Reinforcement learning strategy optimization
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()


@router.get("/rl-strategy-agent")
async def get_rl_strategy_agent_summary():
    """RL Strategy Agent summary endpoint with real reinforcement learning data."""
    try:
        # Using dependencies.rl_strategy_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rl_strategy_agent_service and dependencies.db_pool:
            from services.rl_strategy_agent_service import RLStrategyAgentService
            dependencies.rl_strategy_agent_service = RLStrategyAgentService(dependencies.db_pool)
        
        if dependencies.rl_strategy_agent_service:
            return await dependencies.rl_strategy_agent_service.get_rl_strategy_agent_summary()
        else:
            # Fallback summary
            return {
                "algorithm": "PPO",
                "is_trained": False,
                "training_episodes": 0,
                "current_epsilon": 0.1,
                "experience_buffer_size": 0,
                "model_accuracy": 0.0,
                "last_training_update": datetime.now().isoformat(),
                "performance_metrics": {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "avg_trade_pnl": 0.0,
                    "volatility": 0.0
                },
                "training_metrics": {
                    "algorithm": "PPO",
                    "episodes_trained": 0,
                    "avg_episode_reward": 0.0,
                    "best_episode_reward": 0.0,
                    "training_loss": 0.0,
                    "exploration_rate": 0.1,
                    "experience_buffer_size": 0
                },
                "recent_actions_count": 0,
                "last_updated": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching RL Strategy Agent summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl-strategy-agent/training")
async def get_rl_training_status():
    """Get RL training status from RL Strategy Agent with real data."""
    try:
        # Using dependencies.rl_strategy_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rl_strategy_agent_service and dependencies.db_pool:
            from services.rl_strategy_agent_service import RLStrategyAgentService
            dependencies.rl_strategy_agent_service = RLStrategyAgentService(dependencies.db_pool)
        
        if dependencies.rl_strategy_agent_service:
            return await dependencies.rl_strategy_agent_service.get_rl_training_status()
        else:
            # Fallback training status
            return {
                "algorithm": "PPO",
                "episodes_trained": 0,
                "avg_episode_reward": 0.0,
                "best_episode_reward": 0.0,
                "convergence_episode": None,
                "training_loss": 0.0,
                "exploration_rate": 0.1,
                "experience_buffer_size": 0,
                "model_accuracy": 0.0,
                "training_duration_seconds": 0,
                "is_converged": False,
                "training_status": "not_started",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching RL training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/rl-strategy-agent/performance")
async def get_rl_performance():
    """Get RL performance metrics with real data."""
    try:
        # Using dependencies.rl_strategy_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rl_strategy_agent_service and dependencies.db_pool:
            from services.rl_strategy_agent_service import RLStrategyAgentService
            dependencies.rl_strategy_agent_service = RLStrategyAgentService(dependencies.db_pool)
        
        if dependencies.rl_strategy_agent_service:
            return await dependencies.rl_strategy_agent_service.get_rl_performance()
        else:
            # Fallback performance data
            return {
                "performance_30d": {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0
                },
                "performance_7d": {
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0
                }
            }
    except Exception as e:
        logger.error(f"Error fetching RL performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rl-strategy-agent/actions")
async def get_rl_actions(limit: int = 50):
    """Get recent RL actions from RL Strategy Agent with real data."""
    try:
        # Using dependencies.rl_strategy_agent_service, dependencies.db_pool
        # Initialize service if not already done
        if not dependencies.rl_strategy_agent_service and dependencies.db_pool:
            from services.rl_strategy_agent_service import RLStrategyAgentService
            dependencies.rl_strategy_agent_service = RLStrategyAgentService(dependencies.db_pool)
        
        if dependencies.rl_strategy_agent_service:
            return await dependencies.rl_strategy_agent_service.get_rl_actions(limit)
        else:
            # Fallback actions
            return []
    except Exception as e:
        logger.error(f"Error fetching RL actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RL Training Pipeline Management endpoints


@router.post("/rl-training/start")
async def start_rl_training():
    """Start the RL training pipeline."""
    try:
        # Using dependencies.rl_training_service
        if dependencies.rl_training_service:
            await dependencies.rl_training_service.start_training()
            return {"status": "success", "message": "RL training started"}
        else:
            return {"status": "error", "message": "RL training service not available"}
    except Exception as e:
        logger.error(f"Error starting RL training: {e}")
        return {"status": "error", "message": str(e)}



@router.get("/rl-training/history")
async def get_rl_training_history():
    """Get RL training history and performance metrics."""
    try:
        # Using dependencies.rl_training_service
        if dependencies.rl_training_service:
            # Ensure JSON serialization
            training_history = []
            for episode in dependencies.rl_training_service.training_history[-100:]:
                serialized_episode = {}
                for key, value in episode.items():
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            serialized_episode[key] = 0.0
                        else:
                            serialized_episode[key] = float(value)
                    elif isinstance(value, datetime):
                        serialized_episode[key] = value.isoformat()
                    else:
                        serialized_episode[key] = str(value)
                training_history.append(serialized_episode)
            
            performance_history = []
            for eval_result in dependencies.rl_training_service.performance_history[-50:]:
                serialized_eval = {}
                for key, value in eval_result.items():
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            serialized_eval[key] = 0.0
                        else:
                            serialized_eval[key] = float(value)
                    elif isinstance(value, datetime):
                        serialized_eval[key] = value.isoformat()
                    else:
                        serialized_eval[key] = str(value)
                performance_history.append(serialized_eval)
            
            return {
                "training_history": training_history,
                "performance_history": performance_history,
                "current_episode": int(dependencies.rl_training_service.current_episode),
                "total_episodes": int(dependencies.rl_training_service.config.max_episodes),
                "is_training": bool(dependencies.rl_training_service.is_training)
            }
        else:
            return {"error": "RL training service not available"}
    except Exception as e:
        logger.error(f"Error getting RL training history: {e}")
        return {"error": str(e)}

# Meta-Evaluation Agent endpoints