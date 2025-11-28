"""
Dependency injection for route modules.
All global services are accessed through this module.
"""
from typing import Optional

# Global service references - will be set by main app
db_pool: Optional[any] = None
real_data_service: Optional[any] = None
enhanced_data_manager: Optional[any] = None
alternative_data_manager: Optional[any] = None
data_quality_validator: Optional[any] = None
advanced_ml_manager: Optional[any] = None
model_interpretability: Optional[any] = None
real_time_learning_manager: Optional[any] = None
agent_performance_service: Optional[any] = None
agent_router_service: Optional[any] = None
execution_agent_service: Optional[any] = None
rag_event_agent_service: Optional[any] = None
rl_strategy_agent_service: Optional[any] = None
rl_data_collector: Optional[any] = None
rl_training_service: Optional[any] = None
meta_evaluation_service: Optional[any] = None
latent_pattern_service: Optional[any] = None
individual_agent_service: Optional[any] = None
ensemble_blender_service: Optional[any] = None
enhanced_forecasting_service: Optional[any] = None
automated_rag_service: Optional[any] = None


def init_services(**services):
    """Initialize all service references."""
    global db_pool, real_data_service, enhanced_data_manager, alternative_data_manager
    global data_quality_validator, advanced_ml_manager, model_interpretability
    global real_time_learning_manager, agent_performance_service, agent_router_service
    global execution_agent_service, rag_event_agent_service, rl_strategy_agent_service
    global rl_data_collector, rl_training_service, meta_evaluation_service
    global latent_pattern_service, individual_agent_service, ensemble_blender_service
    global enhanced_forecasting_service, automated_rag_service
    
    for key, value in services.items():
        globals()[key] = value

