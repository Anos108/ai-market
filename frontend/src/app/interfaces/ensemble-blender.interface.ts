// Ensemble Signal Blender Interfaces

export interface AgentWeightData {
  agent_name: string;
  base_weight: number;
  performance_multiplier: number;
  regime_multiplier: number;
  last_updated: string;
}

export interface EnsembleSignalData {
  symbol: string;
  signal_type: string;
  confidence: number;
  blended_confidence: number;
  contributing_agents: string[];
  blend_mode: string;
  regime: string;
  quality_score: number;
  consistency_score: number;
  agreement_score: number;
  timestamp: string;
}

export interface SignalQualityData {
  consistency_score: number;
  agreement_score: number;
  confidence_variance: number;
  regime_alignment: number;
  historical_accuracy: number;
  overall_quality: number;
}

export interface EnsembleBlenderSummary {
  agent_name: string;
  blend_mode: string;
  current_regime: string;
  total_signals_generated: number;
  avg_quality_score: number;
  recent_quality_scores: number[];
  agent_weights: { [key: string]: AgentWeightData };
  regime_history: Array<{ timestamp: string; regime: string }>;
  performance_metrics: {
    total_signals_blended: number;
    avg_contributing_agents: number;
    signal_quality_trend: string;
    regime_adaptation_score: number;
    consistency_score: number;
    agreement_score: number;
    false_positive_reduction: number;
    risk_adjusted_improvement: number;
  };
  last_updated: string;
}

export interface EnsemblePerformanceData {
  signal_quality: {
    avg_quality_score: number;
    quality_trend: string;
    high_quality_signals: number;
    low_quality_signals: number;
    quality_consistency: number;
  };
  blending_effectiveness: {
    avg_contributing_agents: number;
    consensus_rate: number;
    disagreement_rate: number;
    blend_mode_effectiveness: {
      weighted_average: number;
      majority: number;
      max_confidence: number;
      average: number;
    };
  };
  regime_adaptation: {
    current_regime: string;
    regime_accuracy: number;
    regime_transitions: number;
    adaptation_speed: number;
    regime_performance: {
      bull: number;
      bear: number;
      sideways: number;
      volatile: number;
      trending: number;
    };
  };
  risk_management: {
    false_positive_reduction: number;
    false_negative_reduction: number;
    risk_adjusted_improvement: number;
    volatility_reduction: number;
    drawdown_improvement: number;
  };
  agent_contribution: {
    top_contributors: Array<{
      agent: string;
      contribution: number;
    }>;
    weight_stability: number;
    performance_correlation: number;
  };
  last_updated: string;
}
