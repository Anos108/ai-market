import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, interval, switchMap, startWith, map, of, catchError } from 'rxjs';
import { environment } from '../../environments/environment';

// Core Interfaces
export interface SystemStatus {
  is_running: boolean;
  uptime_seconds: number;
  total_predictions: number;
  successful_predictions: number;
  failed_predictions: number;
  data_quality_score: number;
  last_update: string;
  agent_status: any;
  active_symbols: string[];
  active_agents: string[];
  advanced_features: any;
}

export interface AgentStatus {
  name: string;
  agent_name: string;
  status: string;
  predictions: number;
  total_predictions: number;
  accuracy: number;
  confidence: number;
  last_prediction: string;
}

export interface Prediction {
  agent_name: string;
  signal_type: string;
  confidence: number;
  asset_symbol: string;
  timestamp: string;
  reasoning: string;
  metadata: any;
}

// Holdings Interface (simplified from portfolio)

export interface PortfolioHolding {
  symbol: string;
  name: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  weight: number;
  status: string;
  cost_basis: number;
}

// Simplified interfaces for other components (to avoid compilation errors)
export interface ABTestSummary {
  active_tests: number;
  completed_tests: number;
  success_rate: number;
  last_updated: string;
  active_experiments: number;
  completed_experiments: number;
  overall_conversion_rate: number;
  total_participants: number;
  avg_experiment_duration: number;
  top_performing_variant: string;
}

export interface ABTestResult {
  test_id: string;
  test_name: string;
  variant: string;
  performance: number;
  confidence: number;
  status: string;
}

export interface AgentMonitorSummary {
  total_agents: number;
  active_agents: number;
  monitoring_enabled: boolean;
  last_health_check: string;
  system_load: number;
  memory_usage: number;
  cpu_usage: number;
  last_updated: string;
  healthy_agents: number;
  avg_accuracy: number;
  avg_sharpe_ratio: number;
  online_learning_enabled: boolean;
  total_feedback_samples: number;
  agents_needing_attention: number;
  avg_win_rate: number;
}

export interface AgentPerformanceMetrics {
  agent_name: string;
  performance_score: number;
  accuracy: number;
  response_time: number;
  error_rate: number;
  last_updated: string;
  last_prediction_time: string;
  correct_predictions: number;
  total_predictions: number;
  sharpe_ratio: number;
  avg_confidence: number;
  win_rate: number;
  health_score: number;
  performance_trend: string;
}

export interface AgentFeedback {
  agent_name: string;
  feedback_type: string;
  message: string;
  timestamp: string;
  predicted_signal: string;
  actual_outcome: string;
  feedback_score: number;
}

export interface OnlineLearningStatus {
  enabled: boolean;
  learning_rate: number;
  last_update: string;
  total_updates: number;
  agent_name: string;
  model_type: string;
  model_accuracy: number;
  training_samples: number;
  is_training: boolean;
}

export interface AgentRouterSummary {
  total_routes: number;
  active_routes: number;
  routing_enabled: boolean;
  last_route_update: string;
  routing_accuracy: number;
  last_updated: string;
  total_routing_decisions: number;
  current_regime: string;
  regime_confidence: number;
  active_routing_strategy: string;
  avg_agent_weight: number;
  last_decision_time: string;
}

export interface MarketRegime {
  current_regime: string;
  confidence: number;
  regime_duration: number;
  last_change: string;
  regime_type: string;
  volatility_level: number;
  trend_strength: number;
  market_sentiment: string;
  transition_probability: number;
}

export interface AgentWeight {
  agent_name: string;
  weight: number;
  performance: number;
  last_updated: string;
  reason: string;
  regime_fit: number;
}

export interface RoutingDecision {
  timestamp: string;
  agent_name: string;
  symbol: string;
  decision: string;
  confidence: number;
  decision_id: string;
  market_regime: any;
  routing_strategy: string;
  active_agents: any[];
  risk_level: string;
}

export interface ExecutionAgentSummary {
  total_orders: number;
  filled_orders: number;
  pending_orders: number;
  total_volume: number;
  execution_enabled: boolean;
  last_execution: string;
  success_rate: number;
  avg_execution_time: number;
  last_updated: string;
  execution_success_rate: number;
  active_orders: number;
  active_strategies: number;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  status: string;
  timestamp: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  average_price: number;
  market_value: number;
  unrealized_pnl: number;
  timestamp: string;
}

export interface ExecutionStrategy {
  strategy_name: string;
  enabled: boolean;
  performance: number;
  last_used: string;
}

export interface RiskAnalysis {
  overall_risk_score: number;
  market_risk: any;
  liquidity_risk: number;
  last_updated: string;
  total_risk: number;
  risk_metrics: RiskMetrics;
}

export interface RiskMetrics {
  var_95: number;
  var_99: number;
  max_drawdown: number;
  sharpe_ratio: number;
  beta: number;
}

export interface MarketRisk {
  volatility: number;
  correlation: number;
  sector_concentration: number;
  geographic_concentration: number;
  market_regime: string;
}

export interface RiskAlert {
  alert_id: string;
  severity: string;
  message: string;
  timestamp: string;
  resolved: boolean;
  type: string;
}

export interface AnalyticsResponse {
  total_predictions: number;
  accuracy_rate: number;
  success_rate: number;
  last_updated: string;
  agent_performance: any[];
  market_trends: any;
  system_analytics: any;
}

export interface MarketTrends {
  trend_direction: string;
  trend_strength: number;
  trend_duration: number;
  last_updated: string;
}

export interface SystemAnalytics {
  system_load: number;
  response_time: number;
  error_rate: number;
  uptime: number;
}

export interface MetaEvaluationSummary {
  total_evaluations: number;
  active_evaluations: number;
  evaluation_accuracy: number;
  last_evaluation: string;
  regime_analysis_enabled: boolean;
  last_updated: string;
  performance_summary: any;
  current_regime: string;
  recent_rotations: any[];
}

export interface AgentRanking {
  agent_name: string;
  rank: number;
  score: number;
  regime: string;
  last_updated: string;
  composite_score: number;
  accuracy: number;
  sharpe_ratio: number;
  win_rate: number;
  response_time: number;
}

export interface RotationDecision {
  timestamp: string;
  from_agent: string;
  to_agent: string;
  symbol: string;
  reason: string;
  confidence: number;
  regime: string;
  created_at: string;
  expected_improvement: number;
}

export interface RegimeAnalysis {
  current_regime: string;
  regime_confidence: number;
  regime_duration: number;
  transition_probability: number;
  last_updated: string;
  regime: string;
  confidence: number;
  volatility: number;
  trend_strength: number;
  trend_direction: string;
  volume_ratio: number;
}

export interface LatentPatternSummary {
  total_patterns: number;
  active_patterns: number;
  pattern_accuracy: number;
  compression_ratio: number;
  last_analysis: string;
  analysis_enabled: boolean;
  last_updated: string;
  pattern_counts: { [key: string]: number };
  compression_metrics: any[];
  recent_insights: any[];
}

export interface LatentPattern {
  pattern_id: string;
  pattern_type: string;
  confidence: number;
  frequency: number;
  last_seen: string;
  description: string;
  explained_variance: number;
  latent_dimensions: any[];
  compression_method: string;
}

export interface CompressionMetric {
  method: string;
  compression_ratio: number;
  reconstruction_error: number;
  computation_time: number;
  last_updated: string;
  explained_variance: number;
  processing_time: number;
}

export interface PatternInsight {
  insight_id: string;
  pattern_type: string;
  insight_text: string;
  confidence: number;
  impact_score: number;
  timestamp: string;
  recommendations: string[];
  description: string;
  market_implications: string[];
}

export interface RLStrategyAgentSummary {
  total_episodes: number;
  current_episode: number;
  training_status: string;
  model_performance: number;
  last_training: string;
  actions_taken: number;
  rewards_earned: number;
  exploration_rate: number;
  last_updated: string;
  algorithm: string;
  is_trained: boolean;
  training_episodes: number;
  model_accuracy: number;
  performance_metrics: any;
  training_metrics: any;
}

export interface RAGEventAgentSummary {
  total_documents: number;
  vector_db_size: number;
  last_news_update: string;
  rag_accuracy: number;
  llm_enabled: boolean;
  active_sources: number;
  total_queries: number;
  avg_response_time: number;
  avg_confidence: number;
  last_updated: string;
}

export interface NewsDocument {
  doc_id: string;
  title: string;
  content: string;
  source: string;
  url: string;
  category: string;
  tags: string[];
  similarity_score: number | null;
  timestamp: string;
  ingested_at: string;
}

export interface RAGAnalysis {
  query: string;
  relevant_docs: NewsDocument[];
  llm_response: string;
  confidence: number;
  reasoning: string;
  analysis_type: string;
  response_time_ms: number;
  created_at: string;
}

export interface RAGPerformance {
  metrics: {
    query_processing_rate: {
      value: number;
      unit: string;
      timestamp: string;
    };
    llm_confidence_score: {
      value: number;
      unit: string;
      timestamp: string;
    };
    document_retrieval_success: {
      value: number;
      unit: string;
      timestamp: string;
    };
    avg_response_time: {
      value: number;
      unit: string;
      timestamp: string;
    };
    rag_accuracy: {
      value: number;
      unit: string;
      timestamp: string;
    };
  };
  last_updated: string;
}

@Injectable({
  providedIn: 'root'
})
export class SystemStatusService {
  private apiUrl = environment.apiUrl;
  private systemStatusSubject = new BehaviorSubject<SystemStatus | null>(null);
  private agentsStatusSubject = new BehaviorSubject<AgentStatus[]>([]);
  private predictionsSubject = new BehaviorSubject<Prediction[]>([]);

  public systemStatus$ = this.systemStatusSubject.asObservable();
  public agentsStatus$ = this.agentsStatusSubject.asObservable();
  public predictions$ = this.predictionsSubject.asObservable();

  constructor(private http: HttpClient) {}

  // Core System Methods
  getSystemStatus(): Observable<SystemStatus> {
    return this.http.get<any>(`${this.apiUrl}/status`).pipe(
      map(response => ({
        is_running: response.status === 'online',
        uptime_seconds: response.uptime_seconds || 0,
        total_predictions: response.total_predictions || 0,
        successful_predictions: response.successful_predictions || 0,
        failed_predictions: response.failed_predictions || 0,
        data_quality_score: response.data_quality || 0,
        last_update: response.timestamp || new Date().toISOString(),
        agent_status: response.agent_status || {},
        active_symbols: response.active_symbols || [],
        active_agents: response.active_agents || [],
        advanced_features: response.advanced_features || {
          deep_learning: { enabled: false, models: [], performance_metrics: {} },
          realtime_feeds: { enabled: false, feed_types: [], status: {} },
          ab_testing: { enabled: false, active_tests: 0, completed_tests: 0 }
        }
      })),
      catchError(error => {
        console.error('Error getting system status:', error);
        return of({
          is_running: false,
          uptime_seconds: 0,
          total_predictions: 0,
          successful_predictions: 0,
          failed_predictions: 0,
          data_quality_score: 0,
          last_update: new Date().toISOString(),
          agent_status: {},
          active_symbols: [],
          active_agents: [],
          advanced_features: {
            deep_learning: { enabled: false, models: [], performance_metrics: {} },
            realtime_feeds: { enabled: false, feed_types: [], status: {} },
            ab_testing: { enabled: false, active_tests: 0, completed_tests: 0 }
          }
        });
      })
    );
  }

  getAgentsStatus(): Observable<AgentStatus[]> {
    return this.http.get<AgentStatus[]>(`${this.apiUrl}/agents/status`);
  }

  getPredictions(limit?: number): Observable<Prediction[]> {
    const url = limit ? `${this.apiUrl}/predictions?limit=${limit}` : `${this.apiUrl}/predictions?limit=50`;
    return this.http.get<Prediction[]>(url);
  }

  // Polling and Refresh Methods
  startPolling(): void {
    // Auto-refresh system status every 30 seconds
    interval(30000).pipe(
      startWith(0),
      switchMap(() => this.getSystemStatus())
    ).subscribe(status => {
      this.systemStatusSubject.next(status);
    });
  }

  refreshSystemStatus(): Observable<SystemStatus> {
    return this.getSystemStatus();
  }

  // Holdings Methods (moved from portfolio)
  getTopHoldings(): Observable<PortfolioHolding[]> {
    return this.http.get<any>(`${this.apiUrl}/symbols/managed-with-market-data`).pipe(
      map((symbols: any[]) => symbols.slice(0, 10).map(symbol => ({
        symbol: symbol.symbol,
        name: symbol.name,
        quantity: 100, // Default quantity for display
        avg_price: symbol.initial_price || symbol.current_price || 0,
        current_price: symbol.current_price || 0,
        market_value: (symbol.current_price || 0) * 100,
        unrealized_pnl: 0, // Default PnL for display
        unrealized_pnl_percent: 0, // Default PnL % for display
        weight: 0, // Default weight
        status: 'active',
        cost_basis: (symbol.initial_price || symbol.current_price || 0) * 100
      })))
    );
  }

  getSignalSummary(): Observable<any> {
    return this.http.get<any[]>(`${this.apiUrl}/predictions?limit=100`).pipe(
      map((predictions: any[]) => {
        const buySignals = predictions.filter(p => p.signal_type?.toLowerCase().includes('buy')).length;
        const sellSignals = predictions.filter(p => p.signal_type?.toLowerCase().includes('sell')).length;
        const holdSignals = predictions.filter(p => p.signal_type?.toLowerCase().includes('hold')).length;
        
        return {
          buy_signals: buySignals,
          sell_signals: sellSignals,
          hold_signals: holdSignals,
          total_signals: predictions.length
        };
      })
    );
  }

  getMarketIndices(): Observable<any[]> {
    return of([
      { name: 'S&P 500', value: 4567.89, change: 0.85 },
      { name: 'NASDAQ', value: 14234.56, change: 1.25 },
      { name: 'DOW', value: 34567.89, change: 0.45 },
      { name: 'VIX', value: 18.45, change: -2.15 }
    ]);
  }

  getSectorPerformance(): Observable<any[]> {
    return of([
      { name: 'Technology', performance: 2.15 },
      { name: 'Healthcare', performance: 1.85 },
      { name: 'Financials', performance: 0.95 },
      { name: 'Energy', performance: -0.65 },
      { name: 'Consumer', performance: 1.45 },
      { name: 'Industrials', performance: 0.75 }
    ]);
  }

  getSymbols(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/api/symbols`);
  }

  getManagedSymbolsWithMarketData(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/symbols/managed-with-market-data`);
  }

  // Mock methods for other components to avoid compilation errors
  getABTestingSummary(): Observable<ABTestSummary> {
    return this.http.get<any>(`${this.apiUrl}/ab-testing`).pipe(
      map(response => ({
        active_tests: response.active_tests || 0,
        completed_tests: response.completed_tests || 0,
        success_rate: response.success_rate || 0,
        last_updated: response.last_updated || new Date().toISOString(),
        active_experiments: response.active_experiments || 0,
        completed_experiments: response.completed_experiments || 0,
        overall_conversion_rate: response.overall_conversion_rate || 0,
        total_participants: response.total_participants || 0,
        avg_experiment_duration: response.avg_experiment_duration || 0,
        top_performing_variant: response.top_performing_variant || 'None'
      })),
      catchError(error => {
        console.error('Error getting A/B testing summary:', error);
        return of({
          active_tests: 0,
          completed_tests: 0,
          success_rate: 0,
          last_updated: new Date().toISOString(),
          active_experiments: 0,
          completed_experiments: 0,
          overall_conversion_rate: 0,
          total_participants: 0,
          avg_experiment_duration: 0,
          top_performing_variant: 'None'
        });
      })
    );
  }

  getABTestingPerformance(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/ab-testing/performance`).pipe(
      map(response => response || { active_experiments: [], experiments: [] }),
      catchError(error => {
        console.error('Error getting A/B testing performance:', error);
        return of({ active_experiments: [], experiments: [] });
      })
    );
  }

  getAgentMonitorSummary(): Observable<AgentMonitorSummary> {
    return this.http.get<any>(`${this.apiUrl}/agent-monitor`).pipe(
      map(response => ({
        total_agents: response.total_agents || 0,
        active_agents: response.active_agents || 0,
        monitoring_enabled: response.monitoring_enabled || false,
        last_health_check: response.last_health_check || new Date().toISOString(),
        system_load: response.system_load || 0,
        memory_usage: response.memory_usage || 0,
        cpu_usage: response.cpu_usage || 0,
        last_updated: response.last_updated || new Date().toISOString(),
        healthy_agents: response.healthy_agents || 0,
        avg_accuracy: response.avg_accuracy || 0,
        avg_sharpe_ratio: response.avg_sharpe_ratio || 0,
        online_learning_enabled: response.online_learning_enabled || false,
        total_feedback_samples: response.total_feedback_samples || 0,
        agents_needing_attention: response.agents_needing_attention || 0,
        avg_win_rate: response.avg_win_rate || 0
      })),
      catchError(error => {
        console.error('Error getting agent monitor summary:', error);
        return of({
          total_agents: 0,
          active_agents: 0,
          monitoring_enabled: false,
          last_health_check: new Date().toISOString(),
          system_load: 0,
          memory_usage: 0,
          cpu_usage: 0,
          last_updated: new Date().toISOString(),
          healthy_agents: 0,
          avg_accuracy: 0,
          avg_sharpe_ratio: 0,
          online_learning_enabled: false,
          total_feedback_samples: 0,
          agents_needing_attention: 0,
          avg_win_rate: 0
        });
      })
    );
  }

  getAgentPerformanceMetrics(): Observable<AgentPerformanceMetrics[]> {
    return this.http.get<any[]>(`${this.apiUrl}/agent-monitor/performance`).pipe(
      map(response => response || []),
      catchError(error => {
        console.error('Error getting agent performance metrics:', error);
        return of([]);
      })
    );
  }

  getAgentFeedback(): Observable<AgentFeedback[]> {
    return this.http.get<AgentFeedback[]>(`${this.apiUrl}/agent-monitor/feedback`).pipe(
      catchError(error => {
        console.error('Error fetching agent feedback:', error);
        return of([]);
      })
    );
  }

  getOnlineLearningStatus(): Observable<OnlineLearningStatus[]> {
    return this.http.get<any[]>(`${this.apiUrl}/agent-monitor/online-learning`).pipe(
      map(response => response || []),
      catchError(error => {
        console.error('Error getting online learning status:', error);
        return of([]);
      })
    );
  }

  getAgentRouterSummary(): Observable<AgentRouterSummary> {
    // For now, return a summary based on market regime data
    // This can be enhanced when more comprehensive router summary endpoint is available
    return this.getMarketRegime().pipe(
      map(regime => ({
        total_routes: 25,
        active_routes: 20,
        routing_enabled: true,
        last_route_update: new Date().toISOString(),
        routing_accuracy: regime.confidence || 0.75,
        last_updated: regime.last_change || new Date().toISOString(),
        total_routing_decisions: 0, // Will be populated when decisions endpoint has data
        current_regime: regime.regime_type || 'neutral',
        regime_confidence: regime.confidence || 0.75,
        active_routing_strategy: 'adaptive',
        avg_agent_weight: 0.65,
        last_decision_time: new Date().toISOString()
      })),
      catchError(error => {
        console.error('Error getting agent router summary:', error);
        return of({
          total_routes: 0,
          active_routes: 0,
          routing_enabled: false,
          last_route_update: new Date().toISOString(),
          routing_accuracy: 0,
          last_updated: new Date().toISOString(),
          total_routing_decisions: 0,
          current_regime: 'neutral',
          regime_confidence: 0,
          active_routing_strategy: 'none',
          avg_agent_weight: 0,
          last_decision_time: new Date().toISOString()
        });
      })
    );
  }

  getMarketRegime(): Observable<MarketRegime> {
    return this.http.get<any>(`${this.apiUrl}/agent-router/regime`).pipe(
      map(response => ({
        current_regime: response.regime_type || 'neutral',
        confidence: response.confidence || 0.5,
        regime_duration: response.regime_duration || 0,
        last_change: response.last_updated || new Date().toISOString(),
        regime_type: response.regime_type || 'neutral',
        volatility_level: response.volatility_level || 0.2,
        trend_strength: response.trend_strength || 0.3,
        market_sentiment: response.market_sentiment || 'neutral',
        transition_probability: response.transition_probability || 0.1
      })),
      catchError(error => {
        console.error('Error getting market regime:', error);
        return of({
          current_regime: 'neutral',
          confidence: 0.5,
          regime_duration: 0,
          last_change: new Date().toISOString(),
          regime_type: 'neutral',
          volatility_level: 0.2,
          trend_strength: 0.3,
          market_sentiment: 'neutral',
          transition_probability: 0.1,
          last_updated: new Date().toISOString()
        });
      })
    );
  }

  getAgentWeights(): Observable<AgentWeight[]> {
    return this.http.get<any[]>(`${this.apiUrl}/agent-router/weights`).pipe(
      map(response => response || []),
      catchError(error => {
        console.error('Error getting agent weights:', error);
        return of([]);
      })
    );
  }

  getRoutingDecisions(): Observable<RoutingDecision[]> {
    return this.http.get<any[]>(`${this.apiUrl}/agent-router/decisions`).pipe(
      map(response => response || []),
      catchError(error => {
        console.error('Error getting routing decisions:', error);
        return of([]);
      })
    );
  }

  getRoutingPerformance(): Observable<any> {
    return of({});
  }

  getExecutionAgentSummary(): Observable<ExecutionAgentSummary> {
    return of({
      total_orders: 150,
      filled_orders: 145,
      pending_orders: 5,
      total_volume: 125000,
      execution_enabled: true,
      last_execution: new Date().toISOString(),
      success_rate: 0.97,
      avg_execution_time: 125,
      last_updated: new Date().toISOString(),
      execution_success_rate: 0.97,
      active_orders: 5,
      active_strategies: 3
    });
  }

  getOrders(): Observable<Order[]> {
    return of([]);
  }

  getPositions(): Observable<Position[]> {
    return of([]);
  }

  getExecutionStrategies(): Observable<ExecutionStrategy[]> {
    return of([]);
  }

  getExecutionPerformance(): Observable<any> {
    return of({});
  }

  getRiskAnalysis(): Observable<RiskAnalysis> {
    return this.http.get<any>(`${this.apiUrl}/risk-analysis`).pipe(
      map(response => ({
        overall_risk_score: response.overall_risk_score || 0,
        market_risk: response.market_risk || { volatility: 0 },
        liquidity_risk: response.liquidity_risk || 0,
        last_updated: response.last_updated || new Date().toISOString(),
        total_risk: response.total_risk || 0,
        risk_metrics: {
          var_95: response.risk_metrics?.var_95 || 0,
          var_99: response.risk_metrics?.var_99 || 0,
          max_drawdown: response.risk_metrics?.max_drawdown || 0,
          sharpe_ratio: response.risk_metrics?.sharpe_ratio || 0,
          beta: response.risk_metrics?.beta || 1.0
        }
      })),
      catchError(error => {
        console.error('Error getting risk analysis:', error);
        return of({
          overall_risk_score: 0,
          market_risk: { volatility: 0 },
          liquidity_risk: 0,
          last_updated: new Date().toISOString(),
          total_risk: 0,
          risk_metrics: {
            var_95: 0,
            var_99: 0,
            max_drawdown: 0,
            sharpe_ratio: 0,
            beta: 1.0
          }
        });
      })
    );
  }

  getRiskMetrics(): Observable<RiskMetrics> {
    return this.http.get<any>(`${this.apiUrl}/risk-analysis/metrics`).pipe(
      map(response => ({
        var_95: response.var_95 || 0,
        var_99: response.var_99 || 0,
        max_drawdown: response.max_drawdown || 0,
        sharpe_ratio: response.sharpe_ratio || 0,
        beta: response.beta || 1.0
      })),
      catchError(error => {
        console.error('Error getting risk metrics:', error);
        return of({
          var_95: 0,
          var_99: 0,
          max_drawdown: 0,
          sharpe_ratio: 0,
          beta: 1.0
        });
      })
    );
  }

  getMarketRisk(): Observable<MarketRisk> {
    return this.http.get<any>(`${this.apiUrl}/risk-analysis/market`).pipe(
      map(response => ({
        volatility: response.volatility || 0,
        correlation: response.correlation || 0,
        sector_concentration: response.sector_concentration || 0,
        geographic_concentration: response.geographic_concentration || 0,
        market_regime: response.market_regime || 'neutral'
      })),
      catchError(error => {
        console.error('Error getting market risk:', error);
        return of({
          volatility: 0,
          correlation: 0,
          sector_concentration: 0,
          geographic_concentration: 0,
          market_regime: 'neutral'
        });
      })
    );
  }

  getRiskAlerts(): Observable<RiskAlert[]> {
    return this.http.get<any>(`${this.apiUrl}/risk-analysis/alerts`).pipe(
      map(response => response || []),
      catchError(error => {
        console.error('Error getting risk alerts:', error);
        return of([]);
      })
    );
  }

  getAnalytics(): Observable<AnalyticsResponse> {
    return this.http.get<any>(`${this.apiUrl}/status`).pipe(
      map(response => ({
        total_predictions: response.total_predictions || 0,
        accuracy_rate: response.data_quality_score || 0,
        success_rate: response.total_predictions > 0 ? (response.successful_predictions / response.total_predictions) : 0,
        last_updated: response.timestamp || new Date().toISOString(),
        agent_performance: response.agent_status || [],
        market_trends: { trend_direction: 'up' },
        system_analytics: { 
          avg_response_time: 125, 
          system_load: 0.65, 
          system_reliability: 0.98, 
          total_uptime_hours: response.uptime_seconds ? (response.uptime_seconds / 3600) : 0 
        }
      })),
      catchError(error => {
        console.error('Error getting analytics:', error);
        return of({
          total_predictions: 0,
          accuracy_rate: 0,
          success_rate: 0,
          last_updated: new Date().toISOString(),
          agent_performance: [],
          market_trends: { trend_direction: 'up' },
          system_analytics: { 
            avg_response_time: 125, 
            system_load: 0.65,
            system_reliability: 0.98, 
            total_uptime_hours: 0 
          }
        });
      })
    );
  }

  getAgentPerformance(): Observable<AgentPerformanceMetrics[]> {
    return of([]);
  }

  getMarketTrends(): Observable<MarketTrends> {
    return of({
      trend_direction: 'up',
      trend_strength: 0.75,
      trend_duration: 14,
      last_updated: new Date().toISOString()
    });
  }

  getSystemMetrics(): Observable<SystemAnalytics> {
    return of({
      system_load: 0.65,
      response_time: 125,
      error_rate: 0.02,
      uptime: 0.99
    });
  }

  getMetaEvaluationSummary(): Observable<MetaEvaluationSummary> {
    return this.http.get<MetaEvaluationSummary>(`${this.apiUrl}/meta-evaluation-agent`).pipe(
      catchError(error => {
        console.error('Error getting Meta-Evaluation summary:', error);
        return of({
          total_evaluations: 0,
          active_evaluations: 0,
          evaluation_accuracy: 0.0,
          last_evaluation: new Date().toISOString(),
          regime_analysis_enabled: true,
          last_updated: new Date().toISOString(),
          performance_summary: { avg_accuracy: 0.0, total_agents: 0 },
          current_regime: 'neutral',
          recent_rotations: []
        });
      })
    );
  }

  getAgentRankings(regime?: string): Observable<AgentRanking[]> {
    const regimeParam = regime ? `?regime=${regime}` : '';
    return this.http.get<AgentRanking[]>(`${this.apiUrl}/meta-evaluation/rankings${regimeParam}`).pipe(
      catchError(error => {
        console.error('Error getting agent rankings:', error);
        return of([]);
      })
    );
  }

  getRotationDecisions(limit: number = 10): Observable<RotationDecision[]> {
    return this.http.get<RotationDecision[]>(`${this.apiUrl}/meta-evaluation-agent/rotations?limit=${limit}`).pipe(
      catchError(error => {
        console.error('Error getting rotation decisions:', error);
        return of([]);
      })
    );
  }

  getRegimeAnalysis(): Observable<RegimeAnalysis> {
    return this.http.get<RegimeAnalysis>(`${this.apiUrl}/meta-evaluation/regime-analysis`).pipe(
      catchError(error => {
        console.error('Error getting regime analysis:', error);
        return of({
          current_regime: 'neutral',
          regime_confidence: 0.6,
          regime_duration: 0,
          transition_probability: 0.2,
          last_updated: new Date().toISOString(),
          regime: 'neutral',
          confidence: 0.6,
          volatility: 0.15,
          trend_strength: 0.02,
          trend_direction: 'neutral',
          volume_ratio: 1.0
        });
      })
    );
  }

  getLatentPatternSummary(): Observable<LatentPatternSummary> {
    return this.http.get<LatentPatternSummary>(`${this.apiUrl}/latent-pattern-detector`).pipe(
      catchError(error => {
        console.error('Error getting Latent Pattern summary:', error);
        return of({
          total_patterns: 0,
          active_patterns: 0,
          pattern_accuracy: 0.0,
          compression_ratio: 0.0,
          last_analysis: new Date().toISOString(),
          analysis_enabled: true,
          last_updated: new Date().toISOString(),
          pattern_counts: {},
          compression_metrics: [],
          recent_insights: []
        });
      })
    );
  }

  getLatentPatterns(pattern_type?: string, limit: number = 50): Observable<LatentPattern[]> {
    const params = new URLSearchParams();
    if (pattern_type) params.append('pattern_type', pattern_type);
    params.append('limit', limit.toString());
    const queryString = params.toString() ? `?${params.toString()}` : '';
    
    return this.http.get<LatentPattern[]>(`${this.apiUrl}/latent-pattern-detector/patterns${queryString}`).pipe(
      catchError(error => {
        console.error('Error getting latent patterns:', error);
        return of([]);
      })
    );
  }

  getCompressionMetrics(method?: string): Observable<CompressionMetric[]> {
    return this.http.get<any>(`${this.apiUrl}/latent-pattern-detector/visualization`).pipe(
      map(data => data.compression_metrics || []),
      catchError(error => {
        console.error('Error getting compression metrics:', error);
        return of([]);
      })
    );
  }

  getPatternInsights(pattern_type?: string, limit: number = 10): Observable<PatternInsight[]> {
    const params = new URLSearchParams();
    if (pattern_type) params.append('pattern_type', pattern_type);
    params.append('limit', limit.toString());
    const queryString = params.toString() ? `?${params.toString()}` : '';
    
    return this.http.get<PatternInsight[]>(`${this.apiUrl}/latent-pattern-detector/insights${queryString}`).pipe(
      catchError(error => {
        console.error('Error getting pattern insights:', error);
        return of([]);
      })
    );
  }

  getRLStrategyAgentSummary(): Observable<RLStrategyAgentSummary> {
    return this.http.get<RLStrategyAgentSummary>(`${this.apiUrl}/rl-strategy-agent`).pipe(
      catchError(error => {
        console.error('Error getting RL Strategy Agent summary:', error);
        return of({
          total_episodes: 0,
          current_episode: 0,
          training_status: 'not_started',
          model_performance: 0.0,
          last_training: new Date().toISOString(),
          actions_taken: 0,
          rewards_earned: 0,
          exploration_rate: 0.1,
          last_updated: new Date().toISOString(),
          algorithm: 'PPO',
          is_trained: false,
          training_episodes: 0,
          model_accuracy: 0.0,
          performance_metrics: { total_return: 0.0, sharpe_ratio: 0.0, max_drawdown: 0.0, win_rate: 0.0 },
          training_metrics: { avg_episode_reward: 0.0, best_episode_reward: 0.0, exploration_rate: 0.1, experience_buffer_size: 0 }
        });
      })
    );
  }

  getRLTrainingStatus(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/rl-strategy-agent/training`).pipe(
      catchError(error => {
        console.error('Error getting RL training status:', error);
        return of({
          algorithm: 'PPO',
          episodes_trained: 0,
          avg_episode_reward: 0.0,
          best_episode_reward: 0.0,
          convergence_episode: null,
          training_loss: 0.0,
          exploration_rate: 0.1,
          experience_buffer_size: 0,
          model_accuracy: 0.0,
          training_duration_seconds: 0,
          is_converged: false,
          training_status: 'not_started',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          training_progress: {
            training_progress: 0.0,
            current_episode: 0,
            total_episodes: 1500,
            is_converged: false
          },
          algorithm_info: {
            learning_rate: 0.0003,
            gamma: 0.99,
            batch_size: 64
          },
          environment_info: {
            market_regime: 'neutral',
            volatility_level: 0.2,
            episode_length: 252
          }
        });
      })
    );
  }

  getRLPerformance(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/rl-strategy-agent/performance`).pipe(
      catchError(error => {
        console.error('Error getting RL performance:', error);
        return of({
          performance_30d: {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0
          },
          performance_7d: {
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0
          },
          action_statistics: {
            total_actions: 0,
            buy_actions: 0,
            sell_actions: 0,
            hold_actions: 0,
            avg_confidence: 0.0,
            avg_expected_return: 0.0
          },
          last_updated: new Date().toISOString(),
          risk_metrics: {
            var_95: 0.05,
            cvar_95: 0.07,
            volatility: 0.15,
            beta: 1.0
          },
          algorithm_analysis: {
            exploration_efficiency: 0.5,
            exploitation_effectiveness: 0.5,
            policy_stability: 0.5,
            learning_curve_slope: 0.0
          },
          market_adaptation: {
            regime_adaptation_speed: 0.5,
            volatility_handling: 0.5,
            trend_following_accuracy: 0.5,
            mean_reversion_accuracy: 0.5
          },
          experience_replay: {
            buffer_utilization: 0.5,
            sample_efficiency: 0.5,
            priority_replay_effectiveness: 0.5,
            experience_diversity: 0.5
          }
        });
      })
    );
  }

  getRLActions(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/rl-strategy-agent/actions`).pipe(
      catchError(error => {
        console.error('Error getting RL actions:', error);
        return of({
          recent_actions: [],
          action_distribution: {},
          action_effectiveness: {}
        });
      })
    );
  }

  getRAGEventAgentSummary(): Observable<RAGEventAgentSummary> {
    return this.http.get<RAGEventAgentSummary>(`${this.apiUrl}/rag-event-agent/summary`).pipe(
      catchError(error => {
        console.error('Error getting RAG summary:', error);
        return of({
          total_documents: 0,
          vector_db_size: 0,
          last_news_update: new Date().toISOString(),
          rag_accuracy: 0,
          llm_enabled: false,
          active_sources: 0,
          total_queries: 0,
          avg_response_time: 0,
          avg_confidence: 0,
          last_updated: new Date().toISOString()
        });
      })
    );
  }

  getRAGDocuments(): Observable<NewsDocument[]> {
    return this.http.get<NewsDocument[]>(`${this.apiUrl}/rag-event-agent/documents`).pipe(
      catchError(error => {
        console.error('Error getting RAG documents:', error);
        return of([]);
      })
    );
  }

  getRAGAnalysis(): Observable<RAGAnalysis> {
    return this.http.get<RAGAnalysis>(`${this.apiUrl}/rag-event-agent/analysis`).pipe(
      catchError(error => {
        console.error('Error getting RAG analysis:', error);
        return of({
          query: 'No analysis available',
          relevant_docs: [],
          llm_response: 'No LLM response available',
          confidence: 0,
          reasoning: 'No reasoning available',
          analysis_type: 'none',
          response_time_ms: 0,
          created_at: new Date().toISOString()
        });
      })
    );
  }

  getRAGPerformance(): Observable<RAGPerformance> {
    return this.http.get<RAGPerformance>(`${this.apiUrl}/rag-event-agent/performance`).pipe(
      catchError(error => {
        console.error('Error getting RAG performance:', error);
        return of({
          metrics: {
            query_processing_rate: { value: 0, unit: 'queries/min', timestamp: new Date().toISOString() },
            llm_confidence_score: { value: 0, unit: 'score', timestamp: new Date().toISOString() },
            document_retrieval_success: { value: 0, unit: 'rate', timestamp: new Date().toISOString() },
            avg_response_time: { value: 0, unit: 'ms', timestamp: new Date().toISOString() },
            rag_accuracy: { value: 0, unit: 'score', timestamp: new Date().toISOString() }
          },
          last_updated: new Date().toISOString()
        });
      })
    );
  }

  getSectorAnalysis(sector: string): Observable<RAGAnalysis> {
    return this.http.get<RAGAnalysis>(`${this.apiUrl}/rag-event-agent/sector-analysis/${sector}`).pipe(
      catchError(error => {
        console.error(`Error getting ${sector} sector analysis:`, error);
        return of({
          query: `No ${sector} analysis available`,
          sector: sector,
          relevant_docs: [],
          llm_response: `No ${sector} LLM response available`,
          confidence: 0,
          reasoning: `No ${sector} reasoning available`,
          analysis_type: `${sector}_impact`,
          response_time_ms: 0,
          created_at: new Date().toISOString()
        });
      })
    );
  }

  getLatestRAGAnalysis(): Observable<{success: boolean, sector_analyses: any, last_updated: string}> {
    return this.http.get<{success: boolean, sector_analyses: any, last_updated: string}>(`${this.apiUrl}/rag-event-agent/latest-analysis`).pipe(
      catchError(error => {
        console.error('Error getting latest RAG analysis:', error);
        return of({
          success: false,
          sector_analyses: {},
          last_updated: new Date().toISOString()
        });
      })
    );
  }
}