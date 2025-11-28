import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: '/dashboard',
    pathMatch: 'full'
  },
  {
    path: 'dashboard',
    loadComponent: () => import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent)
  },
  {
    path: 'system-status',
    loadComponent: () => import('./pages/system-status/system-status.component').then(m => m.SystemStatusComponent)
  },
  {
    path: 'predictions',
    loadComponent: () => import('./pages/predictions/predictions.component').then(m => m.PredictionsComponent)
  },
  {
    path: 'agents',
    loadComponent: () => import('./pages/agents/agents.component').then(m => m.AgentsComponent)
  },
  {
    path: 'analytics',
    loadComponent: () => import('./pages/analytics/analytics.component').then(m => m.AnalyticsComponent)
  },
  {
    path: 'risk-analysis',
    loadComponent: () => import('./pages/risk-analysis/risk.component').then(m => m.RiskComponent)
  },
  {
    path: 'ab-testing',
    loadComponent: () => import('./pages/ab-testing/ab-testing.component').then(m => m.AbTestingComponent)
  },
  {
    path: 'agent-monitor',
    loadComponent: () => import('./pages/agent-monitor/agent-monitor.component').then(m => m.AgentMonitorComponent)
  },
  {
    path: 'agent-router',
    loadComponent: () => import('./pages/agent-router/agent-router.component').then(m => m.AgentRouterComponent)
  },
  {
    path: 'execution-agent',
    loadComponent: () => import('./pages/execution-agent/execution-agent.component').then(m => m.ExecutionAgentComponent)
  },
      {
        path: 'rag-event-agent',
        loadComponent: () => import('./pages/rag-event-agent/rag-event-agent.component').then(m => m.RAGEventAgentComponent)
      },
      {
        path: 'rl-strategy-agent',
        loadComponent: () => import('./pages/rl-strategy-agent/rl-strategy-agent.component').then(m => m.RLStrategyAgentComponent)
      },
      {
        path: 'meta-evaluation-agent',
        loadComponent: () => import('./pages/meta-evaluation-agent/meta-evaluation-agent.component').then(m => m.MetaEvaluationAgentComponent)
      },
      {
        path: 'latent-pattern-detector',
        loadComponent: () => import('./pages/latent-pattern-detector/latent-pattern-detector.component').then(m => m.LatentPatternDetectorComponent)
      },
  {
    path: 'ensemble-blender',
    loadComponent: () => import('./pages/ensemble-blender/ensemble-blender.component').then(m => m.EnsembleBlenderComponent)
  },
    {
      path: 'forecasting-dashboard',
      loadComponent: () => import('./pages/forecasting-dashboard/forecasting-dashboard.component').then(m => m.ForecastingDashboardComponent)
    },
  {
    path: 'ticker-discovery',
    loadComponent: () => import('./pages/ticker-discovery/ticker-discovery.component').then(m => m.TickerDiscoveryComponent)
  },
  {
    path: 'reports',
    loadComponent: () => import('./pages/reports/reports.component').then(m => m.ReportsComponent)
  },
  {
    path: 'symbol-management',
    loadComponent: () => import('./pages/symbol-management/symbol-management.component').then(m => m.SymbolManagementComponent)
  },
  {
    path: '**',
    redirectTo: '/dashboard'
  }
];
