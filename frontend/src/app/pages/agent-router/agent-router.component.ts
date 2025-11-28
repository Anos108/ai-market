import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SystemStatusService, AgentRouterSummary, MarketRegime, AgentWeight, RoutingDecision } from '../../services/system-status.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-agent-router',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './agent-router.component.html',
  styles: []
})
export class AgentRouterComponent implements OnInit {
  agentRouterSummary$: Observable<AgentRouterSummary> | undefined;
  marketRegime$: Observable<MarketRegime> | undefined;
  agentWeights$: Observable<AgentWeight[]> | undefined;
  routingDecisions$: Observable<RoutingDecision[]> | undefined;

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.agentRouterSummary$ = this.systemStatusService.getAgentRouterSummary();
    this.marketRegime$ = this.systemStatusService.getMarketRegime();
    this.agentWeights$ = this.systemStatusService.getAgentWeights();
    this.routingDecisions$ = this.systemStatusService.getRoutingDecisions();
  }

  getRegimeColor(regime: string): string {
    switch (regime.toLowerCase()) {
      case 'bull':
        return 'bg-green-100 text-green-800';
      case 'bear':
        return 'bg-red-100 text-red-800';
      case 'sideways':
        return 'bg-gray-100 text-gray-800';
      case 'volatile':
        return 'bg-yellow-100 text-yellow-800';
      case 'trending':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getVolatilityLevel(volatility: number | string): string {
    const vol = typeof volatility === 'string' ? parseFloat(volatility) : volatility;
    if (vol < 0.2) return 'low';
    if (vol < 0.4) return 'medium';
    return 'high';
  }

  getVolatilityColor(volatility: string): string {
    switch (volatility.toLowerCase()) {
      case 'low':
        return 'bg-green-100 text-green-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'high':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getSentimentColor(sentiment: string): string {
    switch (sentiment.toLowerCase()) {
      case 'bullish':
        return 'bg-green-100 text-green-800';
      case 'bearish':
        return 'bg-red-100 text-red-800';
      case 'neutral':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  getRiskColor(risk: string): string {
    switch (risk.toLowerCase()) {
      case 'low':
        return 'bg-green-100 text-green-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'high':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  formatNumber(value: number): string {
    return new Intl.NumberFormat('en-US').format(value);
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }
}