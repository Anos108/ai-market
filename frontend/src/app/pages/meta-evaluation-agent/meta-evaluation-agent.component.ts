import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Observable } from 'rxjs';
import { SystemStatusService, MetaEvaluationSummary, AgentRanking, RotationDecision, RegimeAnalysis } from '../../services/system-status.service';

@Component({
  selector: 'app-meta-evaluation-agent',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './meta-evaluation-agent.component.html',
  styles: []
})
export class MetaEvaluationAgentComponent implements OnInit {
  metaEvaluationSummary$: Observable<MetaEvaluationSummary> | undefined;
  agentRankings$: Observable<AgentRanking[]> | undefined;
  rotationDecisions$: Observable<RotationDecision[]> | undefined;
  regimeAnalysis$: Observable<RegimeAnalysis> | undefined;

  // Loading states
  isLoadingSummary = true;
  isLoadingRankings = true;
  isLoadingRotations = true;
  isLoadingRegime = true;

  Object = Object; // Make Object available in template

  constructor(private systemStatusService: SystemStatusService) {}

  ngOnInit() {
    this.loadMetaEvaluationData();
  }

  loadMetaEvaluationData() {
    // Load summary
    this.metaEvaluationSummary$ = this.systemStatusService.getMetaEvaluationSummary();
    this.metaEvaluationSummary$.subscribe({
      next: () => this.isLoadingSummary = false,
      error: () => this.isLoadingSummary = false
    });

    // Load rankings
    this.agentRankings$ = this.systemStatusService.getAgentRankings();
    this.agentRankings$.subscribe({
      next: () => this.isLoadingRankings = false,
      error: () => this.isLoadingRankings = false
    });

    // Load rotations
    this.rotationDecisions$ = this.systemStatusService.getRotationDecisions();
    this.rotationDecisions$.subscribe({
      next: () => this.isLoadingRotations = false,
      error: () => this.isLoadingRotations = false
    });

    // Load regime analysis
    this.regimeAnalysis$ = this.systemStatusService.getRegimeAnalysis();
    this.regimeAnalysis$.subscribe({
      next: () => this.isLoadingRegime = false,
      error: () => this.isLoadingRegime = false
    });
  }

  getRegimeColor(regime: string): string {
    switch (regime.toLowerCase()) {
      case 'bull': return 'text-green-600 bg-green-100';
      case 'bear': return 'text-red-600 bg-red-100';
      case 'volatile': return 'text-yellow-600 bg-yellow-100';
      case 'trending': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  }

  getPerformanceColor(score: number): string {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  }

  getRankBadgeColor(rank: number): string {
    if (rank === 1) return 'bg-yellow-100 text-yellow-800';
    if (rank === 2) return 'bg-gray-100 text-gray-800';
    if (rank === 3) return 'bg-orange-100 text-orange-800';
    return 'bg-blue-100 text-blue-800';
  }
}
